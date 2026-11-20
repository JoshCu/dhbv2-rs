//! δHBV 2.0 (daily), a port of `dhbv2.bmi.DeltaModelBmi` for inference.
//!
//! Hourly forcing is aggregated to calendar days. Once `rho + 1` days (default 366) are
//! buffered, the model is warmed up over all of them in one pass; after that each completed day
//! advances the LSTM and HBV states by one step. The daily prediction is reported for each of the
//! following 24 hours.

use std::path::Path;

use crate::BmiError;
use crate::bmi::{Engine, HourlyForcing, StaticAttributes};
use crate::config::{self, BmiConfig, DateTime};
use crate::hbv::{self, Catchment, HBV2_PARAMS, ParamLayout, Params, States};
use crate::nn::{LstmState, Mlp, ReluLstm};
use crate::ring::RingBuffer;
use crate::weights::Checkpoint;
use crate::{FORCINGS, Normalizer, aggregate_day, normalize_static, pairwise_sum};

/// AnnModel layer names in δMG (`ann.py`).
const ANN_LAYERS: [&str; 8] = ["i2h", "h2h1", "h2h2", "h2h3", "h2h4", "h2h5", "h2h6", "h2o"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PetMethod {
    /// Hourly Penman-Monteith, summed over the day.
    PenmanMonteith,
    /// Daily Hargreaves from the day's min/max/mean temperature.
    Hargreaves {
        latitude_rad: f64,
        /// Days since 1970-01-01 of the first complete calendar day.
        first_day: i64,
    },
}

pub struct DailyNetworks {
    pub lstm: ReluLstm,
    pub mlp: Mlp,
}

impl DailyNetworks {
    pub fn load(path: &Path, n_in: usize, hidden: usize) -> Result<Self, BmiError> {
        let mut ckpt = Checkpoint::open(path)?;
        let lstm = ReluLstm {
            linear_in: ckpt.dense("nn_model.lstminv.linear_in", n_in, hidden)?,
            lstm: ckpt.dmg_lstm("nn_model.lstminv.lstm", hidden, hidden)?,
            linear_out: ckpt.dense_any("nn_model.lstminv.linear_out")?,
        };
        let mlp = ckpt.named_mlp("nn_model.ann", &ANN_LAYERS)?;
        ckpt.finish()?;
        Ok(Self { lstm, mlp })
    }
}

pub struct DailyEngine {
    lstm: ReluLstm,
    layout: ParamLayout,
    static_params: Vec<Params>,
    nearzero: f32,
    norm: Normalizer,
    c_nn_norm: Vec<f32>,
    catchment: Catchment,
    pet_method: PetMethod,
    rho: usize,

    daily_buffer: RingBuffer,
    day_accumulator: [[f32; 3]; 24],
    start_hour: u32,
    day_aligned: bool,
    day_count: i64,
    timestep: u64,
    is_warm: bool,

    lstm_state: Option<LstmState>,
    states: Option<Vec<States>>,
    /// Latest daily streamflow (mm/d).
    prediction: Option<f32>,
}

impl DailyEngine {
    pub fn is_warm(&self) -> bool {
        self.is_warm
    }

    fn aggregate(&self) -> [f32; 3] {
        let (prcp, temp, pet) = aggregate_day(&self.day_accumulator);
        let pet = match self.pet_method {
            PetMethod::PenmanMonteith => pet,
            PetMethod::Hargreaves {
                latitude_rad,
                first_day,
            } => {
                let temps: Vec<f32> = self.day_accumulator.iter().map(|r| r[1]).collect();
                let tmin = temps.iter().copied().fold(f32::INFINITY, f32::min);
                let tmax = temps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let tmean = pairwise_sum(&temps) / 24.0;
                let doy = config::day_of_year(first_day + self.day_count) as f64;
                hargreaves_f32_mix(tmin, tmax, tmean, latitude_rad, doy) as f32
            }
        };
        [prcp, temp, pet]
    }

    /// Runs the LSTM and HBV over `rows` from the cached states; returns the last day's
    /// streamflow (mm/d).
    fn forward(&mut self, rows: &[f32]) -> f32 {
        let n_in = FORCINGS.len() + self.c_nn_norm.len();
        let mut x = Vec::with_capacity(rows.len() / 3 * n_in);
        for row in rows.chunks_exact(3) {
            x.extend(self.norm.apply(row));
            x.extend_from_slice(&self.c_nn_norm);
        }
        let mut lstm_state = self
            .lstm_state
            .take()
            .unwrap_or_else(|| LstmState::zeros(self.lstm.hidden()));
        let dynamic = self.lstm.run(&x, n_in, &mut lstm_state);
        self.lstm_state = Some(lstm_state);

        let mut states = self
            .states
            .take()
            .unwrap_or_else(|| vec![States::default(); self.layout.nmul]);
        let mut params = self.static_params.clone();
        let mut q = vec![0.0f32; self.layout.nmul];
        let mut streamflow = 0.0;
        for (row, dyn_t) in rows
            .chunks_exact(3)
            .zip(dynamic.chunks_exact(self.layout.n_dynamic()))
        {
            self.layout.apply_dynamic(dyn_t, &mut params);
            for ((s, p), q) in states.iter_mut().zip(&params).zip(q.iter_mut()) {
                *q = hbv::hbv2_step(s, p, &self.catchment, row[0], row[1], row[2], self.nearzero);
            }
            streamflow = hbv::lane_mean(&q);
        }
        self.states = Some(states);
        streamflow
    }
}

/// `hargreaves_pet` as NumPy evaluates it on the BMI's f32 temperatures: the temperature terms
/// stay in f32 until they meet the f64 extraterrestrial radiation.
fn hargreaves_f32_mix(tmin: f32, tmax: f32, tmean: f32, lat: f64, doy: f64) -> f64 {
    let trange = (tmax - tmin).max(0.0);
    let temp_term = 0.0023f32 * (tmean + 17.8f32) * trange.sqrt() * 0.408f32;
    (temp_term as f64 * crate::pet::extraterrestrial_radiation(lat, doy)).max(0.0)
}

impl Engine for DailyEngine {
    const COMPONENT_NAME: &'static str = "δHBV2.0 Daily";

    fn initialize(config: &BmiConfig, statics: &StaticAttributes) -> Result<Self, BmiError> {
        let model_dir = config.model_dir()?;
        let model_config = config::read_yaml(&model_dir.join("config.yaml"))?;

        let start = DateTime::parse(
            &config
                .string("start_time")?
                .ok_or("'start_time' (YYYY/MM/DD HH[[:MM]:SS]) is required in BMI config.")?,
        )?;

        let pet_method = match config
            .string("pet_method")?
            .as_deref()
            .unwrap_or("penman_monteith")
        {
            "penman_monteith" => PetMethod::PenmanMonteith,
            "hargreaves" => {
                let latitude = config
                    .float("latitude")?
                    .ok_or("'latitude' (degrees) is required in BMI config when pet_method is 'hargreaves'.")?;
                let first_day = start.days_from_epoch() + if start.hour > 0 { 1 } else { 0 };
                PetMethod::Hargreaves {
                    latitude_rad: latitude.to_radians(),
                    first_day,
                }
            }
            other => {
                return Err(format!(
                    "Unknown pet_method '{other}'. Must be 'penman_monteith' or 'hargreaves'."
                )
                .into());
            }
        };

        let forcings = config::req_str_list(&model_config, &["model", "phy", "forcings"])?;
        let nn_forcings = config::req_str_list(&model_config, &["model", "nn", "forcings"])?;
        if forcings != FORCINGS || nn_forcings != FORCINGS {
            return Err(format!("unsupported forcings {forcings:?}; expected {FORCINGS:?}").into());
        }
        let attributes = config::req_str_list(&model_config, &["model", "nn", "attributes"])?;
        let rho = config::get(&model_config, &["model", "rho"])
            .and_then(|v| v.as_u64())
            .unwrap_or(365) as usize;

        let phy_name = config::get(&model_config, &["model", "phy", "name"])
            .and_then(|v| v.as_sequence())
            .and_then(|s| s.first())
            .and_then(|v| v.as_str())
            .unwrap_or("Hbv_2")
            .to_string();
        if phy_name != "Hbv_2" {
            return Err(format!(
                "unsupported physics model '{phy_name}'; only Hbv_2 (δHBV 2.0) is implemented"
            )
            .into());
        }
        if config::opt_bool(&model_config, &["model", "phy", "routing"], false)? {
            return Err("routing is not supported by the daily BMI".into());
        }
        let nmul = config::get(&model_config, &["model", "phy", "nmul"])
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let dynamic =
            config::req_str_list(&model_config, &["model", "phy", "dynamic_params", "Hbv_2"])?;
        let layout = ParamLayout::new(&HBV2_PARAMS, &dynamic, nmul)?;
        let nearzero = config::get(&model_config, &["model", "phy", "nearzero"])
            .and_then(config::as_float)
            .unwrap_or(1e-5) as f32;

        let epoch = config::get(&model_config, &["test", "test_epoch"]).and_then(|v| v.as_u64());
        let weights = config::find_weights(&model_dir, &phy_name.to_lowercase(), epoch)?;
        let n_in = FORCINGS.len() + attributes.len();
        let nets = DailyNetworks::load(
            &weights,
            n_in,
            config::req_usize(&model_config, &["model", "nn", "lstm_hidden_size"])?,
        )?;
        if nets.lstm.linear_out.d_out != layout.n_dynamic() {
            return Err("LSTM output size does not match the dynamic parameters".into());
        }

        // Normalization: each entry is [min, max, mean, std].
        let stats = config::read_json(&model_dir.join("normalization_statistics.json"))?;
        let stat = |name: &str| -> Result<(f64, f64), BmiError> {
            let entry =
                config::json_f64_list(&stats[name], &format!("normalization_statistics.{name}"))?;
            if entry.len() != 4 {
                return Err(format!(
                    "normalization_statistics.{name} must be [min, max, mean, std]"
                )
                .into());
            }
            Ok((entry[2], entry[3]))
        };
        let forcing_stats = FORCINGS
            .iter()
            .map(|f| stat(f))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = Normalizer::f64_stats(
            &forcing_stats.iter().map(|s| s.0).collect::<Vec<_>>(),
            &forcing_stats.iter().map(|s| s.1).collect::<Vec<_>>(),
        );
        let attr_stats = attributes
            .iter()
            .map(|a| stat(a))
            .collect::<Result<Vec<_>, _>>()?;
        let attr_values = attributes
            .iter()
            .map(|a| statics.get(a))
            .collect::<Result<Vec<_>, _>>()?;
        let c_nn_norm = normalize_static(
            &attr_values,
            &attr_stats.iter().map(|s| s.0).collect::<Vec<_>>(),
            &attr_stats.iter().map(|s| s.1).collect::<Vec<_>>(),
        )?;

        // Static parameters depend only on the attributes: run the (large) MLP once.
        let raw = nets.mlp.forward(&c_nn_norm);
        if raw.len() < layout.n_static() {
            return Err(
                "static-parameter MLP output is smaller than the physics model needs".into(),
            );
        }
        // A checkpoint trained with routing has extra trailing outputs, which δMG drops.
        let static_params = layout.static_params(&raw[..layout.n_static()]);

        Ok(Self {
            lstm: nets.lstm,
            layout,
            static_params,
            nearzero,
            norm,
            c_nn_norm,
            catchment: Catchment {
                ac: statics.get("uparea")? as f32,
                elevation: statics.get("meanelevation")? as f32,
            },
            pet_method,
            rho,
            daily_buffer: RingBuffer::new(rho + 1, 3),
            day_accumulator: [[0.0; 3]; 24],
            start_hour: start.hour,
            day_aligned: start.hour == 0,
            day_count: 0,
            timestep: 0,
            is_warm: false,
            lstm_state: None,
            states: None,
            prediction: None,
        })
    }

    fn update(&mut self, forcing: &HourlyForcing) -> Result<f64, BmiError> {
        let hour = ((self.start_hour as u64 + self.timestep) % 24) as usize;
        let pet = match self.pet_method {
            PetMethod::PenmanMonteith => forcing.penman_monteith_pet() as f32,
            PetMethod::Hargreaves { .. } => 0.0,
        };
        self.day_accumulator[hour] = [forcing.precip as f32, forcing.temp as f32, pet];

        if hour == 23 {
            if self.day_aligned {
                let day = self.aggregate();
                self.daily_buffer.push(&day);
                self.day_count += 1;

                if self.is_warm || self.daily_buffer.len() > self.rho {
                    let rows = if self.is_warm {
                        self.daily_buffer.last()
                    } else {
                        self.daily_buffer.ordered()
                    };
                    self.prediction = Some(self.forward(&rows));
                    self.is_warm = true;
                }
            } else {
                // First partial day (start hour != 0): discard and align.
                self.day_aligned = true;
            }
        }
        self.timestep += 1;

        Ok(match (self.is_warm, self.prediction) {
            // mm d-1 -> m h-1
            (true, Some(q)) => q as f64 / 1000.0 / 24.0,
            _ => 0.0,
        })
    }
}
