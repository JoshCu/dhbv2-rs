//! δHBV 2.1 MTS (hourly), a port of `dhbv2.mts_bmi.MtsDeltaModelBmi` for inference.
//!
//! A daily HBV 2.0 (the low-frequency, "lof" model) is spun up over ~1 year of daily-aggregated
//! forcing and hands its states to an hourly HBV 2.1 (the high-frequency, "hif" model), which is
//! spun up over the preceding hours and then stepped hour by hour. Every `cycle_days` the
//! hand-off is repeated from the rolling forcing buffers. Both models are parameterized by
//! LSTM + MLP networks (δMG `StackLstmMlpModel`); the daily LSTM's final hidden state seeds the
//! hourly LSTM through the learned `h_transfer` / `c_transfer` layers.

use std::path::Path;

use crate::BmiError;
use crate::bmi::{Engine, HourlyForcing, StaticAttributes};
use crate::config::{self, BmiConfig};
use crate::hbv::{self, Catchment, HBV2_PARAMS, HBV21_HOURLY_PARAMS, ParamLayout, Params, States};
use crate::nn::{Dense, LstmState, Mlp, ProjLstm};
use crate::ring::RingBuffer;
use crate::weights::Checkpoint;
use crate::{FORCINGS, Normalizer, aggregate_day, normalize_static};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DailyMode {
    /// Re-spin the daily model from default states over the full daily window at every
    /// hand-off (the validated, legacy behavior).
    Periodic,
    /// Spin up once, then advance `cycle_days` at a time from cached states.
    Once,
    /// Never spin up; advance `cycle_days` at a time from default states.
    Cold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HourlyMode {
    /// Take the daily states and spin up over `hourly_warmup_hours` before simulating.
    Periodic,
    /// Take the daily states and simulate immediately.
    Cold,
}

/// The `warmup` block of the BMI config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Warmup {
    pub daily_mode: DailyMode,
    pub hourly_mode: HourlyMode,
    /// Days of daily forcing in a full daily warmup.
    pub daily_days: usize,
    /// Hours of hourly forcing in an hourly warmup (1 under `hourly_mode: cold`).
    pub hourly_hours: usize,
    /// Hours between hand-offs (`cycle_days * 24`).
    pub interval: usize,
}

impl Warmup {
    pub fn from_config(config: &BmiConfig) -> Result<Self, BmiError> {
        let cfg = match config.get("warmup") {
            None => serde_yaml::Value::Mapping(Default::default()),
            Some(v) if v.is_mapping() => v.clone(),
            Some(_) => return Err("BMI config key 'warmup' must be a mapping".into()),
        };
        let text = |key: &str, default: &str| -> Result<String, BmiError> {
            match cfg.get(key) {
                None | Some(serde_yaml::Value::Null) => Ok(default.to_string()),
                Some(serde_yaml::Value::String(s)) => Ok(s.to_lowercase()),
                Some(_) => Err(format!("warmup.{key} must be a string").into()),
            }
        };
        let int = |key: &str, default: i64| -> Result<i64, BmiError> {
            match cfg.get(key) {
                None | Some(serde_yaml::Value::Null) => Ok(default),
                Some(v) => v
                    .as_i64()
                    .ok_or_else(|| format!("warmup.{key} must be an integer").into()),
            }
        };

        let daily_mode =
            match text("daily_mode", "periodic")?.as_str() {
                "periodic" => DailyMode::Periodic,
                "once" => DailyMode::Once,
                "cold" => DailyMode::Cold,
                other => return Err(format!(
                    "Unknown warmup.daily_mode '{other}'. Expected one of: cold, once, periodic."
                )
                .into()),
            };
        let hourly_mode = match text("hourly_mode", "periodic")?.as_str() {
            "periodic" => HourlyMode::Periodic,
            "cold" => HourlyMode::Cold,
            other => {
                return Err(format!(
                    "Unknown warmup.hourly_mode '{other}'. Expected one of: cold, periodic."
                )
                .into());
            }
        };
        let cycle_days = int("cycle_days", 7)?;
        if cycle_days < 1 {
            return Err(format!("'warmup.cycle_days' must be >= 1 (got {cycle_days}).").into());
        }
        let interval = cycle_days as usize * 24;

        let hourly_hours = if hourly_mode == HourlyMode::Cold {
            if cfg.get("hourly_warmup_hours").is_some() {
                eprintln!(
                    "rust-dhbv2: warmup.hourly_warmup_hours is inactive when hourly_mode is 'cold'"
                );
            }
            1
        } else {
            let hours = int("hourly_warmup_hours", 168)?;
            if hours < 24 || hours % 24 != 0 {
                return Err(format!(
                    "'warmup.hourly_warmup_hours' must be a positive multiple of 24 (got {hours})."
                )
                .into());
            }
            hours as usize
        };

        let daily_days = if daily_mode == DailyMode::Cold {
            if cfg.get("daily_warmup_days").is_some() {
                eprintln!(
                    "rust-dhbv2: warmup.daily_warmup_days is inactive when daily_mode is 'cold'"
                );
            }
            interval / 24
        } else {
            let days = int("daily_warmup_days", 351)?;
            if days < 1 {
                return Err(
                    format!("'warmup.daily_warmup_days' must be >= 1 (got {days}).").into(),
                );
            }
            days as usize
        };

        Ok(Self {
            daily_mode,
            hourly_mode,
            daily_days,
            hourly_hours,
            interval,
        })
    }

    /// Days the daily buffer keeps after the daily window, so the daily and hourly windows don't
    /// overlap. With no hourly spin-up there is no hourly window to avoid.
    fn offset_days(&self) -> usize {
        match self.hourly_mode {
            HourlyMode::Cold => 0,
            HourlyMode::Periodic => self.hourly_hours / 24,
        }
    }
}

/// The parameterization networks of the MTS model.
pub struct MtsNetworks {
    /// Daily LSTM (`nn_model.lstm_mlp`).
    pub lof_lstm: ProjLstm,
    /// Daily static-parameter MLP (`nn_model.lstm_mlp.ann`).
    pub lof_mlp: Mlp,
    /// Hourly LSTM (`nn_model.lstm_mlp2`).
    pub hif_lstm: ProjLstm,
    /// Hourly static-parameter MLP (`nn_model.lstm_mlp2.ann1`).
    pub hif_mlp: Mlp,
    /// Daily -> hourly LSTM state transfer (`nn_model.{h,c}_transfer`).
    pub transfer: Option<(Dense, Dense)>,
}

impl MtsNetworks {
    pub fn load(
        path: &Path,
        n_in: usize,
        lof_hidden: usize,
        hif_hidden: usize,
        use_in_proj: (bool, bool),
    ) -> Result<Self, BmiError> {
        let mut ckpt = Checkpoint::open(path)?;
        let p = "nn_model";
        let proj_lstm = |ckpt: &mut Checkpoint,
                         name: &str,
                         hidden: usize,
                         in_proj: bool|
         -> Result<ProjLstm, BmiError> {
            let (proj, lstm_in) = if in_proj {
                (
                    Some(ckpt.dense(&format!("{p}.{name}.in_proj"), n_in, hidden)?),
                    hidden,
                )
            } else {
                (None, n_in)
            };
            Ok(ProjLstm {
                in_proj: proj,
                lstm: ckpt.torch_lstm(&format!("{p}.{name}.lstm_inv"), lstm_in, hidden)?,
                linear_out: ckpt.dense_any(&format!("{p}.{name}.linear_out"))?,
            })
        };
        let lof_lstm = proj_lstm(&mut ckpt, "lstm_mlp", lof_hidden, use_in_proj.0)?;
        let hif_lstm = proj_lstm(&mut ckpt, "lstm_mlp2", hif_hidden, use_in_proj.1)?;
        let lof_mlp = ckpt.sequential_mlp(&format!("{p}.lstm_mlp.ann"), &[0, 3, 6, 9])?;
        let hif_mlp = ckpt.sequential_mlp(&format!("{p}.lstm_mlp2.ann1"), &[0, 3, 6, 9])?;
        // The distributed-routing MLP (`ann2`) only feeds gage routing, which the BMI does not
        // run (its output is unrouted catchment runoff). Load it to validate the checkpoint.
        let _ = ckpt.sequential_mlp(&format!("{p}.lstm_mlp2.ann2"), &[0, 3, 6, 9])?;
        let transfer = if ckpt.shape(&format!("{p}.h_transfer.weight")).is_ok() {
            Some((
                ckpt.dense(&format!("{p}.h_transfer"), lof_hidden, hif_hidden)?,
                ckpt.dense(&format!("{p}.c_transfer"), lof_hidden, hif_hidden)?,
            ))
        } else {
            None
        };
        ckpt.finish()?;
        Ok(Self {
            lof_lstm,
            lof_mlp,
            hif_lstm,
            hif_mlp,
            transfer,
        })
    }
}

pub struct MtsEngine {
    nets: MtsNetworks,
    warmup: Warmup,
    b_offset: usize,

    hourly_norm: Normalizer,
    daily_norm: Normalizer,
    /// Normalized static attributes, appended to every LSTM input row.
    c_nn_norm: Vec<f32>,
    catchment: Catchment,

    lof_layout: ParamLayout,
    hif_layout: ParamLayout,
    lof_static: Vec<Params>,
    hif_static: Vec<Params>,
    lof_nearzero: f32,
    hif_nearzero: f32,

    hourly_buffer: RingBuffer,
    daily_buffer: RingBuffer,
    day_accumulator: [[f32; 3]; 24],
    day_ptr: usize,

    timestep: u64,
    steps_since_warmup: u64,
    is_warm: bool,

    lof_lstm_state: Option<LstmState>,
    hif_lstm_state: Option<LstmState>,
    lof_states: Option<Vec<States>>,
    hif_states: Option<Vec<States>>,
}

impl MtsEngine {
    pub fn is_warm(&self) -> bool {
        self.is_warm
    }

    fn n_in(&self) -> usize {
        FORCINGS.len() + self.c_nn_norm.len()
    }

    /// Normalized forcing rows with the static attributes appended.
    fn nn_inputs(&self, rows: &[f32], norm: &Normalizer) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows.len() / 3 * self.n_in());
        for row in rows.chunks_exact(3) {
            out.extend(norm.apply(row));
            out.extend_from_slice(&self.c_nn_norm);
        }
        out
    }

    fn daily_rollout(&self) -> bool {
        if !self.is_warm {
            return false;
        }
        match self.warmup.daily_mode {
            DailyMode::Once | DailyMode::Cold => true,
            DailyMode::Periodic => self.daily_buffer.len() < self.warmup.daily_days + self.b_offset,
        }
    }

    fn can_run_warmup(&self) -> bool {
        self.daily_buffer.len() >= self.warmup.daily_days + self.b_offset
            && self.hourly_buffer.len() >= self.warmup.hourly_hours
    }

    fn is_warmup_trigger_step(&self) -> bool {
        if self.hourly_buffer.len() <= self.warmup.hourly_hours {
            return false;
        }
        if !self.timestep.is_multiple_of(24) {
            return false;
        }
        if !self
            .steps_since_warmup
            .is_multiple_of(self.warmup.interval as u64)
        {
            return false;
        }
        let daily_days = if self.daily_rollout() {
            self.warmup.interval / 24
        } else {
            self.warmup.daily_days
        };
        self.daily_buffer.len() >= daily_days + self.b_offset
    }

    fn push_forcing(&mut self, row: [f32; 3]) {
        self.hourly_buffer.push(&row);
        self.day_accumulator[self.day_ptr] = row;
        self.day_ptr += 1;
        if self.day_ptr == 24 {
            let (prcp, temp, pet) = aggregate_day(&self.day_accumulator);
            self.daily_buffer.push(&[prcp, temp, pet]);
            self.day_ptr = 0;
        }
    }

    /// Runs the daily LSTM over `rows` (continuing from its cached state unless `reset`), and
    /// returns its dynamic parameters `[days, n]`. Leaves the final state cached.
    fn run_lof_lstm(&mut self, rows: &[f32], reset: bool) -> Vec<f32> {
        let x = self.nn_inputs(rows, &self.daily_norm);
        let mut state = match (&self.lof_lstm_state, reset) {
            (Some(s), false) => s.clone(),
            _ => LstmState::zeros(self.nets.lof_lstm.hidden()),
        };
        let dynamic = self.nets.lof_lstm.run(&x, self.n_in(), &mut state);
        self.lof_lstm_state = Some(state);
        dynamic
    }

    /// The hourly LSTM's initial state, transferred from the daily LSTM's final state.
    fn transferred_lstm_state(&self) -> Option<LstmState> {
        let (h_transfer, c_transfer) = self.nets.transfer.as_ref()?;
        let lof = self.lof_lstm_state.as_ref()?;
        Some(LstmState {
            h: h_transfer.forward(&lof.h),
            c: c_transfer.forward(&lof.c),
        })
    }

    /// Runs the daily HBV over `rows` with `dynamic` parameters, continuing from its cached
    /// states only on rollout. Returns the final states.
    fn run_lof_physics(&mut self, rows: &[f32], dynamic: &[f32], rollout: bool) -> Vec<States> {
        let nmul = self.lof_layout.nmul;
        let mut states = match (&self.lof_states, rollout) {
            (Some(s), true) => s.clone(),
            _ => vec![States::default(); nmul],
        };
        let n_dyn = self.lof_layout.n_dynamic();
        let mut params = self.lof_static.clone();
        for (row, dyn_t) in rows.chunks_exact(3).zip(dynamic.chunks_exact(n_dyn)) {
            self.lof_layout.apply_dynamic(dyn_t, &mut params);
            for (s, p) in states.iter_mut().zip(&params) {
                hbv::hbv2_step(
                    s,
                    p,
                    &self.catchment,
                    row[0],
                    row[1],
                    row[2],
                    self.lof_nearzero,
                );
            }
        }
        self.lof_states = Some(states.clone());
        states
    }

    /// Hourly HBV states initialized from daily states (identity transfer of the five shared
    /// stores; the infiltration-excess stores start at their default).
    fn transfer_states(lof: &[States]) -> Vec<States> {
        lof.iter()
            .map(|s| States {
                sie1: hbv::INITIAL_STATE,
                sie2: hbv::INITIAL_STATE,
                ..*s
            })
            .collect()
    }

    /// One hourly step of the physics; returns unit-basin runoff in mm/h.
    fn hif_physics_step(&self, states: &mut [States], dynamic: &[f32], forcing: &[f32]) -> f32 {
        let mut params = self.hif_static.clone();
        self.hif_layout.apply_dynamic(dynamic, &mut params);
        let q: Vec<f32> = states
            .iter_mut()
            .zip(&params)
            .map(|(s, p)| {
                hbv::hbv21_hourly_step(
                    s,
                    p,
                    &self.catchment,
                    forcing[0],
                    forcing[1],
                    forcing[2],
                    self.hif_nearzero,
                )
            })
            .collect();
        hbv::lane_mean(&q) * hbv::HOURLY_DT as f32
    }

    fn run_warmup(&mut self) {
        let rollout = self.daily_rollout();
        let lof_reset = !rollout;

        let mut daily = self.daily_buffer.ordered();
        daily.truncate(daily.len() - 3 * self.b_offset);
        if rollout {
            let keep = 3 * (self.warmup.interval / 24);
            daily.drain(..daily.len().saturating_sub(keep));
        }

        let lof_dynamic = self.run_lof_lstm(&daily, lof_reset);
        let lof_states = self.run_lof_physics(&daily, &lof_dynamic, rollout);
        let mut hif_states = Self::transfer_states(&lof_states);
        let mut hif_lstm = self.transferred_lstm_state();

        if self.warmup.hourly_mode == HourlyMode::Periodic {
            // Spin the hourly model up over the hours before the current one.
            let mut hourly = self.hourly_buffer.ordered();
            hourly.truncate(hourly.len() - 3);
            let x = self.nn_inputs(&hourly, &self.hourly_norm);
            let mut state =
                hif_lstm.unwrap_or_else(|| LstmState::zeros(self.nets.hif_lstm.hidden()));
            let n_dyn = self.hif_layout.n_dynamic();
            let dynamic = self.nets.hif_lstm.run(&x, self.n_in(), &mut state);
            for (row, dyn_t) in hourly.chunks_exact(3).zip(dynamic.chunks_exact(n_dyn)) {
                self.hif_physics_step(&mut hif_states, dyn_t, row);
            }
            hif_lstm = Some(state);
        }

        self.hif_lstm_state = hif_lstm;
        self.hif_states = Some(hif_states);
        self.is_warm = true;
        self.steps_since_warmup = 0;
    }

    /// One hourly prediction for the newest forcing row (mm/h).
    fn step(&mut self) -> Result<f32, BmiError> {
        let row = self.hourly_buffer.last();
        let x = self.nn_inputs(&row, &self.hourly_norm);
        let mut state = self
            .hif_lstm_state
            .take()
            .unwrap_or_else(|| LstmState::zeros(self.nets.hif_lstm.hidden()));
        let dynamic = self.nets.hif_lstm.step(&x, &mut state);
        self.hif_lstm_state = Some(state);

        let mut states = self.hif_states.take().ok_or("hourly model has no states")?;
        let runoff = self.hif_physics_step(&mut states, &dynamic, &row);
        self.hif_states = Some(states);
        Ok(runoff)
    }
}

impl Engine for MtsEngine {
    const COMPONENT_NAME: &'static str = "δHBV2.0 MTS";

    fn initialize(config: &BmiConfig, statics: &StaticAttributes) -> Result<Self, BmiError> {
        let model_dir = config.model_dir()?;
        let model_config = config::read_yaml(&model_dir.join("config.yaml"))?;
        let warmup = Warmup::from_config(config)?;

        fn hif_nn(key: &str) -> [&str; 4] {
            ["model", "nn", "hif_model", key]
        }
        fn lof_nn(key: &str) -> [&str; 4] {
            ["model", "nn", "lof_model", key]
        }
        let forcings = config::req_str_list(&model_config, &hif_nn("forcings"))?;
        if forcings != FORCINGS {
            return Err(format!("unsupported forcings {forcings:?}; expected {FORCINGS:?}").into());
        }
        if config::req_str_list(&model_config, &lof_nn("forcings"))? != FORCINGS {
            return Err("low-frequency model forcings must be [P, T, PET]".into());
        }
        let attributes = config::req_str_list(&model_config, &hif_nn("attributes"))?;
        if config::req_str_list(&model_config, &lof_nn("attributes"))? != attributes {
            return Err("low- and high-frequency models must share their attribute list".into());
        }

        // Physics layouts
        let layout = |freq: &str,
                      model: &str,
                      bounds: &'static [(&'static str, f64, f64)]|
         -> Result<(ParamLayout, f32), BmiError> {
            let nmul = config::get(&model_config, &["model", "phy", freq, "nmul"])
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            let dynamic = config::req_str_list(
                &model_config,
                &["model", "phy", freq, "dynamic_params", model],
            )?;
            let nearzero = config::get(&model_config, &["model", "phy", freq, "nearzero"])
                .and_then(config::as_float)
                .unwrap_or(1e-5);
            Ok((ParamLayout::new(bounds, &dynamic, nmul)?, nearzero as f32))
        };
        let hif_name = config::get(&model_config, &["model", "phy", "hif_model", "name"])
            .and_then(|v| v.as_sequence())
            .and_then(|s| s.first())
            .and_then(|v| v.as_str())
            .unwrap_or("Hbv_2_1_hourly")
            .to_string();
        if hif_name != "Hbv_2_1_hourly" {
            return Err(format!("unsupported high-frequency model '{hif_name}'; only Hbv_2_1_hourly (δHBV 2.1 MTS) is implemented").into());
        }
        let (lof_layout, lof_nearzero) = layout("lof_model", "Hbv_2", &HBV2_PARAMS)?;
        let (hif_layout, hif_nearzero) =
            layout("hif_model", "Hbv_2_1_hourly", &HBV21_HOURLY_PARAMS)?;
        if lof_layout.nmul != hif_layout.nmul {
            return Err("low- and high-frequency models must use the same nmul".into());
        }

        // Networks
        let stem = config::get(&model_config, &["model", "phy", "name"])
            .and_then(|v| v.as_sequence())
            .and_then(|s| s.first())
            .and_then(|v| v.as_str())
            .map(|name| format!("d{}", name.to_lowercase()))
            .unwrap_or_else(|| "dhbv_2_mts".into());
        let epoch = config::get(&model_config, &["test", "test_epoch"]).and_then(|v| v.as_u64());
        let weights = config::find_weights(&model_dir, &stem, epoch)?;
        let n_in = FORCINGS.len() + attributes.len();
        let nets = MtsNetworks::load(
            &weights,
            n_in,
            config::req_usize(&model_config, &lof_nn("lstm_hidden_size"))?,
            config::req_usize(&model_config, &hif_nn("lstm_hidden_size"))?,
            (
                config::opt_bool(&model_config, &lof_nn("use_in_proj"), true)?,
                config::opt_bool(&model_config, &hif_nn("use_in_proj"), true)?,
            ),
        )?;
        let use_transfer = config::opt_bool(&model_config, &["model", "nn", "use_transfer"], true)?;
        let nets = MtsNetworks {
            transfer: if use_transfer { nets.transfer } else { None },
            ..nets
        };
        if nets.lof_lstm.linear_out.d_out != lof_layout.n_dynamic()
            || nets.hif_lstm.linear_out.d_out != hif_layout.n_dynamic()
        {
            return Err(
                "LSTM output sizes do not match the physics models' dynamic parameters".into(),
            );
        }

        // Normalization
        let stats = config::read_json(&model_dir.join("normalization_statistics.json"))?;
        let stat = |split: &str, key: &str| {
            config::json_f64_list(
                &stats[split][key],
                &format!("normalization_statistics.{split}.{key}"),
            )
        };
        let hourly_norm =
            Normalizer::f32_stats(&stat("mean", "dyn_input")?, &stat("std", "dyn_input")?)?;
        let daily_norm = Normalizer::f32_stats(
            &stat("mean", "dyn_input_daily")?,
            &stat("std", "dyn_input_daily")?,
        )?;
        let attr_values = attributes
            .iter()
            .map(|a| statics.get(a))
            .collect::<Result<Vec<_>, _>>()?;
        let c_nn_norm = normalize_static(
            &attr_values,
            &stat("mean", "static_input")?,
            &stat("std", "static_input")?,
        )?;

        let catchment = Catchment {
            ac: statics.get("uparea")? as f32,
            elevation: statics.get("meanelevation")? as f32,
        };

        // Static parameters depend only on the attributes, so compute them once.
        let lof_raw = nets.lof_mlp.forward(&c_nn_norm);
        let hif_raw = nets.hif_mlp.forward(&c_nn_norm);
        if lof_raw.len() < lof_layout.n_static() || hif_raw.len() < hif_layout.n_static() {
            return Err(
                "static-parameter MLP outputs are smaller than the physics models need".into(),
            );
        }
        let lof_static = lof_layout.static_params(&lof_raw);
        let hif_static = hif_static_params(&lof_layout, &hif_layout, &lof_raw, &hif_raw)?;

        let b_offset = warmup.offset_days();
        Ok(Self {
            hourly_buffer: RingBuffer::new(warmup.hourly_hours + 1, 3),
            daily_buffer: RingBuffer::new(warmup.daily_days + b_offset, 3),
            day_accumulator: [[0.0; 3]; 24],
            day_ptr: 0,
            nets,
            warmup,
            b_offset,
            hourly_norm,
            daily_norm,
            c_nn_norm,
            catchment,
            lof_layout,
            hif_layout,
            lof_static,
            hif_static,
            lof_nearzero,
            hif_nearzero,
            timestep: 0,
            steps_since_warmup: 0,
            is_warm: false,
            lof_lstm_state: None,
            hif_lstm_state: None,
            lof_states: None,
            hif_states: None,
        })
    }

    fn update(&mut self, forcing: &HourlyForcing) -> Result<f64, BmiError> {
        let pet = forcing.penman_monteith_pet();
        self.push_forcing([forcing.precip as f32, forcing.temp as f32, pet as f32]);

        let mut runoff = 0.0f32;
        if self.is_warm || self.can_run_warmup() {
            if self.is_warmup_trigger_step() {
                self.run_warmup();
            }
            if self.is_warm {
                runoff = self.step()?;
                self.steps_since_warmup += 1;
            }
        }
        self.timestep += 1;
        // mm h-1 -> m h-1
        Ok(runoff as f64 / 1000.0)
    }
}

/// Hourly static parameters: the static parameters shared with the daily model come from the
/// daily MLP (`Hbv_2_mts.param_transfer`); the hourly-only ones (infiltration excess) from the
/// hourly MLP.
fn hif_static_params(
    lof: &ParamLayout,
    hif: &ParamLayout,
    lof_raw: &[f32],
    hif_raw: &[f32],
) -> Result<Vec<Params>, BmiError> {
    let shared = lof.statics.len();
    if hif.statics[..shared] != lof.statics[..] {
        return Err(format!(
            "static parameter order mismatch between low- and high-frequency models: {:?} vs {:?}",
            lof.statics, hif.statics
        )
        .into());
    }
    let nmul = hif.nmul;
    let mut raw = lof_raw[..shared * nmul].to_vec();
    for (i, name) in hif.statics.iter().enumerate().skip(shared) {
        debug_assert!(!lof.statics.contains(name));
        raw.extend_from_slice(&hif_raw[i * nmul..(i + 1) * nmul]);
    }
    Ok(hif.static_params(&raw))
}
