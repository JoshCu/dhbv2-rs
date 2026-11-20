//! BMI plumbing shared by the daily and MTS models: variable names and units, `set_value` /
//! `get_value`, and the hourly clock. The model-specific work lives behind [`Engine`].

use std::collections::HashMap;

use bmi_rs::bmi::{Bmi, BmiResult, Location, RefValues, ValueType, Values};

use crate::BmiError;
use crate::config::BmiConfig;

/// Dynamic (forcing) inputs: (internal name, CSDMS standard name, units).
pub const DYNAMIC_INPUTS: [(&str, &str, &str); 8] = [
    (
        "P",
        "atmosphere_water__liquid_equivalent_precipitation_rate",
        "mm h-1",
    ),
    ("T", "land_surface_air__temperature", "degC"),
    (
        "SPFH",
        "atmosphere_air_water~vapor__relative_saturation",
        "g g-1",
    ),
    (
        "DLWRF",
        "land_surface_radiation~incoming~longwave__energy_flux",
        "W m-2",
    ),
    (
        "DSWRF",
        "land_surface_radiation~incoming~shortwave__energy_flux",
        "W m-2",
    ),
    ("PRES", "land_surface_air__pressure", "Pa"),
    ("U", "land_surface_wind__x_component_of_velocity", "m s-1"),
    ("V", "land_surface_wind__y_component_of_velocity", "m s-1"),
];

/// Static catchment attributes: (BMI config key, CSDMS standard name, units).
pub const STATIC_INPUTS: [(&str, &str, &str); 30] = [
    (
        "aridity",
        "ratio__mean_potential_evapotranspiration__mean_precipitation",
        "-",
    ),
    (
        "meanP",
        "atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate",
        "mm d-1",
    ),
    (
        "ETPOT_Hargr",
        "land_surface_water__Hargreaves_potential_evaporation_volume_flux",
        "mm d-1",
    ),
    (
        "NDVI",
        "land_vegetation__normalized_diff_vegetation_index",
        "-",
    ),
    ("FW", "free_land_surface_water", "mm d-1"),
    ("meanslope", "basin__mean_of_slope", "m km-1"),
    ("SoilGrids1km_sand", "soil_sand__grid", "km2"),
    ("SoilGrids1km_clay", "soil_clay__grid", "km2"),
    ("SoilGrids1km_silt", "soil_silt__grid", "km2"),
    (
        "glaciers",
        "land_surface_water__glacier_fraction",
        "percent",
    ),
    ("HWSD_clay", "soil_clay__attr", "percent"),
    ("HWSD_gravel", "soil_gravel__attr", "percent"),
    ("HWSD_sand", "soil_sand__attr", "percent"),
    ("HWSD_silt", "soil_silt__attr", "percent"),
    ("meanelevation", "basin__mean_of_elevation", "m"),
    (
        "meanTa",
        "atmosphere_water__daily_mean_of_temperature",
        "degC",
    ),
    ("permafrost", "land_surface_water__permafrost_fraction", "-"),
    ("permeability", "bedrock__permeability", "m2"),
    ("seasonality_P", "p_seasonality", "-"),
    (
        "seasonality_PET",
        "land_surface_water__potential_evaporation_volume_flux_seasonality",
        "-",
    ),
    (
        "snow_fraction",
        "land_surface_water__snow_fraction",
        "percent",
    ),
    (
        "snowfall_fraction",
        "atmosphere_water__precipitation_falling_as_snow_fraction",
        "percent",
    ),
    ("T_clay", "soil_clay__volume_fraction", "percent"),
    ("T_gravel", "soil_gravel__volume_fraction", "percent"),
    ("T_sand", "soil_sand__volume_fraction", "percent"),
    ("T_silt", "soil_silt__volume_fraction", "percent"),
    ("Porosity", "soil_active-layer__porosity", "-"),
    ("uparea", "basin__area", "km2"),
    ("catchsize", "catchment__area", "km2"),
    ("lengthkm", "basin__length", "km"),
];

/// The single output: runoff depth rate. The name says "volume flux" for ngen compatibility,
/// but, like the Python BMIs, the value is a depth per hour (m h-1).
pub const OUTPUT: (&str, &str) = ("land_surface_water__runoff_volume_flux", "m h-1");

const INPUT_NAMES: [&str; 8] = [
    DYNAMIC_INPUTS[0].1,
    DYNAMIC_INPUTS[1].1,
    DYNAMIC_INPUTS[2].1,
    DYNAMIC_INPUTS[3].1,
    DYNAMIC_INPUTS[4].1,
    DYNAMIC_INPUTS[5].1,
    DYNAMIC_INPUTS[6].1,
    DYNAMIC_INPUTS[7].1,
];
const OUTPUT_NAMES: [&str; 1] = [OUTPUT.0];

/// One hour of forcing, in the order of [`DYNAMIC_INPUTS`].
#[derive(Clone, Copy, Debug, Default)]
pub struct HourlyForcing {
    pub precip: f64,
    pub temp: f64,
    pub spfh: f64,
    pub dlwrf: f64,
    pub dswrf: f64,
    pub pres: f64,
    pub ugrd: f64,
    pub vgrd: f64,
}

impl HourlyForcing {
    pub fn penman_monteith_pet(&self) -> f64 {
        crate::pet::penman_monteith(
            self.temp, self.spfh, self.dlwrf, self.dswrf, self.pres, self.ugrd, self.vgrd,
        )
    }
}

/// Static attributes read from the BMI config, keyed by config name. Missing attributes read as
/// 0.0 (with a warning when the model uses them), as in the Python BMIs.
#[derive(Clone, Debug)]
pub struct StaticAttributes {
    values: HashMap<&'static str, Option<f64>>,
}

impl StaticAttributes {
    pub fn from_config(config: &BmiConfig) -> Result<Self, BmiError> {
        let mut values = HashMap::new();
        for (key, _, _) in STATIC_INPUTS {
            values.insert(key, config.float(key)?);
        }
        Ok(Self { values })
    }

    /// The attribute's value, 0.0 (with a warning) if the BMI config does not set it.
    pub fn get(&self, key: &str) -> Result<f64, BmiError> {
        match self.values.get(key) {
            Some(Some(v)) => Ok(*v),
            Some(None) => {
                eprintln!("rust-dhbv2: static attribute '{key}' not in BMI config, using 0.0");
                Ok(0.0)
            }
            None => Err(format!("model uses unknown static attribute '{key}'").into()),
        }
    }

    /// The attribute's value for exposing through BMI (0.0 if unset, without a warning).
    pub fn value_or_zero(&self, key: &str) -> f64 {
        self.values.get(key).copied().flatten().unwrap_or(0.0)
    }
}

/// A model run hour by hour behind the BMI.
pub trait Engine: Sized {
    const COMPONENT_NAME: &'static str;

    fn initialize(config: &BmiConfig, statics: &StaticAttributes) -> Result<Self, BmiError>;

    /// Ingests one hour of forcing and returns the output in external units (m h-1).
    fn update(&mut self, forcing: &HourlyForcing) -> Result<f64, BmiError>;
}

pub struct DhbvBmi<E: Engine> {
    engine: Option<E>,
    values: HashMap<&'static str, Vec<f64>>,
    timestep: u64,
    time_step_size: f64,
}

impl<E: Engine> Default for DhbvBmi<E> {
    fn default() -> Self {
        Self::new()
    }
}

fn not_found(name: &str) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("Variable {name} not found"),
    ))
}

impl<E: Engine> DhbvBmi<E> {
    pub fn new() -> Self {
        let values = DYNAMIC_INPUTS
            .iter()
            .map(|(_, name, _)| *name)
            .chain(STATIC_INPUTS.iter().map(|(_, name, _)| *name))
            .chain(std::iter::once(OUTPUT.0))
            .map(|name| (name, vec![0.0]))
            .collect();
        Self {
            engine: None,
            values,
            timestep: 0,
            time_step_size: 3600.0,
        }
    }

    pub fn engine(&self) -> Option<&E> {
        self.engine.as_ref()
    }

    fn value(&self, name: &str) -> f64 {
        self.values[name][0]
    }

    fn current_forcing(&self) -> HourlyForcing {
        let v = |i: usize| self.value(DYNAMIC_INPUTS[i].1);
        HourlyForcing {
            precip: v(0),
            temp: v(1),
            spfh: v(2),
            dlwrf: v(3),
            dswrf: v(4),
            pres: v(5),
            ugrd: v(6),
            vgrd: v(7),
        }
    }

    fn units(name: &str) -> Option<&'static str> {
        DYNAMIC_INPUTS
            .iter()
            .chain(STATIC_INPUTS.iter())
            .find(|(_, n, _)| *n == name)
            .map(|(_, _, u)| *u)
            .or((name == OUTPUT.0).then_some(OUTPUT.1))
    }

    fn is_exposed(name: &str) -> bool {
        INPUT_NAMES.contains(&name) || name == OUTPUT.0
    }
}

impl<E: Engine> Bmi for DhbvBmi<E> {
    fn initialize(&mut self, config_file: &str) -> BmiResult<()> {
        let config = BmiConfig::load(config_file)?;
        config.check_dtype()?;
        if let Some(step) = config.float("time_step_size")?
            && step != 3600.0
        {
            return Err(format!("time_step_size must be 3600 (hourly), got {step}").into());
        }
        let statics = StaticAttributes::from_config(&config)?;
        for (key, name, _) in STATIC_INPUTS {
            self.values.insert(name, vec![statics.value_or_zero(key)]);
        }
        self.engine = Some(E::initialize(&config, &statics)?);
        self.timestep = 0;
        Ok(())
    }

    fn update(&mut self) -> BmiResult<()> {
        let forcing = self.current_forcing();
        let engine = self.engine.as_mut().ok_or("BMI not initialized")?;
        let output = engine.update(&forcing)?;
        self.values.insert(OUTPUT.0, vec![output]);
        self.timestep += 1;
        Ok(())
    }

    fn update_until(&mut self, then: f64) -> BmiResult<()> {
        let now = self.get_current_time();
        if then < now {
            eprintln!("rust-dhbv2: no update performed: end_time ({then}) < current time ({now})");
            return Ok(());
        }
        let n_steps = ((then - now) / self.time_step_size).floor() as u64;
        for _ in 0..n_steps {
            self.update()?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> BmiResult<()> {
        self.engine = None;
        Ok(())
    }

    fn get_component_name(&self) -> &str {
        E::COMPONENT_NAME
    }

    fn get_input_var_names(&self) -> &[&str] {
        &INPUT_NAMES
    }

    fn get_output_var_names(&self) -> &[&str] {
        &OUTPUT_NAMES
    }

    fn get_var_grid(&self, name: &str) -> BmiResult<i32> {
        if Self::is_exposed(name) {
            Ok(0)
        } else {
            Err(not_found(name))
        }
    }

    fn get_var_type(&self, name: &str) -> BmiResult<ValueType> {
        if self.values.contains_key(name) {
            Ok(ValueType::F64)
        } else {
            Err(not_found(name))
        }
    }

    fn get_var_units(&self, name: &str) -> BmiResult<&str> {
        Self::units(name).ok_or_else(|| not_found(name))
    }

    fn get_var_nbytes(&self, name: &str) -> BmiResult<u32> {
        let values = self.values.get(name).ok_or_else(|| not_found(name))?;
        Ok((values.len() * std::mem::size_of::<f64>()) as u32)
    }

    fn get_var_location(&self, name: &str) -> BmiResult<Location> {
        if Self::is_exposed(name) {
            Ok(Location::Node)
        } else {
            Err(not_found(name))
        }
    }

    fn get_current_time(&self) -> f64 {
        self.timestep as f64 * self.time_step_size
    }

    fn get_start_time(&self) -> f64 {
        0.0
    }

    fn get_end_time(&self) -> f64 {
        f64::MAX
    }

    fn get_time_units(&self) -> &str {
        "s"
    }

    fn get_time_step(&self) -> f64 {
        self.time_step_size
    }

    fn get_value_ptr(&self, name: &str) -> BmiResult<RefValues<'_>> {
        self.values
            .get(name)
            .map(|v| RefValues::F64(v))
            .ok_or_else(|| not_found(name))
    }

    fn get_value_at_indices(&self, name: &str, inds: &[u32]) -> BmiResult<Values> {
        let values = self.values.get(name).ok_or_else(|| not_found(name))?;
        let mut out = Vec::with_capacity(inds.len());
        for &i in inds {
            out.push(
                *values
                    .get(i as usize)
                    .ok_or_else(|| format!("Index {i} out of bounds"))?,
            );
        }
        Ok(Values::F64(out))
    }

    fn set_value(&mut self, name: &str, src: RefValues) -> BmiResult<()> {
        let slot = self.values.get_mut(name).ok_or_else(|| not_found(name))?;
        let src: Vec<f64> = match src {
            RefValues::F64(v) => v.to_vec(),
            RefValues::F32(v) => v.iter().map(|x| *x as f64).collect(),
            RefValues::I32(v) => v.iter().map(|x| *x as f64).collect(),
            RefValues::I64(v) => v.iter().map(|x| *x as f64).collect(),
            _ => return Err("unsupported value type for set_value".into()),
        };
        if src.len() != slot.len() {
            return Err(format!(
                "set_value({name}): expected {} value(s), got {}",
                slot.len(),
                src.len()
            )
            .into());
        }
        slot.copy_from_slice(&src);
        Ok(())
    }

    fn set_value_at_indices(&mut self, name: &str, inds: &[u32], src: RefValues) -> BmiResult<()> {
        let RefValues::F64(src) = src else {
            return Err("set_value_at_indices expects f64 values".into());
        };
        if src.len() != inds.len() {
            return Err("source array size doesn't match indices count".into());
        }
        let slot = self.values.get_mut(name).ok_or_else(|| not_found(name))?;
        for (&i, v) in inds.iter().zip(src) {
            *slot
                .get_mut(i as usize)
                .ok_or_else(|| format!("Index {i} out of bounds"))? = *v;
        }
        Ok(())
    }
}
