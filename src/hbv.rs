//! HBV 2.0 (daily) and HBV 2.1 (hourly) physics, ported from hydrodl2
//! (`hydrodl2/models/hbv/hbv_2.py` and `hbv_2_1_hourly.py`) for a single catchment.
//!
//! Each model runs `nmul` parallel parameter sets ("lanes") whose discharge is averaged. The
//! arithmetic follows the PyTorch code operation by operation in f32 so the outputs track the
//! Python BMI closely.

use crate::BmiError;

/// Parameter bounds of HBV 2.0, in the order the networks emit them.
pub const HBV2_PARAMS: [(&str, f64, f64); 16] = [
    ("parBETA", 1.0, 6.0),
    ("parFC", 50.0, 1000.0),
    ("parK0", 0.05, 0.9),
    ("parK1", 0.01, 0.5),
    ("parK2", 0.001, 0.2),
    ("parLP", 0.2, 1.0),
    ("parPERC", 0.0, 10.0),
    ("parUZL", 0.0, 100.0),
    ("parTT", -2.5, 2.5),
    ("parCFMAX", 0.5, 10.0),
    ("parCFR", 0.0, 0.1),
    ("parCWH", 0.0, 0.2),
    ("parBETAET", 0.3, 5.0),
    ("parC", 0.0, 1.0),
    ("parRT", 0.0, 20.0),
    ("parAC", 0.0, 2500.0),
];

/// Hourly timestep of HBV 2.1 as a fraction of a day (`dt = 1.0 / 24` in Python).
pub const HOURLY_DT: f64 = 1.0 / 24.0;

/// Parameter bounds of hourly HBV 2.1 (with infiltration excess), in network order.
pub const HBV21_HOURLY_PARAMS: [(&str, f64, f64); 19] = [
    ("parBETA", 1.0, 6.0),
    ("parFC", 50.0, 1000.0),
    ("parK0", 0.05, 12.0),
    ("parK1", 0.01, 0.5),
    ("parK2", 0.001, 0.2),
    ("parLP", 0.2, 1.0),
    ("parPERC", 0.0, 10.0),
    ("parUZL", 0.0, 100.0),
    ("parTT", -2.5, 2.5),
    ("parCFMAX", 0.5, 10.0),
    ("parCFR", 0.0, 0.1),
    ("parCWH", 0.0, 0.2),
    ("parBETAET", 0.3, 5.0),
    ("parC", 0.0, 1.0),
    ("parRT", 0.0, 20.0),
    ("parAC", 0.0, 2500.0),
    ("parF0", 5.0 / HOURLY_DT, 120.0 / HOURLY_DT),
    ("parALPHA", 0.5, 1.0),
    ("parKS", 0.05, 10.0),
];

/// Initial value of every HBV state (`BasePhysicsModel.initial_state_value`).
pub const INITIAL_STATE: f32 = 0.001;

/// `param * (upper - lower) + lower`, with the bounds rounded to f32 as PyTorch does when a
/// Python float scalar meets an f32 tensor.
#[inline]
pub fn descale(param: f32, lower: f64, upper: f64) -> f32 {
    param * ((upper - lower) as f32) + lower as f32
}

/// Physics parameters of one lane, resolved by name.
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub beta: f32,
    pub fc: f32,
    pub k0: f32,
    pub k1: f32,
    pub k2: f32,
    pub lp: f32,
    pub perc: f32,
    pub uzl: f32,
    pub tt: f32,
    pub cfmax: f32,
    pub cfr: f32,
    pub cwh: f32,
    pub betaet: f32,
    pub c: f32,
    pub rt: f32,
    pub ac: f32,
    pub f0: f32,
    pub alpha: f32,
    pub ks: f32,
}

impl Params {
    fn set(&mut self, name: &str, value: f32) {
        match name {
            "parBETA" => self.beta = value,
            "parFC" => self.fc = value,
            "parK0" => self.k0 = value,
            "parK1" => self.k1 = value,
            "parK2" => self.k2 = value,
            "parLP" => self.lp = value,
            "parPERC" => self.perc = value,
            "parUZL" => self.uzl = value,
            "parTT" => self.tt = value,
            "parCFMAX" => self.cfmax = value,
            "parCFR" => self.cfr = value,
            "parCWH" => self.cwh = value,
            "parBETAET" => self.betaet = value,
            "parC" => self.c = value,
            "parRT" => self.rt = value,
            "parAC" => self.ac = value,
            "parF0" => self.f0 = value,
            "parALPHA" => self.alpha = value,
            "parKS" => self.ks = value,
            _ => unreachable!("unknown HBV parameter {name}"),
        }
    }
}

/// Which parameters are time-dynamic (from the LSTM) and which static (from the MLP), and how
/// the network outputs map onto them.
#[derive(Clone, Debug)]
pub struct ParamLayout {
    pub bounds: &'static [(&'static str, f64, f64)],
    pub nmul: usize,
    /// Dynamic parameter names, in bounds order (hydrodl2 makes them order-independent).
    pub dynamic: Vec<&'static str>,
    /// Static parameter names, in bounds order.
    pub statics: Vec<&'static str>,
}

impl ParamLayout {
    pub fn new(
        bounds: &'static [(&'static str, f64, f64)],
        dynamic: &[String],
        nmul: usize,
    ) -> Result<Self, BmiError> {
        for name in dynamic {
            if !bounds.iter().any(|(n, _, _)| n == name) {
                return Err(format!("unknown dynamic parameter {name}").into());
            }
        }
        let is_dynamic = |n: &str| dynamic.iter().any(|d| d == n);
        Ok(Self {
            bounds,
            nmul,
            dynamic: bounds
                .iter()
                .filter(|(n, _, _)| is_dynamic(n))
                .map(|(n, _, _)| *n)
                .collect(),
            statics: bounds
                .iter()
                .filter(|(n, _, _)| !is_dynamic(n))
                .map(|(n, _, _)| *n)
                .collect(),
        })
    }

    pub fn n_dynamic(&self) -> usize {
        self.dynamic.len() * self.nmul
    }

    pub fn n_static(&self) -> usize {
        self.statics.len() * self.nmul
    }

    fn bounds_of(&self, name: &str) -> (f64, f64) {
        let (_, lo, hi) = self
            .bounds
            .iter()
            .find(|(n, _, _)| *n == name)
            .expect("parameter is in the bounds table");
        (*lo, *hi)
    }

    /// Descaled static parameters per lane from raw (sigmoid) network outputs laid out as
    /// `[n_static_params, nmul]` (PyTorch `.view(ngrid, n, nmul)`).
    pub fn static_params(&self, raw: &[f32]) -> Vec<Params> {
        let mut lanes = vec![Params::default(); self.nmul];
        for (i, name) in self.statics.iter().enumerate() {
            let (lo, hi) = self.bounds_of(name);
            for (m, lane) in lanes.iter_mut().enumerate() {
                lane.set(name, descale(raw[i * self.nmul + m], lo, hi));
            }
        }
        lanes
    }

    /// Overwrites the dynamic parameters of each lane from one timestep of raw LSTM outputs laid
    /// out as `[n_dynamic_params, nmul]`.
    pub fn apply_dynamic(&self, raw: &[f32], lanes: &mut [Params]) {
        for (i, name) in self.dynamic.iter().enumerate() {
            let (lo, hi) = self.bounds_of(name);
            for (m, lane) in lanes.iter_mut().enumerate() {
                lane.set(name, descale(raw[i * self.nmul + m], lo, hi));
            }
        }
    }
}

/// States of one lane: snowpack, meltwater, soil moisture, upper and lower zone, and the two
/// infiltration-excess stores of HBV 2.1 (unused by HBV 2.0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct States {
    pub snowpack: f32,
    pub meltwater: f32,
    pub sm: f32,
    pub suz: f32,
    pub slz: f32,
    pub sie1: f32,
    pub sie2: f32,
}

impl Default for States {
    fn default() -> Self {
        Self {
            snowpack: INITIAL_STATE,
            meltwater: INITIAL_STATE,
            sm: INITIAL_STATE,
            suz: INITIAL_STATE,
            slz: INITIAL_STATE,
            sie1: INITIAL_STATE,
            sie2: INITIAL_STATE,
        }
    }
}

/// Static catchment data the physics reads.
#[derive(Clone, Copy, Debug)]
pub struct Catchment {
    /// Upstream area (`uparea`, km2).
    pub ac: f32,
    /// Mean elevation (`meanelevation`, m).
    pub elevation: f32,
}

#[inline]
fn threshold_temperature(elevation: f32, tt: f32) -> f32 {
    // (elev >= 2000) * 4.0 + (elev < 2000) * parTT
    if elevation >= 2000.0 { 4.0 } else { tt }
}

#[inline]
fn lateral_flow(ac: f32, p: &Params) -> f32 {
    if ac < 2500.0 {
        ((ac - p.ac) / 1000.0).clamp(-1.0, 1.0) * p.rt
    } else {
        (-(ac - 2500.0) / 50.0).clamp(-10.0, 0.0).exp() * p.rt
    }
}

/// One daily step of HBV 2.0 for one lane; returns `Q0 + Q1 + Q2` (mm/d).
pub fn hbv2_step(
    s: &mut States,
    p: &Params,
    catchment: &Catchment,
    prcp: f32,
    temp: f32,
    pet: f32,
    nearzero: f32,
) -> f32 {
    let tt = threshold_temperature(catchment.elevation, p.tt);
    let rain = prcp * if temp >= tt { 1.0 } else { 0.0 };
    let snow = prcp * if temp < tt { 1.0 } else { 0.0 };

    // Snow
    s.snowpack += snow;
    let melt = (p.cfmax * (temp - tt)).max(0.0).min(s.snowpack);
    s.meltwater += melt;
    s.snowpack -= melt;
    let refreezing = (p.cfr * p.cfmax * (tt - temp)).max(0.0).min(s.meltwater);
    s.snowpack += refreezing;
    s.meltwater -= refreezing;
    let tosoil = (s.meltwater - p.cwh * s.snowpack).max(0.0);
    s.meltwater -= tosoil;

    // Soil and evaporation
    let soil_wetness = (s.sm / p.fc).powf(p.beta).clamp(0.0, 1.0);
    let recharge = (rain + tosoil) * soil_wetness;
    s.sm = s.sm + rain + tosoil - recharge;
    let excess = (s.sm - p.fc).max(0.0);
    s.sm -= excess;
    let evapfactor = (s.sm / (p.lp * p.fc)).powf(p.betaet).clamp(0.0, 1.0);
    let et_act = s.sm.min(pet * evapfactor);
    s.sm = (s.sm - et_act).max(nearzero);

    // Capillary rise
    let capillary = s.slz.min(p.c * s.slz * (1.0 - (s.sm / p.fc).min(1.0)));
    s.sm = (s.sm + capillary).max(nearzero);
    s.slz = (s.slz - capillary).max(nearzero);

    // Groundwater
    s.suz = s.suz + recharge + excess;
    let perc = s.suz.min(p.perc);
    s.suz -= perc;
    let q0 = p.k0 * (s.suz - p.uzl).max(0.0);
    s.suz -= q0;
    let q1 = p.k1 * s.suz;
    s.suz -= q1;
    s.slz += perc;

    let lf = lateral_flow(catchment.ac, p);
    s.slz = (s.slz + lf).max(nearzero);
    let q2 = p.k2 * s.slz;
    s.slz -= q2;

    q0 + q1 + q2
}

/// One hourly step of HBV 2.1 (with infiltration excess) for one lane. Forcings are per hour
/// (mm/h, degC, mm/h); returns the unrouted discharge rate `Q0 + Q1 + Q2 + Qie` in mm/d.
pub fn hbv21_hourly_step(
    s: &mut States,
    p: &Params,
    catchment: &Catchment,
    prcp: f32,
    temp: f32,
    pet: f32,
    nearzero: f32,
) -> f32 {
    let dt = HOURLY_DT as f32;

    // Numerical guardrail for long sequences
    s.snowpack = s.snowpack.max(0.0);
    s.meltwater = s.meltwater.max(0.0);
    s.sm = s.sm.max(nearzero);
    s.suz = s.suz.max(nearzero);
    s.slz = s.slz.max(nearzero);
    s.sie1 = s.sie1.max(nearzero);
    s.sie2 = s.sie2.max(nearzero);

    let prcp = prcp / dt;
    let pet = pet / dt;

    let tt = threshold_temperature(catchment.elevation, p.tt);
    let rain = prcp * if temp >= tt { 1.0 } else { 0.0 };
    let snow = prcp * if temp < tt { 1.0 } else { 0.0 };

    // Snow
    s.snowpack += snow * dt;
    let melt = (p.cfmax * (temp - tt)).max(0.0);
    let melt = (melt * dt).min(s.snowpack);
    s.meltwater += melt;
    s.snowpack -= melt;
    let refreezing = (p.cfr * p.cfmax * (tt - temp)).max(0.0);
    let refreezing = (refreezing * dt).min(s.meltwater);
    s.snowpack += refreezing;
    s.meltwater -= refreezing;
    let tosoil = ((s.meltwater - p.cwh * s.snowpack) / dt).max(0.0);
    s.meltwater -= tosoil * dt;

    // Saturation excess plus Hortonian infiltration excess over the unsaturated fraction
    let w = (rain + tosoil).max(0.0);
    let sat = (s.sm / p.fc).clamp(0.0, 1.0 - 1e-4);
    let saturated_fraction = sat.powf(p.beta).clamp(0.0, 1.0);
    let unsaturated_fraction = 1.0 - saturated_fraction;
    let recharge = w * saturated_fraction;
    let d_star = (w / p.f0).powf(1.0 / p.alpha).clamp(0.0, 1.0);
    let f_integral = p.f0 * d_star.powf(p.alpha + 1.0) / (p.alpha + 1.0);
    let ie = (unsaturated_fraction * (w * d_star - f_integral)).max(0.0);
    let to_soil = (w * unsaturated_fraction - ie).max(0.0);
    s.sm += to_soil * dt;

    let excess = ((s.sm - p.fc) / dt).max(0.0);
    s.sm -= excess * dt;
    let evapfactor = (s.sm / (p.lp * p.fc)).powf(p.betaet).clamp(0.0, 1.0);
    let et_act = pet * evapfactor;
    let et_act = s.sm.min(et_act * dt) / dt;
    s.sm = (s.sm - et_act * dt).max(nearzero);

    // Capillary rise
    let capillary = s.slz.min(p.c * s.slz * (1.0 - (s.sm / p.fc).min(1.0)) * dt) / dt;
    s.sm = (s.sm + capillary * dt).max(nearzero);
    s.slz = (s.slz - capillary * dt).max(nearzero);

    // Groundwater boxes
    s.suz += (recharge + excess) * dt;
    let perc = s.suz.min(p.perc * dt) / dt;
    s.suz -= perc * dt;
    let q0 = p.k0 * (s.suz - p.uzl).max(0.0);
    s.suz -= q0 * dt;
    let q1 = p.k1 * s.suz;
    s.suz -= q1 * dt;
    s.slz += perc * dt;

    let lf = lateral_flow(catchment.ac, p);
    s.slz = (s.slz + lf * dt).max(nearzero);
    let q2 = p.k2 * s.slz;
    s.slz -= q2 * dt;

    // Infiltration excess through a two-store Nash cascade
    let ks = p.ks * 2.0;
    s.sie1 += ie * dt;
    let q12 = (ks * s.sie1 * dt).min(s.sie1) / dt;
    s.sie1 -= q12 * dt;
    s.sie2 += q12 * dt;
    let qie = (ks * s.sie2 * dt).min(s.sie2) / dt;
    s.sie2 -= qie * dt;

    q0 + q1 + q2 + qie
}

/// Mean over lanes, summed in order like `Tensor.mean(-1)` on a short axis.
pub fn lane_mean(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |acc, v| acc + v) / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_orders_dynamic_parameters_by_bounds() {
        let layout = ParamLayout::new(
            &HBV2_PARAMS,
            &["parBETAET".into(), "parBETA".into(), "parK0".into()],
            3,
        )
        .unwrap();
        assert_eq!(layout.dynamic, vec!["parBETA", "parK0", "parBETAET"]);
        assert_eq!(layout.n_dynamic(), 9);
        assert_eq!(layout.n_static(), 39);
        assert_eq!(layout.statics[0], "parFC");
    }

    #[test]
    fn hourly_layout_matches_hydrodl2_counts() {
        let dynamic: Vec<String> = ["parBETA", "parK0", "parBETAET", "parKS"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let layout = ParamLayout::new(&HBV21_HOURLY_PARAMS, &dynamic, 1).unwrap();
        assert_eq!(layout.n_dynamic(), 4);
        assert_eq!(layout.n_static(), 15);
        assert_eq!(&layout.statics[13..], &["parF0", "parALPHA"]);
    }

    #[test]
    fn dry_cold_step_conserves_snow() {
        let layout = ParamLayout::new(&HBV2_PARAMS, &[], 1).unwrap();
        let p = layout.static_params(&[0.5; 16])[0];
        let mut s = States::default();
        let catchment = Catchment {
            ac: 10.0,
            elevation: 100.0,
        };
        hbv2_step(&mut s, &p, &catchment, 5.0, -20.0, 0.0, 1e-5);
        assert!((s.snowpack - (INITIAL_STATE + 5.0)).abs() < 1e-3);
    }
}
