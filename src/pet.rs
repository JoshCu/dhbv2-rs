//! Potential evapotranspiration, ported from `dhbv2/pet.py`. Computed in f64, as in Python.

use std::f64::consts::PI;

/// Hourly FAO-56 Penman-Monteith PET (mm/h) from AORC forcings.
///
/// `temp` degC, `spfh` g/g (values > 0.02 are taken as g/kg), radiation W/m2, `pres` Pa, wind m/s.
#[allow(clippy::too_many_arguments)]
pub fn penman_monteith(
    temp: f64,
    spfh: f64,
    dlwrf: f64,
    dswrf: f64,
    pres: f64,
    ugrd: f64,
    vgrd: f64,
) -> f64 {
    const ALBEDO: f64 = 0.23;
    const EPS_S: f64 = 0.98;

    let spfh = if spfh > 0.02 { spfh / 1000.0 } else { spfh };
    let pressure = pres / 1000.0;

    let u10 = (ugrd * ugrd + vgrd * vgrd).sqrt();
    let u2 = u10 * 4.87 / (67.8 * 10.0 - 5.42f64).ln();

    let gamma = 0.000665 * pressure;
    let es = 0.6108 * ((17.27 * temp) / (temp + 237.3)).exp();
    let ea = (spfh * pressure) / (0.622 + 0.378 * spfh);
    let delta = (4098.0 * es) / (temp + 237.3).powi(2);

    let rs = dswrf * 0.0036;
    let rl_down = dlwrf * 0.0036;
    let sigma = 4.903e-9 / 24.0;
    let rl_up = EPS_S * sigma * (temp + 273.15).powf(4.0);
    let rn = (1.0 - ALBEDO) * rs + (rl_down - rl_up);

    let g = if rn > 0.0 { 0.1 * rn } else { 0.5 * rn };
    let cd = if rn > 0.0 { 0.24 } else { 0.96 };

    let num = 0.408 * delta * (rn - g) + gamma * (37.0 / (temp + 273.15)) * u2 * (es - ea);
    let denom = delta + gamma * (1.0 + cd * u2);
    (num / denom).max(0.0)
}

/// Daily Hargreaves PET (mm/d). `lat` in radians, `day_of_year` 1-366.
pub fn hargreaves(tmin: f64, tmax: f64, tmean: f64, lat: f64, day_of_year: f64) -> f64 {
    let trange = (tmax - tmin).max(0.0);
    (0.0023 * (tmean + 17.8) * trange.sqrt() * 0.408 * extraterrestrial_radiation(lat, day_of_year))
        .max(0.0)
}

/// Extraterrestrial radiation term of the Hargreaves equation (MJ m-2 d-1).
pub fn extraterrestrial_radiation(lat: f64, day_of_year: f64) -> f64 {
    const SOLAR_CONSTANT: f64 = 0.0820;
    let sol_dec = 0.409 * ((2.0 * PI / 365.0) * day_of_year - 1.39).sin();
    let sha = (-lat.tan() * sol_dec.tan()).clamp(-1.0, 1.0).acos();
    let ird = 1.0 + 0.033 * ((2.0 * PI / 365.0) * day_of_year).cos();
    (24.0 * 60.0 / PI)
        * SOLAR_CONSTANT
        * ird
        * (sha * lat.sin() * sol_dec.sin() + lat.cos() * sol_dec.cos() * sha.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penman_monteith_is_non_negative_at_night() {
        assert_eq!(
            penman_monteith(-10.0, 0.002, 200.0, 0.0, 101325.0, 1.0, 1.0),
            0.0
        );
    }

    #[test]
    fn hargreaves_summer_exceeds_winter() {
        let lat = 40f64.to_radians();
        let summer = hargreaves(15.0, 30.0, 22.5, lat, 180.0);
        let winter = hargreaves(-5.0, 5.0, 0.0, lat, 15.0);
        assert!(summer > winter && winter >= 0.0);
    }
}
