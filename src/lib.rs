//! Rust BMI adapters for δHBV 2.0 (daily) and δHBV 2.1 MTS (hourly) inference.
//!
//! These read the published PyTorch weights, the model `config.yaml` and
//! `normalization_statistics.json` directly, and reproduce the forward pass of the Python BMIs in
//! [mhpi/dhbv2](https://github.com/mhpi/dhbv2) (`dhbv2.bmi.DeltaModelBmi` and
//! `dhbv2.mts_bmi.MtsDeltaModelBmi`) for use as a `bmi_c` / `bmi_rust` module in NextGen.
//!
//! Two registration functions are exported from the shared library:
//! - [`register_bmi_dhbv2_mts`]: hourly δHBV 2.1 MTS.
//! - [`register_bmi_dhbv2`]: daily δHBV 2.0.

use bmi_rs::bmi::register_model;

pub mod bmi;
pub mod config;
pub mod daily;
pub mod hbv;
pub mod mts;
pub mod nn;
pub mod pet;
pub mod ring;
pub mod weights;

pub use bmi::{DhbvBmi, Engine, HourlyForcing};
pub use daily::DailyEngine;
pub use mts::MtsEngine;

pub type BmiError = Box<dyn std::error::Error>;

/// Hourly δHBV 2.1 MTS BMI.
pub type MtsBmi = DhbvBmi<MtsEngine>;
/// Daily δHBV 2.0 BMI.
pub type DailyBmi = DhbvBmi<DailyEngine>;

/// The forcings both models are trained on, in order.
pub const FORCINGS: [&str; 3] = ["P", "T", "PET"];

/// Added to normalization standard deviations (`self.eps` in the Python BMIs).
pub const EPS: f32 = 1e-6;

/// Gaussian input normalization `(x - mean) / (std + eps)`.
#[derive(Clone, Debug)]
pub enum Normalizer {
    /// Statistics held in f32 (MTS BMI).
    F32 { mean: Vec<f32>, std_eps: Vec<f32> },
    /// Statistics held in f64 (daily BMI), result rounded to f32.
    F64 { mean: Vec<f64>, std_eps: Vec<f64> },
}

impl Normalizer {
    pub fn f32_stats(mean: &[f64], std: &[f64]) -> Result<Self, BmiError> {
        if mean.len() != std.len() {
            return Err("normalization mean/std lengths differ".into());
        }
        Ok(Self::F32 {
            mean: mean.iter().map(|m| *m as f32).collect(),
            std_eps: std.iter().map(|s| *s as f32 + EPS).collect(),
        })
    }

    pub fn f64_stats(mean: &[f64], std: &[f64]) -> Self {
        Self::F64 {
            mean: mean.to_vec(),
            std_eps: std.iter().map(|s| s + EPS as f64).collect(),
        }
    }

    pub fn apply(&self, row: &[f32]) -> Vec<f32> {
        match self {
            Self::F32 { mean, std_eps } => row
                .iter()
                .zip(mean)
                .zip(std_eps)
                .map(|((x, m), s)| (x - m) / s)
                .collect(),
            Self::F64 { mean, std_eps } => row
                .iter()
                .zip(mean)
                .zip(std_eps)
                .map(|((x, m), s)| ((*x as f64 - m) / s) as f32)
                .collect(),
        }
    }
}

/// Normalizes static attributes as the Python BMIs do: values in f64, statistics rounded to f32,
/// result rounded to f32.
pub fn normalize_static(values: &[f64], mean: &[f64], std: &[f64]) -> Result<Vec<f32>, BmiError> {
    if values.len() != mean.len() || values.len() != std.len() {
        return Err(format!(
            "{} static attributes but {} means and {} stds in normalization_statistics.json",
            values.len(),
            mean.len(),
            std.len()
        )
        .into());
    }
    Ok(values
        .iter()
        .zip(mean)
        .zip(std)
        .map(|((v, m), s)| ((v - *m as f32 as f64) / (*s as f32 + EPS) as f64) as f32)
        .collect())
}

/// NumPy's pairwise summation of f32 values (`np.sum` on a float32 array).
pub fn pairwise_sum(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        a.iter().fold(0.0f32, |acc, v| acc + v)
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &a[i..] {
            res += v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// Daily aggregate of 24 hourly `[P, T, PET]` rows: precipitation and PET summed, temperature
/// averaged.
pub fn aggregate_day(hours: &[[f32; 3]; 24]) -> (f32, f32, f32) {
    let column = |i: usize| hours.iter().map(|r| r[i]).collect::<Vec<f32>>();
    (
        pairwise_sum(&column(0)),
        pairwise_sum(&column(1)) / 24.0,
        pairwise_sum(&column(2)),
    )
}

/// Registers the hourly δHBV 2.1 MTS BMI (`registration_function` in the ngen realization).
#[unsafe(no_mangle)]
pub extern "C" fn register_bmi_dhbv2_mts(handle: *mut ffi::Bmi) -> *mut ffi::Bmi {
    register_model(handle, MtsBmi::new());
    handle
}

/// Registers the daily δHBV 2.0 BMI (`registration_function` in the ngen realization).
#[unsafe(no_mangle)]
pub extern "C" fn register_bmi_dhbv2(handle: *mut ffi::Bmi) -> *mut ffi::Bmi {
    register_model(handle, DailyBmi::new());
    handle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_matches_plain_sum_on_integers() {
        let values: Vec<f32> = (1..=300).map(|v| v as f32).collect();
        for n in [0, 5, 8, 24, 129, 300] {
            let expected: f32 = values[..n].iter().sum();
            assert_eq!(pairwise_sum(&values[..n]), expected);
        }
    }
}
