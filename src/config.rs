//! Minimal config loading: the per-catchment BMI YAML, the model's `config.yaml` (shipped with the
//! weights), and `normalization_statistics.json`.

use std::fs;
use std::path::{Path, PathBuf};

use serde_yaml::Value;

use crate::BmiError;

/// Looks up a nested key, e.g. `get(&cfg, &["model", "nn", "forcings"])`.
pub fn get<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(value, |v, key| v.get(*key))
}

pub fn req<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value, BmiError> {
    get(value, path).ok_or_else(|| format!("model config is missing {}", path.join(".")).into())
}

pub fn req_str_list(value: &Value, path: &[&str]) -> Result<Vec<String>, BmiError> {
    req(value, path)?
        .as_sequence()
        .ok_or_else(|| format!("{} must be a list", path.join(".")))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{} must hold strings", path.join(".")).into())
        })
        .collect()
}

pub fn req_usize(value: &Value, path: &[&str]) -> Result<usize, BmiError> {
    req(value, path)?
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("{} must be a non-negative integer", path.join(".")).into())
}

/// A float that PyYAML may have read as a string (YAML 1.1 parses `1e-5` as a string; the Python
/// BMI calls `float()` on it).
pub fn as_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

pub fn opt_bool(value: &Value, path: &[&str], default: bool) -> Result<bool, BmiError> {
    match get(value, path) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| format!("{} must be true or false", path.join(".")).into()),
    }
}

pub fn read_yaml(path: &Path) -> Result<Value, BmiError> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_yaml::from_str(&text)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()).into())
}

/// The per-catchment BMI config passed to `initialize`.
pub struct BmiConfig {
    pub path: PathBuf,
    pub raw: Value,
}

impl BmiConfig {
    pub fn load(path: &str) -> Result<Self, BmiError> {
        let path = PathBuf::from(path);
        let raw = read_yaml(&path)?;
        if !raw.is_mapping() {
            return Err(format!("{} must be a YAML mapping", path.display()).into());
        }
        Ok(Self { path, raw })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.raw.get(key).filter(|v| !v.is_null())
    }

    pub fn float(&self, key: &str) -> Result<Option<f64>, BmiError> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => as_float(v)
                .map(Some)
                .ok_or_else(|| format!("BMI config key '{key}' must be a number").into()),
        }
    }

    pub fn string(&self, key: &str) -> Result<Option<String>, BmiError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(v @ Value::Number(_)) => Ok(Some(serde_yaml::to_string(v)?.trim().to_string())),
            Some(_) => Err(format!("BMI config key '{key}' must be a string").into()),
        }
    }

    /// `model_dir`, used as given (i.e. relative to the working directory, as ngen runs it) and,
    /// failing that, relative to the BMI config's own directory.
    pub fn model_dir(&self) -> Result<PathBuf, BmiError> {
        let dir = self
            .string("model_dir")?
            .ok_or("BMI config is missing 'model_dir'")?;
        let as_given = PathBuf::from(&dir);
        if as_given.is_dir() {
            return Ok(as_given);
        }
        if let Some(parent) = self.path.parent() {
            let beside = parent.join(&dir);
            if beside.is_dir() {
                return Ok(beside);
            }
        }
        Err(format!("model_dir '{dir}' does not exist").into())
    }

    /// Warns about dtypes other than float32: inference always runs in f32, like the trained
    /// weights and the shipped configs.
    pub fn check_dtype(&self) -> Result<(), BmiError> {
        match self.string("dtype")?.as_deref() {
            None | Some("float32") => {}
            Some(other) => eprintln!(
                "rust-dhbv2: dtype '{other}' requested; inference runs in float32 (the weights' precision)"
            ),
        }
        Ok(())
    }
}

/// Finds the checkpoint in `model_dir`: `{stem}_ep{epoch}.pt` when the config names a test epoch,
/// otherwise the only `.pt` file in the directory.
pub fn find_weights(model_dir: &Path, stem: &str, epoch: Option<u64>) -> Result<PathBuf, BmiError> {
    if let Some(epoch) = epoch {
        let path = model_dir.join(format!("{stem}_ep{epoch}.pt"));
        if path.is_file() {
            return Ok(path);
        }
    }
    let mut candidates: Vec<PathBuf> = fs::read_dir(model_dir)
        .map_err(|e| format!("failed to list {}: {e}", model_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "pt"))
        .collect();
    candidates.sort();
    match candidates.len() {
        1 => Ok(candidates.remove(0)),
        0 => Err(format!("no .pt weights found in {}", model_dir.display()).into()),
        _ => Err(format!(
            "several .pt files in {} and none named {stem}_ep{}.pt",
            model_dir.display(),
            epoch
                .map(|e| e.to_string())
                .unwrap_or_else(|| "<epoch>".into())
        )
        .into()),
    }
}

pub fn read_json(path: &Path) -> Result<serde_json::Value, BmiError> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()).into())
}

pub fn json_f64_list(value: &serde_json::Value, what: &str) -> Result<Vec<f64>, BmiError> {
    value
        .as_array()
        .ok_or_else(|| format!("{what} must be a list"))?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| format!("{what} must hold numbers").into())
        })
        .collect()
}

/// A calendar date-time parsed from `YYYY/MM/DD HH[:MM[:SS]]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DateTime {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
}

impl DateTime {
    pub fn parse(text: &str) -> Result<Self, BmiError> {
        let err =
            || format!("'start_time' must be in format 'YYYY/MM/DD HH[[:MM]:SS]', got '{text}'");
        let (date, time) = text.trim().split_once(' ').ok_or_else(err)?;
        let mut date_parts = date.split('/');
        let year: i64 = date_parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(err)?;
        let month: u32 = date_parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(err)?;
        let day: u32 = date_parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(err)?;
        let time_parts: Vec<&str> = time.trim().split(':').collect();
        if date_parts.next().is_some() || time_parts.is_empty() || time_parts.len() > 3 {
            return Err(err().into());
        }
        let nums: Vec<u32> = time_parts
            .iter()
            .map(|s| s.parse().map_err(|_| err()))
            .collect::<Result<_, _>>()?;
        let valid = (1..=12).contains(&month)
            && day >= 1
            && day <= days_in_month(year, month)
            && nums[0] < 24
            && nums.iter().skip(1).all(|v| *v < 60);
        if !valid {
            return Err(err().into());
        }
        Ok(Self {
            year,
            month,
            day,
            hour: nums[0],
        })
    }

    /// Days since 1970-01-01 of this date (proleptic Gregorian).
    pub fn days_from_epoch(&self) -> i64 {
        days_from_civil(self.year, self.month, self.day)
    }
}

fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        _ => 28,
    }
}

/// Howard Hinnant's `days_from_civil`.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = month as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Inverse of [`days_from_civil`]: (year, month, day).
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    (year, month, day)
}

/// Day of year (1-366) of the date `days` after 1970-01-01.
pub fn day_of_year(days: i64) -> u32 {
    let (year, _, _) = civil_from_days(days);
    (days - days_from_civil(year, 1, 1) + 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_start_times() {
        let dt = DateTime::parse("2008/01/09 00").unwrap();
        assert_eq!((dt.year, dt.month, dt.day, dt.hour), (2008, 1, 9, 0));
        assert_eq!(DateTime::parse("2008/01/09 13:30").unwrap().hour, 13);
        assert_eq!(DateTime::parse("2008/02/29 05:00:00").unwrap().day, 29);
        assert!(DateTime::parse("2009/02/29 00").is_err());
        assert!(DateTime::parse("2008-01-09 00").is_err());
    }

    #[test]
    fn day_of_year_round_trips() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(day_of_year(days_from_civil(2008, 12, 31)), 366);
        assert_eq!(day_of_year(days_from_civil(2009, 3, 1)), 60);
        for d in [-1000, 0, 13879, 20000] {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
    }
}
