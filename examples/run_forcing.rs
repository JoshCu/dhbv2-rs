//! Drives a BMI over a forcing CSV, standing in for ngen.
//!
//! ```text
//! cargo run --release --example run_forcing -- <mts|daily> <bmi_config.yaml> <forcing.csv> <output.csv>
//! ```
//!
//! The forcing CSV needs a header and the columns `precip_rate` (mm/h), `TMP_2maboveground`
//! (degC), `SPFH_2maboveground`, `DLWRF_surface`, `DSWRF_surface`, `PRES_surface`,
//! `UGRD_10maboveground`, `VGRD_10maboveground` (any extra columns, e.g. `time`, are ignored).
//! `scripts/prepare_forcing.py` converts an ngen CSV (mm/s, K) into this form.

use std::error::Error;
use std::fs;
use std::io::Write;
use std::time::Instant;

use bmi_rs::bmi::{Bmi, RefValues, Values};
use rust_dhbv2::bmi::{DYNAMIC_INPUTS, OUTPUT};
use rust_dhbv2::{DailyBmi, MtsBmi};

const COLUMNS: [&str; 8] = [
    "precip_rate",
    "TMP_2maboveground",
    "SPFH_2maboveground",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
];

fn run(bmi: &mut dyn Bmi, config: &str, forcing: &str, output: &str) -> Result<(), Box<dyn Error>> {
    let t0 = Instant::now();
    bmi.initialize(config)?;
    let t_init = t0.elapsed();

    let text = fs::read_to_string(forcing)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or("empty forcing file")?
        .split(',')
        .collect();
    let idx: Vec<usize> = COLUMNS
        .iter()
        .map(|c| {
            header
                .iter()
                .position(|h| h == c)
                .ok_or(format!("forcing has no column {c}"))
        })
        .collect::<Result<_, _>>()?;

    let mut out = fs::File::create(output)?;
    writeln!(out, "step,runoff_m_per_h")?;
    let t1 = Instant::now();
    let mut steps = 0;
    for (step, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        for ((_, name, _), i) in DYNAMIC_INPUTS.iter().zip(&idx) {
            let value: f64 = fields[*i].trim().parse()?;
            bmi.set_value(name, RefValues::F64(&[value]))?;
        }
        bmi.update()?;
        let Values::F64(v) = bmi.get_value_at_indices(OUTPUT.0, &[0])? else {
            return Err("unexpected output type".into());
        };
        writeln!(out, "{step},{:e}", v[0])?;
        steps += 1;
    }
    bmi.finalize()?;
    eprintln!(
        "initialize: {:.3}s, {steps} updates: {:.3}s",
        t_init.as_secs_f64(),
        t1.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "usage: {} <mts|daily> <bmi_config.yaml> <forcing.csv> <output.csv>",
            args[0]
        );
        std::process::exit(2);
    }
    match args[1].as_str() {
        "mts" => run(&mut MtsBmi::new(), &args[2], &args[3], &args[4]),
        "daily" => run(&mut DailyBmi::new(), &args[2], &args[3], &args[4]),
        other => Err(format!("unknown model {other}; use mts or daily").into()),
    }
}
