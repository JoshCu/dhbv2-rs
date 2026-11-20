# rust-dhbv2

A Rust BMI (Basic Model Interface) adapter for running **δHBV 2.0** (daily) and **δHBV 2.1 MTS**
(hourly) inference in NextGen with the published PyTorch weights. It replaces the Python BMIs in
[mhpi/dhbv2](https://github.com/mhpi/dhbv2) (`dhbv2.bmi.DeltaModelBmi` and
`dhbv2.mts_bmi.MtsDeltaModelBmi`). You don't need Python, PyTorch, δMG or hydrodl2 at runtime.

It works like [rust-lstm](https://github.com/CIROH-UA/rust-lstm). The library reads the model's
`.pt` checkpoint, `config.yaml` and `normalization_statistics.json` from the downloaded model
directory with no conversion step. It builds as a shared library that ngen loads through
`bmi_c`.

This adapter only runs inference. It does not train, and it does not support other model types.

## Build

```bash
curl https://sh.rustup.rs -sSf | bash -s -- -y && . "$HOME/.cargo/env"
cargo build --release
# shared object: target/release/librust_dhbv2.so
```

## Model weights

Use the same downloads as the Python package (see the dhbv2 docs, `1-module_setup.md`):

| Model | Download | Registration function |
| --- | --- | --- |
| δHBV 2.1 MTS (hourly) | `s3://mhpi-spatial/mhpi-release/models/owp/dhbv_2.1_mts.zip` | `register_bmi_dhbv2_mts` |
| δHBV 2.0 (daily) | `s3://mhpi-spatial/mhpi-release/models/owp/dhbv_2.zip` | `register_bmi_dhbv2` |

Unzip each archive. Then point `model_dir` in the BMI config at the folder that holds
`config.yaml`, `normalization_statistics.json` and the `.pt` file.

## BMI config

The adapter takes the same per-catchment YAML as the Python BMIs. The configs in
`dhbv2/ngen_resources/data/dhbv_2{_mts}/config/` work unchanged.

| Key | Used by | Notes |
| --- | --- | --- |
| static attributes (`aridity`, `meanP`, ..., `catchsize`, `lengthkm`) | both | A missing attribute is read as 0.0, with a warning if the model uses it. |
| `model_dir` | both | Resolved against the working directory first, then against the config file's directory. |
| `warmup.{cycle_days, daily_mode, daily_warmup_days, hourly_mode, hourly_warmup_hours}` | MTS | Same meaning, defaults and validation as the Python MTS BMI. |
| `start_time` (`YYYY/MM/DD HH[:MM[:SS]]`) | daily | Required. Sets the hour of day for daily aggregation and the day of year for Hargreaves. |
| `pet_method` (`penman_monteith` or `hargreaves`) and `latitude` | daily | The MTS model always uses Penman-Monteith, as in Python. |
| `dtype` | both | Inference always runs in float32, the weights' precision. Other values trigger a warning. |

The adapter ignores other keys, such as `serialization`, `verbose` and `catchment_id`.

## NextGen realization

```json
{
  "name": "bmi_c",
  "params": {
    "name": "bmi_c",
    "model_type_name": "dhbv2_mts_rust",
    "library_file": "/dmod/shared_libs/librust_dhbv2.so",
    "registration_function": "register_bmi_dhbv2_mts",
    "init_config": "./data/dhbv_2_mts/config/bmi_{{id}}.yaml",
    "main_output_variable": "land_surface_water__runoff_volume_flux",
    "uses_forcing_file": false,
    "allow_exceed_end_time": true,
    "fixed_time_step": true,
    "variables_names_map": {
      "atmosphere_water__liquid_equivalent_precipitation_rate": "precip_rate",
      "land_surface_air__temperature": "TMP_2maboveground",
      "atmosphere_air_water~vapor__relative_saturation": "SPFH_2maboveground",
      "land_surface_radiation~incoming~longwave__energy_flux": "DLWRF_surface",
      "land_surface_radiation~incoming~shortwave__energy_flux": "DSWRF_surface",
      "land_surface_air__pressure": "PRES_surface",
      "land_surface_wind__x_component_of_velocity": "UGRD_10maboveground",
      "land_surface_wind__y_component_of_velocity": "VGRD_10maboveground",
      "land_surface_water__runoff_volume_flux": "streamflow"
    }
  }
}
```

For the daily model, use `"registration_function": "register_bmi_dhbv2"` and set the daily
`output_interval` (86400).

### Inputs and outputs

The inputs are the eight AORC forcings, which are the same CSDMS names and units as the Python
BMIs: precipitation in mm h-1, temperature in degC, specific humidity, longwave and shortwave
radiation, pressure, and both wind components. Both models need all eight unless the daily model
uses Hargreaves PET. In that case it only needs precipitation and temperature.

The single output is `land_surface_water__runoff_volume_flux`. As in Python, its value is a
runoff depth rate in **m h-1**, despite the variable's name. Both models emit zero until they
have warmed up:

- MTS: ~365 days with the shipped config.
- Daily: `rho + 1` = 366 days.

## How it maps onto the Python implementation

| Python | Rust |
| --- | --- |
| `MtsDeltaModelBmi` rolling buffers, warmup trigger and modes (`periodic` / `once` / `cold`) | `src/mts.rs` |
| `DeltaModelBmi` daily aggregation, batched warmup, then one step per day | `src/daily.rs` |
| δMG `StackLstmMlpModel` / `LstmMlpModel` / `LstmMlp2Model`, `LstmModel` + `AnnModel` | `src/nn.rs` (weights loaded by `src/weights.rs` through burn's `PytorchStore`) |
| hydrodl2 `Hbv_2`, `Hbv_2_1_hourly`, `Hbv_2_mts` parameter and state transfer | `src/hbv.rs`, `src/mts.rs` |
| `dhbv2/pet.py` | `src/pet.rs` |

Differences from the Python BMIs:

- **No checkpoint/restore**: the ngen serialization protocol (`ngen::serialization_*`) is not
  implemented.
- **Static parameters are computed once**: the static-parameter MLPs depend only on catchment
  attributes, so they run once in `initialize`. This includes the daily model's 4096-wide MLP,
  which Python reruns every day. Setting static attributes through `set_value` after
  `initialize` therefore has no effect. Put them in the config instead.
- **No distributed-routing MLP output**: the MTS distributed-routing MLP (`ann2`) is loaded and
  checked but not evaluated. The Python BMI outputs unrouted catchment runoff (`Qs`), so that
  MLP never affects the output.

## Validation

`examples/run_forcing.rs` stands in for ngen. It sets the forcings, calls `update`, and reads
the output. `scripts/` holds a matching Python driver and a comparison script:

```bash
python scripts/prepare_forcing.py "<ngen forcing csv>" forcing.csv
cargo run --release --example run_forcing -- mts bmi_cat-2453.yaml forcing.csv rust.csv
python scripts/run_python_reference.py mts bmi_cat-2453.yaml forcing.csv python.csv  # needs dhbv2
python scripts/compare.py python.csv rust.csv
```

Results against the Python BMIs (dhbv2 `add93d5`, torch 2.15 CPU) on `cat-2453` with the
shipped forcing (2008-01-09 to 2010-12-30, 26,088 hourly steps):

| Model / settings | First output step (Rust = Python) | Max relative error | NSE vs Python | Rust time | Python time |
| --- | --- | --- | --- | --- | --- |
| MTS, shipped config (periodic/periodic, 14-day cycle) | 8760 | 1.8e-6 | 1.000000000 | 0.54 s | 258 s |
| MTS, `daily_mode: once` | 8760 | 2.1e-6 | 1.000000000 | 0.4 s | n/a |
| MTS, `daily_mode: cold`, `hourly_mode: cold` | 336 | 2.0e-5 | 1.000000000 | 0.4 s | n/a |
| Daily, Hargreaves PET | 8783 | 1.1e-6 | 1.000000000 | 1.3 s init + 0.14 s | 2.5 s init + 209 s |
| Daily, Penman-Monteith PET | 8783 | 1.1e-6 | 1.000000000 | 1.2 s init + 0.13 s | 2.4 s init + 211 s |

The remaining differences come from float32 rounding (operation order in matrix products, `exp`
and `pow`). The Python timings were measured with several runs going at once. The "n/a" runs
were not timed in isolation.

## Acknowledgments

The model architecture, physics, and the Python BMIs this adapter ports come from
[mhpi/dhbv2](https://github.com/mhpi/dhbv2), [mhpi/generic_deltamodel](https://github.com/mhpi/generic_deltamodel)
and [mhpi/hydrodl2](https://github.com/mhpi/hydrodl2). Please cite their publications
(Song et al., 2025; Yang et al., 2026). BMI bindings come from
[bmi-rs](https://github.com/aaraney/bmi-rs).

## License

License pending. This adapter is a port of the dhbv2, δMG and hydrodl2 code, so check their
license terms (dhbv2 uses a non-commercial PSU license) before you choose one.
