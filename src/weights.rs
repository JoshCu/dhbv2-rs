//! Reading PyTorch checkpoints (`torch.save(model.state_dict())`) with burn's PyTorch reader.
//!
//! Tensors are taken by their state-dict names and copied, in PyTorch layout, into the plain
//! f32 kernels of [`crate::nn`]. Every tensor in a checkpoint must be consumed by exactly one
//! layer, so a checkpoint that does not match the expected architecture is rejected instead of
//! being silently half-loaded.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use burn::store::{ModuleStore, PytorchStore, bridge};

use crate::BmiError;
use crate::nn::{Dense, Lstm, Mlp};

pub struct Checkpoint {
    path: PathBuf,
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    used: BTreeSet<String>,
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self, BmiError> {
        let mut store = PytorchStore::from_file(path).map_indices_contiguous(false);
        let all = store
            .get_all_tensors()
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let mut tensors = BTreeMap::new();
        for (name, tensor) in all {
            let data = bridge::to_data(tensor).map_err(|e| {
                format!("failed to read tensor {name} from {}: {e}", path.display())
            })?;
            let shape = data.shape().to_vec();
            let values = data.convert::<f32>().try_to_vec::<f32>().map_err(|e| {
                format!("tensor {name} in {} is not numeric: {e:?}", path.display())
            })?;
            tensors.insert(name.clone(), (shape, values));
        }
        Ok(Self {
            path: path.to_path_buf(),
            tensors,
            used: BTreeSet::new(),
        })
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], BmiError> {
        self.tensors
            .get(name)
            .map(|(shape, _)| shape.as_slice())
            .ok_or_else(|| format!("{} has no tensor {name}", self.path.display()).into())
    }

    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, BmiError> {
        let (actual, values) = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("{} has no tensor {name}", self.path.display()))?;
        if actual.as_slice() != shape {
            return Err(format!(
                "tensor {name} in {} has shape {actual:?}, expected {shape:?}",
                self.path.display()
            )
            .into());
        }
        self.used.insert(name.to_string());
        Ok(values.clone())
    }

    /// `torch.nn.Linear` at `prefix` (`{prefix}.weight: [out, in]`, `{prefix}.bias: [out]`).
    pub fn dense(&mut self, prefix: &str, d_in: usize, d_out: usize) -> Result<Dense, BmiError> {
        let weight = self.take(&format!("{prefix}.weight"), &[d_out, d_in])?;
        let bias = self.take(&format!("{prefix}.bias"), &[d_out])?;
        Ok(Dense::new(d_in, d_out, weight, bias))
    }

    /// `torch.nn.Linear` at `prefix`, sized from the checkpoint.
    pub fn dense_any(&mut self, prefix: &str) -> Result<Dense, BmiError> {
        let shape = self.shape(&format!("{prefix}.weight"))?.to_vec();
        if shape.len() != 2 {
            return Err(format!("{prefix}.weight is not a matrix").into());
        }
        self.dense(prefix, shape[1], shape[0])
    }

    /// `torch.nn.LSTM` (single layer) at `prefix`.
    pub fn torch_lstm(
        &mut self,
        prefix: &str,
        d_in: usize,
        hidden: usize,
    ) -> Result<Lstm, BmiError> {
        let g = 4 * hidden;
        let w_ih = self.take(&format!("{prefix}.weight_ih_l0"), &[g, d_in])?;
        let w_hh = self.take(&format!("{prefix}.weight_hh_l0"), &[g, hidden])?;
        let b_ih = self.take(&format!("{prefix}.bias_ih_l0"), &[g])?;
        let b_hh = self.take(&format!("{prefix}.bias_hh_l0"), &[g])?;
        Ok(Lstm::new(
            Dense::new(d_in, g, w_ih, b_ih),
            Dense::new(hidden, g, w_hh, b_hh),
        ))
    }

    /// δMG's custom `Lstm` (`w_ih`, `w_hh`, `b_ih`, `b_hh`) at `prefix`.
    pub fn dmg_lstm(&mut self, prefix: &str, d_in: usize, hidden: usize) -> Result<Lstm, BmiError> {
        let g = 4 * hidden;
        let w_ih = self.take(&format!("{prefix}.w_ih"), &[g, d_in])?;
        let w_hh = self.take(&format!("{prefix}.w_hh"), &[g, hidden])?;
        let b_ih = self.take(&format!("{prefix}.b_ih"), &[g])?;
        let b_hh = self.take(&format!("{prefix}.b_hh"), &[g])?;
        Ok(Lstm::new(
            Dense::new(d_in, g, w_ih, b_ih),
            Dense::new(hidden, g, w_hh, b_hh),
        ))
    }

    /// A `torch.nn.Sequential` MLP whose `Linear` layers sit at the given indices.
    pub fn sequential_mlp(&mut self, prefix: &str, indices: &[usize]) -> Result<Mlp, BmiError> {
        let layers = indices
            .iter()
            .map(|i| self.dense_any(&format!("{prefix}.{i}")))
            .collect::<Result<Vec<_>, _>>()?;
        check_chain(prefix, &layers)?;
        Ok(Mlp { layers })
    }

    /// An MLP made of named `Linear` layers (e.g. δMG's `AnnModel`).
    pub fn named_mlp(&mut self, prefix: &str, names: &[&str]) -> Result<Mlp, BmiError> {
        let layers = names
            .iter()
            .map(|n| self.dense_any(&format!("{prefix}.{n}")))
            .collect::<Result<Vec<_>, _>>()?;
        check_chain(prefix, &layers)?;
        Ok(Mlp { layers })
    }

    /// Fails if any tensor in the checkpoint was not loaded into the model.
    pub fn finish(self) -> Result<(), BmiError> {
        let unused: Vec<&String> = self
            .tensors
            .keys()
            .filter(|k| !self.used.contains(*k))
            .collect();
        if unused.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} holds tensors the model does not use: {unused:?}",
                self.path.display()
            )
            .into())
        }
    }
}

fn check_chain(prefix: &str, layers: &[Dense]) -> Result<(), BmiError> {
    for pair in layers.windows(2) {
        if pair[0].d_out != pair[1].d_in {
            return Err(format!(
                "layers of {prefix} do not chain ({} -> {})",
                pair[0].d_out, pair[1].d_in
            )
            .into());
        }
    }
    Ok(())
}
