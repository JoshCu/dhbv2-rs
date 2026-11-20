//! Plain-f32 inference kernels for the parameterization networks.
//!
//! Weights are loaded from the PyTorch checkpoints with burn (see [`crate::weights`]) and copied
//! into these structs. Every network here runs with a batch of one catchment and hidden sizes of
//! 32-64, where hand-written loops are faster than dispatching tensor ops; the only large layer
//! (the daily model's 4096-wide MLP) depends on static attributes alone and is run once at
//! initialization.

/// Fully connected layer `y = W x + b`, with `W` stored row-major as `[out, in]` (PyTorch layout).
#[derive(Clone, Debug)]
pub struct Dense {
    pub d_in: usize,
    pub d_out: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Dense {
    pub fn new(d_in: usize, d_out: usize, weight: Vec<f32>, bias: Vec<f32>) -> Self {
        assert_eq!(weight.len(), d_in * d_out, "dense weight shape");
        assert_eq!(bias.len(), d_out, "dense bias shape");
        Self {
            d_in,
            d_out,
            weight,
            bias,
        }
    }

    pub fn forward_into(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.d_in);
        debug_assert_eq!(y.len(), self.d_out);
        for (o, out) in y.iter_mut().enumerate() {
            let row = &self.weight[o * self.d_in..(o + 1) * self.d_in];
            *out = dot(row, x) + self.bias[o];
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0; self.d_out];
        self.forward_into(x, &mut y);
        y
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight independent accumulators let the compiler vectorise the loop.
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        for k in 0..8 {
            acc[k] += a[c * 8 + k] * b[c * 8 + k];
        }
    }
    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in chunks * 8..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Hidden and cell state of a single-layer LSTM.
#[derive(Clone, Debug, PartialEq)]
pub struct LstmState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

impl LstmState {
    pub fn zeros(hidden: usize) -> Self {
        Self {
            h: vec![0.0; hidden],
            c: vec![0.0; hidden],
        }
    }
}

/// Single-layer LSTM with PyTorch's fused gate layout (`i, f, g, o` stacked along the rows of
/// `weight_ih_l0: [4H, I]` and `weight_hh_l0: [4H, H]`).
#[derive(Clone, Debug)]
pub struct Lstm {
    pub hidden: usize,
    pub input_transform: Dense,
    pub hidden_transform: Dense,
}

impl Lstm {
    pub fn new(input_transform: Dense, hidden_transform: Dense) -> Self {
        let hidden = hidden_transform.d_in;
        assert_eq!(input_transform.d_out, 4 * hidden, "LSTM input gate width");
        assert_eq!(hidden_transform.d_out, 4 * hidden, "LSTM hidden gate width");
        Self {
            hidden,
            input_transform,
            hidden_transform,
        }
    }

    /// Advances `state` by one timestep of input `x`.
    pub fn step(&self, x: &[f32], state: &mut LstmState) {
        let h = self.hidden;
        let gx = self.input_transform.forward(x);
        let gh = self.hidden_transform.forward(&state.h);
        for j in 0..h {
            let i = sigmoid(gx[j] + gh[j]);
            let f = sigmoid(gx[h + j] + gh[h + j]);
            let g = (gx[2 * h + j] + gh[2 * h + j]).tanh();
            let o = sigmoid(gx[3 * h + j] + gh[3 * h + j]);
            let c = f * state.c[j] + i * g;
            state.c[j] = c;
            state.h[j] = o * c.tanh();
        }
    }
}

/// A stack of dense layers with ReLU between them and a sigmoid on the output, i.e. the
/// `torch.nn.Sequential(Linear, ReLU, Dropout, ..., Linear, Sigmoid)` MLPs of δMG (dropout is
/// inactive at inference).
#[derive(Clone, Debug)]
pub struct Mlp {
    pub layers: Vec<Dense>,
}

impl Mlp {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = x.to_vec();
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out);
            if i < last {
                out.iter_mut().for_each(|v| *v = relu(*v));
            }
        }
        out.iter_mut().for_each(|v| *v = sigmoid(*v));
        out
    }
}

/// δMG `LstmMlpModel` / `LstmMlp2Model` (from `stack_lstm_mlp.py`): an input projection, an LSTM
/// and a sigmoid head producing the time-dynamic physics parameters. The static-parameter MLPs
/// are held separately.
#[derive(Clone, Debug)]
pub struct ProjLstm {
    /// `in_proj` (no activation) when `use_in_proj`, otherwise `None`.
    pub in_proj: Option<Dense>,
    pub lstm: Lstm,
    pub linear_out: Dense,
}

impl ProjLstm {
    pub fn hidden(&self) -> usize {
        self.lstm.hidden
    }

    /// One step: returns the sigmoid-activated dynamic parameters.
    pub fn step(&self, x: &[f32], state: &mut LstmState) -> Vec<f32> {
        match &self.in_proj {
            Some(proj) => self.lstm.step(&proj.forward(x), state),
            None => self.lstm.step(x, state),
        }
        let mut out = self.linear_out.forward(&state.h);
        out.iter_mut().for_each(|v| *v = sigmoid(*v));
        out
    }

    /// Runs over a sequence of `x.len() / n_in` steps; returns `[steps, n_out]` row-major.
    pub fn run(&self, x: &[f32], n_in: usize, state: &mut LstmState) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len() / n_in * self.linear_out.d_out);
        for row in x.chunks_exact(n_in) {
            out.extend(self.step(row, state));
        }
        out
    }
}

/// δMG `LstmModel` (from `lstm.py`), used by the daily δHBV 2.0 model:
/// `linear_out(LSTM(relu(linear_in(x))))`, followed by a sigmoid in `LstmMlpModel`.
#[derive(Clone, Debug)]
pub struct ReluLstm {
    pub linear_in: Dense,
    pub lstm: Lstm,
    pub linear_out: Dense,
}

impl ReluLstm {
    pub fn hidden(&self) -> usize {
        self.lstm.hidden
    }

    pub fn step(&self, x: &[f32], state: &mut LstmState) -> Vec<f32> {
        let mut z = self.linear_in.forward(x);
        z.iter_mut().for_each(|v| *v = relu(*v));
        self.lstm.step(&z, state);
        let mut out = self.linear_out.forward(&state.h);
        out.iter_mut().for_each(|v| *v = sigmoid(*v));
        out
    }

    pub fn run(&self, x: &[f32], n_in: usize, state: &mut LstmState) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len() / n_in * self.linear_out.d_out);
        for row in x.chunks_exact(n_in) {
            out.extend(self.step(row, state));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_matches_naive() {
        let d_in = 13;
        let d_out = 3;
        let weight: Vec<f32> = (0..d_in * d_out).map(|i| (i as f32 * 0.37).sin()).collect();
        let bias = vec![0.1, -0.2, 0.3];
        let x: Vec<f32> = (0..d_in).map(|i| i as f32 * 0.1 - 0.5).collect();
        let layer = Dense::new(d_in, d_out, weight.clone(), bias.clone());
        let y = layer.forward(&x);
        for o in 0..d_out {
            let naive: f32 = (0..d_in).map(|i| weight[o * d_in + i] * x[i]).sum::<f32>() + bias[o];
            assert!((y[o] - naive).abs() < 1e-5);
        }
    }

    #[test]
    fn lstm_zero_weights_keep_zero_state() {
        let h = 4;
        let lstm = Lstm::new(
            Dense::new(2, 4 * h, vec![0.0; 8 * h], vec![0.0; 4 * h]),
            Dense::new(h, 4 * h, vec![0.0; 4 * h * h], vec![0.0; 4 * h]),
        );
        let mut state = LstmState::zeros(h);
        lstm.step(&[1.0, -1.0], &mut state);
        assert!(state.h.iter().all(|v| *v == 0.0));
        assert!(state.c.iter().all(|v| *v == 0.0));
    }
}
