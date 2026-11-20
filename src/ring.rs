//! Fixed-capacity circular buffer of forcing rows (`dhbv2.utils.RingBuffer` for one catchment).

#[derive(Clone, Debug)]
pub struct RingBuffer {
    width: usize,
    capacity: usize,
    data: Vec<f32>,
    ptr: usize,
    full: bool,
}

impl RingBuffer {
    pub fn new(capacity: usize, width: usize) -> Self {
        assert!(capacity > 0 && width > 0);
        Self {
            width,
            capacity,
            data: vec![0.0; capacity * width],
            ptr: 0,
            full: false,
        }
    }

    pub fn push(&mut self, row: &[f32]) {
        assert_eq!(row.len(), self.width);
        self.data[self.ptr * self.width..(self.ptr + 1) * self.width].copy_from_slice(row);
        self.ptr = (self.ptr + 1) % self.capacity;
        if self.ptr == 0 {
            self.full = true;
        }
    }

    pub fn len(&self) -> usize {
        if self.full { self.capacity } else { self.ptr }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rows from oldest to newest, flattened.
    pub fn ordered(&self) -> Vec<f32> {
        if !self.full {
            return self.data[..self.ptr * self.width].to_vec();
        }
        let split = self.ptr * self.width;
        let mut out = Vec::with_capacity(self.data.len());
        out.extend_from_slice(&self.data[split..]);
        out.extend_from_slice(&self.data[..split]);
        out
    }

    /// The most recent row (zeros if empty).
    pub fn last(&self) -> Vec<f32> {
        if self.is_empty() {
            return vec![0.0; self.width];
        }
        let idx = (self.ptr + self.capacity - 1) % self.capacity;
        self.data[idx * self.width..(idx + 1) * self.width].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::RingBuffer;

    #[test]
    fn keeps_the_newest_rows_in_order() {
        let mut ring = RingBuffer::new(3, 1);
        assert_eq!(ring.last(), vec![0.0]);
        for v in 1..=2 {
            ring.push(&[v as f32]);
        }
        assert_eq!(ring.ordered(), vec![1.0, 2.0]);
        for v in 3..=5 {
            ring.push(&[v as f32]);
        }
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.ordered(), vec![3.0, 4.0, 5.0]);
        assert_eq!(ring.last(), vec![5.0]);
    }
}
