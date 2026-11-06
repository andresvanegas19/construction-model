//! Centered PCA by randomized SVD (Halko et al.): project onto a random subspace a little wider than k, sharpen with
//! two power iterations, then solve the small eigenproblem exactly. Accurate for the leading components, which is
//! all a 2-D view and "how many PCs reach 90 %" need, at a fraction of a full (2048 x 2048) eigendecomposition per
//! (layer, point).
use anyhow::Result;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::linalg::{orthonormalize_columns, symmetric_eigen, tensor, to_rows};

pub struct Pca {
    pub mean: Vec<f32>,
    /// (dim x k) row-major: column c is component c.
    pub components: Vec<f32>,
    pub k: usize,
    pub dim: usize,
    /// Variance along each component.
    pub variance: Vec<f64>,
    /// Total variance of the centered data (sum over all dims), the denominator of explained ratios.
    pub total_variance: f64,
}

impl Pca {
    pub fn fit(data: &[f32], n: usize, dim: usize, k: usize, seed: u64) -> Result<Self> {
        let mean: Vec<f32> = (0..dim).map(|j| ((0..n).map(|i| data[i * dim + j] as f64).sum::<f64>() / n.max(1) as f64) as f32).collect();
        let centered: Vec<f32> = data.iter().enumerate().map(|(i, v)| v - mean[i % dim]).collect();
        let total_variance = centered.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64;
        let k = k.min(n).min(dim).max(1);
        let l = (k + 10).min(n).min(dim);
        let x = tensor(&centered, n, dim)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let omega: Vec<f32> = (0..dim * l).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let mut q = to_rows(&x.matmul(&tensor(&omega, dim, l)?)?)?;
        orthonormalize_columns(&mut q, n, l);
        for _ in 0..2 {
            let z = to_rows(&x.t()?.matmul(&tensor(&q, n, l)?)?)?;
            q = to_rows(&x.matmul(&tensor(&z, dim, l)?)?)?;
            orthonormalize_columns(&mut q, n, l);
        }
        // B = Qᵀ X is (l x dim); the eigenvectors of B Bᵀ give B's left singular vectors.
        let b = tensor(&q, n, l)?.t()?.matmul(&x)?;
        let bbt: Vec<f64> = to_rows(&b.matmul(&b.t()?)?)?.into_iter().map(f64::from).collect();
        let (vals, vecs) = symmetric_eigen(&bbt, l);
        let b_rows = to_rows(&b)?;
        let mut components = vec![0f32; dim * k];
        let mut variance = Vec::with_capacity(k);
        for c in 0..k {
            let sigma = vals[c].max(0.0).sqrt();
            variance.push(vals[c].max(0.0) / (n.max(2) - 1) as f64);
            if sigma < 1e-12 {
                continue;
            }
            // v_c = Bᵀ u_c / sigma_c
            for j in 0..dim {
                let s: f64 = (0..l).map(|r| b_rows[r * dim + j] as f64 * vecs[r * l + c]).sum();
                components[j * k + c] = (s / sigma) as f32;
            }
        }
        Ok(Self { mean, components, k, dim, variance, total_variance })
    }

    pub fn explained_ratio(&self) -> Vec<f64> {
        self.variance.iter().map(|v| if self.total_variance > 0.0 { v / self.total_variance } else { 0.0 }).collect()
    }

    /// Components needed to reach `frac` of the total variance (k + 1 when the fitted ones do not reach it).
    pub fn components_for(&self, frac: f64) -> usize {
        let mut acc = 0.0;
        for (i, r) in self.explained_ratio().iter().enumerate() {
            acc += r;
            if acc >= frac {
                return i + 1;
            }
        }
        self.k + 1
    }

    /// (n x k) scores, row-major.
    pub fn transform(&self, data: &[f32], n: usize) -> Result<Vec<f32>> {
        let centered: Vec<f32> = data.iter().enumerate().map(|(i, v)| v - self.mean[i % self.dim]).collect();
        to_rows(&tensor(&centered, n, self.dim)?.matmul(&tensor(&self.components, self.dim, self.k)?)?)
    }
}

/// Keeps the first `k` columns of an (n x m) row-major matrix.
pub fn take_cols(m: &[f32], n: usize, cols: usize, k: usize) -> Vec<f32> {
    (0..n).flat_map(|i| m[i * cols..i * cols + k.min(cols)].iter().copied()).collect()
}

pub fn to_xy(scores: &[f32], n: usize, k: usize) -> Vec<[f32; 2]> {
    (0..n).map(|i| [scores[i * k], if k > 1 { scores[i * k + 1] } else { 0.0 }]).collect()
}
