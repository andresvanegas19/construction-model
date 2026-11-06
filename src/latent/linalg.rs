//! Small dense linear algebra the analysis needs beyond candle's matmul: orthonormalising a tall matrix and the
//! eigen-decomposition of a small symmetric one (randomized SVD reduces PCA to an l x l problem, l <= ~70).
use anyhow::Result;
use candle_core::{Device, Tensor};

pub fn tensor(data: &[f32], rows: usize, cols: usize) -> Result<Tensor> {
    Ok(Tensor::from_slice(data, (rows, cols), &Device::Cpu)?)
}

pub fn to_rows(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

/// Modified Gram-Schmidt on the columns of an (n x l) row-major matrix, in place; a column that collapses (rank
/// deficiency) is zeroed rather than amplified noise.
pub fn orthonormalize_columns(m: &mut [f32], n: usize, l: usize) {
    for j in 0..l {
        for k in 0..j {
            let dot: f64 = (0..n).map(|i| m[i * l + j] as f64 * m[i * l + k] as f64).sum();
            for i in 0..n {
                m[i * l + j] -= (dot as f32) * m[i * l + k];
            }
        }
        let norm: f64 = (0..n).map(|i| (m[i * l + j] as f64).powi(2)).sum::<f64>().sqrt();
        let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
        for i in 0..n {
            m[i * l + j] = (m[i * l + j] as f64 * inv) as f32;
        }
    }
}

/// Cyclic Jacobi for a symmetric (l x l) matrix. Returns eigenvalues sorted descending and the matching eigenvectors
/// as columns of a row-major (l x l) matrix.
pub fn symmetric_eigen(a: &[f64], l: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0; l * l];
    for i in 0..l {
        v[i * l + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..l).flat_map(|i| (0..l).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i * l + j].powi(2)).sum();
        if off < 1e-18 {
            break;
        }
        for p in 0..l {
            for q in p + 1..l {
                let apq = a[p * l + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * l + q] - a[p * l + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..l {
                    let (akp, akq) = (a[k * l + p], a[k * l + q]);
                    a[k * l + p] = c * akp - s * akq;
                    a[k * l + q] = s * akp + c * akq;
                }
                for k in 0..l {
                    let (apk, aqk) = (a[p * l + k], a[q * l + k]);
                    a[p * l + k] = c * apk - s * aqk;
                    a[q * l + k] = s * apk + c * aqk;
                }
                for k in 0..l {
                    let (vkp, vkq) = (v[k * l + p], v[k * l + q]);
                    v[k * l + p] = c * vkp - s * vkq;
                    v[k * l + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..l).collect();
    order.sort_by(|&i, &j| a[j * l + j].partial_cmp(&a[i * l + i]).unwrap_or(std::cmp::Ordering::Equal));
    let values = order.iter().map(|&i| a[i * l + i]).collect();
    let mut vectors = vec![0.0; l * l];
    for (new, &old) in order.iter().enumerate() {
        for k in 0..l {
            vectors[k * l + new] = v[k * l + old];
        }
    }
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_of_diagonalisable() {
        let a = [2.0, 1.0, 1.0, 2.0];
        let (vals, vecs) = symmetric_eigen(&a, 2);
        assert!((vals[0] - 3.0).abs() < 1e-9 && (vals[1] - 1.0).abs() < 1e-9);
        assert!((vecs[0].abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
    }
}
