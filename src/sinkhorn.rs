//! Sinkhorn algorithm for entropic optimal transport.
//!
//! Computes soft permutation matrices via entropy-regularized optimal transport.
//! This is the "entropic" counterpart to PAVA's "L2" projection.
//!
//! # Mathematical Background
//!
//! Solves the regularized optimal transport problem:
//!
//! ```text
//! min_P <P, C> - ε H(P)  s.t. P is doubly-stochastic
//! ```
//!
//! The Sinkhorn algorithm alternates row/column normalization:
//!
//! ```text
//! u_i = 1 / Σⱼ K_ij v_j
//! v_j = 1 / Σᵢ K_ij u_i
//! ```
//!
//! where K_ij = exp(-C_ij / ε) is the Gibbs kernel.
//!
//! # Two Approaches to Differentiable Sorting
//!
//! | Method | Regularization | Projection Target | Complexity |
//! |--------|---------------|-------------------|------------|
//! | PAVA   | L2            | Permutahedron     | O(n)       |
//! | Sinkhorn | Entropy     | Birkhoff Polytope | O(n² × iter) |
//!
//! PAVA is faster but produces piecewise-linear gradients.
//! Sinkhorn is smoother but more expensive.
//!
//! # References
//!
//! - Cuturi (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
//! - Mena et al. (2018). "Learning Latent Permutations with Gumbel-Sinkhorn Networks"
//! - Adams & Zemel (2011). "Ranking via Sinkhorn Propagation"

use crate::{Error, Result};

/// Configuration for Sinkhorn iterations.
#[derive(Debug, Clone)]
pub struct SinkhornConfig {
    /// Regularization strength (epsilon). Smaller = sharper permutations.
    pub epsilon: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence threshold for marginal error.
    pub tol: f64,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iter: 100,
            tol: 1e-6,
        }
    }
}

/// Compute soft permutation matrix via Sinkhorn algorithm.
///
/// Given input values, computes a doubly-stochastic matrix P such that
/// P @ sorted_positions gives soft ranks.
///
/// # Algorithm
///
/// 1. Build cost matrix: C[i,j] = (x_i - sorted_x_j)²
/// 2. Gibbs kernel: K = exp(-C / ε)
/// 3. Sinkhorn iterations until doubly-stochastic
///
/// # Arguments
///
/// * `values` - Input values to sort
/// * `config` - Sinkhorn configuration
///
/// # Returns
///
/// A doubly-stochastic matrix P as a flat Vec (row-major, n×n).
///
/// # Example
///
/// ```rust
/// use fynch::sinkhorn::{sinkhorn_permutation, SinkhornConfig};
///
/// let values = vec![3.0, 1.0, 2.0];
/// let config = SinkhornConfig { epsilon: 0.1, ..Default::default() };
/// let p = sinkhorn_permutation(&values, &config).unwrap();
///
/// // P is 3x3, each row and column sums to ~1
/// let n = values.len();
/// for i in 0..n {
///     let row_sum: f64 = (0..n).map(|j| p[i * n + j]).sum();
///     assert!((row_sum - 1.0).abs() < 0.01);
/// }
/// ```
pub fn sinkhorn_permutation(values: &[f64], config: &SinkhornConfig) -> Result<Vec<f64>> {
    let n = values.len();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if config.epsilon <= 0.0 {
        return Err(Error::InvalidTemperature(config.epsilon));
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }

    // Get sorted order
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Cost matrix: C[i,j] = |value_i - sorted_value_j|²
    let mut cost = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let target_idx = sorted_indices[j];
            let diff = values[i] - values[target_idx];
            cost[i * n + j] = diff * diff;
        }
    }

    // Gibbs kernel: K[i,j] = exp(-C[i,j] / epsilon)
    let mut kernel = vec![0.0; n * n];
    let inv_eps = 1.0 / config.epsilon;
    for idx in 0..n * n {
        kernel[idx] = (-cost[idx] * inv_eps).exp();
    }

    // Initialize scaling vectors
    let mut u = vec![1.0; n];
    let mut v = vec![1.0; n];

    // Sinkhorn iterations
    for _ in 0..config.max_iter {
        // Update u: u_i = 1 / Σⱼ K_ij v_j
        let mut u_new = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += kernel[i * n + j] * v[j];
            }
            u_new[i] = if sum > 1e-10 { 1.0 / sum } else { 1.0 };
        }

        // Update v: v_j = 1 / Σᵢ K_ij u_i
        let mut v_new = vec![0.0; n];
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                sum += kernel[i * n + j] * u_new[i];
            }
            v_new[j] = if sum > 1e-10 { 1.0 / sum } else { 1.0 };
        }

        // Check convergence
        let mut max_err: f64 = 0.0;
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += u_new[i] * kernel[i * n + j] * v_new[j];
            }
            max_err = max_err.max((row_sum - 1.0).abs());
        }

        u = u_new;
        v = v_new;

        if max_err < config.tol {
            break;
        }
    }

    // Build final doubly-stochastic matrix P = diag(u) @ K @ diag(v)
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            p[i * n + j] = u[i] * kernel[i * n + j] * v[j];
        }
    }

    Ok(p)
}

/// Compute soft ranks via Sinkhorn optimal transport.
///
/// # Arguments
///
/// * `values` - Input values
/// * `epsilon` - Regularization strength. Smaller = sharper ranks.
///
/// # Returns
///
/// Soft ranks in [0, n-1]. Lower rank = smaller value.
///
/// # Example
///
/// ```rust
/// use fynch::sinkhorn::sinkhorn_rank;
///
/// let values = vec![3.0, 1.0, 2.0];
/// let ranks = sinkhorn_rank(&values, 0.1).unwrap();
///
/// // 1.0 is smallest → rank ~0
/// // 2.0 is middle → rank ~1  
/// // 3.0 is largest → rank ~2
/// assert!(ranks[1] < ranks[2]);
/// assert!(ranks[2] < ranks[0]);
/// ```
pub fn sinkhorn_rank(values: &[f64], epsilon: f64) -> Result<Vec<f64>> {
    let n = values.len();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if n == 1 {
        return Ok(vec![0.0]);
    }

    let config = SinkhornConfig {
        epsilon,
        ..Default::default()
    };

    let p = sinkhorn_permutation(values, &config)?;

    // Soft rank = P @ [0, 1, 2, ..., n-1]
    let mut ranks = vec![0.0; n];
    for i in 0..n {
        let mut rank = 0.0;
        for j in 0..n {
            rank += p[i * n + j] * j as f64;
        }
        ranks[i] = rank;
    }

    Ok(ranks)
}

/// Compute soft sorted values via Sinkhorn optimal transport.
///
/// # Arguments
///
/// * `values` - Input values
/// * `epsilon` - Regularization strength
///
/// # Returns
///
/// Soft sorted values (approximately monotonic).
///
/// # Example
///
/// ```rust
/// use fynch::sinkhorn::sinkhorn_sort;
///
/// let values = vec![3.0, 1.0, 2.0];
/// let sorted = sinkhorn_sort(&values, 0.1).unwrap();
///
/// // Should be approximately [1.0, 2.0, 3.0]
/// assert!(sorted[0] < sorted[1]);
/// assert!(sorted[1] < sorted[2]);
/// ```
pub fn sinkhorn_sort(values: &[f64], epsilon: f64) -> Result<Vec<f64>> {
    let n = values.len();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if n == 1 {
        return Ok(values.to_vec());
    }

    let config = SinkhornConfig {
        epsilon,
        ..Default::default()
    };

    let p = sinkhorn_permutation(values, &config)?;

    // Soft sorted at position j = Σᵢ P[i,j] * values[i]
    let mut sorted = vec![0.0; n];
    for j in 0..n {
        let mut val = 0.0;
        for i in 0..n {
            val += p[i * n + j] * values[i];
        }
        sorted[j] = val;
    }

    Ok(sorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinkhorn_rank_basic() {
        let values = vec![3.0, 1.0, 2.0];
        let ranks = sinkhorn_rank(&values, 0.1).unwrap();

        // Hard ranks: [2, 0, 1]
        assert!(
            ranks[0] > 1.5,
            "largest should have rank ~2, got {}",
            ranks[0]
        );
        assert!(
            ranks[1] < 0.5,
            "smallest should have rank ~0, got {}",
            ranks[1]
        );
        assert!(
            ranks[2] > 0.5 && ranks[2] < 1.5,
            "middle should have rank ~1, got {}",
            ranks[2]
        );
    }

    #[test]
    fn test_sinkhorn_sort_basic() {
        let values = vec![3.0, 1.0, 2.0];
        let sorted = sinkhorn_sort(&values, 0.1).unwrap();

        // Should be approximately monotonic
        assert!(sorted[0] < sorted[1] + 0.5);
        assert!(sorted[1] < sorted[2] + 0.5);
    }

    #[test]
    fn test_doubly_stochastic() {
        let values = vec![3.0, 1.0, 2.0, 4.0];
        let config = SinkhornConfig::default();
        let p = sinkhorn_permutation(&values, &config).unwrap();
        let n = values.len();

        // Check row sums ≈ 1
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| p[i * n + j]).sum();
            assert!(
                (row_sum - 1.0).abs() < 0.01,
                "row {} sum = {}, expected 1.0",
                i,
                row_sum
            );
        }

        // Check column sums ≈ 1
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| p[i * n + j]).sum();
            assert!(
                (col_sum - 1.0).abs() < 0.01,
                "col {} sum = {}, expected 1.0",
                j,
                col_sum
            );
        }
    }

    #[test]
    fn test_epsilon_effect() {
        let values = vec![3.0, 1.0, 2.0];

        // Small epsilon = sharp
        let ranks_sharp = sinkhorn_rank(&values, 0.01).unwrap();
        // Large epsilon = smooth
        let ranks_smooth = sinkhorn_rank(&values, 1.0).unwrap();

        // Sharp ranks should be closer to integers
        let sharp_var: f64 = ranks_sharp
            .iter()
            .map(|&r| (r - r.round()).powi(2))
            .sum::<f64>()
            / ranks_sharp.len() as f64;

        let smooth_var: f64 = ranks_smooth
            .iter()
            .map(|&r| (r - r.round()).powi(2))
            .sum::<f64>()
            / ranks_smooth.len() as f64;

        assert!(
            sharp_var < smooth_var,
            "sharp_var={} should be < smooth_var={}",
            sharp_var,
            smooth_var
        );
    }

    #[test]
    fn test_empty_input() {
        assert!(sinkhorn_rank(&[], 0.1).is_err());
        assert!(sinkhorn_sort(&[], 0.1).is_err());
    }

    #[test]
    fn test_single_element() {
        let ranks = sinkhorn_rank(&[42.0], 0.1).unwrap();
        assert_eq!(ranks, vec![0.0]);

        let sorted = sinkhorn_sort(&[42.0], 0.1).unwrap();
        assert_eq!(sorted, vec![42.0]);
    }
}
