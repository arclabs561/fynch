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

fn standardize(values: &[f64]) -> Vec<f64> {
    // Permutation-invariant affine normalization: z_i = (x_i - mean) / (std + eps).
    //
    // This lets us compare values to a fixed target grid without smuggling in a
    // discontinuous hard sort.
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut var = 0.0;
    for &v in values {
        var += (v - mean) * (v - mean);
    }
    var /= n as f64;
    let std = var.sqrt();
    let denom = (std + 1e-12).max(1e-12);
    values.iter().map(|&v| (v - mean) / denom).collect()
}

fn target_grid(n: usize) -> Vec<f64> {
    // Fixed targets in [-1, 1]. Critically, this must NOT depend on the input `values`.
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    (0..n)
        .map(|j| -1.0 + 2.0 * (j as f64) / ((n - 1) as f64))
        .collect()
}

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
/// # Convergence
///
/// This function does not expose the iteration count or convergence residual.
/// It runs until `config.tol` is satisfied or `config.max_iter` is reached,
/// with no indication of which occurred. If you need to observe convergence,
/// reimplement the iteration loop with logging.
///
/// # Algorithm
///
/// 1. Standardize inputs: z_i = (x_i - mean(x)) / std(x)
/// 2. Build cost matrix against a fixed target grid t_j ∈ \[-1,1\]:
///    C\[i,j\] = (z_i - t_j)²
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

    // IMPORTANT: avoid using a hard sort inside the operator.
    // If we set the column targets to `sorted(values)`, we reintroduce a
    // discontinuity and destroy the reason we’re using Sinkhorn in the first place.
    let z = standardize(values);
    let t = target_grid(n);

    // Cost matrix: C[i,j] = (z_i - t_j)^2
    let mut cost = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let diff = z[i] - t[j];
            cost[i * n + j] = diff * diff;
        }
    }

    // We compute Sinkhorn in log-space to avoid underflow when epsilon is small.
    //
    // Let logK[i,j] = -(C[i,j] - min(C)) / epsilon.
    // Then u,v updates become log-domain normalization steps via log-sum-exp.
    fn log_sum_exp(xs: &[f64]) -> f64 {
        let mut m = f64::NEG_INFINITY;
        for &x in xs {
            if x > m {
                m = x;
            }
        }
        if !m.is_finite() {
            return f64::NEG_INFINITY;
        }
        let mut s = 0.0;
        for &x in xs {
            s += (x - m).exp();
        }
        m + s.ln()
    }

    let inv_eps = 1.0 / config.epsilon;
    let min_cost = cost.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));

    let mut log_k = vec![0.0; n * n];
    for idx in 0..n * n {
        log_k[idx] = -((cost[idx] - min_cost) * inv_eps);
    }

    let mut log_u = vec![0.0_f64; n];
    let mut log_v = vec![0.0_f64; n];
    // Reusable buffers: avoids allocating two Vec<f64> per iteration.
    let mut log_u_new = vec![0.0_f64; n];
    let mut log_v_new = vec![0.0_f64; n];

    // Sinkhorn iterations in log-space.
    // Convergence check is deferred every `check_every` iterations to amortize
    // the extra logsumexp pass (3x work per iter otherwise).
    let check_every = 5usize;
    let mut scratch = vec![0.0_f64; n];
    for iter in 0..config.max_iter {
        // log_u[i] = -logsumexp_j (log_k[i,j] + log_v[j])
        for i in 0..n {
            let row = &log_k[i * n..i * n + n];
            let mut m = f64::NEG_INFINITY;
            for j in 0..n {
                let x = row[j] + log_v[j];
                scratch[j] = x;
                if x > m {
                    m = x;
                }
            }
            let s: f64 = scratch.iter().map(|&x| (x - m).exp()).sum();
            log_u_new[i] = -(m + s.ln());
        }

        // log_v[j] = -logsumexp_i (log_k[i,j] + log_u_new[i])
        for j in 0..n {
            let mut m = f64::NEG_INFINITY;
            for i in 0..n {
                let x = log_k[i * n + j] + log_u_new[i];
                scratch[i] = x;
                if x > m {
                    m = x;
                }
            }
            let s: f64 = scratch.iter().map(|&x| (x - m).exp()).sum();
            log_v_new[j] = -(m + s.ln());
        }

        std::mem::swap(&mut log_u, &mut log_u_new);
        std::mem::swap(&mut log_v, &mut log_v_new);

        // Deferred convergence: check every `check_every` iterations.
        // row_sum[i] = Σ_j exp(log_u[i] + log_k[i,j] + log_v[j])
        if (iter + 1) % check_every == 0 || iter + 1 == config.max_iter {
            let mut max_err: f64 = 0.0;
            for i in 0..n {
                let row = &log_k[i * n..i * n + n];
                let lui = log_u[i];
                let mut m = f64::NEG_INFINITY;
                for j in 0..n {
                    let x = lui + row[j] + log_v[j];
                    scratch[j] = x;
                    if x > m {
                        m = x;
                    }
                }
                let s: f64 = scratch.iter().map(|&x| (x - m).exp()).sum();
                let row_sum = (m + s.ln()).exp();
                max_err = max_err.max((row_sum - 1.0).abs());
            }
            if max_err < config.tol {
                break;
            }
        }
    }

    // Build final doubly-stochastic matrix P = exp(log_u + logK + log_v)
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            p[i * n + j] = (log_u[i] + log_k[i * n + j] + log_v[j]).exp();
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
/// Soft ranks in [1, n]. Lower rank = smaller value.
///
/// # Example
///
/// ```rust
/// use fynch::sinkhorn::sinkhorn_rank;
///
/// let values = vec![3.0, 1.0, 2.0];
/// let ranks = sinkhorn_rank(&values, 0.1).unwrap();
///
/// // 1.0 is smallest → rank ~1
/// // 2.0 is middle → rank ~2
/// // 3.0 is largest → rank ~3
/// assert!(ranks[1] < ranks[2]);
/// assert!(ranks[2] < ranks[0]);
/// ```
pub fn sinkhorn_rank(values: &[f64], epsilon: f64) -> Result<Vec<f64>> {
    let n = values.len();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }

    let config = SinkhornConfig {
        epsilon,
        // Small epsilon (sharp permutations) needs more Sinkhorn iterations to converge.
        max_iter: if epsilon < 0.05 { 2000 } else { 200 },
        ..Default::default()
    };

    let p = sinkhorn_permutation(values, &config)?;

    // Soft rank = P @ [1, 2, ..., n]  (1-indexed, consistent with IR metrics and `soft_rank`)
    let mut ranks = vec![0.0; n];
    for i in 0..n {
        let mut rank = 0.0;
        for j in 0..n {
            rank += p[i * n + j] * (j as f64 + 1.0);
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
        max_iter: if epsilon < 0.05 { 2000 } else { 200 },
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
    fn test_sinkhorn_coupling_invariant_to_row_col_cost_shifts() {
        // This mirrors the invariance used in `wass`:
        // shifting costs by c_i + d_j should not change the optimal coupling P,
        // since Sinkhorn scalings absorb diagonal factors.
        //
        // We test it at the log-kernel level to avoid changing public APIs.

        fn log_sum_exp(xs: &[f64]) -> f64 {
            let mut m = f64::NEG_INFINITY;
            for &x in xs {
                if x > m {
                    m = x;
                }
            }
            if !m.is_finite() {
                return f64::NEG_INFINITY;
            }
            let mut s = 0.0;
            for &x in xs {
                s += (x - m).exp();
            }
            m + s.ln()
        }

        fn sinkhorn_from_log_k(log_k: &[f64], n: usize, config: &SinkhornConfig) -> Vec<f64> {
            let mut log_u = vec![0.0; n];
            let mut log_v = vec![0.0; n];
            let mut scratch = vec![0.0; n];

            for _ in 0..config.max_iter {
                // log_u[i] = -logsumexp_j(log_k[i,j] + log_v[j])
                let mut log_u_new = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        scratch[j] = log_k[i * n + j] + log_v[j];
                    }
                    log_u_new[i] = -log_sum_exp(&scratch);
                }

                // log_v[j] = -logsumexp_i(log_k[i,j] + log_u_new[i])
                let mut log_v_new = vec![0.0; n];
                for j in 0..n {
                    for i in 0..n {
                        scratch[i] = log_k[i * n + j] + log_u_new[i];
                    }
                    log_v_new[j] = -log_sum_exp(&scratch);
                }

                // Convergence: max row-sum deviation from 1.
                let mut max_err = 0.0f64;
                for i in 0..n {
                    for j in 0..n {
                        scratch[j] = log_u_new[i] + log_k[i * n + j] + log_v_new[j];
                    }
                    let row_sum = log_sum_exp(&scratch).exp();
                    max_err = max_err.max((row_sum - 1.0).abs());
                }

                log_u = log_u_new;
                log_v = log_v_new;
                if max_err < config.tol {
                    break;
                }
            }

            let mut p = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    p[i * n + j] = (log_u[i] + log_k[i * n + j] + log_v[j]).exp();
                }
            }
            p
        }

        let values = vec![3.0, 1.0, 2.0, 4.0, 0.5];
        let n = values.len();
        let config = SinkhornConfig {
            epsilon: 0.2,
            max_iter: 600,
            tol: 1e-10,
        };

        let z = standardize(&values);
        let t = target_grid(n);

        let mut cost = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let diff = z[i] - t[j];
                cost[i * n + j] = diff * diff;
            }
        }

        let inv_eps = 1.0 / config.epsilon;
        let min_cost = cost.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
        let mut log_k = vec![0.0; n * n];
        for idx in 0..n * n {
            log_k[idx] = -((cost[idx] - min_cost) * inv_eps);
        }

        // Apply a separable shift in log-space: logK' = logK + a_i + b_j.
        // This corresponds to a row/col shift of the cost.
        let row = [0.3, -0.2, 0.1, 0.0, 0.15];
        let col = [-0.25, 0.05, 0.2, -0.1, 0.0];
        let mut log_k_shift = log_k.clone();
        for i in 0..n {
            for j in 0..n {
                log_k_shift[i * n + j] += row[i] + col[j];
            }
        }

        let p1 = sinkhorn_from_log_k(&log_k, n, &config);
        let p2 = sinkhorn_from_log_k(&log_k_shift, n, &config);

        let mut max_abs = 0.0f64;
        for (a, b) in p1.iter().zip(p2.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 1e-7,
            "expected coupling invariant to separable shifts: max_abs={max_abs}"
        );
    }

    #[test]
    fn test_sinkhorn_rank_basic() {
        let values = vec![3.0, 1.0, 2.0];
        let ranks = sinkhorn_rank(&values, 0.1).unwrap();

        // Hard ranks (1-indexed): [3, 1, 2]
        assert!(
            ranks[0] > 2.5,
            "largest should have rank ~3, got {}",
            ranks[0]
        );
        assert!(
            ranks[1] < 1.5,
            "smallest should have rank ~1, got {}",
            ranks[1]
        );
        assert!(
            ranks[2] > 1.5 && ranks[2] < 2.5,
            "middle should have rank ~2, got {}",
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
    fn test_sinkhorn_sort_permutation_invariant() {
        // Sorting should not depend on the input order.
        let a = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        let b = vec![1.0, 3.0, 5.0, 2.0, 4.0]; // permutation of a
        let sa = sinkhorn_sort(&a, 0.2).unwrap();
        let sb = sinkhorn_sort(&b, 0.2).unwrap();
        assert_eq!(sa.len(), sb.len());
        for i in 0..sa.len() {
            assert!(
                (sa[i] - sb[i]).abs() < 1e-6,
                "i={} sa={} sb={}",
                i,
                sa[i],
                sb[i]
            );
        }
    }

    #[test]
    fn test_sinkhorn_sort_preserves_sum() {
        // If P is doubly stochastic, then Σ_j (Σ_i P_ij x_i) = Σ_i x_i.
        let values = vec![3.0, 1.0, 2.0, 4.0];
        let sorted = sinkhorn_sort(&values, 0.2).unwrap();
        let s_in: f64 = values.iter().sum();
        let s_out: f64 = sorted.iter().sum();
        assert!((s_in - s_out).abs() < 1e-6, "in={} out={}", s_in, s_out);
    }

    #[test]
    fn test_sinkhorn_rank_range() {
        let values = vec![3.0, 1.0, 2.0, 4.0];
        let ranks = sinkhorn_rank(&values, 0.2).unwrap();
        for &r in &ranks {
            assert!(r >= 1.0 - 1e-6);
            assert!(r <= values.len() as f64 + 1e-6);
        }
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
        assert_eq!(ranks, vec![1.0]);

        let sorted = sinkhorn_sort(&[42.0], 0.1).unwrap();
        assert_eq!(sorted, vec![42.0]);
    }
}
