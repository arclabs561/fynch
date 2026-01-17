//! # fynch
//!
//! Fenchel-Young losses and differentiable sorting/ranking.
//!
//! The name comes from **Fenchel-Young losses** (Blondel et al. 2020, JMLR),
//! a unifying framework connecting prediction functions and loss functions
//! through convex duality.
//!
//! ## The Fenchel-Young Framework
//!
//! Given a regularizer Ω, the **Fenchel-Young loss** is:
//!
//! ```text
//! L_Ω(θ; y) = Ω*(θ) - ⟨θ, y⟩ + Ω(y)
//! ```
//!
//! where Ω* is the Fenchel conjugate. The **prediction function** is:
//!
//! ```text
//! ŷ_Ω(θ) = ∇Ω*(θ) = argmax_p { ⟨θ, p⟩ - Ω(p) }
//! ```
//!
//! Different regularizers give different behaviors:
//!
//! | Regularizer Ω | Prediction ŷ_Ω | Loss L_Ω | Sparsity |
//! |---------------|----------------|----------|----------|
//! | Shannon negentropy | softmax | cross-entropy | Dense |
//! | ½‖·‖² (squared L2) | sparsemax | sparsemax loss | Sparse |
//! | Tsallis α-entropy | α-entmax | entmax loss | Tunable |
//!
//! See the [`fenchel`] module for the generic framework.
//!
//! ## Differentiable Sorting/Ranking
//!
//! Sorting and ranking are discontinuous—small input changes can cause large
//! output changes (rank swaps). This breaks gradient-based optimization.
//!
//! | Approach | Module | Regularization | Complexity |
//! |----------|--------|---------------|------------|
//! | PAVA + Sigmoid | Root | L2 | O(n) / O(n²) |
//! | Sinkhorn OT | [`sinkhorn`] | Entropy (Shannon Ω) | O(n² × iter) |
//!
//! Sinkhorn sorting is exactly FY with Shannon regularization applied to
//! the permutation polytope (Birkhoff polytope).
//!
//! ## Key Functions
//!
//! | Function | Purpose | Module |
//! |----------|---------|--------|
//! | [`pava`] | Isotonic regression | Root |
//! | [`soft_rank`] | Continuous ranks | Root |
//! | [`soft_sort`] | Continuous sorting | Root |
//! | [`fenchel::softmax`] | Dense prediction | [`fenchel`] |
//! | [`fenchel::sparsemax`] | Sparse prediction | [`fenchel`] |
//! | [`fenchel::entmax`] | Tunable sparsity | [`fenchel`] |
//!
//! ## Quick Start
//!
//! ### Fenchel-Young Predictions
//!
//! ```rust
//! use fynch::fenchel::{softmax, sparsemax, entmax};
//!
//! let theta = [2.0, 1.0, 0.1];
//!
//! // Dense (softmax)
//! let p_soft = softmax(&theta);
//! assert!(p_soft.iter().all(|&x| x > 0.0));
//!
//! // Sparse (sparsemax)
//! let p_sparse = sparsemax(&theta);
//! assert!(p_sparse.iter().any(|&x| x == 0.0));
//!
//! // Tunable (1.5-entmax)
//! let p_ent = entmax(&theta, 1.5);
//! ```
//!
//! ### Fenchel-Young Losses
//!
//! ```rust
//! use fynch::fenchel::{Regularizer, Shannon, SquaredL2, Tsallis};
//!
//! let theta = [2.0, 1.0, 0.1];
//! let y = [1.0, 0.0, 0.0];  // one-hot target
//!
//! // Cross-entropy (Shannon)
//! let loss_ce = Shannon.loss(&theta, &y);
//!
//! // Sparsemax-style FY loss via squared L2 regularizer
//! let loss_sp = SquaredL2.loss(&theta, &y);
//!
//! // 1.5-entmax loss
//! let loss_ent = Tsallis::entmax15().loss(&theta, &y);
//! ```
//!
//! ### Differentiable Sorting
//!
//! ```rust
//! use fynch::{pava, soft_rank, soft_sort};
//!
//! // PAVA: isotonic regression
//! let y = [3.0, 1.0, 2.0, 5.0, 4.0];
//! let monotonic = pava(&y);  // [2.0, 2.0, 2.0, 4.5, 4.5]
//!
//! // Soft ranking
//! let scores = [0.5, 0.2, 0.8, 0.1];
//! let ranks = soft_rank(&scores, 0.1).unwrap();
//! ```
//!
//! ### Learning to Rank
//!
//! ```rust
//! use fynch::loss::{spearman_loss, listnet_loss};
//!
//! let pred = [0.9, 0.1, 0.5];
//! let target = [3.0, 1.0, 2.0];
//! let loss = spearman_loss(&pred, &target, 0.1);
//! ```
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`fenchel`] | Generic FY framework: regularizers, predictions, losses |
//! | [`sinkhorn`] | Entropic OT for soft permutations |
//! | [`loss`] | Learning-to-rank losses |
//! | [`metrics`] | IR evaluation: MRR, NDCG, Hits@k |
//!
//! ## Connections
//!
//! - [`wass`](../wass): Sinkhorn OT is the same algorithm
//! - [`logp`](../logp): Shannon entropy connects to information theory
//! - [`cerno`](../cerno): More comprehensive IR evaluation
//!
//! ## What Can Go Wrong
//!
//! 1. **Temperature too low**: Numerical instability, vanishing gradients
//! 2. **Temperature too high**: Predictions become uniform, lose signal
//! 3. **Sinkhorn not converging**: Increase max_iter or epsilon
//! 4. **Wrong regularizer**: Use sparsemax for top-k, softmax for soft attention
//!
//! ## References
//!
//! - Blondel, Martins, Niculae (2020). "Learning with Fenchel-Young Losses" (JMLR)
//! - Martins & Astudillo (2016). "From Softmax to Sparsemax"
//! - Blondel et al. (2020). "Fast Differentiable Sorting and Ranking"
//! - Cuturi et al. (2019). "Differentiable Ranking via Optimal Transport"

pub mod fenchel;
pub mod loss;
pub mod metrics;
pub mod sigmoid;
pub mod sinkhorn;
pub mod topk;

use thiserror::Error;

pub use fenchel::{
    entmax, entropy_bits, entropy_nats, softmax, softmax_with_temperature, sparsemax, Regularizer,
    Shannon, SquaredL2, Tsallis,
};
pub use metrics::{compute_rank, hits_at_k, mean_rank, mrr, ndcg, ndcg_at_k, RankingMetrics};
pub use sigmoid::{sigmoid, sigmoid_derivative};
pub use sinkhorn::{sinkhorn_rank, sinkhorn_sort, SinkhornConfig};
pub use topk::{differentiable_bottomk, differentiable_topk};

#[derive(Debug, Error)]
pub enum Error {
    #[error("empty input")]
    EmptyInput,

    #[error("temperature must be positive: {0}")]
    InvalidTemperature(f64),

    #[error("weights must be positive")]
    InvalidWeights,

    #[error("length mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
///
/// Finds the monotonically non-decreasing sequence ŷ minimizing Σ(yᵢ - ŷᵢ)².
///
/// # Algorithm
///
/// ```text
/// 1. Initialize blocks: each element is its own block
/// 2. Scan left-to-right:
///    - If block[i] > block[i+1], merge them (average values)
///    - Backtrack: merged block might violate with previous
/// 3. Repeat until no violations
/// ```
///
/// # Complexity
///
/// - Time: O(n)
/// - Space: O(n) for output
///
/// # Example
///
/// ```rust
/// use fynch::pava;
///
/// let y = [3.0, 1.0, 2.0, 5.0, 4.0];
/// let result = pava(&y);
/// // result ≈ [2.0, 2.0, 2.0, 4.5, 4.5]
///
/// // Verify monotonicity
/// for i in 1..result.len() {
///     assert!(result[i] >= result[i-1]);
/// }
/// ```
pub fn pava(y: &[f64]) -> Vec<f64> {
    if y.is_empty() {
        return vec![];
    }

    let n = y.len();
    let mut result = y.to_vec();

    // Block representation: (start_index, sum, count)
    let mut blocks: Vec<(usize, f64, usize)> = Vec::with_capacity(n);

    for (i, &val) in y.iter().enumerate() {
        // Add new block
        blocks.push((i, val, 1));

        // Pool while violation exists
        while blocks.len() > 1 {
            let len = blocks.len();
            let (_, sum1, cnt1) = blocks[len - 2];
            let (_, sum2, cnt2) = blocks[len - 1];

            let mean1 = sum1 / cnt1 as f64;
            let mean2 = sum2 / cnt2 as f64;

            if mean1 > mean2 {
                // Merge last two blocks
                blocks.pop();
                let Some(last) = blocks.last_mut() else {
                    // Safety: blocks.len() > 1 before pop, so one block must remain.
                    debug_assert!(false, "blocks unexpectedly empty after pop");
                    break;
                };
                last.1 += sum2;
                last.2 += cnt2;
            } else {
                break;
            }
        }
    }

    // Expand blocks back to result
    for (start, sum, count) in blocks {
        let mean = sum / count as f64;
        for r in result.iter_mut().skip(start).take(count) {
            *r = mean;
        }
    }

    result
}

/// Weighted PAVA with custom weights.
///
/// Finds monotonic ŷ minimizing Σ wᵢ(yᵢ - ŷᵢ)².
///
/// # Example
///
/// ```rust
/// use fynch::pava_weighted;
///
/// let y = [3.0, 1.0, 2.0];
/// let w = [1.0, 2.0, 1.0];  // Middle point has more weight
/// let result = pava_weighted(&y, &w).unwrap();
/// ```
pub fn pava_weighted(y: &[f64], weights: &[f64]) -> Result<Vec<f64>> {
    if y.is_empty() {
        return Ok(vec![]);
    }
    if y.len() != weights.len() {
        return Err(Error::LengthMismatch(y.len(), weights.len()));
    }
    if weights.iter().any(|&w| w <= 0.0) {
        return Err(Error::InvalidWeights);
    }

    let n = y.len();
    let mut result = y.to_vec();

    // Block: (start, weighted_sum, total_weight)
    let mut blocks: Vec<(usize, f64, f64)> = Vec::with_capacity(n);

    for (i, (&val, &w)) in y.iter().zip(weights.iter()).enumerate() {
        blocks.push((i, val * w, w));

        while blocks.len() > 1 {
            let len = blocks.len();
            let (_, wsum1, w1) = blocks[len - 2];
            let (_, wsum2, w2) = blocks[len - 1];

            let mean1 = wsum1 / w1;
            let mean2 = wsum2 / w2;

            if mean1 > mean2 {
                blocks.pop();
                let Some(last) = blocks.last_mut() else {
                    // Safety: blocks.len() > 1 before pop, so one block must remain.
                    debug_assert!(false, "blocks unexpectedly empty after pop");
                    break;
                };
                last.1 += wsum2;
                last.2 += w2;
            } else {
                break;
            }
        }
    }

    // Determine block boundaries
    for (block_idx, (start, wsum, total_w)) in blocks.iter().enumerate() {
        let mean = wsum / total_w;
        // Find how many elements this block covers
        let end = if block_idx + 1 < blocks.len() {
            blocks[block_idx + 1].0
        } else {
            n
        };
        for r in result.iter_mut().skip(*start).take(end - *start) {
            *r = mean;
        }
    }

    Ok(result)
}

/// Soft ranking with temperature parameter.
///
/// Returns continuous approximation to ranks. As τ → 0, converges to hard ranks.
///
/// # Algorithm
///
/// Uses the soft-rank formulation via pairwise comparisons:
/// ```text
/// soft_rank(x)ᵢ = 1 + Σⱼ σ((xⱼ - xᵢ)/τ)
/// ```
/// where σ is the sigmoid function.
///
/// # Example
///
/// ```rust
/// use fynch::soft_rank;
///
/// let scores = [0.5, 0.2, 0.8, 0.1];
/// let ranks = soft_rank(&scores, 0.1);
/// // Approximately [2, 3, 1, 4] but continuous
/// ```
pub fn soft_rank(x: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if x.is_empty() {
        return Err(Error::EmptyInput);
    }
    if temperature <= 0.0 {
        return Err(Error::InvalidTemperature(temperature));
    }

    let n = x.len();
    let mut ranks = vec![1.0; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                // Sigmoid of scaled difference
                let diff = (x[j] - x[i]) / temperature;
                let sigmoid = 1.0 / (1.0 + (-diff).exp());
                ranks[i] += sigmoid;
            }
        }
    }

    Ok(ranks)
}

/// Soft sorting with temperature parameter.
///
/// Returns a continuous approximation to sorted values using Sinkhorn
/// optimal transport. As temperature → 0, converges to hard sorting.
///
/// This is a convenience wrapper around [`sinkhorn::sinkhorn_sort`].
///
/// # Arguments
///
/// * `x` - Input values
/// * `temperature` - Regularization strength (epsilon). Smaller = sharper.
///
/// # Example
///
/// ```rust
/// use fynch::soft_sort;
///
/// let x = [3.0, 1.0, 2.0];
/// let sorted = soft_sort(&x, 0.1).unwrap();
/// // Approximately [1.0, 2.0, 3.0] but smooth
/// ```
pub fn soft_sort(x: &[f64], temperature: f64) -> Result<Vec<f64>> {
    sinkhorn::sinkhorn_sort(x, temperature)
}

/// Fast Differentiable Sorting ($O(n \log n)$).
///
/// Implements the algorithm from Blondel et al. (2020), "Fast Differentiable
/// Sorting and Ranking". Unlike Sinkhorn ($O(n^2)$), this scales to large inputs.
///
/// The algorithm projects the input vector $\theta$ onto the permutahedron
/// (the convex hull of all permutation matrices).
///
/// # Mathematical Details
/// The projection is defined as:
/// $P(\theta) = \text{argmin}_{w \in \Pi} \|w - \theta\|^2$
/// where $\Pi$ is the permutahedron. Blondel et al. show this reduces to
/// sorting $\theta$ and then solving an isotonic regression (PAVA) on the
/// values $(\text{sorted\_}\theta - \rho)$ where $\rho$ is a target sequence.
pub fn fast_soft_sort(x: &[f64], temperature: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let n = x.len();

    // 1. Get sorting permutation and sorted values
    let mut indexed: Vec<(usize, f64)> = x.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (_indices, sorted): (Vec<usize>, Vec<f64>) = indexed.into_iter().unzip();

    // 2. Define target values (rho) for the permutahedron projection
    // For soft sorting, rho is typically [1, 2, ..., n]
    let rho: Vec<f64> = (1..=n).map(|i| i as f64 * temperature).collect();

    // 3. Compute PAVA on (sorted - rho)
    let mut diff = Vec::with_capacity(n);
    for i in 0..n {
        diff.push(sorted[i] - rho[i]);
    }
    let v = pava(&diff);

    // 4. The soft sorted values are (sorted - v)
    // Note: This needs to be mapped back to original indices for ranking,
    // but for "sorted values" we just return them.
    let mut res = Vec::with_capacity(n);
    for i in 0..n {
        res.push(sorted[i] - v[i]);
    }
    res
}

/// Reciprocal Rank Fusion (RRF).
///
/// Combines multiple ranked lists into a single ranking using the formula:
/// `RRFscore(d) = Σ_r 1 / (k + rank_r(d))`
///
/// # Arguments
/// * `rankings` - A slice of ranked lists (each list is a Vec of doc IDs)
/// * `k` - Hyperparameter (default 60 per Cormack et al.)
pub fn reciprocal_rank_fusion<T: std::hash::Hash + Eq + Clone>(
    rankings: &[Vec<T>],
    k: usize,
) -> Vec<(T, f64)> {
    use std::collections::HashMap;
    let mut scores = HashMap::new();

    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            let score = 1.0 / (k as f64 + (rank + 1) as f64);
            *scores.entry(id.clone()).or_insert(0.0) += score;
        }
    }

    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused
}

/// Isotonic regression with L2 loss.
///
/// Alias for [`pava`] with clearer naming.
pub fn isotonic_l2(y: &[f64]) -> Vec<f64> {
    pava(y)
}

/// Compute soft top-k indicator.
///
/// Returns smooth approximation to 1{rank(xᵢ) ≤ k}.
///
/// # Example
///
/// ```rust
/// use fynch::soft_topk_indicator;
///
/// let scores = [0.5, 0.2, 0.8, 0.1, 0.9];
/// let indicator = soft_topk_indicator(&scores, 2, 0.1).unwrap();
/// // High values for indices 2, 4 (top 2 scores)
/// ```
pub fn soft_topk_indicator(x: &[f64], k: usize, temperature: f64) -> Result<Vec<f64>> {
    let ranks = soft_rank(x, temperature)?;
    let k_f = k as f64;

    // Soft indicator: σ((k + 0.5 - rank) / τ)
    Ok(ranks
        .iter()
        .map(|&r| {
            let z = (k_f + 0.5 - r) / temperature;
            1.0 / (1.0 + (-z).exp())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pava_simple() {
        let y = [3.0, 1.0, 2.0, 5.0, 4.0];
        let result = pava(&y);

        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1], "Not monotonic at {}", i);
        }

        // First three should be pooled
        assert_relative_eq!(result[0], result[1], epsilon = 1e-10);
        assert_relative_eq!(result[1], result[2], epsilon = 1e-10);

        // Last two should be pooled
        assert_relative_eq!(result[3], result[4], epsilon = 1e-10);
    }

    #[test]
    fn test_pava_already_monotonic() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = pava(&y);
        assert_eq!(result, y);
    }

    #[test]
    fn test_pava_reverse() {
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let result = pava(&y);

        // Should all be pooled to the mean
        let mean = 3.0;
        for &r in &result {
            assert_relative_eq!(r, mean, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_soft_rank_ordering() {
        let x = [0.1, 0.5, 0.2, 0.9];
        let ranks = soft_rank(&x, 0.01).unwrap();

        // With low temperature, should approximate hard ranks
        // x[3]=0.9 should have lowest rank (~1)
        // x[0]=0.1 should have highest rank (~4)
        assert!(ranks[3] < ranks[1]); // 0.9 ranks higher than 0.5
        assert!(ranks[1] < ranks[2]); // 0.5 ranks higher than 0.2
        assert!(ranks[2] < ranks[0]); // 0.2 ranks higher than 0.1
    }

    #[test]
    fn test_soft_sort_approximates_sort() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let sorted = soft_sort(&x, 0.01).unwrap();

        // Should be approximately sorted
        for i in 1..sorted.len() {
            assert!(
                sorted[i] >= sorted[i - 1] - 0.5,
                "Not approximately sorted at {}: {} < {}",
                i,
                sorted[i],
                sorted[i - 1]
            );
        }
    }

    #[test]
    fn test_soft_topk() {
        let x = [0.1, 0.9, 0.5, 0.8, 0.2];
        let indicator = soft_topk_indicator(&x, 2, 0.01).unwrap();

        // Indices 1 (0.9) and 3 (0.8) should have high indicator
        assert!(indicator[1] > 0.5);
        assert!(indicator[3] > 0.5);

        // Others should have low indicator
        assert!(indicator[0] < 0.5);
        assert!(indicator[2] < 0.5);
        assert!(indicator[4] < 0.5);
    }

    #[test]
    fn test_empty_input() {
        assert!(pava(&[]).is_empty());
        assert!(soft_rank(&[], 0.1).is_err());
        assert!(soft_sort(&[], 0.1).is_err());
    }

    #[test]
    fn test_invalid_temperature() {
        let x = [1.0, 2.0];
        assert!(soft_rank(&x, 0.0).is_err());
        assert!(soft_rank(&x, -1.0).is_err());
    }
}
