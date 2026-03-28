//! Differentiable Top-K selection.
//!
//! Provides soft selection of the k largest (or smallest) elements,
//! allowing gradients to flow through the selection operation.
//!
//! # Two Approaches
//!
//! ## Rank-based (simple)
//!
//! [`differentiable_topk`] and [`differentiable_bottomk`] use soft ranks
//! and sigmoid thresholding. Returns per-element indicators only.
//!
//! ## Sorting-network-based (Petersen et al., ICML 2022)
//!
//! [`sparse_topk_matrix`] runs a differentiable sorting network forward,
//! then propagates a selector matrix backward through the recorded soft
//! comparator weights. Produces an n x k **attribution matrix** A where
//! `A[i][j]` is the soft probability that input element i is the (j+1)-th
//! largest. This is the core contribution of difftopk.
//!
//! The attribution matrix is more informative than indicators: it tells you
//! not just *whether* an element is in the top-k, but *which rank* it likely
//! holds within the top-k.
//!
//! [`topk_cross_entropy_loss`] uses the attribution matrix to compute a
//! top-k aware classification loss (TopKCE from the paper).
//!
//! # References
//!
//! - Petersen et al. (2022). "Differentiable Top-k Classification Learning" (ICML)
//! - Petersen et al. (2021). "Differentiable Sorting Networks" (ICML)
//! - Xie et al. (2020). "Differentiable Top-K Operator with Optimal Transport"

use crate::sigmoid::sigmoid;
use crate::soft_rank;
use crate::sorting_network::{relaxed_sigmoid, DiffSortNet, NetworkType, RelaxDist};
use crate::{Error, Result};

/// Differentiable Top-K selection (k largest values).
///
/// Returns soft indicator weights for each element indicating membership
/// in the top-k set. As temperature decreases, indicators approach {0, 1}.
///
/// # Note on Soft Ranks
///
/// `soft_rank` assigns **lower** ranks to **higher** values:
/// - Value 0.9 (highest) -> rank ~ 1
/// - Value 0.1 (lowest) -> rank ~ n
///
/// So top-k elements have ranks <= k.
///
/// # Arguments
///
/// * `values` - Input values
/// * `k` - Number of top elements to select
/// * `temperature` - Controls sharpness (smaller = sharper)
///
/// # Returns
///
/// Tuple of (weighted_values, indicators) where:
/// - `weighted_values[i]` = values\[i\] * indicator\[i\]
/// - `indicators[i]` in (0, 1) indicates soft membership in top-k
///
/// # Example
///
/// ```rust
/// use fynch::topk::differentiable_topk;
///
/// let values = [0.1, 0.9, 0.5, 0.8, 0.2];
/// let (weighted, indicators) = differentiable_topk(&values, 2, 0.1);
///
/// // Indices 1 (0.9) and 3 (0.8) should have high indicators
/// assert!(indicators[1] > 0.5);
/// assert!(indicators[3] > 0.5);
///
/// // Others should have low indicators
/// assert!(indicators[0] < 0.5);
/// assert!(indicators[2] < 0.5);
/// assert!(indicators[4] < 0.5);
/// ```
pub fn differentiable_topk(values: &[f64], k: usize, temperature: f64) -> (Vec<f64>, Vec<f64>) {
    let n = values.len();
    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }

    // If k >= n, everything is in top-k
    if k >= n {
        let indicators = vec![1.0; n];
        return (values.to_vec(), indicators);
    }

    // Get soft ranks (higher value = LOWER rank, starting at 1)
    let ranks = match soft_rank(values, temperature) {
        Ok(r) => r,
        Err(_) => return (vec![0.0; n], vec![0.0; n]),
    };

    // Top-k: elements with ranks <= k are in top-k
    // Threshold is k + 0.5 (halfway between k and k+1)
    let threshold = k as f64 + 0.5;

    let mut weighted_values = Vec::with_capacity(n);
    let mut indicators = Vec::with_capacity(n);

    for i in 0..n {
        // Soft indicator: sigmoid((threshold - rank) / temperature)
        // Lower rank = higher indicator (closer to top)
        let indicator = sigmoid((threshold - ranks[i]) / temperature);
        indicators.push(indicator);
        weighted_values.push(values[i] * indicator);
    }

    (weighted_values, indicators)
}

/// Differentiable Bottom-K selection (k smallest values).
///
/// Same as [`differentiable_topk`] but selects smallest values.
///
/// # Note on Soft Ranks
///
/// `soft_rank` assigns **higher** ranks to **lower** values:
/// - Value 0.1 (lowest) -> rank ~ n
/// - Value 0.9 (highest) -> rank ~ 1
///
/// So bottom-k elements have ranks >= n - k + 1.
///
/// # Example
///
/// ```rust
/// use fynch::topk::differentiable_bottomk;
///
/// let values = [0.1, 0.9, 0.5, 0.8, 0.2];
/// let (weighted, indicators) = differentiable_bottomk(&values, 2, 0.1);
///
/// // Indices 0 (0.1) and 4 (0.2) should have high indicators
/// assert!(indicators[0] > 0.5);
/// assert!(indicators[4] > 0.5);
/// ```
pub fn differentiable_bottomk(values: &[f64], k: usize, temperature: f64) -> (Vec<f64>, Vec<f64>) {
    let n = values.len();
    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }

    if k >= n {
        let indicators = vec![1.0; n];
        return (values.to_vec(), indicators);
    }

    // Get soft ranks (higher value = LOWER rank)
    let ranks = match soft_rank(values, temperature) {
        Ok(r) => r,
        Err(_) => return (vec![0.0; n], vec![0.0; n]),
    };

    // Bottom-k: elements with ranks >= n - k + 1 are in bottom-k
    // Threshold is n - k + 0.5
    let threshold = (n - k) as f64 + 0.5;

    let mut weighted_values = Vec::with_capacity(n);
    let mut indicators = Vec::with_capacity(n);

    for i in 0..n {
        // Soft indicator: sigmoid((rank - threshold) / temperature)
        // Higher rank = higher indicator (closer to bottom)
        let indicator = sigmoid((ranks[i] - threshold) / temperature);
        indicators.push(indicator);
        weighted_values.push(values[i] * indicator);
    }

    (weighted_values, indicators)
}

// ============================================================================
// Sorting-network-based sparse top-k (Petersen et al., ICML 2022)
// ============================================================================

/// Sparse top-k attribution matrix via differentiable sorting networks.
///
/// Given n input scores, produces an n x k matrix A where `A[i][j]` is
/// the soft probability that input element i occupies position j within
/// the top-k sorted output. Positions are in ascending order within the
/// top-k: column 0 = rank k (k-th largest), column k-1 = rank 1 (largest).
///
/// This matches the difftopk convention where the last column corresponds
/// to the highest-ranked element.
///
/// The algorithm:
/// 1. Forward pass through the sorting network, recording the soft
///    comparator weight (alpha) at each stage.
/// 2. Initialize a selector matrix X = I_n\[:, n-k:\] (selects the last
///    k output positions, which hold the top-k after ascending sort).
/// 3. Backward pass through comparators in reverse order, propagating X
///    through the inverse soft permutation at each stage.
///
/// This is equivalent to extracting the last k columns of the full n x n
/// soft permutation matrix, but computed in O(n log^2 n * k) instead of
/// O(n^2) space for the full matrix.
///
/// # Arguments
///
/// * `scores` - Input scores (n elements)
/// * `k` - Number of top elements (must be <= n)
/// * `steepness` - Inverse temperature for comparator relaxation (higher = sharper)
/// * `network_type` - Bitonic or OddEven sorting network
/// * `dist` - Distribution family for the comparator relaxation
///
/// # Returns
///
/// An n x k matrix (as `Vec<Vec<f64>>`) where entry \[i\]\[j\] is the soft
/// probability that input i occupies position j in the top-k.
/// Column k-1 = rank 1 (largest), column 0 = rank k (k-th largest).
///
/// # Example
///
/// ```rust
/// use fynch::topk::sparse_topk_matrix;
/// use fynch::sorting_network::{NetworkType, RelaxDist};
///
/// let scores = vec![3.0, 1.0, 4.0, 2.0];
/// let a = sparse_topk_matrix(&scores, 2, 10.0, NetworkType::Bitonic, RelaxDist::Logistic).unwrap();
///
/// assert_eq!(a.len(), 4);     // n rows
/// assert_eq!(a[0].len(), 2);  // k columns
///
/// // Element 2 (score=4.0) should have high weight in column 1 (rank 1 = largest)
/// assert!(a[2][1] > 0.5, "score 4.0 should be rank 1: {}", a[2][1]);
/// // Element 0 (score=3.0) should have high weight in column 0 (rank 2)
/// assert!(a[0][0] > 0.5, "score 3.0 should be rank 2: {}", a[0][0]);
/// ```
pub fn sparse_topk_matrix(
    scores: &[f64],
    k: usize,
    steepness: f64,
    network_type: NetworkType,
    dist: RelaxDist,
) -> Result<Vec<Vec<f64>>> {
    let n = scores.len();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if steepness <= 0.0 {
        return Err(Error::InvalidTemperature(steepness));
    }
    let k = k.min(n);
    if k == 0 {
        return Ok(vec![vec![]; n]);
    }

    // Build the sorting network
    let net = DiffSortNet::new(network_type, n, steepness, dist);
    let padded_n = net.size;

    // Pad input
    let pad_val = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1e6;
    let mut values: Vec<f64> = Vec::with_capacity(padded_n);
    values.extend_from_slice(scores);
    while values.len() < padded_n {
        values.push(pad_val);
    }

    // Forward pass: record alpha at each comparator
    let comparators = net.comparator_pairs();
    let mut alphas = Vec::with_capacity(comparators.len());

    // Also need the values at each stage to compute alphas
    let mut vals = values.clone();
    for &(a, b) in comparators {
        let diff = (vals[a] - vals[b]) * steepness;
        let alpha = relaxed_sigmoid(diff, dist);
        alphas.push(alpha);

        // Apply soft comparator to values
        let va = vals[a];
        let vb = vals[b];
        vals[a] = (1.0 - alpha) * va + alpha * vb;
        vals[b] = alpha * va + (1.0 - alpha) * vb;
    }

    // Initialize selector: last k columns of the padded_n x padded_n identity.
    // After ascending sort, the last k positions hold the k largest values.
    // X is padded_n x k.
    let mut x = vec![vec![0.0; k]; padded_n];
    #[allow(clippy::needless_range_loop)] // j indexes both col computation and inner vec
    for j in 0..k {
        let col = padded_n - k + j;
        x[col][j] = 1.0;
    }

    // Backward pass: propagate X through each comparator in reverse.
    // Each comparator (a, b) with alpha acts as:
    //   out_a = (1 - alpha) * in_a + alpha * in_b      (min)
    //   out_b = alpha * in_a + (1 - alpha) * in_b      (max)
    //
    // The transpose (for backward propagation) is:
    //   in_a = (1 - alpha) * out_a + alpha * out_b
    //   in_b = alpha * out_a + (1 - alpha) * out_b
    for (idx, &(a, b)) in comparators.iter().enumerate().rev() {
        let alpha = alphas[idx];
        #[allow(clippy::needless_range_loop)] // two mutable rows from the same Vec
        for j in 0..k {
            let xa = x[a][j];
            let xb = x[b][j];
            x[a][j] = (1.0 - alpha) * xa + alpha * xb;
            x[b][j] = alpha * xa + (1.0 - alpha) * xb;
        }
    }

    // Trim padding rows, keep only original n rows
    let result: Vec<Vec<f64>> = x.into_iter().take(n).collect();
    Ok(result)
}

/// Convenience wrapper: sparse top-k with bitonic network and logistic relaxation.
///
/// # Example
///
/// ```rust
/// use fynch::topk::sparse_topk;
///
/// let scores = vec![3.0, 1.0, 4.0, 2.0];
/// let a = sparse_topk(&scores, 2, 10.0).unwrap();
/// assert_eq!(a.len(), 4);
/// assert_eq!(a[0].len(), 2);
/// ```
pub fn sparse_topk(scores: &[f64], k: usize, steepness: f64) -> Result<Vec<Vec<f64>>> {
    sparse_topk_matrix(
        scores,
        k,
        steepness,
        NetworkType::Bitonic,
        RelaxDist::Logistic,
    )
}

/// Top-k cross-entropy loss (Petersen et al., ICML 2022).
///
/// Given class logits and a target class, computes a loss that rewards
/// placing the correct class in the top-k, weighted by a distribution
/// P_K over ranks.
///
/// The distribution P_K = \[p_1, p_2, ..., p_k\] specifies the importance
/// of each rank position. Must sum to 1. Common choices:
/// - `[1, 0, ..., 0]`: equivalent to standard cross-entropy (only rank 1 matters)
/// - `[0, 0, ..., 1]`: only care about being in top-k at all
/// - `[0.5, 0, 0, 0, 0.5]`: mix of top-1 and top-5
/// - `[0.2, 0.2, 0.2, 0.2, 0.2]`: uniform over top-5 ranks
///
/// # Algorithm
///
/// 1. Compute attribution matrix A (n x k) via sorting network
/// 2. Build per-class distribution: `d[i] = sum_j p_k[j] * A[i][j..k].sum()`
///    (cumulative: rank j contributes to "in top-j")
/// 3. Loss = -log(d[target] + eps)
///
/// The `top1_mode` from the paper is handled by setting `p_k[0]` to use
/// softmax for the top-1 component. When `p_k[0] > 0`, the top-1 weight
/// uses softmax(logits) instead of the sorting network (more stable).
///
/// # Arguments
///
/// * `logits` - Raw class scores (n classes)
/// * `target` - Index of the correct class
/// * `p_k` - Distribution over top-k ranks (must sum to ~1, length = k)
/// * `steepness` - Inverse temperature for sorting network
/// * `network_type` - Bitonic or OddEven
/// * `dist` - Distribution family for comparator relaxation
///
/// # Returns
///
/// The top-k cross-entropy loss (non-negative scalar).
///
/// # Example
///
/// ```rust
/// use fynch::topk::topk_cross_entropy_loss;
/// use fynch::sorting_network::{NetworkType, RelaxDist};
///
/// let logits = vec![2.0, 0.5, 1.0, 0.1];
/// let target = 0; // class 0 has highest logit
/// let p_k = vec![0.5, 0.0, 0.0, 0.5]; // care about top-1 and top-4
///
/// let loss = topk_cross_entropy_loss(
///     &logits, target, &p_k, 10.0,
///     NetworkType::Bitonic, RelaxDist::Logistic,
/// ).unwrap();
/// assert!(loss >= 0.0);
/// assert!(loss < 1.0); // correct class is rank 1, should have low loss
/// ```
pub fn topk_cross_entropy_loss(
    logits: &[f64],
    target: usize,
    p_k: &[f64],
    steepness: f64,
    network_type: NetworkType,
    dist: RelaxDist,
) -> Result<f64> {
    let n = logits.len();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if target >= n {
        return Err(Error::LengthMismatch(target, n));
    }
    let k = p_k.len();
    if k == 0 || k > n {
        return Err(Error::EmptyInput);
    }

    // Get attribution matrix
    let attr = sparse_topk_matrix(logits, k, steepness, network_type, dist)?;

    // Build top-k distribution over classes.
    // For each class i, compute: d[i] = sum_{j=0}^{k-1} p_k[j] * cumulative_attr[i][j]
    // where cumulative_attr[i][j] = sum_{l=0}^{j} attr[i][l]
    // This means: p_k[j] weights the probability of being in top-(j+1).
    //
    // Following the paper's "sm" mode: use softmax for the top-1 component
    // when p_k[0] > 0, which is more numerically stable.
    let use_softmax_top1 = p_k[0] > 0.0;

    let mut topk_dist = vec![0.0; n];

    if use_softmax_top1 {
        // Softmax for top-1 component
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        for i in 0..n {
            topk_dist[i] += p_k[0] * exps[i] / sum_exp;
        }
    }

    // Sorting-network-based components for ranks 2..k (or 1..k if !use_softmax_top1)
    // Convention: column k-1 = rank 1 (largest), column 0 = rank k.
    // "Being in top-j" means having weight in the last j columns = columns k-j..k-1.
    let start_j = if use_softmax_top1 { 1 } else { 0 };
    for (j, &p_k_j) in p_k.iter().enumerate().take(k).skip(start_j) {
        // p_k[j] weights rank j+1. "In top-(j+1)" = sum of last (j+1) columns.
        let first_col = k - (j + 1);
        for i in 0..n {
            let cumulative: f64 = attr[i][first_col..].iter().sum();
            topk_dist[i] += p_k_j * cumulative;
        }
    }

    // NLL loss: -log(d[target])
    let eps = 1e-7;
    let target_prob = topk_dist[target].clamp(eps, 1.0 - eps);
    Ok(-target_prob.ln())
}

/// Convenience wrapper for top-k cross-entropy with bitonic + logistic defaults.
///
/// # Example
///
/// ```rust
/// use fynch::topk::topk_ce_loss;
///
/// let logits = vec![2.0, 0.5, 1.0, 0.1];
/// let p_k = vec![0.5, 0.0, 0.0, 0.5];
/// let loss = topk_ce_loss(&logits, 0, &p_k, 10.0).unwrap();
/// assert!(loss >= 0.0);
/// ```
pub fn topk_ce_loss(logits: &[f64], target: usize, p_k: &[f64], steepness: f64) -> Result<f64> {
    topk_cross_entropy_loss(
        logits,
        target,
        p_k,
        steepness,
        NetworkType::Bitonic,
        RelaxDist::Logistic,
    )
}

/// Gumbel-Softmax utilities for stochastic top-k selection.
#[cfg(feature = "gumbel")]
pub mod gumbel {
    use rand::Rng;

    /// Generate Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0, 1).
    ///
    /// Used in the Gumbel-Softmax trick for differentiable categorical sampling.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use fynch::topk::gumbel::gumbel_noise;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    /// let noise = gumbel_noise(&mut rng);
    /// // noise follows Gumbel(0, 1) distribution
    /// ```
    pub fn gumbel_noise<R: Rng + ?Sized>(rng: &mut R) -> f64 {
        let u: f64 = rng.random_range(0.0..1.0);
        let u = u.clamp(1e-10, 1.0 - 1e-10);
        -(-u.ln()).ln()
    }

    /// Add Gumbel noise to logits for stochastic sampling.
    ///
    /// Returns noisy_logits where argmax(noisy_logits) samples from
    /// the categorical distribution defined by softmax(logits).
    pub fn add_gumbel_noise<R: Rng + ?Sized>(logits: &[f64], rng: &mut R) -> Vec<f64> {
        logits.iter().map(|&l| l + gumbel_noise(rng)).collect()
    }

    /// Gumbel-Softmax: differentiable approximation to categorical sampling.
    ///
    /// Returns a soft one-hot vector that approaches a hard one-hot as
    /// temperature -> 0.
    ///
    /// # Arguments
    ///
    /// * `logits` - Unnormalized log-probabilities
    /// * `temperature` - Controls sharpness (smaller = sharper)
    /// * `rng` - Random number generator
    pub fn gumbel_softmax<R: Rng + ?Sized>(
        logits: &[f64],
        temperature: f64,
        rng: &mut R,
    ) -> Vec<f64> {
        let noisy = add_gumbel_noise(logits, rng);

        // Softmax with temperature
        let max = noisy.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = noisy
            .iter()
            .map(|&l| ((l - max) / temperature).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Relaxed top-k using iterated Gumbel-Softmax (Kool et al., 2019).
    ///
    /// Delegates to [`kuji::relaxed_topk_gumbel`] for the correct iterated
    /// masked-softmax algorithm that enforces without-replacement structure.
    /// The output sums to approximately k.
    ///
    /// # Arguments
    ///
    /// * `scores` - Input scores
    /// * `k` - Number of top elements
    /// * `temperature` - Controls sharpness
    /// * `scale` - Scaling factor for scores
    /// * `rng` - Random number generator
    pub fn relaxed_topk_gumbel<R: Rng + ?Sized>(
        scores: &[f64],
        k: usize,
        temperature: f64,
        scale: f64,
        rng: &mut R,
    ) -> Vec<f64> {
        kuji::relaxed_topk_gumbel(scores, k, temperature, scale, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sorting_network::{NetworkType, RelaxDist};

    #[test]
    fn test_topk_basic() {
        let values = [0.1, 0.9, 0.5, 0.8, 0.2];
        let (weighted, indicators) = differentiable_topk(&values, 2, 0.1);

        assert_eq!(weighted.len(), 5);
        assert_eq!(indicators.len(), 5);

        // Indices 1 (0.9) and 3 (0.8) should be top-2
        assert!(
            indicators[1] > 0.5,
            "0.9 should be in top-2: {}",
            indicators[1]
        );
        assert!(
            indicators[3] > 0.5,
            "0.8 should be in top-2: {}",
            indicators[3]
        );

        // Others should not be in top-2
        assert!(
            indicators[0] < 0.5,
            "0.1 should not be in top-2: {}",
            indicators[0]
        );
        assert!(
            indicators[2] < 0.5,
            "0.5 should not be in top-2: {}",
            indicators[2]
        );
        assert!(
            indicators[4] < 0.5,
            "0.2 should not be in top-2: {}",
            indicators[4]
        );
    }

    #[test]
    fn test_bottomk_basic() {
        let values = [0.1, 0.9, 0.5, 0.8, 0.2];
        let (_, indicators) = differentiable_bottomk(&values, 2, 0.1);

        // Indices 0 (0.1) and 4 (0.2) should be bottom-2
        assert!(
            indicators[0] > 0.5,
            "0.1 should be in bottom-2: {}",
            indicators[0]
        );
        assert!(
            indicators[4] > 0.5,
            "0.2 should be in bottom-2: {}",
            indicators[4]
        );

        // Others should not be in bottom-2
        assert!(indicators[1] < 0.5);
        assert!(indicators[2] < 0.5);
        assert!(indicators[3] < 0.5);
    }

    #[test]
    fn test_topk_empty() {
        let (w, i) = differentiable_topk(&[], 2, 0.1);
        assert!(w.is_empty());
        assert!(i.is_empty());
    }

    #[test]
    fn test_topk_k_zero() {
        let values = [1.0, 2.0, 3.0];
        let (w, i) = differentiable_topk(&values, 0, 0.1);
        assert!(w.is_empty());
        assert!(i.is_empty());
    }

    #[test]
    fn test_topk_k_geq_n() {
        let values = [1.0, 2.0, 3.0];
        let (w, indicators) = differentiable_topk(&values, 5, 0.1);

        assert_eq!(w, values);
        for &i in &indicators {
            assert_eq!(i, 1.0);
        }
    }

    #[test]
    fn test_temperature_effect() {
        let values = [0.1, 0.9, 0.5];

        // Low temperature = sharp
        let (_, indicators_sharp) = differentiable_topk(&values, 1, 0.01);
        // High temperature = smooth
        let (_, indicators_smooth) = differentiable_topk(&values, 1, 1.0);

        // Sharp should be closer to {0, 1}
        let sharp_entropy: f64 = indicators_sharp
            .iter()
            .map(|&p| {
                if p > 0.0 && p < 1.0 {
                    -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
                } else {
                    0.0
                }
            })
            .sum();

        let smooth_entropy: f64 = indicators_smooth
            .iter()
            .map(|&p| {
                if p > 0.0 && p < 1.0 {
                    -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
                } else {
                    0.0
                }
            })
            .sum();

        assert!(
            sharp_entropy < smooth_entropy,
            "sharp should have lower entropy: {} vs {}",
            sharp_entropy,
            smooth_entropy
        );
    }

    // ========================================================================
    // Sparse top-k attribution matrix tests
    // ========================================================================

    #[test]
    fn sparse_topk_correct_shape() {
        let scores = vec![3.0, 1.0, 4.0, 2.0];
        let a = sparse_topk_matrix(&scores, 2, 10.0, NetworkType::Bitonic, RelaxDist::Logistic)
            .unwrap();
        assert_eq!(a.len(), 4, "should have n=4 rows");
        for row in &a {
            assert_eq!(row.len(), 2, "should have k=2 columns");
        }
    }

    #[test]
    fn sparse_topk_correct_shape_odd_even() {
        let scores = vec![5.0, 2.0, 7.0, 1.0, 3.0];
        let a = sparse_topk_matrix(&scores, 3, 10.0, NetworkType::OddEven, RelaxDist::Logistic)
            .unwrap();
        assert_eq!(a.len(), 5);
        for row in &a {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn sparse_topk_row_sums_le_one() {
        let scores = vec![3.0, 1.0, 4.0, 2.0];
        let a = sparse_topk_matrix(&scores, 2, 20.0, NetworkType::Bitonic, RelaxDist::Logistic)
            .unwrap();
        for (i, row) in a.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                sum <= 1.0 + 0.05,
                "row {} sum should be <= 1: got {}",
                i,
                sum
            );
        }
    }

    #[test]
    fn sparse_topk_column_sums_approx_one() {
        let scores = vec![3.0, 1.0, 4.0, 2.0];
        let k = 2;
        let a = sparse_topk_matrix(&scores, k, 20.0, NetworkType::Bitonic, RelaxDist::Logistic)
            .unwrap();
        let n = scores.len();
        for j in 0..k {
            let col_sum: f64 = (0..n).map(|i| a[i][j]).sum();
            assert!(
                (col_sum - 1.0).abs() < 0.1,
                "column {} sum should be ~1.0, got {}",
                j,
                col_sum
            );
        }
    }

    #[test]
    fn sparse_topk_high_steepness_approaches_hard() {
        // With very high steepness, the matrix should approach a hard permutation.
        // Convention: column k-1 = rank 1 (largest), column 0 = rank k.
        let scores = vec![3.0, 1.0, 4.0, 2.0];
        // Hard top-2: scores 4.0 (idx 2) = rank 1, 3.0 (idx 0) = rank 2
        let k = 2;
        let a = sparse_topk_matrix(&scores, k, 100.0, NetworkType::Bitonic, RelaxDist::Logistic)
            .unwrap();

        // Element 2 (score=4.0) should dominate column k-1=1 (rank 1 = largest)
        assert!(
            a[2][k - 1] > 0.8,
            "score 4.0 should be rank 1 (col {}): got {}",
            k - 1,
            a[2][k - 1]
        );
        // Element 0 (score=3.0) should dominate column 0 (rank 2 = k-th largest)
        assert!(
            a[0][0] > 0.8,
            "score 3.0 should be rank 2 (col 0): got {}",
            a[0][0]
        );
        // Non-top-k elements should have near-zero total attribution
        let sum_1: f64 = a[1].iter().sum();
        assert!(
            sum_1 < 0.2,
            "score 1.0 should not be in top-2: row sum = {}",
            sum_1
        );
        let sum_3: f64 = a[3].iter().sum();
        assert!(
            sum_3 < 0.2,
            "score 2.0 should not be in top-2: row sum = {}",
            sum_3
        );
    }

    #[test]
    fn sparse_topk_k_equals_n_recovers_full_perm() {
        // When k = n, the attribution matrix should be a full soft permutation
        let scores = vec![3.0, 1.0, 4.0, 2.0];
        let n = scores.len();
        let a = sparse_topk_matrix(&scores, n, 20.0, NetworkType::Bitonic, RelaxDist::Logistic)
            .unwrap();

        // Should be n x n
        assert_eq!(a.len(), n);
        for row in &a {
            assert_eq!(row.len(), n);
        }

        // Row sums should be ~1
        for (i, row) in a.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.05,
                "row {} sum = {}, expected ~1.0",
                i,
                sum
            );
        }
    }

    #[test]
    fn sparse_topk_with_cauchy() {
        let scores = vec![3.0, 1.0, 4.0, 2.0];
        let a =
            sparse_topk_matrix(&scores, 2, 10.0, NetworkType::Bitonic, RelaxDist::Cauchy).unwrap();
        assert_eq!(a.len(), 4);
        assert_eq!(a[0].len(), 2);
        // Element 2 (score=4.0) should be rank 1
        assert!(a[2][0] > a[1][0], "4.0 should rank higher than 1.0");
    }

    #[test]
    fn sparse_topk_with_gaussian() {
        let scores = vec![3.0, 1.0, 4.0, 2.0];
        let a = sparse_topk_matrix(&scores, 2, 10.0, NetworkType::Bitonic, RelaxDist::Gaussian)
            .unwrap();
        assert_eq!(a.len(), 4);
        assert_eq!(a[0].len(), 2);
    }

    #[test]
    fn sparse_topk_non_power_of_two() {
        // Bitonic pads to next power of 2; verify correct shape
        let scores = vec![5.0, 2.0, 7.0];
        let a = sparse_topk_matrix(&scores, 2, 10.0, NetworkType::Bitonic, RelaxDist::Logistic)
            .unwrap();
        assert_eq!(a.len(), 3, "should have n=3 rows despite padding");
        for row in &a {
            assert_eq!(row.len(), 2);
        }
    }

    #[test]
    fn sparse_topk_convenience_wrapper() {
        let scores = vec![3.0, 1.0, 4.0, 2.0];
        let a = sparse_topk(&scores, 2, 10.0).unwrap();
        assert_eq!(a.len(), 4);
        assert_eq!(a[0].len(), 2);
    }

    #[test]
    fn sparse_topk_empty_input() {
        assert!(sparse_topk(&[], 2, 10.0).is_err());
    }

    #[test]
    fn sparse_topk_invalid_steepness() {
        assert!(sparse_topk(&[1.0, 2.0], 1, 0.0).is_err());
        assert!(sparse_topk(&[1.0, 2.0], 1, -1.0).is_err());
    }

    // ========================================================================
    // TopK cross-entropy loss tests
    // ========================================================================

    #[test]
    fn topk_ce_loss_nonnegative() {
        let logits = vec![2.0, 0.5, 1.0, 0.1];
        let p_k = vec![0.5, 0.0, 0.0, 0.5];
        let loss = topk_ce_loss(&logits, 0, &p_k, 10.0).unwrap();
        assert!(loss >= 0.0, "loss should be non-negative: {}", loss);
    }

    #[test]
    fn topk_ce_loss_correct_class_low_loss() {
        let logits = vec![5.0, 0.1, 0.2, 0.3];
        let p_k = vec![0.5, 0.0, 0.0, 0.5];
        let loss = topk_ce_loss(&logits, 0, &p_k, 10.0).unwrap();
        assert!(
            loss < 1.0,
            "correct class at rank 1 should have low loss: {}",
            loss
        );
    }

    #[test]
    fn topk_ce_loss_wrong_class_high_loss() {
        let logits = vec![0.1, 5.0, 4.0, 3.0];
        // Target is class 0 which has the lowest logit
        let p_k = vec![1.0]; // only top-1 matters
        let loss_wrong = topk_ce_loss(&logits, 0, &p_k, 10.0).unwrap();
        let loss_right = topk_ce_loss(&logits, 1, &p_k, 10.0).unwrap();
        assert!(
            loss_wrong > loss_right,
            "wrong class should have higher loss: {} vs {}",
            loss_wrong,
            loss_right
        );
    }

    #[test]
    fn topk_ce_loss_uniform_pk() {
        let logits = vec![2.0, 0.5, 1.0, 0.1];
        // k > n is clamped; use k=n
        let p_k = vec![0.25, 0.25, 0.25, 0.25];
        let loss = topk_ce_loss(&logits, 0, &p_k, 10.0).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn topk_ce_loss_decreases_when_in_topk() {
        // Class 0 has a moderate logit -- it's in the top-k but not rank 1.
        // With p_k that weights all positions, the loss should be moderate.
        // With p_k that only weights rank 1, the loss should be higher.
        let logits = vec![2.0, 5.0, 1.0, 0.1];
        let target = 0;

        let p_k_topk = vec![0.0, 1.0]; // only care about being in top-2
        let p_k_top1 = vec![1.0, 0.0]; // only care about rank 1

        let loss_topk = topk_ce_loss(&logits, target, &p_k_topk, 10.0).unwrap();
        let loss_top1 = topk_ce_loss(&logits, target, &p_k_top1, 10.0).unwrap();

        assert!(
            loss_topk < loss_top1,
            "top-k loss should be lower than top-1 loss when target is rank 2: {} vs {}",
            loss_topk,
            loss_top1
        );
    }

    #[test]
    fn topk_ce_loss_invalid_target() {
        let logits = vec![1.0, 2.0, 3.0];
        let p_k = vec![1.0];
        assert!(topk_ce_loss(&logits, 5, &p_k, 10.0).is_err());
    }
}
