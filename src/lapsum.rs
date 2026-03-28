//! LapSum differentiable sorting operator.
//!
//! Constructs soft permutation matrices using Laplacian-based smoothing.
//! A single operator that unifies sorting, ranking, and top-k selection.
//!
//! # Algorithm
//!
//! Given scores `s = [s_1, ..., s_n]` and `temperature > 0`:
//!
//! 1. Compute pairwise absolute differences `|s_i - s_j|`
//! 2. Apply Laplacian kernel: `A_ij = exp(-|s_i - s_j| / temperature)`
//! 3. Row-normalize: `P_ij = A_ij / sum_k A_ik`
//!
//! The resulting `P` is a soft permutation matrix (doubly-stochastic in
//! the low-temperature limit).
//!
//! # Unified Operations
//!
//! | Operation | Formula | Function |
//! |-----------|---------|----------|
//! | Sort | `P * values` | [`lapsum_sort`] |
//! | Rank | `diag(P * [1..n])` | [`lapsum_rank`] |
//! | Top-k | `sum first k rows of P` | [`lapsum_topk`] |
//!
//! # Temperature Behavior
//!
//! - `temperature -> 0`: converges to hard permutation (exact sort/rank)
//! - `temperature -> inf`: uniform matrix (no discrimination)
//!
//! # References
//!
//! - Struski, Bednarczyk, Podolak (2025). "LapSum: One Method to
//!   Differentiate Them All"

use crate::{Error, Result};

/// Compute the LapSum soft permutation matrix.
///
/// Returns an `n x n` row-major matrix where entry `P[i*n + j]` is the
/// soft probability that element `j` should be placed at position `i`
/// in the sorted order.
///
/// The Laplacian kernel assigns high weight to score pairs that are
/// close together, producing a smooth relaxation of the permutation.
///
/// # Arguments
///
/// * `scores` - Input scores to sort
/// * `temperature` - Temperature controlling smoothness (must be positive)
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] for empty input, or
/// [`Error::InvalidTemperature`] if `temperature <= 0`.
pub fn lapsum_permutation(scores: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if scores.is_empty() {
        return Err(Error::EmptyInput);
    }
    if temperature <= 0.0 {
        return Err(Error::InvalidTemperature(temperature));
    }

    let n = scores.len();

    // Sort scores to get the target ordering.
    // sorted_indices[pos] = original index of the element at position `pos`.
    let mut indexed: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    // Build kernel matrix: A[pos][orig] = exp(-|sorted_score[pos] - score[orig]| / temperature)
    // Then row-normalize to get P.
    let inv_sigma = 1.0 / temperature;
    let mut perm = vec![0.0_f64; n * n];

    for (pos, &(_, sorted_val)) in indexed.iter().enumerate() {
        let row_start = pos * n;
        let mut row_sum = 0.0_f64;

        for (orig, &score) in scores.iter().enumerate() {
            let kernel = (-(sorted_val - score).abs() * inv_sigma).exp();
            perm[row_start + orig] = kernel;
            row_sum += kernel;
        }

        // Row-normalize
        if row_sum > 0.0 {
            let inv_sum = 1.0 / row_sum;
            for j in 0..n {
                perm[row_start + j] *= inv_sum;
            }
        }
    }

    Ok(perm)
}

/// LapSum sort: apply the soft permutation to values.
///
/// Computes `P * values` where `P` is the LapSum permutation matrix.
/// At low `temperature`, this approximates hard sorting.
///
/// # Arguments
///
/// * `scores` - Scores determining the sort order
/// * `values` - Values to be sorted (must have same length as `scores`)
/// * `temperature` - Temperature
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if `scores` and `values` differ in length.
pub fn lapsum_sort(scores: &[f64], values: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if scores.len() != values.len() {
        return Err(Error::LengthMismatch(scores.len(), values.len()));
    }
    let perm = lapsum_permutation(scores, temperature)?;
    let n = scores.len();

    // Matrix-vector multiply: result[i] = sum_j P[i][j] * values[j]
    let mut result = vec![0.0; n];
    for (i, res) in result.iter_mut().enumerate().take(n) {
        let row_start = i * n;
        for j in 0..n {
            *res += perm[row_start + j] * values[j];
        }
    }
    Ok(result)
}

/// LapSum rank: compute soft ranks for each element.
///
/// Returns a vector where entry `i` is the soft rank of `scores[i]`.
/// Ranks are 1-based: rank 1 = smallest, rank n = largest.
///
/// Computed as `diag(P^T * [1, 2, ..., n])`, i.e., for each original
/// element, the expected position it occupies in the sorted order.
///
/// # Arguments
///
/// * `scores` - Input scores
/// * `temperature` - Temperature
pub fn lapsum_rank(scores: &[f64], temperature: f64) -> Result<Vec<f64>> {
    let perm = lapsum_permutation(scores, temperature)?;
    let n = scores.len();

    // rank[j] = sum_i P[i][j] * (i + 1)
    // where (i+1) is the 1-based position.
    let mut ranks = vec![0.0; n];
    for i in 0..n {
        let row_start = i * n;
        let position = (i + 1) as f64;
        for j in 0..n {
            ranks[j] += perm[row_start + j] * position;
        }
    }
    Ok(ranks)
}

/// LapSum top-k: soft selection weights for the top-k elements.
///
/// Returns a vector of length `n` where entry `j` is the soft probability
/// that element `j` is among the top-k (largest) elements.
///
/// Computed by summing the last `k` rows of the permutation matrix
/// (the rows corresponding to the k largest sorted positions).
///
/// # Arguments
///
/// * `scores` - Input scores
/// * `k` - Number of top elements to select
/// * `temperature` - Temperature
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] if `k == 0` or `k > n`.
pub fn lapsum_topk(scores: &[f64], k: usize, temperature: f64) -> Result<Vec<f64>> {
    let n = scores.len();
    if k == 0 || k > n {
        return Err(Error::EmptyInput);
    }
    let perm = lapsum_permutation(scores, temperature)?;

    // Sum the last k rows (positions n-k..n correspond to the k largest).
    let mut weights = vec![0.0; n];
    for i in (n - k)..n {
        let row_start = i * n;
        for j in 0..n {
            weights[j] += perm[row_start + j];
        }
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn permutation_rows_sum_to_one() {
        let scores = [3.0, 1.0, 4.0, 1.5, 9.0];
        let perm = lapsum_permutation(&scores, 0.5).unwrap();
        let n = scores.len();

        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| perm[i * n + j]).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn permutation_entries_nonnegative() {
        let scores = [3.0, 1.0, 4.0, 1.5, 9.0];
        let perm = lapsum_permutation(&scores, 0.5).unwrap();
        for &val in &perm {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn low_sigma_approaches_hard_sort() {
        let scores = [3.0, 1.0, 4.0, 2.0];
        let sorted = lapsum_sort(&scores, &scores, 0.001).unwrap();

        // At very low temperature, should be close to [1.0, 2.0, 3.0, 4.0]
        assert_relative_eq!(sorted[0], 1.0, epsilon = 0.1);
        assert_relative_eq!(sorted[1], 2.0, epsilon = 0.1);
        assert_relative_eq!(sorted[2], 3.0, epsilon = 0.1);
        assert_relative_eq!(sorted[3], 4.0, epsilon = 0.1);
    }

    #[test]
    fn low_sigma_ranks_match_hard_ranks() {
        let scores = [0.5, 0.2, 0.8, 0.1];
        let ranks = lapsum_rank(&scores, 0.001).unwrap();

        // Hard ranks (ascending): 0.1 -> 1, 0.2 -> 2, 0.5 -> 3, 0.8 -> 4
        // So: scores[0]=0.5 -> rank 3, scores[1]=0.2 -> rank 2,
        //     scores[2]=0.8 -> rank 4, scores[3]=0.1 -> rank 1
        assert_relative_eq!(ranks[0], 3.0, epsilon = 0.1);
        assert_relative_eq!(ranks[1], 2.0, epsilon = 0.1);
        assert_relative_eq!(ranks[2], 4.0, epsilon = 0.1);
        assert_relative_eq!(ranks[3], 1.0, epsilon = 0.1);
    }

    #[test]
    fn topk_weights_sum_to_k() {
        let scores = [0.5, 0.2, 0.8, 0.1, 0.9];
        let k = 2;
        let weights = lapsum_topk(&scores, k, 0.5).unwrap();

        let total: f64 = weights.iter().sum();
        assert_relative_eq!(total, k as f64, epsilon = 0.1);
    }

    #[test]
    fn topk_low_sigma_selects_top_elements() {
        let scores = [0.5, 0.2, 0.8, 0.1, 0.9];
        let weights = lapsum_topk(&scores, 2, 0.001).unwrap();

        // Top-2 are indices 2 (0.8) and 4 (0.9)
        assert!(weights[2] > 0.9, "weight[2] = {}", weights[2]);
        assert!(weights[4] > 0.9, "weight[4] = {}", weights[4]);
        assert!(weights[0] < 0.1, "weight[0] = {}", weights[0]);
        assert!(weights[1] < 0.1, "weight[1] = {}", weights[1]);
        assert!(weights[3] < 0.1, "weight[3] = {}", weights[3]);
    }

    #[test]
    fn symmetry_swapping_scores_swaps_columns() {
        let scores_a = [1.0, 3.0, 2.0];
        let scores_b = [3.0, 1.0, 2.0]; // swapped indices 0 and 1
        let sigma = 0.5;

        let perm_a = lapsum_permutation(&scores_a, sigma).unwrap();
        let perm_b = lapsum_permutation(&scores_b, sigma).unwrap();
        let n = 3;

        // Column 0 in perm_a should equal column 1 in perm_b (and vice versa)
        for i in 0..n {
            assert_relative_eq!(perm_a[i * n], perm_b[i * n + 1], epsilon = 1e-10);
            assert_relative_eq!(perm_a[i * n + 1], perm_b[i * n], epsilon = 1e-10);
            assert_relative_eq!(perm_a[i * n + 2], perm_b[i * n + 2], epsilon = 1e-10);
        }
    }

    #[test]
    fn sort_preserves_sum() {
        // The soft sort should preserve the weighted sum of values
        let scores = [3.0, 1.0, 4.0, 2.0];
        let values = [10.0, 20.0, 30.0, 40.0];
        let sorted = lapsum_sort(&scores, &values, 0.5).unwrap();

        let orig_sum: f64 = values.iter().sum();
        let sorted_sum: f64 = sorted.iter().sum();
        assert_relative_eq!(orig_sum, sorted_sum, epsilon = 1e-6);
    }

    #[test]
    fn errors_on_empty_input() {
        assert!(lapsum_permutation(&[], 1.0).is_err());
        assert!(lapsum_sort(&[], &[], 1.0).is_err());
        assert!(lapsum_rank(&[], 1.0).is_err());
    }

    #[test]
    fn errors_on_invalid_temperature() {
        let s = [1.0, 2.0];
        assert!(lapsum_permutation(&s, 0.0).is_err());
        assert!(lapsum_permutation(&s, -1.0).is_err());
    }

    #[test]
    fn errors_on_length_mismatch() {
        assert!(lapsum_sort(&[1.0, 2.0], &[1.0], 0.5).is_err());
    }

    #[test]
    fn errors_on_invalid_k() {
        let s = [1.0, 2.0, 3.0];
        assert!(lapsum_topk(&s, 0, 0.5).is_err());
        assert!(lapsum_topk(&s, 4, 0.5).is_err());
    }

    #[test]
    fn single_element() {
        let perm = lapsum_permutation(&[5.0], 1.0).unwrap();
        assert_relative_eq!(perm[0], 1.0, epsilon = 1e-10);

        let sorted = lapsum_sort(&[5.0], &[5.0], 1.0).unwrap();
        assert_relative_eq!(sorted[0], 5.0, epsilon = 1e-10);

        let ranks = lapsum_rank(&[5.0], 1.0).unwrap();
        assert_relative_eq!(ranks[0], 1.0, epsilon = 1e-10);
    }
}
