//! LapSum relaxations for sorting, ranking, and top-k selection.
//!
//! For scores `s_i` and scale `alpha > 0`, define
//!
//! ```text
//! L(b) = sum_i LaplaceCDF((b - s_i) / alpha).
//! ```
//!
//! LapSum inverts this monotone function exactly between consecutive sorted
//! scores. Sorting evaluates `L^-1` at the half-integers, ranking sums pairwise
//! Laplace CDFs, and top-k selection evaluates CDFs around the threshold whose
//! expected selection mass is `k`. Sorting the scores dominates the forward
//! cost: O(n log n) time and O(n) memory.
//!
//! The earlier releases' pairwise Laplacian-kernel smoother is retained under
//! the explicit [`laplacian_kernel_permutation`],
//! [`laplacian_kernel_sort`], [`laplacian_kernel_rank`], and
//! [`laplacian_kernel_topk`] names. The old `lapsum_*` spellings remain
//! deprecated aliases for the 0.3 release line.
//!
//! # Reference
//!
//! Struski, Bednarczyk, Podolak, and Tabor (2025), "LapSum - One Method to
//! Differentiate Them All: Ranking, Sorting and Top-k Selection," ICML.

use crate::{Error, Result};

/// Invert the sum of Laplace CDFs at `mass`.
///
/// `mass` must be finite and lie strictly between zero and `scores.len()`.
/// The result is the threshold `b` satisfying
/// `sum_i LaplaceCDF((b - scores[i]) / scale) = mass`.
pub fn lapsum_threshold(scores: &[f64], mass: f64, scale: f64) -> Result<f64> {
    PreparedLapSum::new(scores, scale)?.threshold(mass)
}

/// LapSum soft top-k weights for the `k` largest scores.
///
/// Every output lies in `[0, 1]`, and the outputs sum to `k` up to floating
/// point error. Complexity is O(n log n) time and O(n) memory.
pub fn lapsum_soft_topk(scores: &[f64], k: usize, scale: f64) -> Result<Vec<f64>> {
    validate(scores, scale)?;
    if k == 0 || k > scores.len() {
        return Err(Error::InvalidWeights);
    }
    if k == scores.len() {
        return Ok(vec![1.0; scores.len()]);
    }

    // Selecting the largest k is equivalent to placing `n-k` mass below the
    // threshold and evaluating the reflected Laplace CDF above it.
    let threshold = lapsum_threshold(scores, (scores.len() - k) as f64, scale)?;
    Ok(scores
        .iter()
        .map(|&score| laplace_cdf((score - threshold) / scale))
        .collect())
}

/// LapSum soft ascending ranks, in input order and 1-based.
///
/// Complexity is O(n log n) time and O(n) memory.
pub fn lapsum_soft_rank(scores: &[f64], scale: f64) -> Result<Vec<f64>> {
    validate(scores, scale)?;
    let n = scores.len();
    let mut indexed: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let offset = indexed[0].1;
    for (_, value) in &mut indexed {
        *value = (*value - offset) / scale;
        if !value.is_finite() {
            return Err(Error::InvalidWeights);
        }
    }

    let mut sorted_ranks = vec![0.0; n];
    let mut carried = 0.0;
    for i in 0..n - 1 {
        carried = (indexed[i].1 - indexed[i + 1].1).exp() * (1.0 + carried);
        sorted_ranks[i] += i as f64;
        sorted_ranks[i + 1] -= 0.5 * carried;
    }
    sorted_ranks[n - 1] += (n - 1) as f64;

    carried = 0.0;
    for i in (1..n).rev() {
        carried = (indexed[i - 1].1 - indexed[i].1).exp() * (1.0 + carried);
        sorted_ranks[i - 1] += 0.5 * carried;
    }

    let mut ranks = vec![0.0; n];
    for ((original, _), rank) in indexed.into_iter().zip(sorted_ranks) {
        ranks[original] = rank + 1.0;
    }
    Ok(ranks)
}

/// LapSum soft ascending sort.
///
/// The i-th output is the inverse LapSum threshold at mass `i + 0.5`.
/// Complexity is O(n log n) time and O(n) memory.
pub fn lapsum_soft_sort(scores: &[f64], scale: f64) -> Result<Vec<f64>> {
    let prepared = PreparedLapSum::new(scores, scale)?;
    (0..scores.len())
        .map(|i| prepared.threshold(i as f64 + 0.5))
        .collect()
}

struct PreparedLapSum {
    sorted: Vec<f64>,
    left_sum: Vec<f64>,
    right_sum: Vec<f64>,
    scale: f64,
    offset: f64,
}

impl PreparedLapSum {
    fn new(scores: &[f64], scale: f64) -> Result<Self> {
        validate(scores, scale)?;
        let n = scores.len();
        let mut sorted = scores.to_vec();
        sorted.sort_by(f64::total_cmp);
        let offset = sorted[0];

        // Work in units of `scale`; adjacent differences are non-positive in
        // both recurrences, so the exponentials cannot overflow.
        for value in &mut sorted {
            *value = (*value - offset) / scale;
            if !value.is_finite() {
                return Err(Error::InvalidWeights);
            }
        }

        let mut left_sum = vec![0.0; n];
        for i in 1..n {
            left_sum[i] = (sorted[i - 1] - sorted[i]).exp() * (1.0 + left_sum[i - 1]);
        }

        let mut right_sum = vec![0.0; n];
        for i in (0..n - 1).rev() {
            right_sum[i] = (sorted[i] - sorted[i + 1]).exp() * (1.0 + right_sum[i + 1]);
        }

        Ok(Self {
            sorted,
            left_sum,
            right_sum,
            scale,
            offset,
        })
    }

    fn threshold(&self, mass: f64) -> Result<f64> {
        let n = self.sorted.len();
        if !mass.is_finite() || mass <= 0.0 || mass >= n as f64 {
            return Err(Error::InvalidWeights);
        }

        let at_score = |i: usize| i as f64 + 0.5 * (1.0 + self.right_sum[i] - self.left_sum[i]);
        let mut low = 0;
        let mut high = n;
        while low < high {
            let middle = low + (high - low) / 2;
            if at_score(middle) < mass {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let upper = low;

        let scaled = if upper == 0 {
            solve_interval(
                self.sorted[0],
                self.sorted[0],
                1.0 + self.right_sum[0],
                0.0,
                0.0,
                mass,
            )
        } else if upper == n {
            solve_interval(
                self.sorted[n - 1],
                self.sorted[n - 1],
                0.0,
                1.0 + self.left_sum[n - 1],
                n as f64,
                mass,
            )
        } else {
            let lower = upper - 1;
            solve_interval(
                self.sorted[lower],
                self.sorted[upper],
                1.0 + self.right_sum[upper],
                1.0 + self.left_sum[lower],
                upper as f64,
                mass,
            )
        };
        let threshold = scaled.mul_add(self.scale, self.offset);
        if threshold.is_finite() {
            Ok(threshold)
        } else {
            Err(Error::InvalidWeights)
        }
    }
}

fn validate(scores: &[f64], scale: f64) -> Result<()> {
    if scores.is_empty() {
        return Err(Error::EmptyInput);
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::InvalidTemperature(scale));
    }
    if !scores.iter().all(|score| score.is_finite()) {
        return Err(Error::InvalidWeights);
    }
    Ok(())
}

fn laplace_cdf(x: f64) -> f64 {
    if x <= 0.0 {
        0.5 * x.exp()
    } else {
        1.0 - 0.5 * (-x).exp()
    }
}

// Closed-form inverse on an interval containing no score. This is Algorithm 1's
// quadratic solve, expressed in scaled coordinates.
fn solve_interval(left: f64, right: f64, a: f64, b: f64, c: f64, mass: f64) -> f64 {
    if mass > c && a > 0.0 {
        let diff = mass - c;
        right - a.ln() + (diff + (diff * diff + (left - right).exp() * a * b).sqrt()).ln()
    } else if mass == c && a > 0.0 && b > 0.0 {
        0.5 * (left + right + b.ln() - a.ln())
    } else if mass < c && b > 0.0 {
        let diff = c - mass;
        left + b.ln() - (diff + (diff * diff + (left - right).exp() * a * b).sqrt()).ln()
    } else {
        0.5 * (left + right)
    }
}

/// Pairwise Laplacian-kernel row-stochastic smoother from fynch 0.3.
///
/// This is retained for compatibility; it is not the ICML LapSum operator and
/// is not generally doubly stochastic. The row-major entry at `i*n + j`
/// weights original element `j` at sorted position `i`.
pub fn laplacian_kernel_permutation(scores: &[f64], scale: f64) -> Result<Vec<f64>> {
    validate(scores, scale)?;
    let n = scores.len();
    let mut sorted = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut matrix = vec![0.0; n * n];
    for (position, &center) in sorted.iter().enumerate() {
        let row = &mut matrix[position * n..(position + 1) * n];
        for (weight, &score) in row.iter_mut().zip(scores) {
            *weight = (-(center - score).abs() / scale).exp();
        }
        let sum: f64 = row.iter().sum();
        for weight in row {
            *weight /= sum;
        }
    }
    Ok(matrix)
}

/// Apply the compatibility Laplacian-kernel smoother to `values`.
pub fn laplacian_kernel_sort(scores: &[f64], values: &[f64], scale: f64) -> Result<Vec<f64>> {
    if scores.len() != values.len() {
        return Err(Error::LengthMismatch(scores.len(), values.len()));
    }
    let matrix = laplacian_kernel_permutation(scores, scale)?;
    let n = scores.len();
    Ok(matrix
        .chunks_exact(n)
        .map(|row| row.iter().zip(values).map(|(p, value)| p * value).sum())
        .collect())
}

/// Expected positions under the compatibility Laplacian-kernel smoother.
pub fn laplacian_kernel_rank(scores: &[f64], scale: f64) -> Result<Vec<f64>> {
    let matrix = laplacian_kernel_permutation(scores, scale)?;
    let n = scores.len();
    let mut ranks = vec![0.0; n];
    for (position, row) in matrix.chunks_exact(n).enumerate() {
        for (rank, probability) in ranks.iter_mut().zip(row) {
            *rank += (position + 1) as f64 * probability;
        }
    }
    Ok(ranks)
}

/// Top-k weights from the compatibility Laplacian-kernel smoother.
///
/// This sums the last `k` rows, corresponding to the largest sorted positions.
pub fn laplacian_kernel_topk(scores: &[f64], k: usize, scale: f64) -> Result<Vec<f64>> {
    let n = scores.len();
    if k == 0 || k > n {
        return Err(Error::EmptyInput);
    }
    let matrix = laplacian_kernel_permutation(scores, scale)?;
    let mut weights = vec![0.0; n];
    for row in matrix.chunks_exact(n).skip(n - k) {
        for (weight, probability) in weights.iter_mut().zip(row) {
            *weight += probability;
        }
    }
    Ok(weights)
}

/// Compatibility alias for [`laplacian_kernel_permutation`].
#[deprecated(since = "0.3.3", note = "use laplacian_kernel_permutation")]
pub fn lapsum_permutation(scores: &[f64], scale: f64) -> Result<Vec<f64>> {
    laplacian_kernel_permutation(scores, scale)
}

/// Compatibility alias for [`laplacian_kernel_sort`].
#[deprecated(since = "0.3.3", note = "use lapsum_soft_sort for ICML LapSum")]
pub fn lapsum_sort(scores: &[f64], values: &[f64], scale: f64) -> Result<Vec<f64>> {
    laplacian_kernel_sort(scores, values, scale)
}

/// Compatibility alias for [`laplacian_kernel_rank`].
#[deprecated(since = "0.3.3", note = "use lapsum_soft_rank for ICML LapSum")]
pub fn lapsum_rank(scores: &[f64], scale: f64) -> Result<Vec<f64>> {
    laplacian_kernel_rank(scores, scale)
}

/// Compatibility alias for [`laplacian_kernel_topk`].
#[deprecated(since = "0.3.3", note = "use lapsum_soft_topk for ICML LapSum")]
pub fn lapsum_topk(scores: &[f64], k: usize, scale: f64) -> Result<Vec<f64>> {
    laplacian_kernel_topk(scores, k, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const SCORES: [f64; 4] = [-1.0, 0.25, 2.0, 3.5];
    const SCALE: f64 = 0.7;

    #[test]
    fn threshold_matches_hand_computable_pair() {
        // Symmetry gives b=1 for scores [0, 2] and one unit of mass.
        assert_relative_eq!(
            lapsum_threshold(&[0.0, 2.0], 1.0, 0.5).unwrap(),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn topk_has_exact_mass() {
        let weights = lapsum_soft_topk(&SCORES, 2, SCALE).unwrap();
        assert_relative_eq!(weights.iter().sum::<f64>(), 2.0, epsilon = 1e-12);
        assert!(weights.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn threshold_matches_independent_cdf_bisection() {
        fn naive(scores: &[f64], mass: f64, scale: f64) -> f64 {
            let mut low = scores.iter().copied().fold(f64::INFINITY, f64::min) - 50.0 * scale;
            let mut high = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 50.0 * scale;
            for _ in 0..200 {
                let middle = 0.5 * (low + high);
                let sum: f64 = scores
                    .iter()
                    .map(|&score| laplace_cdf((middle - score) / scale))
                    .sum();
                if sum < mass {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            0.5 * (low + high)
        }

        for scores in [
            vec![-3.0, -0.5, 0.25, 4.0],
            vec![1.0, 1.0, 1.0, 2.0],
            vec![-100.0, 0.0, 100.0],
        ] {
            for mass in [0.25, 0.5, 1.5, scores.len() as f64 - 0.25] {
                if mass < scores.len() as f64 {
                    let got = lapsum_threshold(&scores, mass, 0.8).unwrap();
                    let expected = naive(&scores, mass, 0.8);
                    assert_relative_eq!(got, expected, epsilon = 2e-12);
                }
            }
        }
    }

    #[test]
    fn rank_matches_independent_pairwise_definition() {
        let scores = [-2.0, 0.0, 0.0, 3.0, 8.0];
        let scale = 1.3;
        let ranks = lapsum_soft_rank(&scores, scale).unwrap();
        for (i, &score) in scores.iter().enumerate() {
            let expected = 1.0
                + scores
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &other)| laplace_cdf((score - other) / scale))
                    .sum::<f64>();
            assert_relative_eq!(ranks[i], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn matches_official_lapsum_reference_values() {
        // Generated from gmum/LapSum commit c277d1a09708b2c1b19eaa3cdf2f0e32632b76f2
        // using its pure-PyTorch SoftTopK equations and CPU soft-sort/rank kernels.
        let topk = lapsum_soft_topk(&SCORES, 2, SCALE).unwrap();
        let expected_topk = [
            0.02349650439272626,
            0.14012935307345586,
            0.8535549532922095,
            0.9828191892416084,
        ];
        for (&got, expected) in topk.iter().zip(expected_topk) {
            assert_relative_eq!(got, expected, epsilon = 1e-12);
        }

        let ranks = lapsum_soft_rank(&SCORES, SCALE).unwrap();
        let expected_ranks = [
            1.091527895733334,
            1.9620189467297526,
            3.0107351903686506,
            3.9357179671682627,
        ];
        for (&got, expected) in ranks.iter().zip(expected_ranks) {
            assert_relative_eq!(got, expected, epsilon = 1e-12);
        }

        let sorted = lapsum_soft_sort(&SCORES, SCALE).unwrap();
        let expected_sorted = [
            -1.1176705214211138,
            0.2933404938310701,
            1.9875185187837796,
            3.584662260410507,
        ];
        for (&got, expected) in sorted.iter().zip(expected_sorted) {
            assert_relative_eq!(got, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn low_scale_approaches_hard_operators() {
        let scores = [3.0, 1.0, 4.0, 2.0];
        let sorted = lapsum_soft_sort(&scores, 1e-3).unwrap();
        for (&got, expected) in sorted.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert_relative_eq!(got, expected, epsilon = 1e-2);
        }
        let ranks = lapsum_soft_rank(&scores, 1e-3).unwrap();
        for (&got, expected) in ranks.iter().zip([3.0, 1.0, 4.0, 2.0]) {
            assert_relative_eq!(got, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn ties_receive_equal_ranks_and_weights() {
        let scores = [1.0, 1.0, 2.0];
        let ranks = lapsum_soft_rank(&scores, 0.5).unwrap();
        assert_relative_eq!(ranks[0], ranks[1], epsilon = 1e-12);
        let weights = lapsum_soft_topk(&scores, 1, 0.5).unwrap();
        assert_relative_eq!(weights[0], weights[1], epsilon = 1e-12);
    }

    #[test]
    fn rejects_invalid_domains() {
        assert!(lapsum_soft_topk(&[], 1, 1.0).is_err());
        assert!(lapsum_soft_topk(&[1.0], 0, 1.0).is_err());
        assert!(lapsum_soft_topk(&[1.0], 2, 1.0).is_err());
        assert!(lapsum_soft_topk(&[f64::NAN], 1, 1.0).is_err());
        assert!(lapsum_soft_topk(&[1.0], 1, f64::NAN).is_err());
        assert!(lapsum_soft_topk(&[1.0], 1, f64::INFINITY).is_err());
        assert!(lapsum_soft_topk(&[1.0], 1, 0.0).is_err());
        assert!(lapsum_threshold(&[1.0, 2.0], 0.0, 1.0).is_err());
        assert!(lapsum_threshold(&[1.0, 2.0], 2.0, 1.0).is_err());
    }

    #[test]
    fn compatibility_kernel_keeps_old_behavior_honestly_named() {
        let scores = [0.0, 1.0, 3.0];
        let matrix = laplacian_kernel_permutation(&scores, 1.0).unwrap();
        for row in matrix.chunks_exact(scores.len()) {
            assert_relative_eq!(row.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        }
        let column_sum: f64 = matrix.chunks_exact(scores.len()).map(|row| row[1]).sum();
        assert!((column_sum - 1.0).abs() > 1e-3);
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_aliases_equal_kernel_operators() {
        let scores = [3.0, 1.0, 4.0, 2.0];
        let values = [30.0, 10.0, 40.0, 20.0];
        let scale = 0.6;

        assert_eq!(
            lapsum_permutation(&scores, scale).unwrap(),
            laplacian_kernel_permutation(&scores, scale).unwrap()
        );
        assert_eq!(
            lapsum_sort(&scores, &values, scale).unwrap(),
            laplacian_kernel_sort(&scores, &values, scale).unwrap()
        );
        assert_eq!(
            lapsum_rank(&scores, scale).unwrap(),
            laplacian_kernel_rank(&scores, scale).unwrap()
        );
        assert_eq!(
            lapsum_topk(&scores, 2, scale).unwrap(),
            laplacian_kernel_topk(&scores, 2, scale).unwrap()
        );
    }
}
