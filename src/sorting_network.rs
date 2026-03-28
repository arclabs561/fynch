//! Differentiable sorting networks: bitonic and odd-even.
//!
//! Classic sorting networks are fixed comparator topologies that sort any input
//! in a data-oblivious pattern. By relaxing each hard comparator (min/max) into
//! a smooth interpolation via a sigmoid-like function, we obtain differentiable
//! sorting that returns soft permutation matrices.
//!
//! This complements the PAVA/Sinkhorn approaches in fynch's root module with a
//! **network-based** relaxation that:
//!
//! - Produces full soft permutation matrices (not just soft ranks)
//! - Has O(n log^2 n) comparators (bitonic) or O(n^2) (odd-even)
//! - Supports monotonicity guarantees (Petersen et al., ICLR 2022)
//! - Parameterizes the comparator relaxation via distribution families
//!
//! # References
//!
//! - Petersen et al. (2021), "Differentiable Sorting Networks for Scalable
//!   Sorting and Ranking Supervision" (ICML)
//! - Petersen et al. (2022), "Monotonic Differentiable Sorting Networks" (ICLR)
//! - Batcher (1968), "Sorting Networks and their Applications"
//!
//! # Example
//!
//! ```rust
//! use fynch::sorting_network::{DiffSortNet, NetworkType, RelaxDist};
//!
//! let x = vec![3.0, 1.0, 4.0, 2.0];
//! let net = DiffSortNet::new(NetworkType::Bitonic, 4, 5.0, RelaxDist::Logistic);
//! let (sorted, perm) = net.sort(&x).unwrap();
//!
//! // sorted is approximately [1, 2, 3, 4]
//! // perm is an n x n soft permutation matrix: sort(x) ≈ x @ perm
//! assert_eq!(perm.len(), 4);
//! assert_eq!(perm[0].len(), 4);
//! ```

use crate::{Error, Result};

/// The type of sorting network topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkType {
    /// Bitonic sorting network. Requires power-of-2 input size.
    /// O(n log^2 n) comparators, O(log^2 n) depth.
    Bitonic,
    /// Odd-even transposition sort. Works with any input size.
    /// O(n^2) comparators, O(n) depth.
    OddEven,
}

/// Distribution family for relaxing the comparator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RelaxDist {
    /// Logistic sigmoid: sigma(x) = 1 / (1 + exp(-x)).
    /// Standard choice, smooth and well-behaved.
    Logistic,
    /// Cauchy CDF: 0.5 + arctan(x)/pi.
    /// Heavier tails, more gradient signal far from the decision boundary.
    Cauchy,
    /// Gaussian CDF (probit): Phi(x).
    /// Lighter tails, sharper transitions.
    Gaussian,
}

/// A differentiable sorting network.
#[derive(Debug, Clone)]
pub struct DiffSortNet {
    /// Network topology type.
    pub network_type: NetworkType,
    /// Input size (padded to power-of-2 for bitonic).
    pub size: usize,
    /// Inverse temperature (steepness). Higher = sharper, closer to hard sort.
    pub steepness: f64,
    /// Distribution for the relaxed comparator.
    pub dist: RelaxDist,
    /// Pre-computed comparator pairs: (i, j) where i < j.
    comparators: Vec<(usize, usize)>,
}

impl DiffSortNet {
    /// Create a new differentiable sorting network.
    ///
    /// For `Bitonic`, the size must be a power of 2. If not, inputs will be
    /// padded with `-inf` to the next power of 2.
    ///
    /// # Arguments
    ///
    /// * `network_type` - Bitonic or OddEven
    /// * `size` - Number of elements to sort
    /// * `steepness` - Inverse temperature (higher = sharper)
    /// * `dist` - Distribution family for comparator relaxation
    pub fn new(network_type: NetworkType, size: usize, steepness: f64, dist: RelaxDist) -> Self {
        let actual_size = match network_type {
            NetworkType::Bitonic => size.next_power_of_two(),
            NetworkType::OddEven => size,
        };

        let comparators = match network_type {
            NetworkType::Bitonic => bitonic_comparators(actual_size),
            NetworkType::OddEven => odd_even_comparators(actual_size),
        };

        Self {
            network_type,
            size: actual_size,
            steepness,
            dist,
            comparators,
        }
    }

    /// Sort the input, returning (sorted_values, soft_permutation_matrix).
    ///
    /// The permutation matrix P satisfies: `sorted ≈ x @ P` (row convention).
    /// P[i][j] is the soft weight of input position i contributing to output position j.
    pub fn sort(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        if x.is_empty() {
            return Err(Error::EmptyInput);
        }
        if self.steepness <= 0.0 {
            return Err(Error::InvalidTemperature(self.steepness));
        }

        let n = self.size;

        // Pad input if needed (bitonic requires power-of-2).
        // Pad with a large finite value so padding sorts to the end and gets trimmed.
        // Using INFINITY causes NaN when two padding values are compared (inf - inf).
        let pad_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1e6;
        let mut values: Vec<f64> = Vec::with_capacity(n);
        values.extend_from_slice(x);
        while values.len() < n {
            values.push(pad_val);
        }

        // Initialize permutation matrix as identity
        // perm[i][j] = probability that original position i ends up at output position j
        let mut perm: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();

        // Apply each comparator (a, b) means "min at a, max at b"
        for &(a, b) in &self.comparators {
            self.apply_soft_comparator(&mut values, &mut perm, a, b);
        }

        // Trim padding
        let out_n = x.len();
        let sorted = values[..out_n].to_vec();
        let trimmed_perm: Vec<Vec<f64>> = perm[..out_n]
            .iter()
            .map(|row| row[..out_n].to_vec())
            .collect();

        Ok((sorted, trimmed_perm))
    }

    /// Apply a single soft comparator to positions a, b.
    ///
    /// Semantics: position `a` should get the min, position `b` the max.
    ///
    /// sigma = sigmoid((x_a - x_b) * steepness):
    /// - sigma ~ 1 when x_a > x_b (swap needed)
    /// - sigma ~ 0 when x_a < x_b (no swap)
    fn apply_soft_comparator(&self, values: &mut [f64], perm: &mut [Vec<f64>], a: usize, b: usize) {
        let diff = (values[a] - values[b]) * self.steepness;
        let sigma = relaxed_sigmoid(diff, self.dist);

        // Interpolate values
        let va = values[a];
        let vb = values[b];
        values[a] = (1.0 - sigma) * va + sigma * vb; // tends toward min
        values[b] = sigma * va + (1.0 - sigma) * vb; // tends toward max

        // Update permutation rows
        let n = perm[0].len();
        #[allow(clippy::needless_range_loop)] // two mutable rows from the same Vec
        for k in 0..n {
            let pa = perm[a][k];
            let pb = perm[b][k];
            perm[a][k] = (1.0 - sigma) * pa + sigma * pb;
            perm[b][k] = sigma * pa + (1.0 - sigma) * pb;
        }
    }

    /// Return the number of comparators in this network.
    pub fn num_comparators(&self) -> usize {
        self.comparators.len()
    }

    /// Return the comparator pairs as a slice.
    ///
    /// Each pair (a, b) means "min at a, max at b".
    pub fn comparator_pairs(&self) -> &[(usize, usize)] {
        &self.comparators
    }

    /// Return the network depth (number of parallel stages).
    pub fn depth(&self) -> usize {
        if self.comparators.is_empty() {
            return 0;
        }
        // Count stages: a new stage starts when a comparator reuses
        // an index from the current stage.
        let mut depth = 1;
        let mut used_in_stage = vec![false; self.size];
        for &(i, j) in &self.comparators {
            if used_in_stage[i] || used_in_stage[j] {
                depth += 1;
                used_in_stage.fill(false);
            }
            used_in_stage[i] = true;
            used_in_stage[j] = true;
        }
        depth
    }
}

/// Relaxed sigmoid for the comparator.
pub(crate) fn relaxed_sigmoid(x: f64, dist: RelaxDist) -> f64 {
    match dist {
        RelaxDist::Logistic => {
            // Standard logistic sigmoid with numerical stability
            if x > 500.0 {
                1.0
            } else if x < -500.0 {
                0.0
            } else {
                1.0 / (1.0 + (-x).exp())
            }
        }
        RelaxDist::Cauchy => {
            // Cauchy CDF: 0.5 + arctan(x) / pi
            0.5 + x.atan() / std::f64::consts::PI
        }
        RelaxDist::Gaussian => {
            // Gaussian CDF via erf
            0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
        }
    }
}

/// Error function approximation.
pub(crate) fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let poly = 0.254_829_592 * t - 0.284_496_736 * t2 + 1.421_413_741 * t3 - 1.453_152_027 * t4
        + 1.061_405_429 * t5;
    sign * (1.0 - poly * (-x * x).exp())
}

/// Generate comparator pairs for a bitonic sorting network.
///
/// Batcher's bitonic merge sort for n elements (n must be power of 2).
fn bitonic_comparators(n: usize) -> Vec<(usize, usize)> {
    debug_assert!(n.is_power_of_two());
    let mut comps = Vec::new();

    // Build bottom-up: size 2, 4, 8, ..., n
    let mut k = 2;
    while k <= n {
        // Bitonic merge with alternating directions
        let mut j = k >> 1;
        while j > 0 {
            for i in 0..n {
                let l = i ^ j;
                if l > i {
                    // Direction: ascending if (i & k) == 0
                    if (i & k) == 0 {
                        comps.push((i, l));
                    } else {
                        comps.push((l, i));
                    }
                }
            }
            j >>= 1;
        }
        k <<= 1;
    }

    comps
}

/// Generate comparator pairs for an odd-even transposition sort.
fn odd_even_comparators(n: usize) -> Vec<(usize, usize)> {
    let mut comps = Vec::new();

    for _ in 0..n {
        // Even phase: compare (0,1), (2,3), (4,5), ...
        let mut i = 0;
        while i + 1 < n {
            comps.push((i, i + 1));
            i += 2;
        }
        // Odd phase: compare (1,2), (3,4), (5,6), ...
        let mut i = 1;
        while i + 1 < n {
            comps.push((i, i + 1));
            i += 2;
        }
    }

    comps
}

/// Convenience: sort with a bitonic network using logistic relaxation.
///
/// Pads to next power-of-2 internally.
///
/// # Example
///
/// ```rust
/// use fynch::sorting_network::bitonic_sort;
///
/// let x = vec![3.0, 1.0, 4.0, 2.0];
/// let (sorted, perm) = bitonic_sort(&x, 10.0).unwrap();
/// // sorted ≈ [1.0, 2.0, 3.0, 4.0]
/// ```
pub fn bitonic_sort(x: &[f64], steepness: f64) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let net = DiffSortNet::new(
        NetworkType::Bitonic,
        x.len(),
        steepness,
        RelaxDist::Logistic,
    );
    net.sort(x)
}

/// Convenience: sort with an odd-even network using logistic relaxation.
///
/// # Example
///
/// ```rust
/// use fynch::sorting_network::odd_even_sort;
///
/// let x = vec![3.0, 1.0, 4.0, 2.0];
/// let (sorted, perm) = odd_even_sort(&x, 10.0).unwrap();
/// // sorted ≈ [1.0, 2.0, 3.0, 4.0]
/// ```
pub fn odd_even_sort(x: &[f64], steepness: f64) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let net = DiffSortNet::new(
        NetworkType::OddEven,
        x.len(),
        steepness,
        RelaxDist::Logistic,
    );
    net.sort(x)
}

/// Extract soft ranks from a permutation matrix.
///
/// Given P where `sorted = x @ P`, the rank of element i is the weighted
/// column index: `rank_i = sum_j j * P[i][j]`.
pub fn ranks_from_permutation(perm: &[Vec<f64>]) -> Vec<f64> {
    perm.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &p)| (j as f64 + 1.0) * p)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitonic_sort_basic() {
        let x = vec![3.0, 1.0, 4.0, 2.0];
        let (sorted, perm) = bitonic_sort(&x, 20.0).unwrap();

        // Check approximately sorted
        assert!(sorted[0] < sorted[1] + 0.1);
        assert!(sorted[1] < sorted[2] + 0.1);
        assert!(sorted[2] < sorted[3] + 0.1);

        // Check permutation matrix is approximately doubly stochastic
        for row in &perm {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Row sum should be ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_odd_even_sort_basic() {
        let x = vec![3.0, 1.0, 4.0, 2.0];
        let (sorted, perm) = odd_even_sort(&x, 20.0).unwrap();

        assert!(sorted[0] < sorted[1] + 0.1);
        assert!(sorted[1] < sorted[2] + 0.1);
        assert!(sorted[2] < sorted[3] + 0.1);

        for row in &perm {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Row sum should be ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_already_sorted() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (sorted, perm) = bitonic_sort(&x, 20.0).unwrap();

        for i in 0..4 {
            assert!(
                (sorted[i] - x[i]).abs() < 0.01,
                "Already sorted input should remain sorted"
            );
            // Permutation should be approximately identity
            assert!(perm[i][i] > 0.9, "Should be near-identity permutation");
        }
    }

    #[test]
    fn test_reverse_sorted() {
        let x = vec![4.0, 3.0, 2.0, 1.0];
        let (sorted, _perm) = bitonic_sort(&x, 20.0).unwrap();

        assert!(sorted[0] < sorted[3]);
    }

    #[test]
    fn test_cauchy_distribution() {
        let x = vec![3.0, 1.0, 4.0, 2.0];
        let net = DiffSortNet::new(NetworkType::Bitonic, 4, 10.0, RelaxDist::Cauchy);
        let (sorted, _) = net.sort(&x).unwrap();

        assert!(sorted[0] < sorted[1] + 0.2);
        assert!(sorted[2] < sorted[3] + 0.2);
    }

    #[test]
    fn test_gaussian_distribution() {
        let x = vec![3.0, 1.0, 4.0, 2.0];
        let net = DiffSortNet::new(NetworkType::Bitonic, 4, 10.0, RelaxDist::Gaussian);
        let (sorted, _) = net.sort(&x).unwrap();

        assert!(sorted[0] < sorted[1] + 0.2);
        assert!(sorted[2] < sorted[3] + 0.2);
    }

    #[test]
    fn test_non_power_of_two() {
        // Bitonic pads to next power of 2
        let x = vec![3.0, 1.0, 5.0];
        let net = DiffSortNet::new(NetworkType::Bitonic, 3, 20.0, RelaxDist::Logistic);
        let (sorted, perm) = net.sort(&x).unwrap();

        assert_eq!(sorted.len(), 3);
        assert_eq!(perm.len(), 3);
        assert!(sorted[0] < sorted[2]);
    }

    #[test]
    fn test_odd_even_non_power_of_two() {
        let x = vec![5.0, 2.0, 7.0, 1.0, 3.0];
        let (sorted, _) = odd_even_sort(&x, 20.0).unwrap();

        assert_eq!(sorted.len(), 5);
        for i in 1..sorted.len() {
            assert!(
                sorted[i] >= sorted[i - 1] - 0.2,
                "Not approximately sorted at {i}"
            );
        }
    }

    #[test]
    fn test_ranks_from_permutation() {
        let x = vec![3.0, 1.0, 4.0, 2.0];
        let (sorted, perm) = bitonic_sort(&x, 50.0).unwrap();
        let ranks = ranks_from_permutation(&perm);

        // Verify sorted output is approximately correct
        assert!(sorted[0] < sorted[3], "min < max: {sorted:?}");

        // All ranks should be in [1, n] range
        for &r in &ranks {
            assert!((0.5..=4.5).contains(&r), "Rank out of range: {r}");
        }
    }

    #[test]
    fn test_steepness_sharpness() {
        let x = vec![3.0, 1.0, 4.0, 2.0];

        // Low steepness: more mixing
        let (_, perm_low) = bitonic_sort(&x, 1.0).unwrap();
        // High steepness: near-discrete
        let (_, perm_high) = bitonic_sort(&x, 100.0).unwrap();

        // High steepness should have entries closer to 0/1
        let max_low: f64 = perm_low
            .iter()
            .flat_map(|r| r.iter())
            .cloned()
            .fold(0.0, f64::max);
        let max_high: f64 = perm_high
            .iter()
            .flat_map(|r| r.iter())
            .cloned()
            .fold(0.0, f64::max);

        assert!(
            max_high > max_low,
            "Higher steepness should produce sharper permutations"
        );
    }

    #[test]
    fn test_empty_input() {
        assert!(bitonic_sort(&[], 10.0).is_err());
    }

    #[test]
    fn test_single_element() {
        let x = vec![42.0];
        let (sorted, perm) = odd_even_sort(&x, 10.0).unwrap();
        assert_eq!(sorted.len(), 1);
        assert!((sorted[0] - 42.0).abs() < 1e-10);
        assert!((perm[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_num_comparators() {
        let net4 = DiffSortNet::new(NetworkType::Bitonic, 4, 1.0, RelaxDist::Logistic);
        assert!(net4.num_comparators() > 0);

        let net8 = DiffSortNet::new(NetworkType::Bitonic, 8, 1.0, RelaxDist::Logistic);
        assert!(net8.num_comparators() > net4.num_comparators());
    }

    #[test]
    fn test_column_sums_doubly_stochastic() {
        let x = vec![3.0, 1.0, 4.0, 2.0];
        let (_, perm) = bitonic_sort(&x, 20.0).unwrap();

        let n = perm.len();
        #[allow(clippy::needless_range_loop)] // j indexes columns across rows
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| perm[i][j]).sum();
            assert!(
                (col_sum - 1.0).abs() < 0.05,
                "Column {j} sum should be ~1.0, got {col_sum}"
            );
        }
    }
}
