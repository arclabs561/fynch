//! Property tests for sorting network module.

use fynch::sorting_network::{
    bitonic_sort, odd_even_sort, ranks_from_permutation, DiffSortNet, NetworkType, RelaxDist,
};
use proptest::prelude::*;

proptest! {
    /// Bitonic sort output should be approximately sorted (monotone within tolerance).
    #[test]
    fn prop_bitonic_approximately_sorted(
        len in 2usize..=16,
        steepness in 5.0f64..=50.0,
    ) {
        // Generate values from index to ensure variety
        let x: Vec<f64> = (0..len).map(|i| ((i * 7 + 3) % len) as f64).collect();
        let _padded_len = len.next_power_of_two();
        let net = DiffSortNet::new(NetworkType::Bitonic, len, steepness, RelaxDist::Logistic);
        let (sorted, _) = net.sort(&x).unwrap();

        // Check approximate monotonicity
        for i in 1..sorted.len() {
            prop_assert!(
                sorted[i] >= sorted[i - 1] - 0.5,
                "Not approximately sorted at {i}: {sorted:?}"
            );
        }
    }

    /// Permutation matrix rows should sum to ~1.0 (doubly stochastic).
    #[test]
    fn prop_permutation_row_sums(
        len in 2usize..=8,
    ) {
        let x: Vec<f64> = (0..len).map(|i| i as f64).rev().collect();
        let (_, perm) = odd_even_sort(&x, 20.0).unwrap();

        for (i, row) in perm.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 0.1,
                "Row {i} sum = {sum}, expected ~1.0"
            );
        }
    }

    /// Permutation matrix columns should sum to ~1.0.
    #[test]
    fn prop_permutation_col_sums(
        len in 2usize..=8,
    ) {
        let x: Vec<f64> = (0..len).map(|i| (i as f64 * 1.5).sin()).collect();
        let (_, perm) = bitonic_sort(&x, 20.0).unwrap();

        let n = perm.len();
        #[allow(clippy::needless_range_loop)] // j indexes columns across rows
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| perm[i][j]).sum();
            prop_assert!(
                (col_sum - 1.0).abs() < 0.1,
                "Column {j} sum = {col_sum}, expected ~1.0"
            );
        }
    }

    /// Higher steepness should produce sharper (more discrete) permutations.
    #[test]
    fn prop_steepness_sharpness(
        len in 2usize..=8,
    ) {
        let x: Vec<f64> = (0..len).map(|i| (len - i) as f64).collect();

        let (_, perm_low) = bitonic_sort(&x, 1.0).unwrap();
        let (_, perm_high) = bitonic_sort(&x, 50.0).unwrap();

        // High steepness: max entry should be closer to 1.0
        let max_low: f64 = perm_low.iter().flat_map(|r| r.iter()).cloned().fold(0.0, f64::max);
        let max_high: f64 = perm_high.iter().flat_map(|r| r.iter()).cloned().fold(0.0, f64::max);

        prop_assert!(max_high >= max_low - 0.01,
            "Higher steepness should be sharper: max_high={max_high}, max_low={max_low}");
    }

    /// Ranks extracted from permutation matrix should be in valid range.
    #[test]
    fn prop_ranks_in_range(
        len in 2usize..=8,
    ) {
        let x: Vec<f64> = (0..len).map(|i| (i as f64 * 0.7).cos()).collect();
        let (_, perm) = bitonic_sort(&x, 20.0).unwrap();
        let ranks = ranks_from_permutation(&perm);

        for (i, &r) in ranks.iter().enumerate() {
            prop_assert!(
                r >= 0.5 && r <= (len as f64 + 0.5),
                "Rank {i} = {r} out of range [0.5, {}]", len as f64 + 0.5
            );
        }
    }
}
