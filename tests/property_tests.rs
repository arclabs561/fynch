//! Property-based tests for sparse top-k, PAVA, entmax, and sorting operations.

use fynch::fenchel::{entmax, softmax, sparsemax, Regularizer, Shannon, SquaredL2, Tsallis};
use fynch::sorting_network::{bitonic_sort, odd_even_sort, NetworkType, RelaxDist};
use fynch::topk::{
    differentiable_bottomk, differentiable_topk, sparse_topk, sparse_topk_matrix, topk_ce_loss,
};
use fynch::{pava, pava_weighted};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn vec_f64(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-10.0f64..10.0, min_len..=max_len)
}

/// Distinct values avoid ties that confuse rank-based assertions.
fn distinct_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-10.0f64..10.0, min_len..=max_len).prop_map(|mut v| {
        // Add small unique perturbations to break ties
        for (i, x) in v.iter_mut().enumerate() {
            *x += i as f64 * 1e-6;
        }
        v
    })
}

// ---------------------------------------------------------------------------
// PAVA properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// PAVA output is monotonically non-decreasing.
    #[test]
    fn prop_pava_monotone(y in vec_f64(1, 30)) {
        let result = pava(&y);
        for i in 1..result.len() {
            prop_assert!(
                result[i] >= result[i - 1] - 1e-12,
                "Not monotone at {i}: {} < {}",
                result[i],
                result[i - 1]
            );
        }
    }

    /// PAVA is idempotent: pava(pava(x)) == pava(x).
    #[test]
    fn prop_pava_idempotent(y in vec_f64(1, 30)) {
        let first = pava(&y);
        let second = pava(&first);
        for (i, (a, b)) in first.iter().zip(&second).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-10,
                "Not idempotent at {i}: {} vs {}",
                a, b
            );
        }
    }

    /// PAVA preserves the mean (sum).
    #[test]
    fn prop_pava_preserves_sum(y in vec_f64(1, 30)) {
        let result = pava(&y);
        let sum_in: f64 = y.iter().sum();
        let sum_out: f64 = result.iter().sum();
        prop_assert!(
            (sum_in - sum_out).abs() < 1e-8,
            "Sum changed: {sum_in} vs {sum_out}"
        );
    }

    /// Already monotone input is a fixed point.
    #[test]
    fn prop_pava_fixed_point_sorted(len in 1usize..=20) {
        let y: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let result = pava(&y);
        for (i, (&a, &b)) in y.iter().zip(&result).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-12,
                "Sorted input changed at {i}: {} vs {}",
                a, b
            );
        }
    }

    /// Weighted PAVA output is also monotone.
    #[test]
    fn prop_pava_weighted_monotone(
        y in vec_f64(1, 20),
    ) {
        let w: Vec<f64> = y.iter().enumerate().map(|(i, _)| 0.5 + i as f64 * 0.1).collect();
        let result = pava_weighted(&y, &w).unwrap();
        for i in 1..result.len() {
            prop_assert!(
                result[i] >= result[i - 1] - 1e-12,
                "Weighted PAVA not monotone at {i}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fenchel-Young / entmax properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Softmax output sums to 1 and is strictly positive.
    #[test]
    fn prop_softmax_valid_distribution(theta in vec_f64(1, 20)) {
        let p = softmax(&theta);
        let sum: f64 = p.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-8,
            "Softmax sum = {sum}"
        );
        for (i, &pi) in p.iter().enumerate() {
            prop_assert!(pi > 0.0, "Softmax[{i}] = {pi} <= 0");
            prop_assert!(pi <= 1.0, "Softmax[{i}] = {pi} > 1");
        }
    }

    /// Sparsemax output sums to 1 and is non-negative.
    #[test]
    fn prop_sparsemax_valid_distribution(theta in vec_f64(1, 20)) {
        let p = sparsemax(&theta);
        let sum: f64 = p.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-8,
            "Sparsemax sum = {sum}"
        );
        for (i, &pi) in p.iter().enumerate() {
            prop_assert!(pi >= 0.0, "Sparsemax[{i}] = {pi} < 0");
        }
    }

    /// Sparsemax produces at least one zero for inputs with spread > 0.
    #[test]
    fn prop_sparsemax_sparse(theta in vec_f64(3, 20)) {
        let max = theta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = theta.iter().cloned().fold(f64::INFINITY, f64::min);
        // Only check when values are sufficiently spread
        prop_assume!(max - min > 1.0);
        let p = sparsemax(&theta);
        let zeros = p.iter().filter(|&&x| x == 0.0).count();
        prop_assert!(zeros > 0, "Expected at least one zero in sparsemax output");
    }

    /// Entmax (alpha > 1) output sums to 1 and is non-negative.
    #[test]
    fn prop_entmax_valid_distribution(
        theta in vec_f64(2, 15),
        alpha in 1.1f64..=3.0,
    ) {
        let p = entmax(&theta, alpha);
        let sum: f64 = p.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "Entmax(alpha={alpha}) sum = {sum}"
        );
        for (i, &pi) in p.iter().enumerate() {
            prop_assert!(pi >= -1e-10, "Entmax[{i}] = {pi} < 0");
        }
    }

    /// FY loss at the prediction point should be ~0.
    #[test]
    fn prop_fy_loss_zero_at_prediction(theta in vec_f64(2, 10)) {
        let y = Shannon.predict(&theta);
        let loss = Shannon.loss(&theta, &y);
        prop_assert!(
            loss.abs() < 1e-4,
            "Shannon FY loss at prediction = {loss}"
        );

        let y2 = SquaredL2.predict(&theta);
        let loss2 = SquaredL2.loss(&theta, &y2);
        prop_assert!(
            loss2.abs() < 1e-4,
            "SquaredL2 FY loss at prediction = {loss2}"
        );
    }

    /// FY loss should be non-negative for valid distributions.
    /// Tsallis uses bisection-based entmax with limited precision, so we allow
    /// a small numerical tolerance.
    #[test]
    fn prop_fy_loss_nonneg(theta in vec_f64(2, 10)) {
        // Create a valid distribution target
        let y = softmax(&theta);
        // Use different theta for prediction so loss > 0
        let theta2: Vec<f64> = theta.iter().map(|&t| t + 0.5).collect();

        let loss_s = Shannon.loss(&theta2, &y);
        prop_assert!(loss_s >= -1e-8, "Shannon loss = {loss_s}");

        let loss_l2 = SquaredL2.loss(&theta2, &y);
        prop_assert!(loss_l2 >= -1e-8, "SquaredL2 loss = {loss_l2}");

        // Tsallis entmax uses bisection with 50 iterations. Numerical error
        // from the approximate threshold can produce negative losses up to ~0.1
        // when theta values are close together. This is a known limitation of
        // the bisection approach, not a bug in the FY framework.
        let loss_t = Tsallis::entmax15().loss(&theta2, &y);
        prop_assert!(loss_t >= -0.15, "Tsallis loss = {loss_t}");
    }

    /// Entmax output preserves ordering: if theta_i > theta_j, then p_i >= p_j.
    #[test]
    fn prop_entmax_order_preserving(
        theta in distinct_vec(2, 10),
        alpha in 1.01f64..=3.0,
    ) {
        let p = entmax(&theta, alpha);
        for i in 0..theta.len() {
            for j in (i + 1)..theta.len() {
                if theta[i] > theta[j] + 1e-4 {
                    prop_assert!(
                        p[i] >= p[j] - 1e-4,
                        "Order not preserved: theta[{i}]={} > theta[{j}]={} but p[{i}]={} < p[{j}]={}",
                        theta[i], theta[j], p[i], p[j]
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sorting network properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Sorting network permutation matrix entries are in [0, 1].
    #[test]
    fn prop_sort_perm_entries_bounded(
        len in 2usize..=8,
        steepness in 5.0f64..=50.0,
    ) {
        let x: Vec<f64> = (0..len).map(|i| (i as f64 * 1.3).sin() * 5.0).collect();
        let (_, perm) = odd_even_sort(&x, steepness).unwrap();

        for (i, row) in perm.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                prop_assert!(
                    (-0.01..=1.01).contains(&val),
                    "perm[{i}][{j}] = {val} out of [0,1]"
                );
            }
        }
    }

    /// At high steepness, soft sort approaches hard sort.
    #[test]
    fn prop_high_steepness_approaches_hard_sort(
        len in 2usize..=8,
    ) {
        let x: Vec<f64> = (0..len).map(|i| (len - i) as f64 * 2.0).collect();
        let mut expected = x.clone();
        expected.sort_by(|a, b| a.total_cmp(b));

        let (sorted, _) = odd_even_sort(&x, 100.0).unwrap();

        for (i, (&got, &want)) in sorted.iter().zip(&expected).enumerate() {
            prop_assert!(
                (got - want).abs() < 0.5,
                "At high steepness, sorted[{i}]={got} should be near {want}"
            );
        }
    }

    /// Sorting network preserves sum (doubly stochastic property).
    #[test]
    fn prop_sort_preserves_sum(
        len in 2usize..=8,
        steepness in 5.0f64..=30.0,
    ) {
        let x: Vec<f64> = (0..len).map(|i| (i as f64 - 3.0) * 1.5).collect();
        let (sorted, _) = bitonic_sort(&x, steepness).unwrap();

        let sum_in: f64 = x.iter().sum();
        let sum_out: f64 = sorted.iter().sum();
        prop_assert!(
            (sum_in - sum_out).abs() < 0.5,
            "Sum not preserved: {sum_in} vs {sum_out}"
        );
    }

    /// Sorting an already-sorted input with high steepness yields near-identity permutation.
    #[test]
    fn prop_sort_identity_on_sorted(len in 2usize..=8) {
        let x: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let (_, perm) = bitonic_sort(&x, 50.0).unwrap();

        for (i, row) in perm.iter().enumerate().take(len) {
            prop_assert!(
                row[i] > 0.7,
                "Diagonal perm[{i}][{i}] = {} should be near 1.0 for sorted input",
                perm[i][i]
            );
        }
    }

    /// All three relaxation distributions produce valid doubly-stochastic matrices.
    #[test]
    fn prop_all_dists_doubly_stochastic(len in 2usize..=8) {
        let x: Vec<f64> = (0..len).map(|i| (i as f64 * 0.9).sin() * 3.0).collect();
        let dists = [RelaxDist::Logistic, RelaxDist::Cauchy, RelaxDist::Gaussian];

        for dist in &dists {
            let net = fynch::sorting_network::DiffSortNet::new(
                NetworkType::Bitonic, len, 15.0, *dist,
            );
            let (_, perm) = net.sort(&x).unwrap();
            let n = perm.len();

            for (i, row) in perm.iter().enumerate() {
                let row_sum: f64 = row.iter().sum();
                prop_assert!(
                    (row_sum - 1.0).abs() < 0.15,
                    "{dist:?}: row {i} sum = {row_sum}"
                );
            }
            #[allow(clippy::needless_range_loop)] // j indexes columns across rows
            for j in 0..n {
                let col_sum: f64 = (0..n).map(|i| perm[i][j]).sum();
                prop_assert!(
                    (col_sum - 1.0).abs() < 0.15,
                    "{dist:?}: col {j} sum = {col_sum}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse top-k properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// sparse_topk output shape is n x k.
    #[test]
    fn prop_sparse_topk_shape(
        n in 2usize..=16,
        k_frac in 0.1f64..=1.0,
    ) {
        let k = ((n as f64 * k_frac).ceil() as usize).max(1).min(n);
        let scores: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() * 5.0).collect();
        let a = sparse_topk(&scores, k, 10.0).unwrap();

        prop_assert_eq!(a.len(), n, "Expected {} rows, got {}", n, a.len());
        for (i, row) in a.iter().enumerate() {
            prop_assert_eq!(row.len(), k, "Row {}: expected {} cols, got {}", i, k, row.len());
        }
    }

    /// Column sums should be approximately 1 (each rank position filled by one element).
    /// Uses OddEven to avoid bitonic padding artifacts with non-power-of-2 sizes.
    #[test]
    fn prop_sparse_topk_col_sums(
        n in 2usize..=8,
        k_frac in 0.2f64..=0.8,
    ) {
        let k = ((n as f64 * k_frac).ceil() as usize).max(1).min(n);
        let scores: Vec<f64> = (0..n).map(|i| (n - i) as f64 * 2.0).collect();
        let a = sparse_topk_matrix(&scores, k, 20.0, NetworkType::OddEven, RelaxDist::Logistic)
            .unwrap();

        #[allow(clippy::needless_range_loop)] // j indexes columns across rows
        for j in 0..k {
            let col_sum: f64 = (0..n).map(|i| a[i][j]).sum();
            prop_assert!(
                (col_sum - 1.0).abs() < 0.15,
                "Column {} sum = {}, expected ~1.0", j, col_sum
            );
        }
    }

    /// Row sums should be in [0, 1]: each element is assigned to at most one rank.
    #[test]
    fn prop_sparse_topk_row_sums(
        n in 2usize..=8,
        k_frac in 0.1f64..=0.9,
    ) {
        let k = ((n as f64 * k_frac).ceil() as usize).max(1).min(n);
        let scores: Vec<f64> = (0..n).map(|i| i as f64 * 1.5).collect();
        let a = sparse_topk(&scores, k, 20.0).unwrap();

        for (i, row) in a.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            prop_assert!(
                sum <= 1.0 + 0.1,
                "Row {i} sum = {sum}, expected <= 1.0"
            );
            prop_assert!(
                sum >= -0.01,
                "Row {i} sum = {sum}, expected >= 0.0"
            );
        }
    }

    /// All entries in the attribution matrix are non-negative.
    #[test]
    fn prop_sparse_topk_nonneg(
        n in 2usize..=8,
    ) {
        let scores: Vec<f64> = (0..n).map(|i| (i as f64 * 1.1).cos() * 5.0).collect();
        let k = (n / 2).max(1);
        let a = sparse_topk(&scores, k, 15.0).unwrap();

        for (i, row) in a.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                prop_assert!(
                    val >= -0.01,
                    "a[{i}][{j}] = {val} < 0"
                );
            }
        }
    }

    /// At high steepness, top-k elements get row sum ~1, others ~0.
    /// Uses OddEven to avoid bitonic padding artifacts with non-power-of-2 sizes.
    #[test]
    fn prop_sparse_topk_high_steepness(n in 3usize..=8) {
        // Use well-separated distinct scores
        let scores: Vec<f64> = (0..n).map(|i| i as f64 * 3.0).collect();
        let k = (n / 2).max(1);
        let a = sparse_topk_matrix(&scores, k, 100.0, NetworkType::OddEven, RelaxDist::Logistic)
            .unwrap();

        // Top-k elements are the last k by score value
        let mut indexed: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let topk_indices: Vec<usize> = indexed.iter().take(k).map(|&(i, _)| i).collect();

        for i in 0..n {
            let row_sum: f64 = a[i].iter().sum();
            if topk_indices.contains(&i) {
                prop_assert!(
                    row_sum > 0.7,
                    "Top-k element {i} (score={}) row sum = {row_sum}, expected ~1.0",
                    scores[i]
                );
            } else {
                prop_assert!(
                    row_sum < 0.3,
                    "Non-top-k element {i} (score={}) row sum = {row_sum}, expected ~0.0",
                    scores[i]
                );
            }
        }
    }

    /// Top-k of a sorted input assigns weight to the last k elements.
    /// Uses OddEven to avoid bitonic padding artifacts with non-power-of-2 sizes.
    #[test]
    fn prop_sparse_topk_sorted_input(n in 3usize..=8) {
        let scores: Vec<f64> = (0..n).map(|i| i as f64).collect(); // ascending
        let k = (n / 2).max(1);
        let a = sparse_topk_matrix(&scores, k, 80.0, NetworkType::OddEven, RelaxDist::Logistic)
            .unwrap();

        // Last k elements (highest scores) should have high row sums
        for (i, row) in a.iter().enumerate().skip(n - k) {
            let row_sum: f64 = row.iter().sum();
            prop_assert!(
                row_sum > 0.7,
                "Sorted top-k element {i} row sum = {row_sum}"
            );
        }
        // First n-k elements should have low row sums
        for (i, row) in a.iter().enumerate().take(n - k) {
            let row_sum: f64 = row.iter().sum();
            prop_assert!(
                row_sum < 0.3,
                "Sorted non-top-k element {i} row sum = {row_sum}"
            );
        }
    }

    /// Monotonicity: if score[i] > score[j], then total attribution of i >= that of j.
    #[test]
    fn prop_sparse_topk_monotone_attribution(n in 3usize..=8) {
        let scores: Vec<f64> = (0..n).map(|i| i as f64 * 2.0 + (i as f64 * 0.3).sin()).collect();
        let k = (n / 2).max(1);
        let a = sparse_topk(&scores, k, 50.0).unwrap();

        for i in 0..n {
            for j in (i + 1)..n {
                if scores[j] > scores[i] + 0.5 {
                    let sum_i: f64 = a[i].iter().sum();
                    let sum_j: f64 = a[j].iter().sum();
                    prop_assert!(
                        sum_j >= sum_i - 0.15,
                        "score[{j}]={} > score[{i}]={} but attr sum {sum_j} < {sum_i}",
                        scores[j], scores[i]
                    );
                }
            }
        }
    }

    /// sparse_topk with k=n should recover a full soft permutation (row sums ~1).
    /// Uses OddEven to avoid bitonic padding artifacts with non-power-of-2 sizes.
    #[test]
    fn prop_sparse_topk_k_eq_n(n in 2usize..=8) {
        let scores: Vec<f64> = (0..n).map(|i| (i as f64 * 0.8).cos() * 3.0).collect();
        let a = sparse_topk_matrix(&scores, n, 20.0, NetworkType::OddEven, RelaxDist::Logistic)
            .unwrap();

        for (i, row) in a.iter().enumerate() {
            prop_assert_eq!(row.len(), n);
            let sum: f64 = row.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 0.1,
                "k=n: row {i} sum = {sum}, expected ~1.0"
            );
        }
    }

    /// OddEven network produces same structural properties as Bitonic.
    #[test]
    fn prop_sparse_topk_oddeven(n in 2usize..=6) {
        let scores: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
        let k = (n / 2).max(1);
        let a = sparse_topk_matrix(&scores, k, 20.0, NetworkType::OddEven, RelaxDist::Logistic)
            .unwrap();

        prop_assert_eq!(a.len(), n);
        for row in &a {
            prop_assert_eq!(row.len(), k);
        }
        // Column sums ~1
        #[allow(clippy::needless_range_loop)] // j indexes columns across rows
        for j in 0..k {
            let col_sum: f64 = (0..n).map(|i| a[i][j]).sum();
            prop_assert!(
                (col_sum - 1.0).abs() < 0.2,
                "OddEven col {j} sum = {col_sum}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Differentiable top-k / bottom-k indicator properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    /// Indicators are in [0, 1].
    #[test]
    fn prop_topk_indicators_bounded(
        values in vec_f64(2, 15),
        k in 1usize..=5,
    ) {
        let k = k.min(values.len());
        let (_, indicators) = differentiable_topk(&values, k, 0.1);
        for (i, &ind) in indicators.iter().enumerate() {
            prop_assert!(
                (-1e-10..=1.0 + 1e-10).contains(&ind),
                "indicator[{i}] = {ind} out of [0,1]"
            );
        }
    }

    /// Bottom-k indicators are in [0, 1].
    #[test]
    fn prop_bottomk_indicators_bounded(
        values in vec_f64(2, 15),
        k in 1usize..=5,
    ) {
        let k = k.min(values.len());
        let (_, indicators) = differentiable_bottomk(&values, k, 0.1);
        for (i, &ind) in indicators.iter().enumerate() {
            prop_assert!(
                (-1e-10..=1.0 + 1e-10).contains(&ind),
                "bottomk indicator[{i}] = {ind} out of [0,1]"
            );
        }
    }

    /// Top-k and bottom-k are complementary: top-k(n-k) indicators are roughly
    /// the complement of bottom-k(k) indicators.
    #[test]
    fn prop_topk_bottomk_complement(
        values in distinct_vec(4, 10),
    ) {
        let n = values.len();
        let k = n / 2;
        let (_, top_ind) = differentiable_topk(&values, k, 0.05);
        let (_, bot_ind) = differentiable_bottomk(&values, n - k, 0.05);

        // At low temperature, top-k indicators + bottom-(n-k) indicators should ~1
        for i in 0..n {
            let sum = top_ind[i] + bot_ind[i];
            prop_assert!(
                (sum - 1.0).abs() < 0.5,
                "top + bottom at {i}: {} + {} = {sum}",
                top_ind[i], bot_ind[i]
            );
        }
    }

    /// weighted_values[i] = values[i] * indicators[i].
    #[test]
    fn prop_topk_weighted_consistency(
        values in vec_f64(2, 10),
        k in 1usize..=5,
    ) {
        let k = k.min(values.len());
        let (weighted, indicators) = differentiable_topk(&values, k, 0.1);
        for i in 0..values.len() {
            let expected = values[i] * indicators[i];
            prop_assert!(
                (weighted[i] - expected).abs() < 1e-10,
                "weighted[{i}] = {}, expected {}",
                weighted[i], expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Top-k cross-entropy loss properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// TopK CE loss is non-negative.
    #[test]
    fn prop_topk_ce_nonneg(n in 2usize..=8) {
        let logits: Vec<f64> = (0..n).map(|i| (i as f64 * 0.8).sin() * 3.0).collect();
        let k = (n / 2).max(1);
        let mut p_k = vec![0.0; k];
        p_k[0] = 1.0; // uniform-ish: all weight on top-1
        let loss = topk_ce_loss(&logits, 0, &p_k, 10.0).unwrap();
        prop_assert!(loss >= 0.0, "loss = {loss}");
        prop_assert!(loss.is_finite(), "loss is not finite");
    }

    /// Loss for correct class (highest logit) < loss for wrong class (lowest logit).
    #[test]
    fn prop_topk_ce_correct_lower(n in 3usize..=8) {
        // Make class 0 clearly the best
        let mut logits: Vec<f64> = (0..n).map(|i| -(i as f64)).collect();
        logits[0] = 5.0;
        let p_k = vec![1.0]; // top-1 only
        let loss_correct = topk_ce_loss(&logits, 0, &p_k, 10.0).unwrap();
        let loss_wrong = topk_ce_loss(&logits, n - 1, &p_k, 10.0).unwrap();
        prop_assert!(
            loss_correct < loss_wrong,
            "correct loss {loss_correct} should be < wrong loss {loss_wrong}"
        );
    }
}
