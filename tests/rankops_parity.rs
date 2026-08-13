#![allow(deprecated)]

use proptest::prelude::*;

fn assert_same(left: f64, right: f64) {
    if left.is_nan() || right.is_nan() {
        assert!(left.is_nan() && right.is_nan());
    } else {
        assert_eq!(left.to_bits(), right.to_bits());
    }
}

proptest! {
    #[test]
    fn legacy_rank_metrics_match_rankops(
        ranks in prop::collection::vec(0usize..128, 0..64),
        k in 0usize..128,
        n_relevant in 0usize..128,
        beta in 0.01f64..8.0,
        persistence in -0.25f64..1.25,
    ) {
        assert_same(fynch::metrics::mrr(&ranks), rankops::metrics::mrr(&ranks));
        assert_same(
            fynch::metrics::hits_at_k(&ranks, k),
            rankops::metrics::hits_at_k(&ranks, k),
        );
        assert_same(
            fynch::metrics::mean_rank(&ranks),
            rankops::metrics::mean_rank(&ranks),
        );
        assert_same(
            fynch::metrics::precision_at_k(&ranks, k),
            rankops::metrics::precision_at_k(&ranks, k),
        );
        assert_same(
            fynch::metrics::recall_at_k(&ranks, n_relevant, k),
            rankops::metrics::recall_at_k(&ranks, n_relevant, k),
        );
        assert_same(
            fynch::metrics::average_precision(&ranks, n_relevant),
            rankops::metrics::average_precision(&ranks, n_relevant),
        );
        assert_same(
            fynch::metrics::f_measure_at_k(&ranks, n_relevant, k, beta),
            rankops::metrics::f_measure_at_k(&ranks, n_relevant, k, beta),
        );
        assert_same(
            fynch::metrics::r_precision(&ranks, n_relevant),
            rankops::metrics::r_precision(&ranks, n_relevant),
        );
        assert_same(
            fynch::metrics::err_at_k(&ranks, k),
            rankops::metrics::err_at_k(&ranks, k),
        );
        assert_same(
            fynch::metrics::rbp_at_k(&ranks, k, persistence),
            rankops::metrics::rbp_at_k(&ranks, k, persistence),
        );

        let legacy = fynch::metrics::RankingMetrics::from_ranks(&ranks);
        let current = rankops::metrics::RankingMetrics::from_ranks(&ranks);
        assert_same(legacy.mrr, current.mrr);
        assert_same(legacy.hits_at_1, current.hits_at_1);
        assert_same(legacy.hits_at_3, current.hits_at_3);
        assert_same(legacy.hits_at_10, current.hits_at_10);
        assert_same(legacy.mean_rank, current.mean_rank);
        prop_assert_eq!(legacy.count, current.count);
    }

    #[test]
    fn legacy_relevance_metrics_match_rankops(
        relevance in prop::collection::vec(-16.0f64..16.0, 0..64),
        ideal in prop::collection::vec(-16.0f64..16.0, 0..64),
        k in 0usize..128,
    ) {
        assert_same(
            fynch::metrics::dcg(&relevance),
            rankops::metrics::dcg(&relevance),
        );
        assert_same(
            fynch::metrics::ndcg(&relevance, &ideal),
            rankops::metrics::ndcg(&relevance, &ideal),
        );
        assert_same(
            fynch::metrics::ndcg_at_k(&relevance, &ideal, k),
            rankops::metrics::ndcg_at_k(&relevance, &ideal, k),
        );
    }

    #[test]
    fn legacy_rank_selection_matches_rankops(
        target in -1_000.0f64..1_000.0,
        scores in prop::collection::vec(-1_000.0f64..1_000.0, 0..128),
        higher_is_better in any::<bool>(),
    ) {
        prop_assert_eq!(
            fynch::metrics::compute_rank(target, &scores, higher_is_better),
            rankops::metrics::compute_rank(target, &scores, higher_is_better),
        );
    }
}
