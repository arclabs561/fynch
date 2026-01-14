//! Ranking evaluation metrics.
//!
//! Standard metrics for evaluating ranking quality in information retrieval,
//! knowledge graph completion, and learning-to-rank tasks.
//!
//! # Metrics Overview
//!
//! | Metric | Range | Interpretation |
//! |--------|-------|----------------|
//! | MRR | [0, 1] | Average reciprocal of correct answer's rank |
//! | Hits@k | [0, 1] | Fraction with correct in top k |
//! | Mean Rank | [1, n] | Average rank of correct answer |
//! | NDCG | [0, 1] | Position-weighted relevance score |
//!
//! # Example
//!
//! ```rust
//! use fynch::metrics::{mrr, hits_at_k, mean_rank, ndcg};
//!
//! // Ranks of correct answers for 4 queries (1-indexed)
//! let ranks = [1, 3, 2, 5];
//!
//! assert!((mrr(&ranks) - 0.508).abs() < 0.01);
//! assert!((hits_at_k(&ranks, 3) - 0.75).abs() < 0.01);
//! assert!((mean_rank(&ranks) - 2.75).abs() < 0.01);
//! ```
//!
//! # Research Background
//!
//! These metrics are standard in:
//! - **Information Retrieval**: Evaluating search engines (Croft et al., 2010)
//! - **Knowledge Graphs**: Link prediction (Bordes et al., 2013)
//! - **Recommendation**: Ranking candidate items
//!
//! MRR was popularized for KG evaluation by Bordes et al. (2013) "Translating
//! Embeddings for Modeling Multi-relational Data".

/// Mean Reciprocal Rank (MRR).
///
/// Average of 1/rank for correct answers. Gives exponentially more weight
/// to top positions: rank 1 → 1.0, rank 2 → 0.5, rank 10 → 0.1.
///
/// # Formula
///
/// ```text
/// MRR = (1/|Q|) Σᵢ 1/rankᵢ
/// ```
///
/// # Parameters
///
/// - `ranks`: Slice of ranks (1-indexed, positive integers)
///
/// # Returns
///
/// MRR in [0, 1]. Higher is better.
///
/// # Example
///
/// ```rust
/// use fynch::metrics::mrr;
///
/// let ranks = [1, 3, 2, 5];
/// let score = mrr(&ranks);
/// // (1/1 + 1/3 + 1/2 + 1/5) / 4 ≈ 0.508
/// assert!((score - 0.508).abs() < 0.01);
/// ```
pub fn mrr(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }

    let sum: f64 = ranks
        .iter()
        .filter(|&&r| r > 0)
        .map(|&r| 1.0 / r as f64)
        .sum();

    sum / ranks.len() as f64
}

/// Hits@k: fraction of queries with correct answer in top k.
///
/// Binary metric: either the answer is in top k (1) or not (0).
///
/// # Formula
///
/// ```text
/// Hits@k = (1/|Q|) Σᵢ 𝟙[rankᵢ ≤ k]
/// ```
///
/// # Parameters
///
/// - `ranks`: Slice of ranks (1-indexed)
/// - `k`: Cutoff (typically 1, 3, or 10)
///
/// # Returns
///
/// Hits@k in [0, 1]. Higher is better.
///
/// # Example
///
/// ```rust
/// use fynch::metrics::hits_at_k;
///
/// let ranks = [1, 3, 2, 5];
/// assert!((hits_at_k(&ranks, 3) - 0.75).abs() < 0.01);  // 3 of 4 in top 3
/// assert!((hits_at_k(&ranks, 1) - 0.25).abs() < 0.01);  // 1 of 4 at rank 1
/// ```
pub fn hits_at_k(ranks: &[usize], k: usize) -> f64 {
    if ranks.is_empty() || k == 0 {
        return 0.0;
    }

    let hits = ranks.iter().filter(|&&r| r > 0 && r <= k).count();
    hits as f64 / ranks.len() as f64
}

/// Mean Rank: average position of correct answers.
///
/// Unlike MRR, this is linear in rank. Lower is better.
///
/// # Formula
///
/// ```text
/// MR = (1/|Q|) Σᵢ rankᵢ
/// ```
///
/// # Returns
///
/// Mean Rank in [1, n]. Lower is better.
///
/// # Example
///
/// ```rust
/// use fynch::metrics::mean_rank;
///
/// let ranks = [1, 3, 2, 5];
/// assert!((mean_rank(&ranks) - 2.75).abs() < 0.01);
/// ```
pub fn mean_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }

    let sum: f64 = ranks.iter().map(|&r| r as f64).sum();
    sum / ranks.len() as f64
}

/// Normalized Discounted Cumulative Gain (NDCG).
///
/// Measures ranking quality with position-weighted relevance.
/// Unlike Hits@k, NDCG considers graded relevance (not just binary).
///
/// # Formula
///
/// ```text
/// DCG = Σᵢ relᵢ / log₂(i + 1)
/// NDCG = DCG / IDCG
/// ```
///
/// where IDCG is DCG of the ideal (perfectly sorted) ranking.
///
/// # Parameters
///
/// - `relevance`: Relevance scores in ranked order (position 1, 2, 3, ...)
/// - `ideal`: Ideal relevance scores (sorted descending)
///
/// # Returns
///
/// NDCG in [0, 1]. Higher is better.
///
/// # Example
///
/// ```rust
/// use fynch::metrics::ndcg;
///
/// // Actual ranking: relevance [3, 2, 1, 0]
/// // Ideal ranking:  relevance [3, 2, 1, 0] (already optimal)
/// let relevance = [3.0, 2.0, 1.0, 0.0];
/// let ideal = [3.0, 2.0, 1.0, 0.0];
/// assert!((ndcg(&relevance, &ideal) - 1.0).abs() < 0.01);
///
/// // Suboptimal ranking
/// let relevance = [1.0, 3.0, 2.0, 0.0];
/// let ideal = [3.0, 2.0, 1.0, 0.0];
/// assert!(ndcg(&relevance, &ideal) < 1.0);
/// ```
pub fn ndcg(relevance: &[f64], ideal: &[f64]) -> f64 {
    let actual_dcg = dcg(relevance);
    let ideal_dcg = dcg(ideal);

    if ideal_dcg == 0.0 {
        0.0
    } else {
        actual_dcg / ideal_dcg
    }
}

/// Discounted Cumulative Gain (DCG).
///
/// Helper for NDCG. Sums relevance weighted by log position.
fn dcg(relevance: &[f64]) -> f64 {
    relevance
        .iter()
        .enumerate()
        .map(|(i, &rel)| {
            let position = i + 1;
            rel / (position as f64 + 1.0).log2()
        })
        .sum()
}

/// NDCG@k: NDCG truncated at position k.
///
/// Only considers the top k positions in the ranking.
///
/// # Example
///
/// ```rust
/// use fynch::metrics::ndcg_at_k;
///
/// let relevance = [3.0, 1.0, 2.0, 0.0];
/// let ideal = [3.0, 2.0, 1.0, 0.0];
/// let score = ndcg_at_k(&relevance, &ideal, 3);
/// ```
pub fn ndcg_at_k(relevance: &[f64], ideal: &[f64], k: usize) -> f64 {
    let rel_k: Vec<f64> = relevance.iter().take(k).copied().collect();
    let ideal_k: Vec<f64> = ideal.iter().take(k).copied().collect();
    ndcg(&rel_k, &ideal_k)
}

/// Compute rank of a target score among all scores.
///
/// Useful for link prediction: given scores for all entities,
/// find the rank of the correct entity.
///
/// # Parameters
///
/// - `target_score`: Score of the correct answer
/// - `all_scores`: Scores of all candidates (including target)
/// - `higher_is_better`: If true, higher scores rank higher
///
/// # Returns
///
/// Rank (1-indexed). Rank 1 = best.
///
/// # Example
///
/// ```rust
/// use fynch::metrics::compute_rank;
///
/// let scores = [0.1, 0.5, 0.3, 0.8];
/// let rank = compute_rank(0.5, &scores, true);
/// // 0.8 > 0.5 > 0.3 > 0.1, so 0.5 is rank 2
/// assert_eq!(rank, 2);
/// ```
pub fn compute_rank(target_score: f64, all_scores: &[f64], higher_is_better: bool) -> usize {
    let mut rank = 1;
    for &score in all_scores {
        if higher_is_better {
            if score > target_score {
                rank += 1;
            }
        } else {
            if score < target_score {
                rank += 1;
            }
        }
    }
    rank
}

/// Aggregate ranking metrics.
///
/// Computes MRR, Hits@1/3/10, and Mean Rank in one pass.
#[derive(Debug, Clone, Default)]
pub struct RankingMetrics {
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Hits@1
    pub hits_at_1: f64,
    /// Hits@3
    pub hits_at_3: f64,
    /// Hits@10
    pub hits_at_10: f64,
    /// Mean Rank
    pub mean_rank: f64,
    /// Number of queries
    pub count: usize,
}

impl RankingMetrics {
    /// Compute all metrics from ranks.
    pub fn from_ranks(ranks: &[usize]) -> Self {
        Self {
            mrr: mrr(ranks),
            hits_at_1: hits_at_k(ranks, 1),
            hits_at_3: hits_at_k(ranks, 3),
            hits_at_10: hits_at_k(ranks, 10),
            mean_rank: mean_rank(ranks),
            count: ranks.len(),
        }
    }

    /// Pretty print metrics.
    pub fn summary(&self) -> String {
        format!(
            "MRR: {:.4}, Hits@1: {:.4}, Hits@3: {:.4}, Hits@10: {:.4}, MR: {:.2}",
            self.mrr, self.hits_at_1, self.hits_at_3, self.hits_at_10, self.mean_rank
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mrr() {
        let ranks = [1, 3, 2, 5];
        let expected = (1.0 + 1.0 / 3.0 + 0.5 + 0.2) / 4.0;
        assert!((mrr(&ranks) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_mrr_empty() {
        assert_eq!(mrr(&[]), 0.0);
    }

    #[test]
    fn test_mrr_all_rank_one() {
        let ranks = [1, 1, 1, 1];
        assert!((mrr(&ranks) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hits_at_k() {
        let ranks = [1, 3, 2, 5, 10, 15];

        assert!((hits_at_k(&ranks, 1) - 1.0 / 6.0).abs() < 1e-6);
        assert!((hits_at_k(&ranks, 3) - 3.0 / 6.0).abs() < 1e-6);
        assert!((hits_at_k(&ranks, 10) - 5.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_hits_at_k_empty() {
        assert_eq!(hits_at_k(&[], 10), 0.0);
    }

    #[test]
    fn test_mean_rank() {
        let ranks = [1, 3, 2, 5];
        let expected = (1.0 + 3.0 + 2.0 + 5.0) / 4.0;
        assert!((mean_rank(&ranks) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg_perfect() {
        let relevance = [3.0, 2.0, 1.0, 0.0];
        let ideal = [3.0, 2.0, 1.0, 0.0];
        assert!((ndcg(&relevance, &ideal) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg_suboptimal() {
        let relevance = [0.0, 3.0, 2.0, 1.0]; // worst item first
        let ideal = [3.0, 2.0, 1.0, 0.0];
        let score = ndcg(&relevance, &ideal);
        assert!(score < 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_compute_rank() {
        let scores = [0.1, 0.5, 0.3, 0.8];

        // Higher is better: 0.8 > 0.5 > 0.3 > 0.1
        assert_eq!(compute_rank(0.8, &scores, true), 1);
        assert_eq!(compute_rank(0.5, &scores, true), 2);
        assert_eq!(compute_rank(0.3, &scores, true), 3);
        assert_eq!(compute_rank(0.1, &scores, true), 4);

        // Lower is better: 0.1 < 0.3 < 0.5 < 0.8
        assert_eq!(compute_rank(0.1, &scores, false), 1);
        assert_eq!(compute_rank(0.8, &scores, false), 4);
    }

    #[test]
    fn test_ranking_metrics_struct() {
        let ranks = [1, 2, 3, 10];
        let metrics = RankingMetrics::from_ranks(&ranks);

        assert!((metrics.mrr - mrr(&ranks)).abs() < 1e-6);
        assert!((metrics.hits_at_1 - hits_at_k(&ranks, 1)).abs() < 1e-6);
        assert!((metrics.hits_at_10 - hits_at_k(&ranks, 10)).abs() < 1e-6);
        assert!((metrics.mean_rank - mean_rank(&ranks)).abs() < 1e-6);
        assert_eq!(metrics.count, 4);
    }
}
