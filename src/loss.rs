//! Differentiable ranking losses for learning to rank.
//!
//! These losses allow gradient-based optimization of ranking objectives.
//!
//! # Overview
//!
//! | Loss | Type | Use Case |
//! |------|------|----------|
//! | [`spearman_loss`] | Correlation | Rank correlation optimization |
//! | [`listnet_loss`] | Listwise | Cross-entropy on rank distributions |
//! | [`listmle_loss`] | Listwise | Maximum likelihood for permutations |
//!
//! # Learning to Rank Paradigms
//!
//! ```text
//! Pointwise: Predict relevance of each item independently
//!   └─ Simple but ignores list structure
//!
//! Pairwise: Predict which of two items ranks higher
//!   └─ LambdaRank, RankNet
//!
//! Listwise: Optimize the entire ranking at once
//!   └─ ListNet, ListMLE, SoftRank
//! ```
//!
//! The losses here are **listwise** - they consider the entire ranking.
//!
//! # References
//!
//! - Cao et al. (2007). "Learning to Rank: From Pairwise Approach to Listwise Approach"
//! - Xia et al. (2008). "Listwise Approach to Learning to Rank"
//! - Taylor et al. (2008). "SoftRank: Optimizing Non-smooth Rank Metrics"

use crate::soft_rank;

/// InfoNCE (Information Noise-Contrastive Estimation) Loss.
/// 
/// L = -log( exp(pos / tau) / sum(exp(all / tau)) )
pub fn info_nce_loss(pos_score: f64, neg_scores: &[f64], temperature: f64) -> f64 {
    let pos = pos_score / temperature;
    
    let mut max_score = pos;
    for &s in neg_scores {
        max_score = max_score.max(s / temperature);
    }
    
    let sum_exp = (pos - max_score).exp() + 
        neg_scores.iter().map(|&s| (s / temperature - max_score).exp()).sum::<f64>();
    
    -(pos - max_score - sum_exp.ln())
}

/// Spearman correlation loss.
///
/// Measures how well the predicted ranking correlates with target ranking.
///
/// ```text
/// Loss = 1 - ρ(soft_rank(pred), soft_rank(target))
/// ```
///
/// where ρ is Pearson correlation.
///
/// # Arguments
///
/// * `predictions` - Predicted scores (higher = more relevant)
/// * `targets` - Target scores/labels
/// * `temperature` - Controls sharpness of soft ranking
///
/// # Returns
///
/// Loss in [0, 2]. Lower is better (0 = perfect positive correlation).
///
/// # Example
///
/// ```rust
/// use fynch::loss::spearman_loss;
///
/// let pred = [0.9, 0.1, 0.5];   // predicts order: 0, 2, 1
/// let target = [3.0, 1.0, 2.0]; // true order: 0, 2, 1
///
/// let loss = spearman_loss(&pred, &target, 0.1);
/// assert!(loss < 0.5); // Should be low (rankings match)
/// ```
pub fn spearman_loss(predictions: &[f64], targets: &[f64], temperature: f64) -> f64 {
    let n = predictions.len();
    if n != targets.len() || n < 2 {
        return 1.0;
    }

    let pred_ranks = match soft_rank(predictions, temperature) {
        Ok(r) => r,
        Err(_) => return 1.0,
    };
    let target_ranks = match soft_rank(targets, temperature) {
        Ok(r) => r,
        Err(_) => return 1.0,
    };

    let pred_mean: f64 = pred_ranks.iter().sum::<f64>() / n as f64;
    let target_mean: f64 = target_ranks.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut pred_var = 0.0;
    let mut target_var = 0.0;

    for i in 0..n {
        let pred_diff = pred_ranks[i] - pred_mean;
        let target_diff = target_ranks[i] - target_mean;
        numerator += pred_diff * target_diff;
        pred_var += pred_diff * pred_diff;
        target_var += target_diff * target_diff;
    }

    let denominator = (pred_var * target_var).sqrt();

    if denominator < 1e-10 {
        return 1.0; // No variance = undefined correlation
    }

    1.0 - (numerator / denominator)
}

/// ListNet loss: cross-entropy between rank probability distributions.
///
/// Converts rankings to probability distributions via softmax, then
/// computes cross-entropy loss.
///
/// ```text
/// P(i) = softmax(ranks)_i
/// Loss = -Σᵢ P_target(i) log P_pred(i)
/// ```
///
/// # Arguments
///
/// * `predictions` - Predicted scores
/// * `targets` - Target scores
/// * `temperature` - Controls sharpness of soft ranking
///
/// # Example
///
/// ```rust
/// use fynch::loss::listnet_loss;
///
/// let pred = [0.9, 0.1, 0.5];
/// let target = [3.0, 1.0, 2.0];
///
/// let loss = listnet_loss(&pred, &target, 0.1);
/// ```
pub fn listnet_loss(predictions: &[f64], targets: &[f64], temperature: f64) -> f64 {
    let n = predictions.len();
    if n == 0 || n != targets.len() {
        return f64::INFINITY;
    }

    let pred_ranks = match soft_rank(predictions, temperature) {
        Ok(r) => r,
        Err(_) => return f64::INFINITY,
    };
    let target_ranks = match soft_rank(targets, temperature) {
        Ok(r) => r,
        Err(_) => return f64::INFINITY,
    };

    let pred_probs = softmax(&pred_ranks);
    let target_probs = softmax(&target_ranks);

    let mut loss = 0.0;
    for i in 0..n {
        if target_probs[i] > 1e-10 {
            loss -= target_probs[i] * (pred_probs[i] + 1e-10).ln();
        }
    }
    loss
}

/// ListMLE loss: negative log-likelihood of target permutation.
///
/// Models ranking as sequential selection and maximizes likelihood
/// of the target ordering.
///
/// ```text
/// P(π) = Π_i exp(s_{π(i)}) / Σ_{j≥i} exp(s_{π(j)})
/// Loss = -log P(π_target)
/// ```
///
/// # Arguments
///
/// * `predictions` - Predicted scores
/// * `targets` - Target scores (used to determine true ordering)
/// * `temperature` - Controls sharpness
///
/// # Example
///
/// ```rust
/// use fynch::loss::listmle_loss;
///
/// let pred = [0.9, 0.1, 0.5];
/// let target = [3.0, 1.0, 2.0];
///
/// let loss = listmle_loss(&pred, &target, 0.1);
/// ```
pub fn listmle_loss(predictions: &[f64], targets: &[f64], temperature: f64) -> f64 {
    let n = predictions.len();
    if n == 0 || n != targets.len() {
        return f64::INFINITY;
    }

    // Get target ranking order (descending by target score)
    let mut target_order: Vec<usize> = (0..n).collect();
    target_order.sort_unstable_by(|&a, &b| {
        targets[b]
            .partial_cmp(&targets[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Use soft ranks for differentiability
    let pred_ranks = match soft_rank(predictions, temperature) {
        Ok(r) => r,
        Err(_) => return f64::INFINITY,
    };

    // ListMLE: log-likelihood of selecting items in target order
    let mut loss = 0.0;
    for i in 0..n {
        let idx = target_order[i];
        let score = pred_ranks[idx];

        // Denominator: sum over remaining items
        let mut log_denom = f64::NEG_INFINITY;
        for &jdx in target_order.iter().skip(i) {
            log_denom = log_sum_exp(log_denom, pred_ranks[jdx]);
        }

        loss -= score - log_denom;
    }
    loss
}

/// Pairwise margin ranking loss.
///
/// For each pair where target[i] > target[j], penalize if pred[i] < pred[j].
///
/// ```text
/// Loss = Σ_{i,j: t_i > t_j} max(0, margin - (s_i - s_j))
/// ```
///
/// # Arguments
///
/// * `predictions` - Predicted scores
/// * `targets` - Target scores
/// * `margin` - Minimum margin between positive and negative pairs
///
/// # Example
///
/// ```rust
/// use fynch::loss::pairwise_margin_loss;
///
/// let pred = [0.9, 0.1, 0.5];
/// let target = [3.0, 1.0, 2.0];
///
/// let loss = pairwise_margin_loss(&pred, &target, 0.1);
/// ```
pub fn pairwise_margin_loss(predictions: &[f64], targets: &[f64], margin: f64) -> f64 {
    let n = predictions.len();
    if n != targets.len() || n < 2 {
        return 0.0;
    }

    let mut loss = 0.0;
    let mut count = 0;

    for i in 0..n {
        for j in 0..n {
            if targets[i] > targets[j] {
                // i should rank higher than j
                let diff = predictions[i] - predictions[j];
                loss += (margin - diff).max(0.0);
                count += 1;
            }
        }
    }

    if count > 0 {
        loss / count as f64
    } else {
        0.0
    }
}

// Helper functions

fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else if a > b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spearman_perfect_correlation() {
        // Perfect positive correlation
        let pred = [1.0, 2.0, 3.0, 4.0];
        let target = [10.0, 20.0, 30.0, 40.0];
        let loss = spearman_loss(&pred, &target, 0.1);
        assert!(
            loss < 0.1,
            "Perfect correlation should give low loss: {}",
            loss
        );
    }

    #[test]
    fn test_spearman_negative_correlation() {
        // Perfect negative correlation
        let pred = [4.0, 3.0, 2.0, 1.0];
        let target = [1.0, 2.0, 3.0, 4.0];
        let loss = spearman_loss(&pred, &target, 0.1);
        assert!(
            loss > 1.5,
            "Negative correlation should give high loss: {}",
            loss
        );
    }

    #[test]
    fn test_listnet_same_ranking() {
        let pred = [0.9, 0.1, 0.5];
        let target = [3.0, 1.0, 2.0];
        let loss = listnet_loss(&pred, &target, 0.1);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_listmle_basic() {
        let pred = [0.9, 0.1, 0.5];
        let target = [3.0, 1.0, 2.0];
        let loss = listmle_loss(&pred, &target, 0.1);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_pairwise_margin() {
        let pred = [0.9, 0.1, 0.5];
        let target = [3.0, 1.0, 2.0];
        let loss = pairwise_margin_loss(&pred, &target, 0.1);
        // Should be low since ranking is correct
        assert!(
            loss < 0.5,
            "Correct ranking should have low margin loss: {}",
            loss
        );
    }

    #[test]
    fn test_pairwise_margin_wrong_order() {
        let pred = [0.1, 0.9, 0.5]; // Wrong order
        let target = [3.0, 1.0, 2.0];
        let loss_wrong = pairwise_margin_loss(&pred, &target, 0.1);

        let pred_right = [0.9, 0.1, 0.5];
        let loss_right = pairwise_margin_loss(&pred_right, &target, 0.1);

        assert!(
            loss_wrong > loss_right,
            "Wrong order should have higher loss: {} vs {}",
            loss_wrong,
            loss_right
        );
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_infonce_non_negative(
            pos in -10.0f64..10.0,
            negs in prop::collection::vec(-10.0f64..10.0, 1..16),
            temp in 0.1f64..2.0
        ) {
            let loss = info_nce_loss(pos, &negs, temp);
            // InfoNCE is cross-entropy, so it should be non-negative.
            prop_assert!(loss >= 0.0);
        }

        #[test]
        fn prop_infonce_higher_pos_lower_loss(
            pos1 in -10.0f64..0.0,
            pos2 in 0.0f64..10.0,
            negs in prop::collection::vec(-10.0f64..10.0, 1..16),
            temp in 0.1f64..2.0
        ) {
            let loss1 = info_nce_loss(pos1, &negs, temp);
            let loss2 = info_nce_loss(pos2, &negs, temp);
            prop_assert!(loss2 < loss1);
        }
    }
}
