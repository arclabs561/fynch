//! Differentiable Top-K selection.
//!
//! Provides soft selection of the k largest (or smallest) elements,
//! allowing gradients to flow through the selection operation.
//!
//! # Background
//!
//! Hard top-k is discontinuous: changing one value slightly can swap which
//! elements are in the top-k set. The differentiable version uses soft masks
//! that approach the hard selection as temperature → 0.
//!
//! # References
//!
//! - Xie et al. (2020). "Differentiable Top-K Operator with Optimal Transport"
//! - Grover et al. (2019). "Stochastic Optimization of Sorting Networks via REINFORCE"

use crate::sigmoid::sigmoid;
use crate::soft_rank;

/// Differentiable Top-K selection (k largest values).
///
/// Returns soft indicator weights for each element indicating membership
/// in the top-k set. As temperature decreases, indicators approach {0, 1}.
///
/// # Note on Soft Ranks
///
/// `soft_rank` assigns **lower** ranks to **higher** values:
/// - Value 0.9 (highest) → rank ≈ 1
/// - Value 0.1 (lowest) → rank ≈ n
///
/// So top-k elements have ranks ≤ k.
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
/// - `weighted_values[i]` = values[i] * indicator[i]
/// - `indicators[i]` ∈ (0, 1) indicates soft membership in top-k
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
/// - Value 0.1 (lowest) → rank ≈ n
/// - Value 0.9 (highest) → rank ≈ 1
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
        let u: f64 = rng.gen_range(0.0..1.0);
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
    /// temperature → 0.
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
