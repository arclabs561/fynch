//! Fenchel-Young losses: a unified framework for prediction functions and losses.
//!
//! # The Framework
//!
//! Given a regularizer Ω defined over a domain (like the probability simplex),
//! the Fenchel-Young loss is:
//!
//! ```text
//! L_Ω(θ; y) = Ω*(θ) - ⟨θ, y⟩ + Ω(y)
//! ```
//!
//! where Ω* is the Fenchel conjugate (convex conjugate) of Ω:
//!
//! ```text
//! Ω*(θ) = max_p { ⟨θ, p⟩ - Ω(p) }
//! ```
//!
//! The prediction function (argmax of the above) is:
//!
//! ```text
//! ŷ_Ω(θ) = ∇Ω*(θ) = argmax_p { ⟨θ, p⟩ - Ω(p) }
//! ```
//!
//! # Why This Matters
//!
//! Different regularizers give different behaviors:
//!
//! | Regularizer | Prediction | Loss | Sparsity |
//! |-------------|------------|------|----------|
//! | Shannon negentropy | softmax | cross-entropy | Dense |
//! | Squared L2 (1/2‖·‖²) | sparsemax | sparsemax loss | Sparse |
//! | Tsallis α-entropy | α-entmax | entmax loss | Tunable |
//!
//! # Mathematical Foundation
//!
//! The key insight is **Moreau's decomposition**: for a proper closed convex Ω,
//!
//! ```text
//! θ = ∇Ω(ŷ_Ω(θ)) + ∇Ω*(θ) × (for Ω = 1/2‖·‖²)
//! ```
//!
//! This means the prediction ŷ_Ω and the loss L_Ω are intimately connected.
//!
//! # References
//!
//! - Blondel, Martins, Niculae (2020). "Learning with Fenchel-Young Losses" (JMLR 21)
//! - Martins, Treviso, et al. (2022). "Sparse Continuous Distributions and FY Losses"
//! - Blondel, Teboul, et al. (2020). "Fast Differentiable Sorting and Ranking"

// Note: This module doesn't use crate-level Error/Result,
// preferring to return NaN/empty for invalid inputs (matching softmax conventions)

// ============================================================================
// Regularizers (Ω)
// ============================================================================

/// Trait for regularization functions Ω.
///
/// A regularizer Ω defines:
/// 1. The regularization value Ω(p)
/// 2. The prediction map ŷ_Ω(θ) = argmax_p { ⟨θ,p⟩ - Ω(p) }
/// 3. The conjugate Ω*(θ) = max_p { ⟨θ,p⟩ - Ω(p) }
pub trait Regularizer {
    /// Compute Ω(p) for a probability distribution p.
    fn omega(&self, p: &[f64]) -> f64;

    /// Compute the prediction map ŷ_Ω(θ) = ∇Ω*(θ).
    fn predict(&self, theta: &[f64]) -> Vec<f64>;

    /// Compute the conjugate Ω*(θ).
    fn conjugate(&self, theta: &[f64]) -> f64;

    /// Compute the Fenchel-Young loss L_Ω(θ; y).
    fn loss(&self, theta: &[f64], y: &[f64]) -> f64 {
        if theta.len() != y.len() {
            return f64::NAN;
        }
        let inner: f64 = theta.iter().zip(y).map(|(t, yi)| t * yi).sum();
        self.conjugate(theta) - inner + self.omega(y)
    }
}

// ============================================================================
// Shannon Negentropy → Softmax + Cross-Entropy
// ============================================================================

/// Shannon negentropy regularizer.
///
/// ```text
/// Ω(p) = Σ pᵢ log pᵢ  (negative entropy)
/// ```
///
/// This gives:
/// - Prediction: softmax(θ)
/// - Loss: cross-entropy
/// - Conjugate: log-sum-exp(θ)
///
/// # Example
///
/// ```rust
/// use fynch::fenchel::{Shannon, Regularizer};
///
/// let shannon = Shannon;
/// let theta = [2.0, 1.0, 0.1];
/// let pred = shannon.predict(&theta);  // softmax
///
/// assert!((pred.iter().sum::<f64>() - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Shannon;

impl Regularizer for Shannon {
    fn omega(&self, p: &[f64]) -> f64 {
        // Negative entropy: Σ pᵢ log pᵢ
        p.iter()
            .filter(|&&pi| pi > 0.0)
            .map(|&pi| pi * pi.ln())
            .sum()
    }

    fn predict(&self, theta: &[f64]) -> Vec<f64> {
        softmax(theta)
    }

    fn conjugate(&self, theta: &[f64]) -> f64 {
        // log-sum-exp
        log_sum_exp(theta)
    }
}

// ============================================================================
// Squared L2 → Sparsemax
// ============================================================================

/// Squared L2 regularizer (half squared norm).
///
/// ```text
/// Ω(p) = (1/2) ‖p‖²
/// ```
///
/// This gives:
/// - Prediction: sparsemax(θ) = Euclidean projection onto simplex
/// - Loss: sparsemax loss
/// - Conjugate: (1/2) ‖sparsemax(θ)‖²
///
/// # Example
///
/// ```rust
/// use fynch::fenchel::{SquaredL2, Regularizer};
///
/// let l2 = SquaredL2;
/// let theta = [2.0, 1.0, 0.1];
/// let pred = l2.predict(&theta);  // sparsemax
///
/// // Sparsemax produces sparse outputs
/// assert!(pred.iter().any(|&p| p == 0.0));
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredL2;

impl Regularizer for SquaredL2 {
    fn omega(&self, p: &[f64]) -> f64 {
        // (1/2) ‖p‖²
        0.5 * p.iter().map(|&x| x * x).sum::<f64>()
    }

    fn predict(&self, theta: &[f64]) -> Vec<f64> {
        sparsemax(theta)
    }

    fn conjugate(&self, theta: &[f64]) -> f64 {
        // Ω*(θ) = (1/2) ‖sparsemax(θ)‖² + ⟨θ - sparsemax(θ), sparsemax(θ)⟩
        // Simplifies to: (1/2) ‖sparsemax(θ)‖²
        let p = sparsemax(theta);
        0.5 * p.iter().map(|&x| x * x).sum::<f64>()
    }

    fn loss(&self, theta: &[f64], y: &[f64]) -> f64 {
        // Sparsemax loss has a cleaner form
        sparsemax_loss(theta, y)
    }
}

// ============================================================================
// Tsallis α-Entropy → α-Entmax
// ============================================================================

/// Tsallis α-entropy regularizer.
///
/// ```text
/// Ω_α(p) = (1/(α(α-1))) (Σ pᵢ^α - 1)  for α ≠ 1
/// ```
///
/// This interpolates between:
/// - α → 1: Shannon (softmax)
/// - α = 2: Squared L2 (sparsemax)
/// - α = 1.5: A popular middle ground
///
/// # Example
///
/// ```rust
/// use fynch::fenchel::{Tsallis, Regularizer};
///
/// let tsallis = Tsallis::new(1.5);  // 1.5-entmax
/// let theta = [2.0, 1.0, 0.1];
/// let pred = tsallis.predict(&theta);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Tsallis {
    /// The α parameter. α=1 → Shannon, α=2 → sparsemax.
    pub alpha: f64,
}

impl Tsallis {
    /// Create a new Tsallis regularizer with parameter α.
    ///
    /// # Panics
    ///
    /// Panics if α ≤ 0.
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0, "alpha must be positive");
        Self { alpha }
    }

    /// Create 1.5-entmax (a popular middle ground).
    pub fn entmax15() -> Self {
        Self::new(1.5)
    }
}

impl Default for Tsallis {
    fn default() -> Self {
        Self::new(1.5)
    }
}

impl Regularizer for Tsallis {
    fn omega(&self, p: &[f64]) -> f64 {
        if (self.alpha - 1.0).abs() < 1e-10 {
            // Shannon limit
            return Shannon.omega(p);
        }
        // (1/(α(α-1))) (Σ pᵢ^α - 1)
        let sum_powers: f64 = p.iter().map(|&pi| pi.powf(self.alpha)).sum();
        (sum_powers - 1.0) / (self.alpha * (self.alpha - 1.0))
    }

    fn predict(&self, theta: &[f64]) -> Vec<f64> {
        if (self.alpha - 1.0).abs() < 1e-10 {
            return softmax(theta);
        }
        if (self.alpha - 2.0).abs() < 1e-10 {
            return sparsemax(theta);
        }
        entmax(theta, self.alpha)
    }

    fn conjugate(&self, theta: &[f64]) -> f64 {
        if (self.alpha - 1.0).abs() < 1e-10 {
            return Shannon.conjugate(theta);
        }
        // For general α, conjugate is computed via the prediction
        let p = self.predict(theta);
        let inner: f64 = theta.iter().zip(&p).map(|(t, pi)| t * pi).sum();
        inner - self.omega(&p)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Softmax: normalize via exp.
///
/// ```text
/// softmax(θ)ᵢ = exp(θᵢ) / Σⱼ exp(θⱼ)
/// ```
pub fn softmax(theta: &[f64]) -> Vec<f64> {
    if theta.is_empty() {
        return vec![];
    }
    let max = theta.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = theta.iter().map(|&t| (t - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp (numerically stable).
fn log_sum_exp(theta: &[f64]) -> f64 {
    if theta.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = theta.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max + theta.iter().map(|&t| (t - max).exp()).sum::<f64>().ln()
}

/// Sparsemax: Euclidean projection onto the probability simplex.
///
/// ```text
/// sparsemax(θ) = argmin_p { ‖p - θ‖² : p ∈ Δ }
/// ```
///
/// where Δ is the probability simplex.
///
/// # Algorithm
///
/// Sort θ descending, find threshold τ, and return [θᵢ - τ]₊.
///
/// # References
///
/// Martins & Astudillo (2016). "From Softmax to Sparsemax"
pub fn sparsemax(theta: &[f64]) -> Vec<f64> {
    if theta.is_empty() {
        return vec![];
    }

    // Sort descending
    let mut sorted: Vec<f64> = theta.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Find k = max { j : 1 + j θ_(j) > Σᵢ≤j θ_(i) }
    let mut cumsum = 0.0;
    let mut k = 0;
    for (j, &s) in sorted.iter().enumerate() {
        cumsum += s;
        if 1.0 + (j + 1) as f64 * s > cumsum {
            k = j + 1;
        }
    }

    // Threshold τ = (Σᵢ≤k θ_(i) - 1) / k
    let tau = (sorted[..k].iter().sum::<f64>() - 1.0) / k as f64;

    // Return [θᵢ - τ]₊
    theta.iter().map(|&t| (t - tau).max(0.0)).collect()
}

/// Sparsemax loss (closed form).
///
/// ```text
/// L(θ; y) = (1/2) (‖y‖² - ‖ŷ‖²) + ⟨ŷ - y, θ⟩
/// ```
///
/// where ŷ = sparsemax(θ).
pub fn sparsemax_loss(theta: &[f64], y: &[f64]) -> f64 {
    if theta.len() != y.len() || theta.is_empty() {
        return f64::NAN;
    }

    let p = sparsemax(theta);

    // ‖y‖² and ‖ŷ‖²
    let y_sq: f64 = y.iter().map(|&yi| yi * yi).sum();
    let p_sq: f64 = p.iter().map(|&pi| pi * pi).sum();

    // ⟨ŷ - y, θ⟩
    let diff_inner: f64 = p
        .iter()
        .zip(y)
        .zip(theta)
        .map(|((&pi, &yi), &ti)| (pi - yi) * ti)
        .sum();

    0.5 * (y_sq - p_sq) + diff_inner
}

/// α-entmax: sparse transformation with tunable sparsity.
///
/// For α = 1: softmax (dense)
/// For α = 2: sparsemax (sparse)
/// For 1 < α < 2: interpolates
///
/// # Algorithm
///
/// Bisection to find threshold τ such that projection sums to 1.
///
/// # References
///
/// Peters, Niculae, Martins (2019). "Sparse Sequence-to-Sequence Models"
pub fn entmax(theta: &[f64], alpha: f64) -> Vec<f64> {
    if theta.is_empty() {
        return vec![];
    }
    if (alpha - 1.0).abs() < 1e-10 {
        return softmax(theta);
    }
    if (alpha - 2.0).abs() < 1e-10 {
        return sparsemax(theta);
    }

    // Find max for numerical stability
    let max_theta = theta.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Bisection to find τ
    let mut tau_lo = max_theta - 10.0;
    let mut tau_hi = max_theta;

    for _ in 0..50 {
        let tau = (tau_lo + tau_hi) / 2.0;
        let sum: f64 = theta
            .iter()
            .map(|&t| ((t - tau).max(0.0)).powf(1.0 / (alpha - 1.0)))
            .sum();

        if sum < 1.0 {
            tau_hi = tau;
        } else {
            tau_lo = tau;
        }
    }

    let tau = (tau_lo + tau_hi) / 2.0;

    // Compute output
    let mut result: Vec<f64> = theta
        .iter()
        .map(|&t| ((t - tau).max(0.0)).powf(1.0 / (alpha - 1.0)))
        .collect();

    // Normalize to ensure sum = 1
    let sum: f64 = result.iter().sum();
    if sum > 0.0 {
        for r in &mut result {
            *r /= sum;
        }
    }

    result
}

/// Compute the Fenchel-Young loss for any regularizer.
///
/// Convenience function that dispatches to the regularizer's loss method.
pub fn fy_loss<R: Regularizer>(reg: &R, theta: &[f64], y: &[f64]) -> f64 {
    reg.loss(theta, y)
}

// ============================================================================
// Temperature Scaling
// ============================================================================

/// Apply temperature scaling to logits before softmax.
///
/// ```text
/// softmax(θ/τ)  where τ = temperature
/// ```
///
/// Temperature affects entropy calibration:
/// - τ > 1: Higher entropy (more diverse/random outputs)
/// - τ < 1: Lower entropy (more peaked/deterministic)
/// - τ = 1: Unmodified softmax
///
/// # Connection to Entropy Calibration
///
/// In LLM generation, temperature is the simplest calibration knob.
/// Cao, Valiant, Liang (2025) show that the "right" temperature is
/// similar across model sizes because miscalibration scales slowly
/// with model capacity for heavy-tailed (text-like) distributions.
///
/// For more principled calibration, see:
/// - `surp::entropy_calibration` for evaluation metrics
/// - `surp::zipf` for understanding why calibration is hard
///
/// # Example
///
/// ```rust
/// use fynch::fenchel::softmax_with_temperature;
///
/// let theta = [2.0, 1.0, 0.1];
///
/// let cold = softmax_with_temperature(&theta, 0.5);  // more peaked
/// let hot = softmax_with_temperature(&theta, 2.0);   // flatter
///
/// // Cold temperature increases max probability
/// assert!(cold.iter().cloned().fold(0.0_f64, f64::max) >
///         hot.iter().cloned().fold(0.0_f64, f64::max));
/// ```
pub fn softmax_with_temperature(theta: &[f64], temperature: f64) -> Vec<f64> {
    if temperature <= 0.0 || !temperature.is_finite() {
        // Invalid temperature: return uniform or empty
        if theta.is_empty() {
            return vec![];
        }
        let n = theta.len() as f64;
        return vec![1.0 / n; theta.len()];
    }

    let scaled: Vec<f64> = theta.iter().map(|&t| t / temperature).collect();
    softmax(&scaled)
}

/// Compute entropy of a probability distribution (in nats).
///
/// H(p) = -Σ pᵢ ln(pᵢ)
///
/// Useful for checking calibration: compare entropy of model outputs
/// to log-loss on reference text.
pub fn entropy_nats(p: &[f64]) -> f64 {
    p.iter()
        .filter(|&&pi| pi > 0.0)
        .map(|&pi| -pi * pi.ln())
        .sum()
}

/// Compute entropy of a probability distribution (in bits).
pub fn entropy_bits(p: &[f64]) -> f64 {
    entropy_nats(p) / std::f64::consts::LN_2
}

// ============================================================================
// Presets
// ============================================================================

/// Softmax + cross-entropy (Shannon regularization).
pub fn softmax_loss(theta: &[f64], y: &[f64]) -> f64 {
    Shannon.loss(theta, y)
}

/// 1.5-entmax loss (Tsallis α=1.5).
pub fn entmax15_loss(theta: &[f64], y: &[f64]) -> f64 {
    Tsallis::entmax15().loss(theta, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let theta = [2.0, 1.0, 0.1, -1.0];
        let p = softmax(&theta);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparsemax_sums_to_one() {
        let theta = [2.0, 1.0, 0.1, -1.0];
        let p = sparsemax(&theta);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparsemax_is_sparse() {
        let theta = [2.0, 1.0, 0.1, -1.0];
        let p = sparsemax(&theta);
        let zeros = p.iter().filter(|&&x| x == 0.0).count();
        assert!(zeros > 0, "sparsemax should produce zeros");
    }

    #[test]
    fn test_entmax_interpolates() {
        let theta = [2.0, 1.0, 0.1];

        // α = 1: softmax (dense)
        let p1 = entmax(&theta, 1.0);
        assert!(p1.iter().all(|&x| x > 0.0), "α=1 should be dense");

        // α = 2: sparsemax (sparse)
        let p2 = entmax(&theta, 2.0);
        let sparse2 = sparsemax(&theta);
        for (a, b) in p2.iter().zip(&sparse2) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_shannon_loss_equals_cross_entropy() {
        let theta = [2.0, 1.0, 0.1];
        let y = [1.0, 0.0, 0.0]; // one-hot

        let loss = Shannon.loss(&theta, &y);

        // Cross-entropy: -Σ yᵢ log(softmax(θ)ᵢ)
        let p = softmax(&theta);
        let ce: f64 = -y.iter().zip(&p).map(|(&yi, &pi)| yi * pi.ln()).sum::<f64>();

        // They should be equal (up to constant)
        // Actually FY loss = CE when y is one-hot
        assert!((loss - ce).abs() < 1e-6, "loss={}, ce={}", loss, ce);
    }

    #[test]
    fn test_fy_loss_nonnegative() {
        let theta = [2.0, 1.0, 0.1];
        let y = [0.5, 0.3, 0.2];

        assert!(Shannon.loss(&theta, &y) >= -1e-10);
        assert!(SquaredL2.loss(&theta, &y) >= -1e-10);
        assert!(Tsallis::entmax15().loss(&theta, &y) >= -1e-10);
    }

    #[test]
    fn test_softmax_temperature_scaling() {
        let theta = [2.0, 1.0, 0.1];

        let cold = softmax_with_temperature(&theta, 0.5);
        let normal = softmax_with_temperature(&theta, 1.0);
        let hot = softmax_with_temperature(&theta, 2.0);

        // All should sum to 1
        assert!((cold.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!((normal.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!((hot.iter().sum::<f64>() - 1.0).abs() < 1e-10);

        // Normal should match regular softmax
        let regular = softmax(&theta);
        for (a, b) in normal.iter().zip(&regular) {
            assert!((a - b).abs() < 1e-10);
        }

        // Cold should be more peaked (lower entropy)
        let h_cold = entropy_bits(&cold);
        let h_normal = entropy_bits(&normal);
        let h_hot = entropy_bits(&hot);

        assert!(h_cold < h_normal, "cold={}, normal={}", h_cold, h_normal);
        assert!(h_normal < h_hot, "normal={}, hot={}", h_normal, h_hot);
    }

    #[test]
    fn test_entropy_uniform() {
        let p = [0.25, 0.25, 0.25, 0.25];
        let h = entropy_bits(&p);
        // log2(4) = 2 bits
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fy_loss_zero_at_prediction() {
        // L_Ω(θ; ŷ_Ω(θ)) = 0
        let theta = [2.0, 1.0, 0.1];

        let y_shannon = Shannon.predict(&theta);
        let loss_shannon = Shannon.loss(&theta, &y_shannon);
        assert!(
            loss_shannon.abs() < 1e-6,
            "Shannon loss at prediction: {}",
            loss_shannon
        );

        // For sparsemax, the loss at prediction should also be 0
        let y_sparse = SquaredL2.predict(&theta);
        let loss_sparse = SquaredL2.loss(&theta, &y_sparse);
        assert!(
            loss_sparse.abs() < 1e-6,
            "Sparsemax loss at prediction: {}",
            loss_sparse
        );
    }
}
