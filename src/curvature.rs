//! Curvature-aware gradients for differentiable sorting/ranking operators.
//!
//! Standard gradient descent ignores the curvature of the soft-rank/soft-sort
//! operators. Newton preconditioning divides the gradient element-wise by the
//! Hessian diagonal, taking larger steps in flat directions and smaller steps
//! in curved directions.
//!
//! The Hessian diagonal of the sigmoid-based soft-rank operator at position i:
//!
//! ```text
//! H_ii = (1/tau) * sum_{j != i} sigma'((x_j - x_i) / tau)
//! ```
//!
//! where `sigma'(z) = sigma(z) * (1 - sigma(z))`.
//!
//! # References
//!
//! - Petersen et al. (2024), "Newton Losses: Using Curvature Information for
//!   Learning with Differentiable Algorithms"
//!
//! # Example
//!
//! ```rust
//! use fynch::curvature::{newton_soft_rank_loss, damped_newton_gradient};
//!
//! let predictions = [0.3, 0.7, 0.1, 0.9];
//! let targets = [1.0, 2.0, 3.0, 4.0];
//! let (loss, newton_grad) = newton_soft_rank_loss(&predictions, &targets, 0.5);
//! assert!(loss >= 0.0);
//! ```

use crate::sigmoid::sigmoid_derivative;
use crate::{soft_rank, Error, Result};

/// Compute the diagonal of the Hessian of the soft-rank operator.
///
/// For each position i, the Hessian diagonal entry is the sum of sigmoid
/// derivatives of pairwise differences, scaled by 1/temperature.
pub fn soft_rank_hessian_diag(x: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if x.is_empty() {
        return Err(Error::EmptyInput);
    }
    if temperature <= 0.0 {
        return Err(Error::InvalidTemperature(temperature));
    }

    let n = x.len();
    let mut hessian = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let z = (x[j] - x[i]) / temperature;
                hessian[i] += sigmoid_derivative(z);
            }
        }
        hessian[i] /= temperature;
    }

    Ok(hessian)
}

/// Compute Newton-preconditioned gradients: `g / (h + damping)`.
///
/// Prevents division by near-zero curvature via the damping term. When
/// `damping` is large relative to `h`, this recovers standard gradient
/// descent. When curvature is high, the gradient is scaled down.
pub fn damped_newton_gradient(gradient: &[f64], hessian_diag: &[f64], damping: f64) -> Vec<f64> {
    gradient
        .iter()
        .zip(hessian_diag.iter())
        .map(|(&g, &h)| g / (h + damping))
        .collect()
}

/// Soft-rank Spearman loss with Newton-preconditioned gradients.
///
/// Computes the MSE between soft ranks of predictions and targets, then
/// preconditions the gradient by the Hessian diagonal of the soft-rank
/// operator.
///
/// Returns `(loss, newton_gradients)`.
pub fn newton_soft_rank_loss(
    predictions: &[f64],
    targets: &[f64],
    temperature: f64,
) -> (f64, Vec<f64>) {
    let n = predictions.len();
    assert_eq!(n, targets.len(), "length mismatch");
    assert!(n >= 2, "need at least 2 elements");
    assert!(temperature > 0.0, "temperature must be positive");

    let pred_ranks = soft_rank(predictions, temperature).expect("valid input");
    let target_ranks = soft_rank(targets, temperature).expect("valid input");

    // Loss: mean squared error between soft ranks.
    let mut loss = 0.0;
    let mut residuals = vec![0.0; n];
    for i in 0..n {
        residuals[i] = pred_ranks[i] - target_ranks[i];
        loss += residuals[i] * residuals[i];
    }
    loss /= n as f64;

    // Raw gradient via chain rule through the soft-rank Jacobian.
    let mut raw_gradient = vec![0.0; n];
    for k in 0..n {
        let mut grad_k = 0.0;
        for i in 0..n {
            let jacobian_ik = if i == k {
                let mut s = 0.0;
                for j in 0..n {
                    if j != i {
                        s += sigmoid_derivative((predictions[j] - predictions[i]) / temperature);
                    }
                }
                s / temperature
            } else {
                -sigmoid_derivative((predictions[k] - predictions[i]) / temperature) / temperature
            };
            grad_k += residuals[i] * jacobian_ik;
        }
        raw_gradient[k] = 2.0 * grad_k / n as f64;
    }

    // Newton preconditioning.
    let hessian_diag = soft_rank_hessian_diag(predictions, temperature).expect("valid input");
    let damping = 1e-8;
    let newton_grad = damped_newton_gradient(&raw_gradient, &hessian_diag, damping);

    (loss, newton_grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn newton_gradients_smaller_than_raw_when_curvature_high() {
        let predictions = vec![0.5, 0.2, 0.8, 0.1, 0.9];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let temperature = 0.1;

        let n = predictions.len();
        let pred_ranks = soft_rank(&predictions, temperature).unwrap();
        let target_ranks = soft_rank(&targets, temperature).unwrap();

        let mut residuals = vec![0.0; n];
        for i in 0..n {
            residuals[i] = pred_ranks[i] - target_ranks[i];
        }

        let mut raw_gradient = vec![0.0; n];
        for k in 0..n {
            let mut grad_k = 0.0;
            for i in 0..n {
                let jacobian_ik = if i == k {
                    let mut s = 0.0;
                    for j in 0..n {
                        if j != i {
                            s +=
                                sigmoid_derivative((predictions[j] - predictions[i]) / temperature);
                        }
                    }
                    s / temperature
                } else {
                    -sigmoid_derivative((predictions[k] - predictions[i]) / temperature)
                        / temperature
                };
                grad_k += residuals[i] * jacobian_ik;
            }
            raw_gradient[k] = 2.0 * grad_k / n as f64;
        }

        let (_, newton_grad) = newton_soft_rank_loss(&predictions, &targets, temperature);

        let raw_norm = l2_norm(&raw_gradient);
        let newton_norm = l2_norm(&newton_grad);

        assert!(
            newton_norm < raw_norm,
            "Newton norm ({newton_norm:.6}) should be < raw norm ({raw_norm:.6})"
        );
    }

    #[test]
    fn damping_prevents_division_by_zero() {
        let gradient = vec![1.0, 2.0, 3.0];
        let hessian_diag = vec![0.0, 0.0, 0.0];
        let damping = 1.0;

        let result = damped_newton_gradient(&gradient, &hessian_diag, damping);
        for (r, g) in result.iter().zip(gradient.iter()) {
            assert!((r - g).abs() < 1e-10);
        }
    }

    #[test]
    fn high_temperature_hessian_approximately_constant() {
        let x = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let temperature = 100.0;

        let hessian = soft_rank_hessian_diag(&x, temperature).unwrap();

        let mean_h: f64 = hessian.iter().sum::<f64>() / hessian.len() as f64;
        for (i, &h) in hessian.iter().enumerate() {
            let rel_diff = (h - mean_h).abs() / mean_h;
            assert!(
                rel_diff < 0.01,
                "Hessian entry {i} ({h:.6}) deviates from mean ({mean_h:.6})"
            );
        }

        let expected = (x.len() - 1) as f64 * 0.25 / temperature;
        assert!(
            (mean_h - expected).abs() / expected < 0.01,
            "mean Hessian ({mean_h:.6}) should be ~ (n-1)*0.25/tau ({expected:.6})"
        );
    }

    #[test]
    fn soft_rank_loss_returns_finite() {
        let predictions = vec![0.3, 0.7, 0.1, 0.9];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let (loss, grad) = newton_soft_rank_loss(&predictions, &targets, 0.5);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
        for g in &grad {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn perfect_ranking_has_zero_loss() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![10.0, 20.0, 30.0, 40.0];

        let (loss, _) = newton_soft_rank_loss(&predictions, &targets, 0.1);
        assert!(loss < 1e-6, "loss should be near zero, got {loss}");
    }
}
