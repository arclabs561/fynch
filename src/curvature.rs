//! Gradients and diagonal curvature for the pairwise-sigmoid soft-rank loss.
//!
//! The helpers here differentiate this crate's [`crate::soft_rank`] formula
//! directly. They do not implement the surrogate construction from the
//! Newton Losses paper.
//!
//! # Example
//!
//! ```rust
//! use fynch::curvature::{newton_soft_rank_loss, damped_newton_gradient};
//!
//! let predictions = [0.3, 0.7, 0.1, 0.9];
//! let targets = [1.0, 2.0, 3.0, 4.0];
//! let (loss, direction) = newton_soft_rank_loss(&predictions, &targets, 0.5)?;
//! assert!(loss >= 0.0);
//! # Ok::<(), fynch::Error>(())
//! ```

use crate::sigmoid::{sigmoid, sigmoid_derivative};
use crate::{soft_rank, Error, Result};

fn validate_input(x: &[f64], temperature: f64) -> Result<()> {
    if x.is_empty() {
        return Err(Error::EmptyInput);
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(Error::InvalidTemperature(temperature));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

/// Compute the magnitude of the diagonal soft-rank Jacobian.
///
/// The true diagonal Jacobian is the negation of this value. This function
/// retains its original name for source compatibility; it is not a Hessian.
#[deprecated(
    since = "0.3.3",
    note = "this is a Jacobian-diagonal magnitude, not a Hessian; use soft_rank_jacobian_diag_magnitude"
)]
pub fn soft_rank_hessian_diag(x: &[f64], temperature: f64) -> Result<Vec<f64>> {
    soft_rank_jacobian_diag_magnitude(x, temperature)
}

/// Compute the magnitude of the diagonal soft-rank Jacobian.
pub fn soft_rank_jacobian_diag_magnitude(x: &[f64], temperature: f64) -> Result<Vec<f64>> {
    validate_input(x, temperature)?;

    let n = x.len();
    let mut sensitivity = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let z = (x[j] - x[i]) / temperature;
                sensitivity[i] += sigmoid_derivative(z);
            }
        }
        sensitivity[i] /= temperature;
    }

    Ok(sensitivity)
}

/// Divide a gradient by diagonal curvature plus damping.
///
/// Prevents division by near-zero curvature via the damping term. When
/// `damping` is large relative to `h`, this recovers standard gradient
/// descent. When curvature is high, the gradient is scaled down.
/// The function retains its original name for source compatibility.
pub fn damped_newton_gradient(gradient: &[f64], hessian_diag: &[f64], damping: f64) -> Vec<f64> {
    gradient
        .iter()
        .zip(hessian_diag.iter())
        .map(|(&g, &h)| g / (h + damping))
        .collect()
}

/// Compute the soft-rank squared loss and its exact gradient.
///
/// Both predictions and targets are converted to pairwise-sigmoid soft ranks
/// at the same temperature. The loss is their mean squared difference.
pub fn soft_rank_loss_gradient(
    predictions: &[f64],
    targets: &[f64],
    temperature: f64,
) -> Result<(f64, Vec<f64>)> {
    let n = predictions.len();
    if n != targets.len() {
        return Err(Error::LengthMismatch(n, targets.len()));
    }
    validate_input(predictions, temperature)?;
    validate_input(targets, temperature)?;

    let pred_ranks = soft_rank(predictions, temperature)?;
    let target_ranks = soft_rank(targets, temperature)?;
    let residuals: Vec<f64> = pred_ranks
        .iter()
        .zip(&target_ranks)
        .map(|(prediction, target)| prediction - target)
        .collect();
    let loss = residuals
        .iter()
        .map(|residual| residual * residual)
        .sum::<f64>()
        / n as f64;

    let mut gradient = vec![0.0; n];
    for k in 0..n {
        for i in 0..n {
            let jacobian = if i == k {
                let diagonal_magnitude: f64 = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| sigmoid_derivative((predictions[j] - predictions[i]) / temperature))
                    .sum();
                -diagonal_magnitude / temperature
            } else {
                sigmoid_derivative((predictions[k] - predictions[i]) / temperature) / temperature
            };
            gradient[k] += residuals[i] * jacobian;
        }
        gradient[k] *= 2.0 / n as f64;
    }

    Ok((loss, gradient))
}

/// Compute the exact diagonal Hessian of the soft-rank squared loss.
pub fn soft_rank_loss_hessian_diag(
    predictions: &[f64],
    targets: &[f64],
    temperature: f64,
) -> Result<Vec<f64>> {
    let n = predictions.len();
    if n != targets.len() {
        return Err(Error::LengthMismatch(n, targets.len()));
    }
    validate_input(predictions, temperature)?;
    validate_input(targets, temperature)?;
    let pred_ranks = soft_rank(predictions, temperature)?;
    let target_ranks = soft_rank(targets, temperature)?;
    let residuals: Vec<f64> = pred_ranks
        .iter()
        .zip(target_ranks)
        .map(|(prediction, target)| prediction - target)
        .collect();
    let inv_temperature_sq = 1.0 / (temperature * temperature);
    let mut hessian_diag = vec![0.0; n];

    for k in 0..n {
        let mut jacobian_sq_sum = 0.0;
        let mut residual_second_sum = 0.0;
        for i in 0..n {
            let (jacobian, second_derivative) = if i == k {
                let mut first = 0.0;
                let mut second = 0.0;
                for j in 0..n {
                    if j != i {
                        let z = (predictions[j] - predictions[i]) / temperature;
                        let derivative = sigmoid_derivative(z);
                        first += derivative;
                        second += derivative * (1.0 - 2.0 * sigmoid(z));
                    }
                }
                (-first / temperature, second * inv_temperature_sq)
            } else {
                let z = (predictions[k] - predictions[i]) / temperature;
                let derivative = sigmoid_derivative(z);
                (
                    derivative / temperature,
                    derivative * (1.0 - 2.0 * sigmoid(z)) * inv_temperature_sq,
                )
            };
            jacobian_sq_sum += jacobian * jacobian;
            residual_second_sum += residuals[i] * second_derivative;
        }
        hessian_diag[k] = 2.0 * (jacobian_sq_sum + residual_second_sum) / n as f64;
    }

    Ok(hessian_diag)
}

/// Soft-rank squared loss with a diagonally preconditioned descent direction.
///
/// Computes the MSE between soft ranks of predictions and targets, then
/// divides its exact gradient by the absolute diagonal curvature of that loss
/// plus a small damping term. The returned direction is not the raw gradient.
///
/// Returns `(loss, preconditioned_direction)`.
pub fn newton_soft_rank_loss(
    predictions: &[f64],
    targets: &[f64],
    temperature: f64,
) -> Result<(f64, Vec<f64>)> {
    let n = predictions.len();
    if n != targets.len() {
        return Err(Error::LengthMismatch(n, targets.len()));
    }
    validate_input(predictions, temperature)?;
    validate_input(targets, temperature)?;

    let (loss, raw_gradient) = soft_rank_loss_gradient(predictions, targets, temperature)?;

    // Exact diagonal of the MSE Hessian. Its residual term can be negative, so
    // use its magnitude to keep the returned vector a descent direction.
    let loss_hessian_diag = soft_rank_loss_hessian_diag(predictions, targets, temperature)?;
    let damping = 1e-8;
    let curvature_magnitude: Vec<f64> = loss_hessian_diag.iter().map(|h| h.abs()).collect();
    let direction = damped_newton_gradient(&raw_gradient, &curvature_magnitude, damping);

    Ok((loss, direction))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loss_only(predictions: &[f64], targets: &[f64], temperature: f64) -> f64 {
        let predicted = soft_rank(predictions, temperature).unwrap();
        let target = soft_rank(targets, temperature).unwrap();
        predicted
            .iter()
            .zip(target)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / predictions.len() as f64
    }

    fn assert_gradient_matches_finite_difference(
        predictions: &[f64],
        targets: &[f64],
        temperature: f64,
    ) {
        let (_, analytic) = soft_rank_loss_gradient(predictions, targets, temperature).unwrap();
        let step = 1e-6;
        for coordinate in 0..predictions.len() {
            let mut plus = predictions.to_vec();
            let mut minus = predictions.to_vec();
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let numeric = (loss_only(&plus, targets, temperature)
                - loss_only(&minus, targets, temperature))
                / (2.0 * step);
            assert!(
                (analytic[coordinate] - numeric).abs() < 2e-8,
                "coordinate {coordinate}: analytic={}, numeric={numeric}",
                analytic[coordinate]
            );
        }
    }

    #[test]
    fn gradient_matches_independent_central_differences() {
        assert_gradient_matches_finite_difference(&[0.2, -0.7], &[1.1, -0.4], 0.6);
        assert_gradient_matches_finite_difference(
            &[0.5, -1.2, 0.1, 2.0],
            &[-0.4, 0.8, 1.5, -1.0],
            0.9,
        );
        assert_gradient_matches_finite_difference(
            &[-2.1, 0.3, 0.31, 1.7, -0.8],
            &[0.4, -1.3, 2.2, 0.0, 0.9],
            0.27,
        );
    }

    #[test]
    fn loss_hessian_diagonal_matches_central_differences() {
        let predictions = [0.5, -1.2, 0.1, 2.0];
        let targets = [-0.4, 0.8, 1.5, -1.0];
        let temperature = 0.9;
        let analytic = soft_rank_loss_hessian_diag(&predictions, &targets, temperature).unwrap();
        let step = 1e-4;
        let center = loss_only(&predictions, &targets, temperature);
        for coordinate in 0..predictions.len() {
            let mut plus = predictions;
            let mut minus = predictions;
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let numeric = (loss_only(&plus, &targets, temperature) - 2.0 * center
                + loss_only(&minus, &targets, temperature))
                / (step * step);
            assert!(
                (analytic[coordinate] - numeric).abs() < 2e-6,
                "coordinate {coordinate}: analytic={}, numeric={numeric}",
                analytic[coordinate]
            );
        }
    }

    #[test]
    fn preconditioned_direction_uses_finite_difference_gradient() {
        let predictions = [0.5, -1.2, 0.1, 2.0];
        let targets = [-0.4, 0.8, 1.5, -1.0];
        let temperature = 0.9;
        let (_, direction) = newton_soft_rank_loss(&predictions, &targets, temperature).unwrap();
        let curvature = soft_rank_loss_hessian_diag(&predictions, &targets, temperature).unwrap();
        let step = 1e-6;
        for coordinate in 0..predictions.len() {
            let mut plus = predictions;
            let mut minus = predictions;
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let numeric_gradient = (loss_only(&plus, &targets, temperature)
                - loss_only(&minus, &targets, temperature))
                / (2.0 * step);
            let expected = numeric_gradient / (curvature[coordinate].abs() + 1e-8);
            assert!(
                (direction[coordinate] - expected).abs() < 2e-7,
                "coordinate {coordinate}: direction={}, expected={expected}",
                direction[coordinate]
            );
        }
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
    fn high_temperature_jacobian_magnitude_is_approximately_constant() {
        let x = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let temperature = 100.0;

        let sensitivity = soft_rank_jacobian_diag_magnitude(&x, temperature).unwrap();

        let mean_h: f64 = sensitivity.iter().sum::<f64>() / sensitivity.len() as f64;
        for (i, &h) in sensitivity.iter().enumerate() {
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

        let (loss, grad) = newton_soft_rank_loss(&predictions, &targets, 0.5).unwrap();
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

        let (loss, _) = newton_soft_rank_loss(&predictions, &targets, 0.1).unwrap();
        assert!(loss < 1e-6, "loss should be near zero, got {loss}");
    }

    #[test]
    fn rejects_invalid_domains() {
        assert!(matches!(
            soft_rank_loss_gradient(&[0.0], &[0.0, 1.0], 1.0),
            Err(Error::LengthMismatch(1, 2))
        ));
        assert!(matches!(
            newton_soft_rank_loss(&[0.0, f64::NAN], &[0.0, 1.0], 1.0),
            Err(Error::NonFiniteInput)
        ));
        assert!(matches!(
            soft_rank_jacobian_diag_magnitude(&[0.0, 1.0], f64::INFINITY),
            Err(Error::InvalidTemperature(value)) if value.is_infinite()
        ));
    }
}
