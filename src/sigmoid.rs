//! Sigmoid function for smooth approximations.
//!
//! The sigmoid function provides a smooth, differentiable approximation to the
//! step function. It's used throughout differentiable ranking as a soft comparison.
//!
//! # Properties
//!
//! - Range: (0, 1)
//! - σ(0) = 0.5
//! - σ(-x) = 1 - σ(x) (symmetry)
//! - Derivative: σ'(x) = σ(x)(1 - σ(x))

/// Sigmoid function: σ(x) = 1 / (1 + exp(-x))
///
/// Provides a smooth, differentiable approximation to the step function.
///
/// # Numerical Stability
///
/// Uses the numerically stable formulation:
/// - For x >= 0: Use 1 / (1 + exp(-x))
/// - For x < 0: Use exp(x) / (1 + exp(x))
/// - Clamps extreme values (|x| > 500) to prevent overflow
///
/// # Example
///
/// ```rust
/// use fynch::sigmoid::sigmoid;
///
/// assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
/// assert!(sigmoid(10.0) > 0.99);
/// assert!(sigmoid(-10.0) < 0.01);
/// ```
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    // Clamp extreme values to prevent overflow/underflow
    if x > 500.0 {
        return 1.0;
    }
    if x < -500.0 {
        return 0.0;
    }

    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
///
/// Maximum at x=0 where σ'(0) = 0.25.
///
/// # Example
///
/// ```rust
/// use fynch::sigmoid::sigmoid_derivative;
///
/// let d0 = sigmoid_derivative(0.0);
/// assert!((d0 - 0.25).abs() < 1e-10);  // maximum at x=0
///
/// // Derivative approaches 0 at extremes
/// assert!(sigmoid_derivative(10.0) < 0.001);
/// assert!(sigmoid_derivative(-10.0) < 0.001);
/// ```
#[inline]
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Log-sigmoid: log(σ(x)) = -log(1 + exp(-x))
///
/// More numerically stable than computing sigmoid then taking log.
///
/// # Example
///
/// ```rust
/// use fynch::sigmoid::log_sigmoid;
///
/// let ls = log_sigmoid(0.0);
/// assert!((ls - (-0.693147)).abs() < 1e-5);  // log(0.5) ≈ -0.693
/// ```
#[inline]
pub fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -softplus(-x)
    } else {
        x - softplus(x)
    }
}

/// Softplus: log(1 + exp(x))
///
/// Smooth approximation to ReLU.
///
/// # Example
///
/// ```rust
/// use fynch::sigmoid::softplus;
///
/// assert!(softplus(0.0) > 0.69 && softplus(0.0) < 0.70);  // log(2) ≈ 0.693
/// assert!(softplus(-100.0) < 1e-40);  // ≈ 0 for large negative
/// assert!((softplus(100.0) - 100.0).abs() < 1e-40);  // ≈ x for large positive
/// ```
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x // log(1 + exp(x)) ≈ x for large x
    } else if x < -20.0 {
        0.0 // log(1 + exp(x)) ≈ 0 for small x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_center() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_extremes() {
        assert!(sigmoid(100.0) > 0.999999);
        assert!(sigmoid(-100.0) < 0.000001);
        assert_eq!(sigmoid(600.0), 1.0);
        assert_eq!(sigmoid(-600.0), 0.0);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        for x in [-5.0, -1.0, 0.5, 2.0, 10.0] {
            let sym_diff = (sigmoid(x) + sigmoid(-x) - 1.0).abs();
            assert!(sym_diff < 1e-10, "symmetry failed for x={}", x);
        }
    }

    #[test]
    fn test_sigmoid_derivative() {
        // Max at 0
        assert!((sigmoid_derivative(0.0) - 0.25).abs() < 1e-10);

        // Decreases away from 0
        assert!(sigmoid_derivative(1.0) < sigmoid_derivative(0.0));
        assert!(sigmoid_derivative(-1.0) < sigmoid_derivative(0.0));
    }

    #[test]
    fn test_log_sigmoid() {
        // log(0.5) = -ln(2)
        let expected = -std::f64::consts::LN_2;
        assert!((log_sigmoid(0.0) - expected).abs() < 1e-10);

        // Should be negative for all x
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            assert!(log_sigmoid(x) < 0.0);
        }
    }

    #[test]
    fn test_softplus() {
        // log(2) at 0
        assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-10);

        // Positive for all x
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            assert!(softplus(x) > 0.0);
        }
    }
}
