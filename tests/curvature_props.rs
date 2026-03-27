//! Property tests for curvature module.

use fynch::curvature::{damped_newton_gradient, newton_soft_rank_loss, soft_rank_hessian_diag};
use proptest::prelude::*;

proptest! {
    /// Hessian diagonal entries should be non-negative (sum of sigmoid derivatives).
    #[test]
    fn prop_hessian_nonnegative(
        len in 2usize..=10,
        temp in 0.01f64..=10.0,
    ) {
        let x: Vec<f64> = (0..len).map(|i| (i as f64 * 0.3).sin()).collect();
        let h = soft_rank_hessian_diag(&x, temp).unwrap();

        for (i, &hi) in h.iter().enumerate() {
            prop_assert!(hi >= 0.0, "Hessian diag[{i}] = {hi} < 0");
            prop_assert!(hi.is_finite(), "Hessian diag[{i}] is not finite");
        }
    }

    /// Newton loss should be non-negative.
    #[test]
    fn prop_loss_nonnegative(
        len in 2usize..=8,
        temp in 0.1f64..=5.0,
    ) {
        let pred: Vec<f64> = (0..len).map(|i| (i as f64 * 0.5).sin()).collect();
        let target: Vec<f64> = (0..len).map(|i| i as f64).collect();

        let (loss, grad) = newton_soft_rank_loss(&pred, &target, temp);
        prop_assert!(loss >= 0.0, "Loss = {loss} < 0");
        prop_assert!(loss.is_finite(), "Loss is not finite");
        for (i, &g) in grad.iter().enumerate() {
            prop_assert!(g.is_finite(), "Gradient[{i}] is not finite");
        }
    }

    /// Damped Newton gradient should be finite for any positive damping.
    #[test]
    fn prop_damped_gradient_finite(
        len in 1usize..=20,
        damping in 1e-10f64..=10.0,
    ) {
        let grad: Vec<f64> = (0..len).map(|i| (i as f64 - 5.0) * 0.3).collect();
        let hess: Vec<f64> = (0..len).map(|i| (i as f64 * 0.1).abs()).collect();

        let result = damped_newton_gradient(&grad, &hess, damping);
        for (i, &r) in result.iter().enumerate() {
            prop_assert!(r.is_finite(), "Result[{i}] = {r} is not finite");
        }
    }

    /// Perfect ranking (same ordering) should have near-zero loss.
    #[test]
    fn prop_perfect_ranking_zero_loss(
        len in 2usize..=8,
        temp in 0.05f64..=1.0,
    ) {
        // Predictions and targets in same order
        let pred: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let target: Vec<f64> = (0..len).map(|i| (i as f64) * 10.0).collect();

        let (loss, _) = newton_soft_rank_loss(&pred, &target, temp);
        // At high temperature, soft ranks are smoother, so "perfect" ranking still has residual loss
        let threshold = if temp > 0.5 { 0.5 } else { 0.1 };
        prop_assert!(loss < threshold, "Loss should be near zero for matching orders: {loss} (temp={temp})");
    }
}
