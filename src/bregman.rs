//! Optional bridge from Fenchel-Young regularizers to `logp` Bregman generators.
//!
//! The bridge is deliberately narrow. `fynch::Shannon` and `fynch::SquaredL2`
//! have exact Bregman-generator counterparts in `logp`; Tsallis and SparseMAP
//! do not get placeholder implementations here.

use crate::fenchel::{Shannon, SquaredL2};

impl logp::BregmanGenerator for SquaredL2 {
    fn f(&self, x: &[f64]) -> logp::Result<f64> {
        <logp::SquaredL2 as logp::BregmanGenerator>::f(&logp::SquaredL2, x)
    }

    fn grad_into(&self, x: &[f64], out: &mut [f64]) -> logp::Result<()> {
        <logp::SquaredL2 as logp::BregmanGenerator>::grad_into(&logp::SquaredL2, x, out)
    }
}

impl logp::BregmanGenerator for Shannon {
    fn f(&self, x: &[f64]) -> logp::Result<f64> {
        <logp::NegEntropy as logp::BregmanGenerator>::f(&logp::NegEntropy, x)
    }

    fn grad_into(&self, x: &[f64], out: &mut [f64]) -> logp::Result<()> {
        <logp::NegEntropy as logp::BregmanGenerator>::grad_into(&logp::NegEntropy, x, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fenchel::Regularizer;

    #[test]
    fn squared_l2_matches_logp_generator() {
        let p = [1.0, 2.0, 3.0];
        let q = [1.5, 1.0, 2.5];

        let fynch = logp::bregman_divergence(&SquaredL2, &p, &q).unwrap();
        let expected = logp::bregman_divergence(&logp::SquaredL2, &p, &q).unwrap();

        assert!((fynch - expected).abs() < 1e-12);
    }

    #[test]
    fn shannon_matches_logp_negative_entropy_generator() {
        let p = [0.2, 0.3, 0.5];
        let q = [0.25, 0.25, 0.5];

        let fynch = logp::bregman_divergence(&Shannon, &p, &q).unwrap();
        let expected = logp::bregman_divergence(&logp::NegEntropy, &p, &q).unwrap();

        assert!((fynch - expected).abs() < 1e-12);
        assert!(
            (Shannon.omega(&p) - logp::BregmanGenerator::f(&Shannon, &p).unwrap()).abs() < 1e-12
        );
    }
}
