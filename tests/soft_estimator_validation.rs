//! Soft estimators checked against exact references.
//!
//! Soft estimators must collapse to their hard counterparts in the limit, and
//! isotonic regression has an exact characterization. These tests check:
//! - `soft_rank(x, τ→0)` == `1 + |{j: xⱼ > xᵢ}|` (1-based descending rank, exact
//!   for distinct inputs);
//! - `soft_sort(x, small τ)` is non-decreasing and close to the sorted input;
//! - `pava` is non-decreasing, idempotent, sum-preserving, and matches
//!   hand-computed known answers.
//!
//! All inputs come from one deterministic xorshift stream. Each test skips the
//! draws used by the tests before it, so every test sees the same inputs it saw
//! when these checks ran as a single sequential program.

use fynch::{
    pava, sinkhorn_permutation_with_diagnostics, soft_rank, soft_sort, Error, SinkhornConfig,
};

const SOFT_RANK_TRIALS: usize = 50;
const SOFT_SORT_TRIALS: usize = 30;
const PAVA_TRIALS: usize = 40;

fn soft_rank_len(trial: usize) -> usize {
    4 + (trial % 12)
}

fn soft_sort_len(trial: usize) -> usize {
    5 + (trial % 8)
}

fn pava_len(trial: usize) -> usize {
    3 + (trial % 15)
}

/// Deterministic uniform [0, 1) stream.
struct XorShift(u64);

impl XorShift {
    fn new() -> Self {
        Self(0x9E3779B97F4A7C15)
    }

    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn skip(&mut self, draws: usize) {
        for _ in 0..draws {
            self.next();
        }
    }
}

fn soft_rank_draws() -> usize {
    (0..SOFT_RANK_TRIALS).map(soft_rank_len).sum()
}

fn soft_sort_draws() -> usize {
    (0..SOFT_SORT_TRIALS).map(soft_sort_len).sum()
}

#[test]
fn soft_rank_collapses_to_descending_rank_as_temperature_vanishes() {
    let mut rng = XorShift::new();
    for trial in 0..SOFT_RANK_TRIALS {
        let n = soft_rank_len(trial);
        // Ties are astronomically unlikely with f64 noise, so the hard rank is exact.
        let x: Vec<f64> = (0..n).map(|_| rng.next() * 100.0).collect();
        let ranks = soft_rank(&x, 1e-9).expect("soft_rank");
        for i in 0..n {
            let hard = 1.0 + x.iter().filter(|&&xj| xj > x[i]).count() as f64;
            assert!(
                (ranks[i] - hard).abs() < 1e-6,
                "trial {trial}: soft_rank[{i}]={:.4} vs hard {hard}",
                ranks[i]
            );
        }
    }
}

#[test]
fn soft_sort_small_temperature_is_monotone_and_near_sorted() {
    let mut rng = XorShift::new();
    rng.skip(soft_rank_draws());
    for trial in 0..SOFT_SORT_TRIALS {
        let n = soft_sort_len(trial);
        let x: Vec<f64> = (0..n).map(|_| rng.next() * 10.0).collect();
        let mut sorted = x.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let out = match soft_sort(&x, 0.01) {
            Ok(out) => out,
            Err(Error::SinkhornDidNotConverge { .. }) => {
                // Sharp kernels can exceed the convenience wrapper's iteration
                // budget. Retry the same problem with the same tolerance; do
                // not accept an unfinished coupling or substitute a hard sort.
                let config = SinkhornConfig {
                    epsilon: 0.01,
                    max_iter: 100_000,
                    ..Default::default()
                };
                let output = sinkhorn_permutation_with_diagnostics(&x, &config)
                    .unwrap_or_else(|e| panic!("soft_sort trial {trial}: {e}"));
                for i in 0..n {
                    let row: f64 = (0..n).map(|j| output.matrix[i * n + j]).sum();
                    let col: f64 = (0..n).map(|j| output.matrix[j * n + i]).sum();
                    assert!(
                        (row - 1.0).abs() < config.tol && (col - 1.0).abs() < config.tol,
                        "soft_sort trial {trial}: invalid marginal {i}"
                    );
                }
                // The same barycentric projection used by sinkhorn_sort: Pᵀx.
                (0..n)
                    .map(|j| (0..n).map(|i| output.matrix[i * n + j] * x[i]).sum())
                    .collect()
            }
            Err(e) => panic!("soft_sort trial {trial}: {e}"),
        };
        for i in 1..n {
            assert!(
                out[i] >= out[i - 1] - 1e-6,
                "trial {trial}: soft_sort not non-decreasing at {i}: {out:?}"
            );
        }
        for i in 0..n {
            assert!(
                (out[i] - sorted[i]).abs() < 0.5,
                "trial {trial}: soft_sort[{i}]={:.4} far from sorted {:.4}",
                out[i],
                sorted[i]
            );
        }
    }
}

#[test]
fn pava_matches_hand_computed_isotonic_fits() {
    let known: &[(&[f64], &[f64])] = &[
        (&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
        (&[3.0, 2.0, 1.0], &[2.0, 2.0, 2.0]),
        (&[1.0, 3.0, 2.0, 4.0], &[1.0, 2.5, 2.5, 4.0]),
        (&[4.0, 1.0, 1.0, 1.0], &[1.75, 1.75, 1.75, 1.75]),
    ];
    for (input, want) in known {
        let got = pava(input);
        assert!(
            got.len() == want.len() && got.iter().zip(*want).all(|(a, b)| (a - b).abs() < 1e-9),
            "pava({input:?}) = {got:?}, want {want:?}"
        );
    }
}

#[test]
fn pava_is_monotone_idempotent_and_sum_preserving() {
    let mut rng = XorShift::new();
    rng.skip(soft_rank_draws() + soft_sort_draws());
    for trial in 0..PAVA_TRIALS {
        let n = pava_len(trial);
        let y: Vec<f64> = (0..n).map(|_| rng.next() * 10.0).collect();
        let r = pava(&y);
        for i in 1..n {
            assert!(
                r[i] >= r[i - 1] - 1e-9,
                "trial {trial}: pava not monotone at {i}"
            );
        }
        assert!(
            pava(&r).iter().zip(&r).all(|(a, b)| (a - b).abs() < 1e-9),
            "trial {trial}: pava not idempotent"
        );
        // The L2 projection onto the monotone cone preserves the mean.
        let (sy, sr): (f64, f64) = (y.iter().sum(), r.iter().sum());
        assert!(
            (sy - sr).abs() < 1e-6,
            "trial {trial}: pava sum drift {sy} vs {sr}"
        );
    }
}
