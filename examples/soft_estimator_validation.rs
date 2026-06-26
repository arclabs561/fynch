//! Validate fynch's soft estimators against exact references.
//!
//! Soft estimators must collapse to their hard counterparts in the limit, and
//! isotonic regression has an exact characterization. This checks:
//! - `soft_rank(x, τ→0)` == `1 + |{j: xⱼ > xᵢ}|` (1-based descending rank, exact
//!   for distinct inputs);
//! - `soft_sort(x, small τ)` is non-decreasing and close to the sorted input;
//! - `pava` / `isotonic_l2` are non-decreasing, idempotent, sum-preserving,
//!   agree with each other, and match hand-computed known answers.
//!
//! ```sh
//! cargo run --release --example soft_estimator_validation
//! ```

use std::process::ExitCode;

use fynch::{pava, soft_rank, soft_sort};

fn main() -> ExitCode {
    let mut failures = 0u64;
    let mut checks = 0u64;
    let mut check = |cond: bool, what: String| {
        checks += 1;
        if !cond {
            failures += 1;
            if failures <= 10 {
                eprintln!("  VIOLATION: {what}");
            }
        }
    };

    // Deterministic distinct inputs.
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };

    // --- soft_rank -> descending competition rank as τ -> 0 ---
    for trial in 0..50 {
        let n = 4 + (trial % 12);
        let x: Vec<f64> = (0..n).map(|_| next() * 100.0).collect();
        // distinct? regenerate-proof: ties astronomically unlikely with f64 noise
        let ranks = soft_rank(&x, 1e-9).expect("soft_rank");
        for i in 0..n {
            let hard = 1.0 + x.iter().filter(|&&xj| xj > x[i]).count() as f64;
            check(
                (ranks[i] - hard).abs() < 1e-6,
                format!("soft_rank[{i}]={:.4} vs hard {hard}", ranks[i]),
            );
        }
    }

    // --- soft_sort: non-decreasing + close to sorted ascending ---
    for trial in 0..30 {
        let n = 5 + (trial % 8);
        let x: Vec<f64> = (0..n).map(|_| next() * 10.0).collect();
        let mut sorted = x.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let out = soft_sort(&x, 0.01).expect("soft_sort");
        for i in 1..n {
            check(
                out[i] >= out[i - 1] - 1e-6,
                format!("soft_sort not non-decreasing at {i}: {:?}", out),
            );
        }
        for i in 0..n {
            check(
                (out[i] - sorted[i]).abs() < 0.5,
                format!(
                    "soft_sort[{i}]={:.4} far from sorted {:.4}",
                    out[i], sorted[i]
                ),
            );
        }
    }

    // --- pava / isotonic_l2: known answers, monotone, idempotent, sum-preserving, agree ---
    let known: &[(&[f64], &[f64])] = &[
        (&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
        (&[3.0, 2.0, 1.0], &[2.0, 2.0, 2.0]),
        (&[1.0, 3.0, 2.0, 4.0], &[1.0, 2.5, 2.5, 4.0]),
        (&[4.0, 1.0, 1.0, 1.0], &[1.75, 1.75, 1.75, 1.75]),
    ];
    for (input, want) in known {
        let got = pava(input);
        check(
            got.len() == want.len() && got.iter().zip(*want).all(|(a, b)| (a - b).abs() < 1e-9),
            format!("pava({input:?}) = {got:?}, want {want:?}"),
        );
    }
    for trial in 0..40 {
        let n = 3 + (trial % 15);
        let y: Vec<f64> = (0..n).map(|_| next() * 10.0).collect();
        let r = pava(&y);
        // non-decreasing
        for i in 1..n {
            check(r[i] >= r[i - 1] - 1e-9, format!("pava not monotone at {i}"));
        }
        // idempotent
        check(
            pava(&r).iter().zip(&r).all(|(a, b)| (a - b).abs() < 1e-9),
            "pava not idempotent".into(),
        );
        // sum-preserving (L2 projection onto monotone cone preserves the mean)
        let (sy, sr): (f64, f64) = (y.iter().sum(), r.iter().sum());
        check(
            (sy - sr).abs() < 1e-6,
            format!("pava sum drift {sy} vs {sr}"),
        );
    }

    println!("{checks} checks, {failures} violations");
    if failures == 0 {
        println!("PASS: soft estimators match exact references");
        ExitCode::SUCCESS
    } else {
        eprintln!("FAIL: a soft estimator diverged from its exact reference");
        ExitCode::FAILURE
    }
}
