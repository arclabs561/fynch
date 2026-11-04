//! fynch's log-sum-exp calls now go to logp. These are the private copies they
//! replaced, kept here as the reference: on finite inputs the results must be
//! bitwise equal, and the empty and all-`-inf` cases must agree.

fn old_slice(theta: &[f64]) -> f64 {
    if theta.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = theta.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max + theta.iter().map(|&t| (t - max).exp()).sum::<f64>().ln()
}

fn old_pair(a: f64, b: f64) -> f64 {
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

// Small xorshift so the test needs no dev-dependency.
fn values(seed: &mut u64, n: usize, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * scale
        })
        .collect()
}

#[test]
fn slice_form_matches_the_old_copy_bitwise() {
    let mut seed = 0x9e37_79b9_7f4a_7c15;
    for n in [1, 2, 3, 7, 64, 1000] {
        for scale in [1e-3, 1.0, 50.0, 1e3] {
            let v = values(&mut seed, n, scale);
            assert_eq!(logp::log_sum_exp(&v).to_bits(), old_slice(&v).to_bits());
        }
    }
    assert_eq!(logp::log_sum_exp(&[]), old_slice(&[]));
    let all_neg_inf = [f64::NEG_INFINITY; 3];
    assert_eq!(logp::log_sum_exp(&all_neg_inf), old_slice(&all_neg_inf));
}

#[test]
fn pair_form_matches_the_old_copy_bitwise_including_the_fold() {
    let mut seed = 0x2545_f491_4f6c_dd1d;
    for scale in [1e-3, 1.0, 50.0, 1e3] {
        let v = values(&mut seed, 200, scale);
        for w in v.windows(2) {
            assert_eq!(
                logp::log_sum_exp2(w[0], w[1]).to_bits(),
                old_pair(w[0], w[1]).to_bits()
            );
        }
        // `loss.rs` folds from -inf, so the -inf seed must behave the same.
        let new = v
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, logp::log_sum_exp2);
        let old = v.iter().copied().fold(f64::NEG_INFINITY, old_pair);
        assert_eq!(new.to_bits(), old.to_bits());
    }
    assert_eq!(logp::log_sum_exp2(f64::NEG_INFINITY, 2.5), 2.5);
    assert_eq!(logp::log_sum_exp2(-1.0, f64::NEG_INFINITY), -1.0);
    assert_eq!(
        logp::log_sum_exp2(f64::NEG_INFINITY, f64::NEG_INFINITY),
        f64::NEG_INFINITY
    );
}
