//! Soft ranking shootout: comparing differentiable ranking methods from fynch and rankit.
//!
//! Generates synthetic data with known ground-truth rankings, then measures how
//! closely each soft ranking method recovers the true order and computes
//! learning-to-rank losses from both crates.
//!
//! Run: `cargo run --example soft_rank_shootout`

fn main() {
    // -- 1. Ground truth: 10 items with known relevance (higher = more relevant) ---------

    let relevance = [9.0, 7.5, 6.0, 5.5, 4.0, 3.5, 2.0, 1.5, 1.0, 0.5];
    let n = relevance.len();

    // Ascending true ranks: 1 = smallest value, n = largest value.
    // Used by fynch::sinkhorn_rank (which assigns low rank to low values).
    let true_ranks_asc: Vec<f64> = {
        let mut indexed: Vec<(usize, f64)> = relevance.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut ranks = vec![0.0; n];
        for (pos, &(orig_idx, _)) in indexed.iter().enumerate() {
            ranks[orig_idx] = (pos + 1) as f64;
        }
        ranks
    };

    // Descending true ranks (IR convention): 1 = highest value (best), n = lowest.
    // Used by fynch::soft_rank (which assigns rank 1 to the largest score).
    let true_ranks_desc: Vec<f64> = true_ranks_asc
        .iter()
        .map(|&r| (n as f64 + 1.0) - r)
        .collect();

    // 0-indexed ascending: 0 = smallest, n-1 = largest.
    // Used by rankit methods.
    let true_ranks_0idx: Vec<f64> = true_ranks_asc.iter().map(|r| r - 1.0).collect();

    // -- 2. Predicted scores: relevance + deliberate noise --------------------------------

    let noise = [0.3, -0.8, 1.2, -0.5, 0.7, -1.0, 0.4, 0.1, -0.3, 0.6];
    let predicted: Vec<f64> = relevance
        .iter()
        .zip(noise.iter())
        .map(|(r, n)| r + n)
        .collect();

    println!("Soft Rank Shootout");
    println!("==================");
    println!();
    println!("Items: {n}");
    println!("Relevance:  [{}]", fmt_vec(&relevance, 1));
    println!("Predicted:  [{}]", fmt_vec(&predicted, 1));
    println!("True ranks (desc): [{}]", fmt_vec(&true_ranks_desc, 0));
    println!();

    // -- 3. Compute soft ranks from 5 methods ---------------------------------------------

    let tau = 0.5; // temperature / regularization for fynch methods
    let alpha = 5.0; // regularization strength for rankit methods

    // fynch methods return 1-indexed ranks in [1, n].
    let fynch_sigmoid = fynch::soft_rank(&predicted, tau).expect("soft_rank");
    let fynch_sinkhorn = fynch::sinkhorn_rank(&predicted, tau).expect("sinkhorn_rank");

    // rankit methods return 0-indexed ranks in [0, n-1].
    let rankit_sigmoid = rankit::soft_rank(&predicted, alpha);
    let rankit_neural = rankit::methods::soft_rank_neural_sort(&predicted, tau);
    let rankit_prob = rankit::methods::soft_rank_probabilistic(&predicted, tau);

    // -- 4. Spearman rank correlation against true ranking --------------------------------

    println!("Ranking accuracy (Spearman rho vs true ranks)");
    println!("----------------------------------------------");
    println!("{:<28} {:>8}  {:>8}  convention", "Method", "rho", "1-rho");
    println!("{}", "-".repeat(62));

    // Each entry: (name, computed ranks, ground truth in matching convention, label).
    //
    // fynch::soft_rank uses IR convention: rank 1 = highest score.
    // fynch::sinkhorn_rank uses ascending convention: rank 1 = lowest score, rank n = highest.
    // rankit methods use 0-indexed ascending: 0 = lowest, n-1 = highest.
    let methods: Vec<(&str, &[f64], &[f64], &str)> = vec![
        (
            "fynch::soft_rank",
            &fynch_sigmoid,
            &true_ranks_desc,
            "[1,n] desc",
        ),
        (
            "fynch::sinkhorn_rank",
            &fynch_sinkhorn,
            &true_ranks_asc,
            "[1,n] asc",
        ),
        (
            "rankit::soft_rank",
            &rankit_sigmoid,
            &true_ranks_0idx,
            "[0,n-1] asc",
        ),
        (
            "rankit::neural_sort",
            &rankit_neural,
            &true_ranks_0idx,
            "[0,n-1] asc",
        ),
        (
            "rankit::probabilistic",
            &rankit_prob,
            &true_ranks_0idx,
            "[0,n-1] asc",
        ),
    ];

    for (name, ranks, truth, convention) in &methods {
        let rho = spearman_rho(ranks, truth);
        println!(
            "{:<28} {:>8.4}  {:>8.4}  {}",
            name,
            rho,
            1.0 - rho,
            convention
        );
    }

    // -- Bonus: fast_soft_sort (returns sorted values, not ranks) -------------------------

    let fss = fynch::fast_soft_sort(&predicted, tau);
    println!();
    println!(
        "fynch::fast_soft_sort output (sorted values, not ranks): [{}]",
        fmt_vec(&fss, 2)
    );

    // -- 5. Learning-to-rank losses -------------------------------------------------------

    println!();
    println!("Learning-to-rank losses");
    println!("-----------------------");

    // fynch losses (temperature-based)
    let loss_temp = 1.0;
    let fynch_spearman = fynch::loss::spearman_loss(&predicted, &relevance, loss_temp);
    let fynch_listnet = fynch::loss::listnet_loss(&predicted, &relevance, loss_temp);

    println!("{:<36} {:>10.6}", "fynch::spearman_loss", fynch_spearman);
    println!("{:<36} {:>10.6}", "fynch::listnet_loss", fynch_listnet);

    // rankit losses (regularization-strength-based)
    let reg = 5.0;
    let rankit_approx_ndcg = rankit::losses::approx_ndcg(&predicted, &relevance, reg, Some(n));
    let rankit_lambda = rankit::losses::lambda_loss(&predicted, &relevance, Some(n));

    println!(
        "{:<36} {:>10.6}  (higher=better)",
        "rankit::approx_ndcg", rankit_approx_ndcg
    );
    println!("{:<36} {:>10.6}", "rankit::lambda_loss", rankit_lambda);

    // -- 6. Summary table -----------------------------------------------------------------

    println!();
    println!("Summary");
    println!("-------");
    println!("fynch provides two soft-rank families: sigmoid (O(n^2)) and Sinkhorn OT");
    println!("(O(n^2 * iter)), plus fast_soft_sort (O(n log n)) for sorted values.");
    println!("rankit offers four sigmoid variants (sigmoid, NeuralSort, probabilistic,");
    println!("SmoothI) with LTR losses like ApproxNDCG and LambdaLoss.");
    println!();
    println!("Key difference: fynch ranks are 1-indexed [1, n]; rankit ranks are");
    println!("0-indexed [0, n-1]. Both preserve ordering; the offset is a convention.");
}

/// Pearson correlation between two vectors (used as Spearman rho on rank vectors).
fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len());
    if n < 2 {
        return 0.0;
    }
    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    num / denom
}

/// Format a float slice for display.
fn fmt_vec(v: &[f64], decimals: usize) -> String {
    v.iter()
        .map(|x| format!("{x:.decimals$}"))
        .collect::<Vec<_>>()
        .join(", ")
}
