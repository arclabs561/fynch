//! The first weight in a two-score soft top-1 selection, at unit scale.
//! Pass the first score as a gap relative to the second score (zero).

use fynch::lapsum_soft_topk;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let gap: f64 = args.next().as_deref().unwrap_or("0").parse()?;
    if args.next().is_some() || !gap.is_finite() {
        return Err("usage: lapsum_tie [finite score gap]".into());
    }

    let weights = lapsum_soft_topk(&[gap, 0.0], 1, 1.0)?;
    println!("weights: {:.6} {:.6}", weights[0], weights[1]);

    // For two scores the threshold is their midpoint. Differentiating the
    // first weight gives half the unit Laplace density at the half-gap.
    let slope = 0.25 * (-0.5 * gap.abs()).exp();
    println!("first weight slope: {slope:.6}");

    let step = 1e-5;
    let lower = gap - step;
    let upper = gap + step;
    if lower < gap && gap < upper {
        let before = lapsum_soft_topk(&[lower, 0.0], 1, 1.0)?;
        let after = lapsum_soft_topk(&[upper, 0.0], 1, 1.0)?;
        let difference = (after[0] - before[0]) / (upper - lower);
        println!("finite difference: {difference:.6}");
    }
    Ok(())
}
