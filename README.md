# fynch

[![crates.io](https://img.shields.io/crates/v/fynch.svg)](https://crates.io/crates/fynch)
[![Documentation](https://docs.rs/fynch/badge.svg)](https://docs.rs/fynch)

Differentiable sorting and ranking.

Dual-licensed under MIT or Apache-2.0.

## What it does

Sorting, ranking, and argmax are discontinuous: a small change in scores can
flip two ranks or move probability mass entirely onto a different element, so
their gradient is zero almost everywhere and undefined at the jumps. That blocks
training any model whose loss runs through a sort or an argmax. fynch provides
smoothed (differentiable) replacements that pass gradients: soft ranks and soft
sorts in place of argsort, and the entmax/sparsemax/softmax family in place of a
hard argmax. These come from the Fenchel-Young framework (Blondel, Martins,
Niculae 2020), which derives a prediction function and a matching convex loss
from one regularizer Omega, so cross-entropy, sparsemax, and entmax are the same
construction with different Omega. Typical uses are learning-to-rank, top-k
selection, attention with sparse weights, and any pipeline where a hard sort or
argmax sits between the model and the loss.

## Quickstart

```toml
[dependencies]
fynch = "0.3.2"
```

```rust
use fynch::fenchel::{entmax, softmax, sparsemax};
use fynch::{pava, soft_rank};

let theta = [2.0, 1.0, 0.1];

// Fenchel-Young predictions: dense, sparse, or tunable sparsity.
let dense = softmax(&theta); // sums to 1, all positive
let sparse = sparsemax(&theta); // exact zeros for low scores
let tunable = entmax(&theta, 1.5); // between the two

// Isotonic regression (PAVA): nearest non-decreasing fit.
let monotonic = pava(&[3.0, 1.0, 2.0, 5.0, 4.0]);

// Differentiable ranks: a continuous, backprop-friendly stand-in for argsort.
let ranks = soft_rank(&[0.5, 0.2, 0.8, 0.1], 0.1).unwrap();
```

Lower `temperature` makes `soft_rank` and `soft_sort` approach the hard
(discrete) result; higher temperature smooths them out.

## Modules

- `fenchel`: the generic framework (regularizers, prediction functions, losses).
- `sinkhorn`: entropic optimal transport for soft permutations.
- `lapsum`: LapSum unified soft sort, rank, and top-k.
- `loss`: learning-to-rank losses (Spearman, ListNet).
- `metrics`: IR evaluation (MRR, NDCG, Hits@k).

## Examples

Runnable examples live in [`examples/`](examples/):

- `soft_rank_shootout` compares fynch and rankit ranking methods on data with a
  known ground-truth order, measuring how closely each recovers the true ranks.
- `soft_estimator_validation` checks the soft estimators against exact
  references: `soft_rank` and `soft_sort` collapsing to their hard counterparts
  as temperature goes to zero, and PAVA against hand-computed isotonic fits.
