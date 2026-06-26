# fynch

[![crates.io](https://img.shields.io/crates/v/fynch.svg)](https://crates.io/crates/fynch)
[![Documentation](https://docs.rs/fynch/badge.svg)](https://docs.rs/fynch)
[![CI](https://github.com/arclabs561/fynch/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/fynch/actions/workflows/ci.yml)

Differentiable sorting and ranking: PAVA isotonic regression,
Fenchel-Young losses, and O(n log n) FastSoftSort.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/fynch) | [docs.rs](https://docs.rs/fynch)

## Quickstart

```toml
[dependencies]
fynch = "0.2.1"
```

```rust
use fynch::{entmax, pava, soft_rank};

let theta = [2.0, 1.0, 0.1];
let p = entmax(&theta, 1.5);

let y = [3.0, 1.0, 2.0, 5.0, 4.0];
let isotonic = pava(&y);
let r = soft_rank(&y, 1.0);

println!("p={p:?}\nisotonic={isotonic:?}\nsoft_rank={r:?}");
```

## Examples

Runnable examples live in [`examples/`](examples/):

- `soft_rank_shootout` compares differentiable ranking methods from fynch and rankit on data with a known ground-truth order, the building block for gradient-based learning-to-rank where a hard sort would block backpropagation.
