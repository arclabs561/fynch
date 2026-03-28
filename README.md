# fynch

[![crates.io](https://img.shields.io/crates/v/fynch.svg)](https://crates.io/crates/fynch)
[![Documentation](https://docs.rs/fynch/badge.svg)](https://docs.rs/fynch)
[![CI](https://github.com/arclabs561/fynch/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/fynch/actions/workflows/ci.yml)

Differentiable sorting and ranking.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/fynch) | [docs.rs](https://docs.rs/fynch)

## Quickstart

```toml
[dependencies]
fynch = "0.1.1"
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
