# fynch

Differentiable sorting and ranking primitives for structured prediction and isotonic regression.
Implements PAVA, Fenchel-Young losses, and entropy-regularized operators.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/fynch) | [docs.rs](https://docs.rs/fynch)

```rust
use fynch::{entmax, pava, soft_rank};

let theta = [2.0, 1.0, 0.1];
let p = entmax(&theta, 1.5);

let y = [3.0, 1.0, 2.0, 5.0, 4.0];
let isotonic = pava(&y);
let r = soft_rank(&y, 1.0);

println!("p={p:?}\nisotonic={isotonic:?}\nsoft_rank={r:?}");
```
