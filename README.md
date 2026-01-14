# fynch

Differentiable sorting and ranking: PAVA, Fenchel-Young losses, soft operators.

Dual-licensed under MIT or Apache-2.0.

```rust
use fynch::{entmax, pava, soft_rank};

let theta = [2.0, 1.0, 0.1];
let p = entmax(&theta, 1.5);

let y = [3.0, 1.0, 2.0, 5.0, 4.0];
let isotonic = pava(&y);
let r = soft_rank(&y, 1.0);

println!("p={p:?}\nisotonic={isotonic:?}\nsoft_rank={r:?}");
```

For more, see the crate docs.