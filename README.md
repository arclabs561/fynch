# fynch

Fenchel-Young losses and differentiable sorting/ranking.

The name comes from **Fenchel-Young losses** (Blondel et al. 2020), a unifying
framework connecting prediction functions and loss functions through convex duality.

Dual-licensed under MIT or Apache-2.0.

```rust
use fynch::fenchel::{softmax, sparsemax, entmax};
use fynch::{pava, soft_rank};

// Fenchel-Young predictions
let theta = [2.0, 1.0, 0.1];
let p_soft = softmax(&theta);    // dense (cross-entropy)
let p_sparse = sparsemax(&theta); // sparse (simplex projection)
let p_ent = entmax(&theta, 1.5);  // tunable sparsity

// Differentiable sorting
let y = [3.0, 1.0, 2.0, 5.0, 4.0];
let monotonic = pava(&y);  // isotonic regression
```

## The Framework

Given a regularizer Ω, the Fenchel-Young loss is:

```
L_Ω(θ; y) = Ω*(θ) - ⟨θ, y⟩ + Ω(y)
```

| Regularizer Ω | Prediction | Loss | Sparsity |
|---------------|------------|------|----------|
| Shannon negentropy | softmax | cross-entropy | Dense |
| ½‖·‖² | sparsemax | sparsemax loss | Sparse |
| Tsallis α-entropy | α-entmax | entmax loss | Tunable |

## Modules

| Module | Contents |
|--------|----------|
| `fenchel` | Generic FY framework: regularizers, predictions, losses |
| `sinkhorn` | Entropic OT for soft permutations |
| `loss` | Learning-to-rank losses (ListNet, ListMLE, Spearman) |
| `metrics` | IR evaluation (MRR, NDCG, Hits@k) |

## Connection to Entropy Calibration

Temperature scaling and truncation (top-k/top-p) are standard methods to calibrate
LLM generation entropy. The FY framework provides a principled view:

- **Shannon (softmax)**: Dense predictions, tends toward high entropy
- **Sparsemax**: Naturally truncates low-probability tokens (like top-k)
- **Temperature**: Equivalent to scaling logits before FY transformation

See `surp::entropy_calibration` for metrics and `surp::zipf` for why heavy-tailed
distributions make calibration difficult.

## References

- Blondel, Martins, Niculae (2020). "Learning with Fenchel-Young Losses"
- Martins & Astudillo (2016). "From Softmax to Sparsemax"
- Blondel et al. (2020). "Fast Differentiable Sorting and Ranking"
- Cao, Valiant, Liang (2025). "On the Entropy Calibration of Language Models"
