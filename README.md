# fynch

Fenchel-Young losses, differentiable sorting, learning-to-rank primitives.

(fynch: from Fenchel-Young)

Dual-licensed under MIT or Apache-2.0.

```rust
use fynch::{softmax, sparsemax, entmax, softmax_with_temperature};
use fynch::{pava, soft_rank};

// Prediction functions with tunable sparsity
let theta = [2.0, 1.0, 0.1];
let p = softmax(&theta);              // dense
let p = sparsemax(&theta);            // sparse
let p = entmax(&theta, 1.5);          // tunable
let p = softmax_with_temperature(&theta, 0.5);  // sharper

// Differentiable sorting
let y = [3.0, 1.0, 2.0, 5.0, 4.0];
let monotonic = pava(&y);  // isotonic regression (PAVA)
```

## The Framework

Fenchel-Young losses unify prediction functions and losses via convex duality:

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
| `fenchel` | Regularizers, predictions, temperature scaling, entropy |
| `sinkhorn` | Entropic OT for soft permutations |
| `loss` | Learning-to-rank: ListNet, ListMLE, ApproxNDCG |
| `metrics` | IR evaluation: MRR, NDCG, Hits@k |
| `sigmoid` | Numerically stable sigmoid, log-sigmoid, softplus |
| `topk` | Differentiable top-k selection, Gumbel-softmax |

## Temperature and Entropy

Temperature scaling and truncation (top-k/top-p) calibrate LLM generation:

- **Shannon (softmax)**: Dense, tends toward high entropy
- **Sparsemax**: Naturally truncates low-probability tokens
- **Temperature < 1**: Sharper distributions, lower entropy

See `surp::entropy_calibration` for entropy-based LLM metrics.

## Connections

- [`surp`](../surp): Temperature affects entropy calibration
- [`wass`](../wass): Sinkhorn algorithm for OT and soft sorting

## References

- Blondel, Martins, Niculae (2020). "Learning with Fenchel-Young Losses"
- Martins & Astudillo (2016). "From Softmax to Sparsemax"
- Blondel et al. (2020). "Fast Differentiable Sorting and Ranking"
