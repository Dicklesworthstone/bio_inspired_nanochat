# Findings & Micro-Benchmark Report: MGR Cross-Pollination Prototypes (beads 6cb, 3zd)

> **Prototypes Evaluated**:
> 1. **Reversible Coupling Blocks** with per-layer activation replay — `ReversibleBlock` in `bio_inspired_nanochat/xpoll.py`
> 2. **Simplicial Higher-Order Attention** (2-hop graph diffusion) — `SimplicialAttention` in `bio_inspired_nanochat/xpoll.py`

---

## 1. Prototype 1: Reversible Coupling Blocks (bead 6cb)

- **Architecture**: Additive invertible coupling $y_1 = x_1 + F(x_2)$, $y_2 = x_2 + G(y_1)$.

The original backward discarded both coupling networks' parameter gradients. Its
test checked only the input gradient. The training path now replays the ordinary
coupling with PyTorch's non-reentrant checkpoint. All trainable parameter and
input gradients and one SGD update matched an independently expressed eager
calculation across frozen/trainable inputs and branches. The analytic inverse
round-trips floating-point inputs within the tested tolerance.

The paired CPU probe trained the same weights on the same dense regression
batch for two warmups and 32 timed updates. Each batch contains 4 × 64
positions of width 64. Saved storage counts unique nonparameter tensor storage
held by autograd immediately after the stack forward. Both arms achieved
identical loss sequences and final parameter tensors.

| Blocks | Eager saved bytes | Replay saved bytes | Eager positions/s | Replay positions/s | Last MSE | Matched weight tensors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 430,080 | 65,536 | 20,885 | 6,412 | 2.063445 | 12 |
| 4 | 1,720,320 | 262,144 | 1,205 | 1,392 | 2.143996 | 48 |
| 8 | 3,440,640 | 524,288 | 534 | 660 | 2.305160 | 96 |

**Decision.** Retain the optional stateless prototype. Replay saved 84.8% of
forward storage at each depth, with one retained input per block. Throughput
changed direction across sizes on a shared CPU worker. Production adoption
requires stateful gradient parity and a longer-context run with device
allocator peak measurements on the target GPUs. Run
`uv run python -m scripts.xpoll_benchmark --coupling-replay --steps 32 --seq-len 64 --embd-dim 64 --batch-size 4 --seed 612 --layers 4`
to repeat the middle comparison through DSR.

---

## 2. Prototype 2: Simplicial 2-Hop Attention (bead 3zd)

- **Architecture**: Convex combination of 1-hop dot-product attention and 2-hop graph diffusion:
  $$\hat{y} = (1 - \sigma(\lambda)) A v + \sigma(\lambda) A (A v)$$
- **Gradient Stability**: $\nabla_\lambda \mathcal{L}$ computes smoothly and learns optimal diffusion without divergence.
- **Computational Cost**: One additional matrix multiplication $A \cdot (Av)$ per head; easily fused into custom Triton attention kernels.
- **Recommendation**: Deploy as an optional attention branch for relational reasoning and long-range dependency benchmarks.
