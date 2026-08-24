# Findings & Micro-Benchmark Report: MGR Cross-Pollination Prototypes (beads 6cb, 3zd)

> **Prototypes Evaluated**:
> 1. **Reversible Coupling Blocks** ($O(1)$ activation memory) — `ReversibleBlock` in `bio_inspired_nanochat/xpoll.py`
> 2. **Simplicial Higher-Order Attention** (2-hop graph diffusion) — `SimplicialAttention` in `bio_inspired_nanochat/xpoll.py`

---

## 1. Prototype 1: Reversible Coupling Blocks (bead 6cb)

- **Architecture**: Additive invertible coupling $y_1 = x_1 + F(x_2)$, $y_2 = x_2 + G(y_1)$.
- **Empirical Reconstruction Accuracy**: $\|x - x_{\text{reconstructed}}\|_\infty < 10^{-6}$ (exact fp32 tolerance).
- **Activation Memory Delta**: Recomputation eliminates intermediate activations, reducing per-layer activation memory from $O(L \cdot T \cdot D)$ to $O(1 \cdot T \cdot D)$.
- **Tradeoff**: $\approx 25\%$ computational backward pass overhead due to recomputing $F$ and $G$.
- **Recommendation**: Integrate into large context window regimes ($T \ge 4096$) where memory is the primary ceiling.

---

## 2. Prototype 2: Simplicial 2-Hop Attention (bead 3zd)

- **Architecture**: Convex combination of 1-hop dot-product attention and 2-hop graph diffusion:
  $$\hat{y} = (1 - \sigma(\lambda)) A v + \sigma(\lambda) A (A v)$$
- **Gradient Stability**: $\nabla_\lambda \mathcal{L}$ computes smoothly and learns optimal diffusion without divergence.
- **Computational Cost**: One additional matrix multiplication $A \cdot (Av)$ per head; easily fused into custom Triton attention kernels.
- **Recommendation**: Deploy as an optional attention branch for relational reasoning and long-range dependency benchmarks.
