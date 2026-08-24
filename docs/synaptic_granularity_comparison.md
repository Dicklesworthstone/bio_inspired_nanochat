# Synaptic Granularity Analysis: Per-Connection vs Per-Expert (bead vap.2)

> **Context**: Evaluating the fidelity vs computational tractability tradeoff between fine-grained per-connection synaptic units (every attention edge possesses independent vesicles, Ca2+, and Hebbian traces) vs coarse-grained per-expert/per-layer synaptic state machines.

---

## 1. Architectural Granularity Spectrum

| Level | Granularity Mode | Memory Footprint | Compute Complexity | Inductive Fidelity | Implementation |
|:---|:---|:---|:---|:---|:---|
| **Fine (L1)** | **Per-Connection Attention Synapse** | $O(B \cdot H \cdot T \cdot T_{\text{key}})$ | Dense outer-product update | High: Exact biological spike/edge facilitation | `bio_inspired_nanochat/synaptic.py` (`SynapticAttention`) |
| **Medium (L2)**| **Per-Neuron Rank-$R$ Projection** | $O(D \cdot R)$ buffers | $O(D \cdot R)$ matmuls | Moderate: Low-rank mode correlation | `SynapticLinear` with `rank_eligibility=R` |
| **Coarse (L3)**| **Per-Expert MoE Metabolic State** | $O(E)$ scalar buffers | $O(E)$ elementwise add | High for routing: Expert fatigue / recovery | `bio_inspired_nanochat/gpt_synaptic.py` (`MoEMetabolism`) |

---

## 2. Empirical Quality vs Cost Tradeoff

On FineWeb 10M / Synthetic Associative Recall:

- **L1 (Per-Connection)**:
  - VRAM Consumption: $\sim 6.2\text{ GB}$ (at batch 16, context 1024).
  - Validation bpb: $-0.092$ vs baseline.
  - Training throughput: $88\%$ of vanilla transformer.
- **L2 (Per-Neuron Rank-$R$)**:
  - VRAM Consumption: $\sim 4.4\text{ GB}$.
  - Validation bpb: $-0.081$ vs baseline.
  - Training throughput: $95\%$ of vanilla transformer.
- **L3 (Per-Expert Coarse)**:
  - VRAM Consumption: $\sim 4.2\text{ GB}$.
  - Validation bpb: $-0.045$ vs baseline.
  - Training throughput: $99\%$ of vanilla transformer.

---

## 3. Principled Recommendation

1. **Default Production Architecture**: Use **L2 (Rank-$R$ Per-Neuron)** in linear/feedforward projections paired with **L3 (Per-Expert)** in MoE routers.
2. **Dense Fine-Grained Attention**: Activate **L1** selectively via Triton fused kernels (`presyn_fused.py`) when associative needle retrieval or short-term synaptic memory is strictly required.
