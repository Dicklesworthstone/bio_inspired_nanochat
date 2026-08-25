# Synaptic Granularity Empirical Comparison & Evaluation (bead `vap.2`)

This document presents the unified architectural granularity switch and an executable apples-to-apples empirical benchmark comparing quality, state footprint, throughput, and loss across synaptic granularities.

---

## 1. Architectural Granularity Spectrum

The configuration switch `SynapticConfig.granularity` (`SynapticGranularity`) unifies the biological state representations across three architectural granularities:

| Level | Granularity mode | State scaling | Implemented path |
|:---|:---|:---|:---|
| **Fine (L1)** | `per_connection` | Full-resolution per-attention-edge & per-connection states | Presynaptic modulation in `SynapticAttention` & full rank-$R$ eligibility in `SynapticLinear` |
| **Medium (L2)** | `per_neuron` | Intermediate rank-$R$ traces scaled across neurons | Intermediate rank-$R$ projections in `SynapticLinear` |
| **Coarse (L3)** | `per_expert` | Pooled per-expert / per-layer scalar state machine | Rank-1 state in `SynapticLinear` & MoE metabolic routing state in `SynapticMoE` |

---

## 2. Executable Apples-to-Apples Evaluation Harness

The benchmark is implemented in [`scripts/eval_synaptic_granularity.py`](file:///data/projects/bio_inspired_nanochat/scripts/eval_synaptic_granularity.py) and verified by unit tests in [`tests/test_synaptic_granularity.py`](file:///data/projects/bio_inspired_nanochat/tests/test_synaptic_granularity.py).

### Benchmark Command
```bash
python -m scripts.eval_synaptic_granularity --output results/granularity_comparison.json
```

---

## 3. Empirical Results & Quality/Cost Tradeoffs

Measurements gathered from multi-seed runs (`seeds=(42, 1337)`) with identical model architecture ($L=2, D=64, H=4, \text{vocab}=128, T=32, B=4$) and optimizer settings on CPU:

| Granularity | State Footprint | State Reduction | Throughput (tok/s) | Val Loss ($\text{mean}\pm\text{std}$) | Val BPB ($\text{mean}\pm\text{std}$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`per_connection` (L1)** | 98.1 KB | baseline (0%) | $95 \pm 6$ | **$4.8632 \pm 0.0088$** | **$7.0162 \pm 0.0127$** |
| **`per_neuron` (L2)** | 58.1 KB | **-40.8%** | $262 \pm 270$ | $4.8659 \pm 0.0118$ | $7.0199 \pm 0.0170$ |
| **`per_expert` (L3)** | 28.1 KB | **-71.4%** | $305 \pm 218$ | $4.8657 \pm 0.0118$ | $7.0197 \pm 0.0170$ |

### Findings & Analysis
1. **Fidelity vs. Cost Frontier**:
   - `per_connection` (fine-grained GPT-5 Pro blueprint) achieves the lowest validation loss ($4.8632$) and best bits-per-byte ($7.0162$), capturing full-rank synaptic plasticity at the cost of higher state footprint (98.1 KB).
   - `per_expert` (coarse Grok blueprint) drastically slashes state buffer footprint by **71.4%** ($98.1\text{ KB} \to 28.1\text{ KB}$) and yields the highest compute throughput ($305\text{ tok/s}$) with minimal loss degradation ($\Delta \text{loss} \approx +0.0025$).
   - `per_neuron` (medium granularity) occupies the intermediate pareto point ($58.1\text{ KB}$ state footprint).

All raw run artifacts and per-seed trajectories are tracked in [`results/granularity_comparison.json`](file:///data/projects/bio_inspired_nanochat/results/granularity_comparison.json).

