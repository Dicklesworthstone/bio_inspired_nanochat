## plan to optimize key hotspots using Triton + CUDA Fusion

We initially considered Rust + PyO3, but moving data between GPU and CPU for high-frequency kernels is a performance killer. To truly unlock the "living fluid" architecture on modern hardware (H100/4090), we must execute the biological dynamics **on-device** using **Triton**.

Below is the revised roadmap that prioritizes **kernel fusion** to eliminate VRAM bandwidth bottlenecks and launch overhead.

---

### phase 0 – groundwork (1 day)
| Goal | Tasks |
| --- | --- |
| **Setup** | Create `nanochat/kernels/` and add `triton>=3.0.0` to `pyproject.toml`. |
| **Feature Flags** | Standardize env vars (`BIO_FUSED_PRESYN`, `BIO_FUSED_METRICS`) to toggle between pure PyTorch and fused Triton kernels. |
| **Verification** | Update `verify_evolution.py` to compare Triton outputs against the PyTorch reference implementation (Golden Tests). |

---

### phase 1 – fused presynaptic kernel
**Hotspot**: `SynapticPresyn.forward` (~25–30% of forward time).
**Bottleneck**: Dozens of element-wise ops (decay, clamp, refill) launch separate kernels, thrashing VRAM bandwidth.

**Deliverables**
1. `presyn_fused.py`: A Triton kernel that:
   - Loads `q, k, state` into SRAM.
   - Computes attention scores `(q@k)` block-wise.
   - Fuses the biological ODEs (Calcium → Release Probability → RRP Depletion).
   - Writes back `logits` and updated `state` in a single pass.
2. Support for stochastic release (using Triton's random number generation).

**Validation**
- `verify_evolution.py` confirms numerical parity (FP16 tolerance).
- Nsight Compute shows a reduction from ~50 micro-kernels to 1 mega-kernel.

---

### phase 2 – fused router + neuroscore
**Pain**: Analyzing expert lineage and routing stats requires gathering data from all GPUs to CPU, stalling the pipeline.

**Deliverables**
1. `metrics_fused.py`: A Triton kernel that accumulates:
   - **Routing Frequency**: Atomic adds for expert usage.
   - **Efficiency/Specialization**: Fused reduction over `last_ctx`.
2. **Async Logging**: A background thread that pulls these small accumulated buffers to CPU for TensorBoard only when needed (decoupling logging from the training loop).

---

### phase 3 – fused genetics/xi
**Pain**: `_get_phenotype` and metabolic updates touch every expert every step, preventing graph capture optimization.

**Deliverables**
1. `genetics_fused.py`:
   - Fuses `Xi` decoding (logits → biological constants) with the `fatigue/energy` EMA updates.
   - Computes `gene_bias` directly in the router kernel to avoid intermediate tensor materialization.

---

### phase 4 – structural plasticity capsule
**Pain**: Split/Merge operations involve complex index maniuplation and copying large parameter tensors.

**Deliverables**
1. **GPU-side Reorganization**: Implement structural moves (split/merge) as a set of masked copy kernels.
2. **Optimizer State patching**: A fused kernel to fix up AdamW/Muon states after an expert splits (e.g., inheriting momentum) without iterating via Python loops.

---

### success metrics
- **Presynaptic Overhead**: < 5% of forward pass (down from 30%).
- **Throughput**: 1.5x - 2x tokens/sec improvement on dense synaptic models.
- **Scalability**: Linear scaling with experts (logging/genetics overhead becomes negligible).
