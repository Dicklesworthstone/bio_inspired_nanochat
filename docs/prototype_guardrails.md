# Risk & Rollback Guardrails for Invasive Prototypes (bead co6)

> **Purpose**: Establish safety gates, canary configurations, and automated rollback criteria before invasive features (reversible flows, simplicial attention, CA init, experimental custom kernels) can be enabled.

---

## 1. Prototype Risk Classification

| Prototype Feature | Risk Level | Primary Failure Mode | Rollback Trigger | Safe Default |
|:---|:---|:---|:---|:---|
| **Reversible Blocks** | High | Activation reconstruction error / numerical drift | $\Delta y > 10^{-4}$ vs forward cache | `reversible=False` |
| **Simplicial 2-Hop Diffusion** | Medium | Gradient explosion on dense graph diffusion | $\nabla \mathcal{L} > 10.0$ or spectral gap collapse | `simplicial=False` |
| **Cellular Automata Init** | Medium | Ill-conditioned singular values ($\kappa > 10.0$) | Early loss spike $\Delta \mathcal{L} > 0.5$ at step 10 | Permanent OFF |
| **Custom Triton Kernels** | High | Silent NaN/Inf or memory corruption under DDP | Non-finite output / divergence guard | Fallback to PyTorch reference |

---

## 2. Canary Rollout Protocol

1. **Step 1: Isolated Micro-Unit Test**: 100% test passing in `tests/test_<feature>.py`.
2. **Step 2: Scoped Canary Run**: 100-step smoke train at small batch size on single device with `divergence_guard=True`.
3. **Step 3: Equal-Compute Paired Comparison**: 10M token paired evaluation vs baseline with statistical significance test (`eval_stats.py`, $p < 0.05$).
4. **Step 4: Rollback Checklist**: If any canary check fails, feature flag remains strictly `False` by default and is archived in `docs/mgr_cross_pollination_playbook.md`.
