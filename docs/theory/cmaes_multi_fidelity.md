# Multi-Fidelity, Multi-Seed & Composite CMA-ES Objective — Design Note (bead `hea.1`)

_Hybrid Optimization (`hea`) · Neuroevolutionary Search Infrastructure. Author: GoldenRiver · 2026-08-24._

## Purpose & Scope

Standard hyperparameter search evaluated on noisy, single-seed training runs suffers from two fatal failure modes:
1. **Noise Domination**: Variance between random seeds ($\sigma \sim 2 \cdot 10^{-3}$) dwarfs the true hyperparameter effect size ($\Delta \sim 10^{-4}$), causing evolutionary search to drift into uninformative local minima.
2. **Compute Inefficiency**: Allocating uniform maximal step budgets to obviously unviable or diverging candidates wastes over 75% of total evaluation FLOPs.

The **Strengthened Multi-Fidelity Objective** addresses this via:
1. **Multi-Seed Averaging**: Each candidate is evaluated across $S \ge 3$ independent seeds; the target metric is the sample mean $\overline{\mathcal{L}} = \frac{1}{S} \sum_{s=1}^S \mathcal{L}_s$, shrinking noise by $1/\sqrt{S}$.
2. **Multi-Fidelity Scheduling (ASHA/Hyperband-style)**:
   - Stage 1 (Screening): Evaluate all $N$ candidates at $R_{\text{min}}$ steps on 1 seed.
   - Stage 2 (Refinement): Promote top $\eta = 1/2$ candidates to $R_{\text{med}}$ steps on 2 seeds.
   - Stage 3 (Final): Evaluate top surviving candidates at $R_{\text{max}}$ steps on $S$ seeds.
3. **Composite Multi-Objective Score**:
   $$J(\theta) = \overline{\mathcal{L}}_{\text{held\_out}}(\theta) + \alpha_{\text{lat}} \cdot \frac{\text{Latency}(\theta)}{\text{Latency}_{\text{base}}} + \beta_{\text{reg}} \cdot \|\theta - \theta_0\|_2^2$$
   balancing held-out task quality against computational latency and biophysical parameter drift.

---

## 1. Mathematical Formulation & Scheduling

```text
Algorithm: Multi-Fidelity Candidate Evaluator (MF-CMA)
Input: Candidate parameters x in R^D, Fidelity r in {1, 2, 3}, Base Seed seed_0

1. If r == 1 (Screening):
     steps = R_min, seeds = [seed_0]
   Else if r == 2 (Refinement):
     steps = R_med, seeds = [seed_0, seed_0 + 1]
   Else (High Fidelity):
     steps = R_max, seeds = [seed_0, seed_0 + 1, seed_0 + 2]

2. For s in seeds:
     res_s = evaluate_candidate(x, seed=s, steps=steps)
     losses.append(res_s.held_out_loss)
     latencies.append(res_s.step_latency_ms)

3. Compute composite loss:
     J = mean(losses) + alpha_lat * (mean(latencies) / base_latency) + beta_reg * ||x||^2

4. Return J, std(losses)
```

---

## 2. API Knobs & Acceptance Criteria

- `seeds_per_candidate`: Integer $S \ge 3$ (default: 3).
- `fidelities`: Tuple $(R_{\text{min}}, R_{\text{med}}, R_{\text{max}})$ (default: $(15, 50, 150)$).
- `composite_alpha_lat`: Weight for throughput penalty (default: 0.01).
- `composite_beta_reg`: Weight for parameter regularization (default: 0.001).
- **Acceptance**: Known-good vs known-bad separation $> 3\sigma$ with statistical significance ($p < 0.01$).
