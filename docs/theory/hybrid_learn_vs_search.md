# Hybrid Learn-vs-Search Optimization Protocol — Design Note (bead `hea.4`)

_Hybrid Optimization (`hea`) · Differentiable Mechanics & Bilevel Evolutionary Architecture. Author: GoldenRiver · 2026-08-24._

## Purpose & Scope

Bio-inspired neural network architectures contain two distinct classes of parameters:
1. **Continuous Differentiable Variables** ($\theta \in \mathbb{R}^D$): Synaptic weight matrices, projection layers, and per-expert Xi kinetic latent vectors ($\tau_c, \tau_{\text{rec}}, \alpha_{\text{fatigue}}$) decoded via `SynapticGenomeDecoder` (`yw9.4`). These have smooth loss gradients $\nabla_\theta \mathcal{L}$ that first-order SGD/Adam optimizes efficiently.
2. **Discrete Non-Differentiable Knobs** ($z \in \mathcal{Z}$): Structural thresholds, stochastic release modes (`stochastic_mode`), discrete attention top-$k$ (`attn_topk`), structural pruning periods, and integer rank eligibility dimensions. These produce step-function discontinuities where $\nabla_z \mathcal{L} \equiv 0$ or is undefined.

Applying CMA-ES to all continuous weights creates an intractable high-dimensional search space, while applying SGD to non-differentiable choices fails due to zero gradients.

The **Hybrid Learn-vs-Search Protocol** solves this via **Bilevel Optimization**:
$$\min_{z \in \mathcal{Z}} \mathcal{L}_{\text{val}}(\theta^*(z), z) \quad \text{subject to} \quad \theta^*(z) = \arg\min_{\theta} \mathcal{L}_{\text{train}}(\theta, z)$$

---

## 1. Bilevel Algorithmic Structure

```text
Algorithm: Bilevel Hybrid Optimizer (BHO)
Input: Discrete search space Z, Population size P, Generations G, Inner SGD steps K

1. Initialize outer evolutionary population of discrete configurations {z_1, ..., z_P}.
2. For generation g in 1 .. G:
   a. For each discrete candidate z_p in population:
      i.   Instantiate model with discrete choices z_p.
      ii.  Initialize continuous weights theta_0.
      iii. Run K inner-loop SGD update steps on training batches:
             theta_{k+1} = theta_k - eta * grad_theta L_train(theta_k, z_p)
      iv.  Evaluate held-out validation fitness:
             fitness(z_p) = - L_val(theta_K, z_p)
   b. Compute natural evolutionary update for discrete distribution params.
   c. Sample next generation of discrete configurations.
3. Return optimal discrete configuration z* and final trained weights theta*(z*).
```

---

## 2. Parameter Division of Labor Ledger

| Subsystem / Parameter | Optimization Mechanism | Rationale |
|:---|:---|:---|
| Weight Matrices ($W_q, W_k, W_v, W_{\text{out}}$) | **Inner Loop: SGD / AdamW** | Dense, continuous, convex local basins |
| Synaptic Xi-Genome ($\Xi$) | **Inner Loop: SGD** | Differentiable via `SynapticGenomeDecoder` |
| `stochastic_mode` (`normal`, `gumbel`, `bernoulli`) | **Outer Loop: NES / CMA-ES** | Categorical choice, non-differentiable |
| `attn_topk` ($k \in \{16, 32, 64\}$) | **Outer Loop: NES / CMA-ES** | Discrete sorting threshold |
| `rank_eligibility` ($r \in \{4, 8, 16\}$) | **Outer Loop: NES / CMA-ES** | Discrete matrix rank dimension |
| `structural_every` ($N \in \{0, 2, 4, 8\}$) | **Outer Loop: NES / CMA-ES** | Discrete hook execution cadence |

---

## 3. Acceptance & Empirical Validation

- The hybrid optimizer strictly beats either SGD-alone (with arbitrary fixed discrete defaults) or Evolution-alone (attempting to search continuous parameters).
- Unit and integration tests verify bilevel convergence and reproducibility.
