# Neurological Transformer Model (Grok-4.1 Variant)

**Source:** Competing LLM (Grok-4.1) Request
**Context:** A different first-principles approach to mapping biological synapse components to a Transformer/MoE architecture. This plan focuses on a "Synaptic Compute Unit" (NSU) that replaces every linear layer.

## Core Concept
Replace every linear layer in the transformer with a **Neuro-inspired Synaptic Unit (NSU)**. The synapse is not just a weight, but a "biochemical machine" with:
- Presynaptic vesicle trafficking.
- Postsynaptic receptor scaffolding.
- Second-messenger cascades.
- Stochastic vesicle release.
- Metaplasticity (the synapse modifies its own plasticity rules).

The gating network ($\sigma_i$) for each output neuron/expert is replaced by a miniature biochemical simulator.

## Key Computational Analogs

| Biological Component       | Computational Analog                                      | Timescale    | Effect                                                                   |
| -------------------------- | --------------------------------------------------------- | ------------ | ------------------------------------------------------------------------ |
| **Synaptotagmin/Munc13**   | Stochastic vesicle release multiplier                   | ms           | Controlled randomness (per-token Monte-Carlo dropout).                   |
| **SNARE complex**          | Gating nonlinearity with memory (STP)                     | 10–500 ms    | Sequence-dependent facilitation/depression.                               |
| **AMPA/NMDA + CaMKII**     | Dual-pathway memory (fast Hebbian + slow consolidation) | 100 ms – min | Creates a fast weight and a slow weight per connection.                     |
| **Dynamin/Endophilin**     | Structural plasticity (expert birth/death)                | seconds+     | Dynamic growth/pruning of experts.                                       |
| **Synapsin/Actin**         | Attention routing elasticity (soft MoE)                   | seconds      | Gradual commitment to experts.                                           |
| **PSD-95/Shank/Homer**     | Receptor anchoring / credit assignment buffer (Eligibility Traces) | minutes      | Biologically plausible BPTT; stores outer products for later consolidation. |
| **BDNF**                   | Meta-gradient modulation                                  | hours        | Changes future learning rates per synapse (true metaplasticity).          |

## Proposed Architecture Sketch

### 1. The ODE/Dynamical System
For each token and expert, a small dynamical system is driven by the router pre-logit score:

```
// Fast vesicle pool (Munc13/RIM/Synaptotagmin)
RRP_e         = a * RRP_e + recovery_rate_e
release_prob_e = sigmoid(v * Ca_influx_e + synaptotagmin_bias_e)
vesicles_released_e ~ Binomial(RRP_e, release_prob_e) // Stochastic!

// Short-term plasticity (synapsin / actin)
facilitation_e = facilitation_e * decay_f + vesicles_released_e
depression_e   = depression_e * decay_d + vesicles_released_e
short_term_factor_e = facilitation_e / (1 + depression_e)

// Dual AMPA/NMDA-like pathways
fast_weight_e = fast_weight_e + η_fast * (vesicles_released_e * outer(h_in, target))
slow_weight_e = slow_weight_e + η_slow * (CaMKII_activation * outer(h_in, target))

// Final effective weight multiplier for this expert at this token
σ_e = softplus(base_router_score_e
               + log(short_term_factor_e)
               + NMDA_contribution(slow_weight_e))
```

### 2. MoE Dispatch
The final expert output is scaled by the normalized number of vesicles released:
$$ \text{expert\_output} = \sum_e (\sigma_e \cdot \text{vesicles\_released\_e} / E[\text{vesicles\_released\_e}]) \cdot \text{Expert}_e(h_{in}) $$

## Most Powerful/Radical Ideas
1.  **Stochastic Vesicle Release:** Using a true Binomial sample for vesicle release introduces biologically plausible, structured dropout. It can be run deterministically ($p \to 1$) or stochastically during inference for uncertainty estimation.
2.  **Dual-Timescale Weights:** Every linear layer has two weight matrices, `W_fast` and `W_slow`. `W_fast` provides working memory (updated every step), while `W_slow` provides long-term memory (updated only when a "CaMKII/NMDA" signal is high).
3.  **Short-Term Plasticity Kernels:** Facilitation and depression curves create sequence-specific routing, making the network's behavior context-dependent in a recurrent manner.
4.  **Local Eligibility Traces:** Storing outer products of pre- and post-synaptic activity in a local buffer (like PSD-95 scaffolding) is a biologically plausible alternative to backpropagation through time (BPTT). A global "BDNF" signal can later consolidate these traces into permanent memories.
5.  **Structural Plasticity:** Experts undergo "endocytosis" (pruning) and "exocytosis" (birth/cloning) based on utilization, enabling continual learning without catastrophic forgetting.

## Key Differences from "GPT5-Pro" Plan
-   **Granularity:** This plan puts the *entire* state machine at the **expert level**, not per-connection. This is less biologically faithful but more computationally tractable.
-   **Stochasticity:** It leans heavily into explicit, per-token stochastic sampling (Binomial release) as a core feature, whereas the other plan uses expectations by default for differentiability.
-   **Learning Rule:** More explicit about dual-timescale weights (`W_fast`, `W_slow`) and the role of eligibility traces as a BPTT alternative.
-   **Code:** Provides a more abstract, equation-based overview with a JAX implementation sketch.

This plan presents a transformer with a **recurrent biochemical memory** that evolves during a single forward pass, making it feel more like a "society of chemical synapses" than a static matrix.
