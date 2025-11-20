# Neurological Transformer Model

**Source:** GPT-5 Pro Request
**Context:** A conceptual proposal for grafting "presynaptic biophysics" onto a standard LLM transformer architecture (like DeepSeek MoE) to create a "living fluid" network where weights are stateful synapses.

## Core Concept
Replace static weights with **Stateful Synapses** that mimic key biological molecules:
*   **SNAREs**: Docking/Priming
*   **Synaptotagmin**: Ca2+ sensing (fast vs facilitating)
*   **Complexin**: Clamping
*   **Munc13/18**: Priming/Scaffolding
*   **V-ATPase/VDAC**: Energy/ATP budget
*   **Clathrin/Dynamin**: Endocytosis (recycling)

## 1. The Synaptic Compute Unit (SCU)

### 1.1 Replacement Rule
For any directed connection $i \to j$ (attention head or router edge), replace the scalar weight $a_{ij,t}$ with:
$$ m_{ij,t} = a_{ij,t} \cdot q_{ij} \cdot n_{ij,t} $$
*   $a_{ij,t}$: Standard attention score (softmax).
*   $q_{ij}$: Quantal amplitude (learnable weight per vesicle).
*   $n_{ij,t}$: Expected vesicles released (from biophysical micro-model).

### 1.2 Presynaptic State Vector ($s_{ij,t}$)
Each synapse maintains a state vector:
*   $D_t$: Docked/primed vesicles (Ready-Releasable Pool, RRP).
*   $R_t$: Reserve + recycling pool.
*   $E_t$: Local ATP/energy budget.
*   $C_t$: Ca2+ nanodomain concentration.
*   $Q_t$: SNARE "zippering" fraction (priming status).
*   $\Xi$: Fixed expression vector (learned "genome" of the synapse defining its proteome profile).

### 1.3 Dynamics Equations
*   **Calcium ($C_t$):** Driven by presynaptic input $I_{ij,t}$ (pre/post match). Buffered by Parvalbumin ($PV$) for fast decay or RIM for strong coupling.
    $$ C_{t+1} = \lambda_C(\Xi) C_t + \alpha_C(\Xi) I_{ij,t} $$
*   **SNARE Priming ($Q_t$):** Munc13 raises priming; Munc18 clamps it. NSF resets it.
*   **Release Probability ($p_t$):** Driven by Synaptotagmin (Syt1/Syt7) sensing Calcium. Complexin clamps it (threshold). Doc2 adds low-Ca background release.
    $$ p_t = \sigma(\kappa_{Syt} \log(C_{t+1} + \epsilon) - \theta_{clamp}) $$
*   **Vesicle Release ($n_{ij,t}$):** Expected release = $D_t \cdot Q_t \cdot p_t$.
*   **Pool Dynamics:** Vesicles cycle from Reserve ($R$) $\to$ Docked ($D$) $\to$ Released $\to$ Uncoated ($U$) $\to$ Refilled ($R$).
    *   Endocytosis (Clathrin/Dynamin) limits recycling rate.
    *   Refilling requires Energy ($E$).
*   **Energy ($E_t$):** VDAC/Mitochondria supply ATP. Release and recycling consume ATP. Low energy throttles docking/refilling (metabolic fatigue).

## 2. Integration into Transformer

### 2.1 Attention
*   Replace attention weights with:
    $$ \tilde{\ell}_{ij,t} = \frac{q_i^\top k_j}{\sqrt{d}} + \log(\epsilon + q_{ij} n_{ij,t}) $$
    (Log-space addition keeps softmax normalization valid).
*   This introduces **Frequency Penalty** naturally: high activity $\to$ vesicle depletion $\to$ lower $n_{ij,t}$ $\to$ lower attention score.

### 2.2 MoE Router
*   Experts get "tired". Routing logits are modulated by the expert's vesicle availability and energy.
*   **Structural Plasticity**: Experts have a "life cycle".
    *   High usage + High Energy $\to$ **Mitosis** (Split into two experts).
    *   Low usage $\to$ **Apoptosis** (Pruning/Death) or **Merge**.

## 3. Emergent Properties
*   **Short-term Plasticity**: Facilitation (Ca2+ buildup) and Depression (Vesicle depletion) naturally emerge.
*   **Energy-Constrained Routing**: Implicit load balancing. The network creates "sparse" paths dynamically based on energy availability.
*   **Timescale Separation**: Fast (Ca2+), Medium (Pools), Slow (Weights/Genome).

## 4. Postsynaptic Upgrades (Hybrid Plan)

The "Hybrid V2" plan integrates ideas from a competing LLM design:
*   **Dual Weights**: $W_{fast}$ (AMPA-like, updated frequently) + $W_{slow}$ (NMDA-like, updated via consolidation).
*   **CaMKII/PP1 Gating**: A bistable switch controls consolidation from fast $\to$ slow weights.
*   **Eligibility Traces**: Local outer-product buffers (PSD-95) store gradients/activity traces for delayed updates (biologically plausible BPTT approximation).
*   **Stochastic Release**: Use expected value ($n_{ij,t}$) for differentiability, but inject noise (Gumbel-Softmax or Binomial) on top-k edges for exploration/regularization.

## 5. Implementation Details
*   **JAX/Flax code provided**: A full `NeuroMoELayer` implementation.
*   **PyTorch code provided**: Drop-in replacements for `nanochat` (`synaptic.py`, `gpt_synaptic.py`).
*   **Tuning**: Use `SynapticConfig` to control timescales ($\tau$), gains ($\alpha$), and energy costs.
*   **Visualization**: `NeuroViz` dashboard (TensorBoard integration) to watch the "brain" evolve (lineage tree, expert health radar, UMAP of router embeddings).

## 6. Structural Plasticity Controller (`synaptic_splitmerge.py`)
*   A standalone controller that monitors expert "health" (utilization * energy).
*   Performs **Split** (clone healthy expert) and **Merge** (fuse weak experts) operations periodically.
*   Uses **Router Embeddings** (learned functional identity) to determine which experts are "similar" enough to merge.

## Key Takeaways
This architecture turns a static LLM into a **dynamic, homeostatic system**. It adds time-dependent state to every connection, allowing the model to "get bored," "save energy," and "learn online" within the context window via synaptic dynamics, without requiring full weight updates.
