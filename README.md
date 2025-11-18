# Bio-Inspired Nanochat

> **"What if a Transformer had a metabolism?"**

This is a fork of [Nanochat](https://github.com/karpathy/nanochat) that replaces standard static weights with **computational analogs of synaptic proteins**.

Standard LLMs are "frozen crystals"—static matrices of `float16` numbers that never change once training is done.
**Bio-Inspired Nanochat** is a "living fluid". Its connections grow, shrink, fatigue, recover, and even reproduce *during inference*, mimicking the energy-constrained efficiency of the biological brain.

## ⚔️ Tale of the Tape: Silicon vs. Carbon

| Feature | Standard Transformer | Bio-Inspired Nanochat |
| :--- | :--- | :--- |
| **Weights** | 🧊 **Static**: Fixed after training. | 🌊 **Fluid**: Evolve in real-time during inference. |
| **Memory** | 📜 **Context Window**: Limited by `seq_len`. | 🧠 **Associative**: Fast-weights "remember" patterns locally. |
| **Diversity** | 🎲 **Randomness**: Temperature sampling. | 🔋 **Metabolism**: Synapses "tire out", forcing new paths. |
| **Capacity** | 🏗️ **Fixed**: Pre-allocated size (e.g., 32 layers). | 🏙️ **Elastic**: Experts multiply/die based on demand. |
| **Learning** | 🏫 **Offline**: Only learns during Backprop. | ⚡ **Online**: "Learns" context via Hebbian consolidation. |

---

## 🧠 The "Wetware" Stack: From Biology to Math

We map specific cellular mechanisms from the [Synaptic Cleft](https://en.wikipedia.org/wiki/Chemical_synapse) directly to tensor operations. This architecture is grounded in the blueprints found in `prompts/Neurological_transformer_model_GPT5_Pro.pdf`.

### 1. Presynaptic Biophysics (The Sender)
*The mechanism of "Fatigue" and "Boredom"*

**The Biology**: Neurons run on batteries (ATP). If a neuron shouts too much (fires continuously), it runs out of neurotransmitter vesicles (chemical ammo). It *must* rest to reload.
**The Math**: We track a fluid reservoir `RRP` (Readily Releasable Pool) for every attention head. High attention scores drain the pool.
**The Effect**: A physically-grounded **frequency penalty**. The model literally *cannot* attend to the same token endlessly. It gets "bored" (depleted) and naturally shifts focus to novel information.

```mermaid
graph LR
    A[Logits] -->|Drive| B(Calcium Influx)
    B -->|Activates| C{Synaptotagmin Sensor}
    D[Vesicle Pool] -->|Limits| E(Release Probability)
    C -->|Gates| E
    E -->|Attenuates| A
    E -->|Consumes| D
    style D fill:#ff9999,stroke:#333,stroke-width:2px
```

### 2. Postsynaptic Density (The Receiver)
*The mechanism of "Working Memory"*

**The Biology**: "Neurons that fire together, wire together." A transient thought becomes a memory only if it is important (high activity) and the brain has energy to "write" it down (Consolidation).
**The Math**: Weights are split into $W_{slow}$ (Long-term) and $W_{fast}$ (Short-term).
$$ y = x(W_{slow} + \underbrace{W_{fast} + \text{Hebb}(x, y)}_{\text{The Scratchpad}}) $$
**The Effect**: **Infinite local context**. The model can define a variable at the start of a sentence and "remember" it at the end via the fast weights, without needing to attend back to it.

### 3. Structural Plasticity (The Life Cycle)
*The mechanism of "Economy & Efficiency"*

**The Biology**: The brain is a ruthlessly efficient economy. It doesn't keep billions of idle neurons on payroll. Useful regions get more resources (Neurogenesis); idle regions are demolished (Pruning).
**The Math**: A **Synaptic Mixture-of-Experts (MoE)** where experts have a "Bank Account" (Energy).
*   **Taxation**: Every forward pass costs Energy.
*   **Income**: Being routed to earns Energy.
*   **Bankruptcy**: Experts with $E \approx 0$ are killed (Merged).
*   **IPO**: Wealthy, overworked experts clone themselves (Split).

**The Effect**: **Neural Architecture Search**. The model starts small and *grows* capacity exactly where the data complexity demands it.

```mermaid
graph TD
    Start((Birth)) --> Healthy[🟢 Healthy Expert]
    Healthy -->|High Usage + Energy| Split{⚡ Split?}
    Split -->|Yes| Clones[Clone into 2 Experts]
    Healthy -->|Low Usage| Starving[🔴 Starving Expert]
    Starving -->|Energy < 0| Merge{💀 Merge?}
    Merge -->|Yes| Absorb[Absorbed by Stronger Neighbor]
    Clones --> Healthy
    Absorb --> Healthy
```

---

## 💉 The Neurosurgeon's Toolkit (Configuration)

You can tweak the personality of the brain by adjusting its chemical balance in `nanochat/synaptic.py` (or via CLI overrides).

| If the model is... | It means... | You should tweak... | Action |
| :--- | :--- | :--- | :--- |
| **Repetitive / Stuck** | Synapses aren't tiring fast enough. | `tau_rrp` (Refill Time) | ⬆️ Increase |
| **Forgetful** | Short-term memory is fading too fast. | `camkii_gain` (Write Strength) | ⬆️ Increase |
| **Scatterbrained** | Firing is too noisy/random. | `syt_fast_kd` (Sensor Sensitivity) | ⬇️ Decrease |
| **Too Small / Dumb** | Experts aren't reproducing. | `split_health_min` (Birth Bar) | ⬇️ Decrease |
| **Bloated / Slow** | Too many lazy experts. | `energy_cost_rel` (Metabolic Tax) | ⬆️ Increase |

**Pro Tip**: Try this "ADHD Mode" override to force high novelty seeking:
```bash
python -m scripts.base_train --syn_cfg.tau_rrp=100.0 --syn_cfg.energy_cost_rel=0.05
```

---

## 🚀 Quick Start (UV Optimized)

We use **Python 3.14** and **uv** for bleeding-edge performance.

### 1. Install the "Wetware"
```bash
# Create the vat (environment)
uv venv .venv --python 3.14
source .venv/bin/activate

# Inject the chemicals (dependencies)
uv sync --extra gpu
```

### 2. Grow a Brain
Train a small bio-model (~4 hours on 1 GPU).
```bash
python -m scripts.base_train \
    --synapses=1 \           # Enable biology
    --depth=12 \             # Layers
    --splitmerge_every=1000  # Run "Life Cycle" every 1k steps
```

### 3. Monitor Vitals (TensorBoard)
```bash
tensorboard --logdir runs/
```
*   **💓 Heartbeat**: `energy_mean` (Should stay > 0.5)
*   **🧠 Map**: `router_embedding` (Should show distinct clusters of expertise)
*   **🌳 Family Tree**: `lineage` (Watch experts split and branch out)

---

## 📂 Anatomy of the Codebase

*   **`nanochat/synaptic.py`** ⚡ **The Physics Engine**: Implements the differential equations for Calcium, ATP, and Vesicle dynamics.
*   **`nanochat/synaptic_splitmerge.py`** 👼 **The God Hand**: The controller that pauses training to perform surgery (splitting/merging experts).
*   **`nanochat/gpt_synaptic.py`** 🏗️ **The Body**: The Transformer skeleton that holds the synaptic organs.
*   **`nanochat/neuroviz.py`** 📸 **The MRI**: Generates beautiful visualizations of the brain's internal state.

---

## 🧬 Legacy Nanochat Features
*(Inherited from the base [Nanochat](https://github.com/karpathy/nanochat) repo)*

This repo remains fully compatible with the original "silicon" workflows:
*   **`speedrun.sh`**: Train a standard static GPT-2.
*   **`scripts/chat_web.py`**: Chat UI.
*   To disable biology, just run without `--synapses`.

## License
MIT
