# Sleep Phases, Memory Replay & Synaptic Consolidation (SHY Hypothesis) — Design Note (bead `cel.1`)

_Continual Learning & Synaptic Consolidation (`cel`) · Biologically Grounded Continual Adaptation. Author: GoldenRiver · 2026-08-24._

## Purpose & Theoretical Foundations

In standard deep neural networks, sequential learning on non-stationary distributions results in **catastrophic forgetting**. Biological brains solve this through two coupled mechanisms:
1. **Two-Stage Memory Architecture (Hippocampal-Neocortical Division)**: Fast, plastic synaptic weights ($W_{\text{fast}}$) rapidly encode experiential context online during active wakefulness. During offline "sleep" states, prioritized replay of high-surprise experiences drives the distillation of $W_{\text{fast}}$ into durable slow weights ($W_{\text{slow}}$).
2. **Synaptic Homeostasis Hypothesis (SHY)** (Tononi & Cirelli, 2003): Net synaptic weight increases during wakeful learning saturate energetic and metabolic budgets. Slow-wave sleep executes a global, multiplicative downscaling of synaptic strengths ($\beta_{\text{down}} < 1.0$), pruning weak or spurious connections while preserving relative potentiated ratios.

---

## 1. The Wake-Sleep Tri-Phasic Architecture

```text
               ┌────────────────────────────────────────┐
               │              WAKE PHASE                │
               │ • Online Fast-Weight Adaptation W_fast │
               │ • Experience stream evaluation         │
               │ • Surprise metric: S = -log P(x_t)     │
               │ • Top-K high-surprise buffer push      │
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │            NREM SLEEP PHASE            │
               │ • Prioritized Experience Replay (PER)  │
               │ • Sharp-Wave Ripple replay forward pass│
               │ • Fast->Slow Consolidation:            │
               │   ΔW_slow = η_c * (W_fast ⊙ Gate)      │
               │ • W_fast reset to zero                 │
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │             REM SLEEP / SHY            │
               │ • Homeostatic Multiplicative Renorm:   │
               │   W_slow ← W_slow * (E_budget / ||W||) │
               │ • Low-salience weight pruning          │
               │ • Thermodynamic energy recovery        │
               └────────────────────────────────────────┘
```

---

## 2. Mathematical Formalization

### 2.1 Wake Surprise Metric & Buffer Push
During the online wake phase, sequence tokens $x_{1:T}$ generate sequence-level surprisal:
$$\mathcal{S}(x_{1:T}) = \frac{1}{T} \sum_{t=1}^T -\log P(x_t \mid x_{<t})$$
Sequences where $\mathcal{S}(x_{1:T}) > \tau_{\text{surprise}}$ are stored in a prioritized episodic replay ring buffer $\mathcal{B}_{\text{replay}}$ with priority $p_i = (\mathcal{S}_i)^\alpha$.

### 2.2 NREM Fast-to-Slow Synaptic Consolidation
During offline NREM replay, replaying stored trajectories activates the CaMKII/PP1 consolidation latch $\Lambda \in [0, 1]^{d_{\text{out}} \times d_{\text{in}}}$. Slow weights distill the accumulated fast representation:
$$W_{\text{slow}}^{(t+1)} = W_{\text{slow}}^{(t)} + \eta_{\text{cons}} \cdot \left( W_{\text{fast}} \odot \Lambda \right)$$
Following consolidation, the fast weights are cleared: $W_{\text{fast}} \leftarrow 0$.

### 2.3 Synaptic Homeostatic Downscaling (SHY)
To maintain bounded total metabolic weight norm $\Omega_{\text{max}}$ without distorting signal direction:
$$W_{\text{slow}} \leftarrow W_{\text{slow}} \cdot \min\left(1.0, \frac{\Omega_{\text{target}}}{\|W_{\text{slow}}\|_F}\right)$$
This contracts noise while preserving consolidated memory traces.

---

## 3. Implementation Blueprint & Acceptance

1. **Replay Buffer**: `HighSurpriseReplayBuffer` tracking sequence perplexity and priority sampling.
2. **Consolidation Controller**: `OfflineConsolidationPhase` executing NREM replay passes and fast $\to$ slow weight transfer.
3. **Homeostasis Renormalizer**: `SynapticHomeostasisModule` enforcing SHY downscaling bounds.
4. **Acceptance**: Tested against standard continual learning benchmarks (sequential task transfer without forgetting).
