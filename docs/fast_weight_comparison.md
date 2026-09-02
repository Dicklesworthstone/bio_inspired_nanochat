# Fast-Weight-Programmer Framing & Literature Comparison (bead `sax.5`)

## 1. Theoretical Framing & Related Literature

The synaptic fast-weight mechanism implemented in `bio_inspired_nanochat` is directly situated within the rich lineage of **Fast Weight Programmers (FWP)**, Linear Transformers, and neuromorphic associative memory.

### Literature Mapping

| Architecture / Framework | Primary Reference | State Update Rule | Distinctive Mechanism |
|:---|:---|:---|:---|
| **Classical Fast Weights** | Schmidhuber (1992, 1993) | $W_t = \lambda W_{t-1} + \eta (v_t \otimes k_t)$ | Simple linear outer-product trace decay. |
| **Linear Transformers** | Katharopoulos et al. (2020) | $S_t = S_{t-1} + \phi(k_t) v_t^T$ | Unnormalized associative feature sum; equivalent to causal attention with kernel trick. |
| **DeltaNet / Error-Correcting FWP** | Schlag et al. (2021), Sun et al. (2024) | $\Delta W = \beta (v_t - W_{t-1} k_t) k_t^T$ | Error-correcting associative delta rule (widely adopted in modern RWKV-7 / DeltaNet). |
| **Bio-Inspired Synaptic Transformer** | This Project (`GPTSynaptic`) | $W_{\text{fast}, t} = \text{NormClip}\left(\gamma W_{\text{fast}, t-1} + \eta \text{BCM}(Ca^{2+}) \cdot \text{Latch}(\text{CaMKII}, \text{PP1})\right)$ | Bounded Hebbian updates, presynaptic vesicle dynamics ($RRP/RES$), and Lisman bistable CaMKII/PP1 latching. |

---

## 2. Key Differences in Architectural Philosophy

1. **Error-Driven vs. Correlation-Driven Plasticity**:
   - **DeltaNet** updates fast weights via an explicit gradient descent delta rule: $\Delta W = \beta (v - W k) k^T$. This precisely avoids overwriting already-stored keys.
   - **Bio-Synaptic** updates fast weights via biophysical Hebbian co-activation gated by presynaptic calcium and postsynaptic BDNF/CaMKII state. It does not require computing an explicit error vector at the synapse, reducing computational overhead and aligning with local biological plausibility.

2. **Retention vs. Overwriting Dynamics**:
   - In linear transformers, old memories linearly decay ($\lambda < 1.0$) or get washed out by continuous unconstrained updates.
   - In `GPTSynaptic`, the **CaMKII / PP1 bistable switch** introduces non-linear hysteresis: once a synaptic trace surpasses the autophosphorylation threshold, it latches into the potentiated state, creating noise-robust retention over long context intervals without ongoing refresh signals.

3. **Normalization and Stability**:
   - Classical fast weights frequently suffer from exploding eigenvalues unless heavily regularized.
   - `GPTSynaptic` enforces strict directional normalization and Frobenius-norm bounds (`fast_weight_normalized=True`, `fast_weight_max_norm=1.0`), ensuring numerical stability during long sequential rollouts.

---

## 3. Empirical Working-Memory Benchmark

`scripts/e2e/fast_weight_comparison_bench.py` scores four **untrained** 2-layer / 64-dim
architectures on the working-memory suite (associative recall with 2/4/8 pairs, variable binding,
NIAH at 16/64 tokens), 5 seeds, and records `results/fast_weight_comparison_evaluation.json`.
Re-derived 2026-09-02 under deterministic evaluation (the file carries `measurement_regime`):

| Architecture | Composite | 95% CI | Recall | Binding | NIAH |
|:---|:---:|:---:|:---:|:---:|:---:|
| Vanilla Transformer | 1.5% | [1.0%, 2.0%] | 1.7% | 0.8% | 2.1% |
| Outer-Product Fast Weights | 1.1% | [-0.3%, 2.5%] | 1.7% | 0.8% | 0.8% |
| DeltaNet Error-Correcting | 1.0% | [-0.4%, 2.3%] | 1.7% | 0.8% | 0.4% |
| Bio-Inspired Synaptic | 1.5% | [-0.1%, 3.0%] | 0.8% | 0.8% | 2.7% |

Chance for recall is 1/97 ≈ 1.0%. Paired against vanilla over the same seeds: bio-synaptic
-0.07 pp (p = 0.91), outer-product -0.42 pp (p = 0.30), DeltaNet
-0.56 pp (p = 0.18); bio vs DeltaNet +0.49 pp (p = 0.57).

### Honest verdict

Every arm sits at chance because no arm is trained: the benchmark measures untrained architectures
and therefore cannot rank them. The 2026-08 reading of this table ("bio wins NIAH at 3.8% vs
1.7%", "DeltaNet wins recall at 2.8% vs 0.7%") was seed noise around 1%. Two further defects applied
to those numbers: the reads were single full forwards, which cannot see within-sequence fast-weight
writes (`docs/online_learning_status.md`), and evaluation ran with stochastic vesicle sampling on
(fixed 2026-09-01). The informative comparison trains each arm under the chunked regime and reads
with `chunk_len` (bead `hwxb.9`); until it runs, this page supports no claim about relative
working-memory quality.

