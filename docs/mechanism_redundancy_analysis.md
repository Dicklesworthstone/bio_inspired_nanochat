# Mechanism interaction experiment (bead `74f.5`)

## What the run measured

**The experiment leaves mechanism redundancy unresolved.** Twelve named configurations produced four distinct vectors of per-seed validation losses. The run used seeds 201, 203 and 207, three optimizer steps per seed and two validation batches on a synthetic repeated-sequence task. Each arm used a two-layer dense `GPTSynaptic` model. The rounded values below preserve the original measurements. [Raw per-seed results](../results/mechanism_redundancy_evaluation.json)

The harness enables Hebbian plasticity whenever it enables the CaMKII/PP1 latch. Consequently, `latch_only` and `hebbian_latch` configure the same model. Glial homeostasis acts on MoE routing, and every arm sets `use_moe=False`. The baseline uses the synaptic scaffold with the four requested flags off. The twelve arms cover the baseline, singles, pairs and all-flags condition, omitting the four triples. [Arm construction and evaluation](../scripts/e2e/mechanism_redundancy_eval.py)

---

## Historical arm measurements

| Recorded arm | Effective configuration | Val loss (mean ± SEM) | Val accuracy |
|:---|:---:|:---:|:---:|
| `vanilla_baseline` | Dense synaptic scaffold, four flags off | $4.1646 \pm 0.0028$ | $1.8\%$ |
| **Presynaptic Only** | Presyn | $4.1662 \pm 0.0024$ | $1.8\%$ |
| **Hebbian Only** | Postsyn Fast Weights | $4.1641 \pm 0.0034$ | $2.0\%$ |
| `latch_only` | Hebbian + latch | $4.1641 \pm 0.0034$ | $2.0\%$ |
| `glial_only` | Dense scaffold, glial flag inert | $4.1646 \pm 0.0028$ | $1.8\%$ |
| **Presyn + Hebbian** | Pre + Post Dual | $4.1657 \pm 0.0033$ | $1.8\%$ |
| **Hebbian + Latch** | Fast Weights + Bistable Latch | $4.1641 \pm 0.0034$ | $2.0\%$ |
| `all_bio_active` | Presyn + Hebbian + latch, glial flag inert | $4.1657 \pm 0.0033$ | $1.8\%$ |

## Historical interaction estimates

Each delta in the interaction estimate measures a mean loss change from the common scaffold.
$$\text{Synergy}(A, B) = \Delta \text{Loss}(A+B) - (\Delta \text{Loss}(A) + \Delta \text{Loss}(B))$$

The JSON assigns an interaction label using a fixed absolute cutoff of $10^{-4}$. It computes paired tests on individual arms against the scaffold, but gives no uncertainty interval for the interaction itself. The results below are descriptive.

| Requested pair | $\Delta \text{Loss}_A$ | $\Delta \text{Loss}_B$ | $\Delta \text{Loss}_{A+B}$ | Interaction estimate | Interpretation limit |
|:---|:---:|:---:|:---:|:---:|:---|
| **Presyn + Hebbian** | $+0.0015$ | $-0.0005$ | $+0.0011$ | $+6.01 \times 10^{-5}$ | Three-seed estimate |
| **Presyn + Latch** | $+0.0015$ | $-0.0005$ | $+0.0011$ | $+6.01 \times 10^{-5}$ | Latch arm also enables Hebbian |
| **Presyn + Glial** | $+0.0015$ | $0.0000$ | $+0.0015$ | $0.0000$ | Glial flag inert |
| **Hebbian + Latch** | $-0.0005$ | $-0.0005$ | $-0.0005$ | $+4.96 \times 10^{-4}$ | Latch arm duplicates combined arm |
| **Hebbian + Glial** | $-0.0005$ | $0.0000$ | $-0.0005$ | $0.0000$ | Glial flag inert |
| **Latch + Glial** | $-0.0005$ | $0.0000$ | $-0.0005$ | $0.0000$ | Glial flag inert |

---

## What would identify an interaction

The harness assigns `AUDITED_FACTORIAL_COMPLETE` and its positive explanatory sentence unconditionally. The recorded values support neither the original mechanism classifications nor a decision to keep or remove a mechanism.

A follow-up can measure the incremental latch effect within a Hebbian-enabled arm and the glial effect within matched MoE arms. On a task the baseline learns, verify that every enabled mechanism changes its intended state, retain the per-seed differences of differences, and report their uncertainty against a prespecified decision threshold. Conclusions about redundancy require enough precision to distinguish a practically small effect from an unresolved one.
