# Mechanism interaction experiment (bead `74f.5`)

## Direct metabolic bias and glial feedback in a trained MoE

The interaction between the two routing signals was small on a learned modulo-16 prediction task. Enabling both changed held-out next-token loss by **−0.0000994 nats/token beyond the sum of the individual changes**. The four paired seed differences were −0.0000217, +0.0001001, −0.0003650 and −0.0001111. Their 95% percentile bootstrap interval, [−0.0002792, +0.0000473], lay inside the specified ±0.01-nat practical margin. Four independent training seeds limit precision. [Per-seed logits, routing decisions and interaction](../results/mechanism_interaction_moe_2026-11-22.json)

| Direct metabolic router bias | Glial feedback | Mean token NLL (nats) | Correct predictions across fits | Changed top-two expert sets vs both off |
|---|---|---:|---:|---:|
| Off | Off | 2.146733 | 125/128 | 0/128 |
| On | Off | 2.146835 | 125/128 | 1/128 |
| Off | On | 2.147554 | 124/128 | 19/128 |
| On | On | 2.147556 | 124/128 | 19/128 |

The 16 fits shared the same initial parameters and training sequences within each quartet. Each one-layer, width-16 model used eight fixed experts and selected two per token. Training ran for 16 AdamW steps on sequences beginning at 12 of the 16 possible tokens. Evaluation used the four reserved starts and eight next-token predictions per start. The 128 table entries are 32 positions scored by four independently trained models. The repeated validation batches add no independent observations. The no-bias baseline cleared the prespecified 0.20-nat improvement over uniform predictions on every seed. [Experiment](../scripts/e2e/mechanism_redundancy_eval.py)

All eight glial-enabled fits recorded 16 observations and developed a nonuniform router bias (logit range 0.213–0.254). All eight direct-metabolic-bias fits developed a logit range of 0.00391–0.00523 across experts. The metabolism switch controls the direct routing term. The expert energy and fatigue estimates continue updating when this term is off, and glial feedback can still read energy. Glial feedback changed expert choices much more often than the direct term on these four held-out inputs, while their interaction in token loss stayed within the stated margin. The experiment leaves BDNF, the CaMKII/PP1 latch, NeuroScore and language-model-scale keep/cut decisions open. [Source and scores](../results/mechanism_interaction_moe_2026-11-22.json)

## Historical dense experiment

**The experiment leaves mechanism redundancy unresolved.** Twelve named configurations produced four distinct vectors of per-seed validation losses. The run used seeds 201, 203 and 207, three optimizer steps per seed and two validation batches on a synthetic repeated-sequence task. Each arm used a two-layer dense `GPTSynaptic` model. The rounded values below preserve the original measurements. [Raw per-seed results](../results/mechanism_redundancy_evaluation.json)

The historical harness enabled Hebbian plasticity whenever it enabled the CaMKII/PP1 latch. Consequently, `latch_only` and `hebbian_latch` configured the same model. Glial homeostasis acts on MoE routing, and every arm set `use_moe=False`. The baseline used the synaptic scaffold with the four requested flags off. The twelve arms covered the baseline, singles, pairs and all-flags condition, omitting the four triples. [Original arm construction and evaluation](https://github.com/Dicklesworthstone/bio_inspired_nanochat/blob/371baba/scripts/e2e/mechanism_redundancy_eval.py)

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

The historical harness assigned `AUDITED_FACTORIAL_COMPLETE` and its explanatory sentence unconditionally. These data leave the mechanism classifications and any decision to keep or remove a mechanism unresolved.

The historical harness reset sequence state before every training forward. On the real Hebbian layer, the next forward normally commits a pending update. Replaying three forwards with the historical resets yielded exactly zero accumulated BDNF trace and zero change in the slow consolidation parameter on four seeds. Three successive forwards with the same inputs and initialization yielded nonzero values on every seed. This isolates a second cause of the historical latch comparison's insensitivity. [Reset witness](../results/mechanism_interaction_moe_2026-11-22.json)

A functional comparison of BDNF and the latch requires Hebbian writes that reach the readout, independently toggled gains on the same scaffold, and held-out task loss. NeuroScore requires a matched lifecycle controller with an opportunity to act. The small MoE routing result above supplies one conditional interaction; these additional comparisons and their D1 language-model measurements remain open.
