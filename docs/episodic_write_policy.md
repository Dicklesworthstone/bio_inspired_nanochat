# Which observations are worth a memory write?

## Prospective decision, November 23, 2026

Bead `72lc`. Start from the real-text empirical bigram that beat the 24-update
vanilla pilot by 0.96053 bits per byte (BPB). Hold the predictor, observations,
storage and write count fixed while changing which observed associations
survive. The question is whether a learner can spend its limited plasticity
on experiences that will improve future predictions. [Matched baseline](../results/vanilla_matched_counts_2026-11-23.json)

Cache language models, variable-order retrieval and surprise-driven memory
are established ideas. The experiment isolates the value of a write by
measuring its effect on later prequential code length. Its candidate accounts
for both a future hit and the probability it takes away from other outcomes.
[Continuous cache](https://arxiv.org/abs/1612.04426) · [Infini-gram](https://arxiv.org/abs/2401.17377) · [Titans](https://arxiv.org/abs/2501.00663)

### Frozen design

Use both archived train-only priors (6,291,456 and 498,073,600 training pairs)
and reproduce their original static BPB numerators before measuring adaptation.
Use every complete, unique document in the archived validation stream. Split
on BOS, break on discontinuities, discard partial documents, and assign by the
low bit of the first byte of the document's SHA-256 digest. These two panels
share a corpus. Neither selects hyperparameters.

Predict each target before changing any persistent state. Clear memory at
each document boundary. All equal-write arms see the same completed blocks
of 32 transitions, commit four entries in chronological order, and retain
64 FIFO entries. Count repeated writes as separate events. For a cue with
memory counts `m_xy, m_x`, predict `(m_xy + 4 p0(y|x))/(m_x + 4)`.
Every slot has room for the preceding context token, cue and outcome.

For a completed block let `n_xy, n_x` be its counts and let
`q = (n_xy + 4 p0)/(n_x + 4)`. Score each prospective write by

```text
n_x [q log2(1 + 1/(m_xy + 4 p0)) - log2(1 + 1/(m_x + 4))]
```

For a future occurrence of the cue drawn from this posterior, the bracket
is the expected reduction in code length from adding *one* occurrence to
the current memory. The factor estimates cue reuse. The expression ignores
interactions between simultaneous admissions and the subsequent FIFO eviction.
Rank once per block, break score ties in favor of the earlier event, then
commit in time order.

The equal-write controls are uniform admission without replacement, the last
four arrivals, rare targets under the train unigram, surprising targets under
the train bigram, and within-block pair frequency times bigram surprisal.
Two stronger controls use the same uniform and frequency-times-surprise
writes with a two-token cue interpolated over the one-token predictor
(pseudocount four). The frozen prior and a dense FIFO that commits every event
at the same block boundary provide reference points. Seed uniform admission
from the document digest. No neural training is part of this experiment.

Advance the proposed rule only if it saves at least **0.005 BPB** against each
of the seven equal-write competitors and the frozen prior in **both panels
and at both training exposures**. Each paired lower bound must exceed zero.
Use 32,768 paired document bootstrap draws per panel, fixed seed 7291 plus
panel number, a one-sided 5% family correction over 32 comparisons, and a
ratio of summed bits to summed bytes. Publish per-document numerators, hashes,
write counts, confidence bounds and all failures. A pass buys replication on
a fresh corpus and a trained neural prior. A miss ends promotion of this rule.

### Why this experiment ranked first

| Rank | Direction | Scientific decision it buys |
| --- | --- | --- |
| 1 | Expected utility of a synaptic write | Intervene on one mechanism on real text; a strong cheap control and a falsifier already exist. |
| 2 | Train the neural residual above a frozen count predictor | Isolate predictive information the count model cannot supply. Requires fresh matched training with and without the prior. |
| 3 | Cross query cues with the values written to fast weights | Distinguish actual binding from recency on the same inputs. A real-text extension needs a matched intervention on future reads. |
| 4 | Intervene on the tied readout early in training | Determine what causes the observed repetitive decoding. Requires independently initialized and trained controls. |
| 5 | Measure retention per write under distribution shifts | Ask whether the mechanism helps when the statistics change again. Requires independent shifts with known boundaries. |

The next ten were document-boundary reset, shuffled write priorities, a second
context order, a dense-write reference, a larger frozen prior, two document
panels, a strong frequency selector, exact byte accounting, cue scrambling,
and independent-domain transfer. The first eight inform this design (uniform
random admission supplies the exchangeable priority control). Cue scrambling
and cross-domain transfer would warrant more work after an advantage.

Fifteen alternatives screened out were another uncalibrated softcap, a larger
MoE, a kernel-only sprint, longer reference training, a larger ablation grid,
synthetic recall alone, a larger memory with unlimited writes, learned decay,
an entropy-only abstainer, more decoding passes, holonomy-only scoring,
sheaf diffusion, a tuned repetition penalty, a new checkpoint format and a new
benchmark dashboard. The existing Beads already cover most of these. None
isolates useful online learning as directly at the available compute budget.
