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

## Prospective transfer test after the write selector failed

Bead `zml5`. The first study passed 11/32 contrasts. The all-write cache
improved on the frozen count model in all four panels. The separate transfer
question is whether *pairing* a cue with its actual successor supplies
information beyond lexical repetition when a pretrained Transformer already
sees the same prefix.

Freeze Pythia-160M at revision `50f5173d932e8e61f858120bcb800b97af589f46`.
Use the first 128 eligible FineWeb documents from the last two validation row
groups, last group first, and every eligible article in WikiText-103-raw test
at `b08601e04326c79dfdd32d625aee71d232d685c3`. Require unique documents,
lossless tokenizer round trips, and 128–512 scored tokens from the first
2,048 Unicode characters. Shorten long prefixes to fit; start with the
model's end-of-text token. Exclude the preceding study's FineWeb documents.
Within each corpus sort by SHA-256 and use every fourth document for
calibration, starting with index zero. The rest are untouched test cases.
[Pythia model and provenance](https://huggingface.co/EleutherAI/pythia-160m) ·
[WikiText](https://huggingface.co/datasets/Salesforce/wikitext)

Every arm predicts before writing. Each sees the same two most recent
completed 32-event blocks. The exact binding predictor is
`(m_xy + 4 p_neural)/(m_x + 4)`. In its counterfactual, replace each
block's `m_xy` by `n_x n_y / 32`. This is the **exact mean probability**
over uniform within-block permutations of values at fixed cues; the read
opportunity, block recency, cue and outcome marginals are identical. Also
compare a target-frequency cache and a continuous cache keyed by cosine
similarity between the neural hidden states that predicted each target.
The latter gets the same 64 past observations and richer stored keys.

Tune each arm's mixture with the frozen model on calibration BPB over
`0, .01, .03, .1, .3, .6, 1`. Tune the continuous cache's inverse
temperature over `1, 5, 15, 50` on that same split. Prefer the lower
mixture and temperature on ties. Freeze all settings before test evaluation.
Report the unmodified LM, every tuned control, and the exchangeable control
at the *binding arm's* mixture. All predictive distributions must normalize.

Advance mechanistic attribution only if binding saves at least 0.005 BPB
against the frozen LM **and** the same-mixture exchangeable control on each
corpus, with positive simultaneous paired lower bounds. Declare competitive
benefit only if it additionally clears that bar against all three separately
tuned controls on each corpus. Bootstrap whole test documents with 32,768
draws, seeds 8911 and 8912 (FineWeb, WikiText), using one-sided 5% / 10
bounds for the ten contrasts. No changes to the write selector or training
defaults follow from this transfer test. The corpus split establishes
evaluation independence from calibration; overlap with Pythia's pretraining
is unknown.

## Results

**Neither mechanism met its registered promotion criterion.** The proposed write
selector beat the frozen count predictor on average in all four panels, but a
simple frequency-times-surprise selector beat the proposal in every panel at
the same write budget. Against pretrained Pythia-160M, preserving the observed
cue–successor pairing gave a small FineWeb gain. The WikiText estimate was
close to zero and its corrected lower bound was negative.

### Which four writes should survive?

The evaluation retained 1,003 unique, complete FineWeb documents with 857,692
targets and 4,160,621 UTF-8 bytes. Each row reports held-out BPB, where lower
is better. Panels A and B contain 489 and 514 documents respectively.

| Training pairs | Panel | Frozen bigram | Proposed selector | Frequency × surprise | Write every event |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 6,291,456 | A | 2.232556 | 2.178113 | 2.172855 | 2.157639 |
| 6,291,456 | B | 2.177801 | 2.133212 | 2.125163 | 2.117528 |
| 498,073,600 | A | 1.816459 | 1.800018 | 1.796915 | 1.772551 |
| 498,073,600 | B | 1.773996 | 1.765945 | 1.763329 | 1.741012 |

The proposal cleared 11 of 32 comparisons. Both smaller-prior panels cleared
five, and the two larger-prior panels cleared one and zero. Writing every
event was a descriptive reference with eight times as many writes per block.
At the same 64-slot capacity, it also evicted old entries eight times faster.
An independent implementation replayed 80 document–arm combinations from
their original token stream, including FIFO eviction, and reconstructed all
40 pooled BPB values. [Complete comparison and corrected bounds](../results/episodic_write_policy_2026-11-23.json) · [Document losses and write counts](../results/episodic_write_policy_2026-11-23.jsonl.gz)

### What does an observed pairing buy over the same words in the same memory?

Calibration selected a 3% binding mixture on both corpora. The table gives
the **control's BPB minus binding BPB**, followed by its one-sided simultaneous
lower bound in parentheses. Positive values favor binding. The registered
minimum saving was 0.005 BPB.

| Held-out corpus | Calibration / test documents | Scored tokens / bytes | Binding BPB | Saving over frozen LM | Saving over same-mixture permutation control |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 32 / 96 | 38,723 / 178,513 | 1.024010 | +0.000378 (+0.000013) | +0.000482 (+0.000142) |
| WikiText-103 | 16 / 45 | 20,110 / 91,660 | 1.090570 | +0.000111 (−0.000213) | +0.000017 (−0.000661) |

Zero of ten comparisons cleared the promotion criterion. FineWeb calibration
selected a zero mixture for both lexical and continuous caches. On WikiText
the lexical and continuous caches scored 1.087844 and 1.087653 BPB,
respectively, both lower than exact binding's 1.090570. Every calibration and
test loss curve was byte-identical on two separate RTX 4090s. An independent
calculation recovered each selected mixture and temperature and all 42
bootstrap decisions across the two studies. The experiments cover one frozen
neural checkpoint, short prefixes and 64 memory slots. The planned large-scale
trained synaptic ablation is still open. [Transfer summary and all controls](../results/associative_credit_2026-11-23.json) · [Per-document curves across every mixture and temperature](../results/associative_credit_2026-11-23.jsonl.gz)

A retrospective token-by-token replay explains the small net effect. At
34,409/38,723 FineWeb positions and 17,836/20,110 WikiText positions, pairing
left the probability of the observed next token exactly equal to the permutation
control. On FineWeb it saved 187.50 bits across 2,678 targets and spent 101.39
extra bits across 1,636 targets. On WikiText it saved 81.46 bits across 1,416
targets and spent 79.87 extra bits across 858 targets, leaving **1.59 bits
saved across 45 articles**. Every loss came from an absent pair whose cue
and target had occurred in the same block. The replay recovered all 141
held-out document differences to within `6e-13` bits per document. These
groups depend on the observed target. A deployable predictor must decide
whether to use memory before seeing it. [Token attribution and verifier](../results/episodic_write_policy_audit_2026-11-23.json)

**An upper bound for a learned gate.** Keep the two calibrated predictors
fixed at their 3% mixtures. Let `Bₜ(y)` be the binding prediction and `Eₜ(y)`
the permutation-control prediction. A gate that uses the prefix to choose
`aₜ` in `[0,1]` assigns probability `aₜ Bₜ(y) + (1 − aₜ) Eₜ(y)`. For every
possible target this is at most `max(Bₜ(y), Eₜ(y))`. Its total saving against
the permutation control is therefore bounded above by the sum of the
positive token-level log ratios, even if training produces a perfect choice
at every observed position. Dividing the replayed positive sums by test
bytes gives ceilings of **0.001051 BPB on FineWeb and 0.000889 BPB on
WikiText** (rounded upward). Both are below the 0.005 target. This rules out
reaching that target on these panels by changing only the switch between
those two fixed predictions. Richer keys, different memory contents and
other mixture strengths change the bound.

### Reproducing the studies

The prospective protocols are in commits `e633c47` (write selection) and
`8426935` (neural transfer). Use a fresh output directory for each DSR check.
The following are the commands for the `checks` array in a DSR quality profile
using the locked project environment. Launch the profile with
`DSR_REPOS_FILE="$PROFILE" dsr quality --tool "$STUDY" --work-dir "$CHECKOUT"`.

```sh
uv run --no-sync python -m scripts.e2e.episodic_write_policy \
  --inputs "$ARCHIVED_COUNTS_AND_VALIDATION" --out "$NEW_WRITE_OUTPUT"

NANOCHAT_BASE_DIR="$PROJECT_TOKENIZER_CACHE" HF_HUB_OFFLINE=1 \
  uv run --no-sync python -m scripts.e2e.associative_credit \
  --model "$PYTHIA_SNAPSHOT" --wiki "$WIKITEXT_TEST_PARQUET" \
  --fineweb "$FINEWEB_SHARD_00027" \
  --previous results/episodic_write_policy_2026-11-23.jsonl.gz \
  --out "$NEW_TRANSFER_OUTPUT"
```

For the first command, place both ranks of `counts-rank{0,1}.npz`,
`prefix-counts-rank{0,1}.npz`, the ordered `eval-pairs.npy`, and the
project's `token_bytes.pt` and `tokenizer.json` in one input directory.
Those intermediate inputs come from the archived project training streams.
Their ordered-pair and per-rank count digests are checked at startup.
The audit's `count_input_producers` entry also contains the original
single-rank validation and two-rank fit programs, their historical prefix
references and the invocation order for rebuilding these inputs from the
project tokenizer and 28 staged shards. This study reused the archived full
fit and independently checked its validation code length.
[Pilot count archive identities](../results/vanilla_matched_counts_2026-11-23.json) · [Full count archive identities](../results/vanilla_ngram_2026-11-22.json)

The second command uses the published previous-document archive, a local
project tokenizer cache for the overlap check, the pinned Pythia snapshot,
and two public parquet inputs. It checks model, tokenizer and data digests
before scoring. The saved document archives contain hashes, byte counts and
loss numerators. They support an independent statistical replay without the
training archives or a GPU. [Pythia snapshot](https://huggingface.co/EleutherAI/pythia-160m/tree/50f5173d932e8e61f858120bcb800b97af589f46) · [WikiText test split](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/test-00000-of-00001.parquet) · [FineWeb validation shard](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/blob/main/shard_00027.parquet) · [Independent verifiers and DSR receipts](../results/episodic_write_policy_audit_2026-11-23.json)

To check both registered decisions using only this checkout, use this DSR
check with a new scratch directory. It rebuilds the calibration choices and
simultaneous bootstrap bounds from the published document losses.

```sh
audit=results/episodic_write_policy_audit_2026-11-23.json
jq -r '.statistics_verifier.source' "$audit" > "$SCRATCH/replay.py" &&
  uv run --no-sync python "$SCRATCH/replay.py" results "$SCRATCH/observed.json" &&
  jq '.statistics_verifier.report' "$audit" > "$SCRATCH/expected.json" &&
  cmp "$SCRATCH/observed.json" "$SCRATCH/expected.json"
```
