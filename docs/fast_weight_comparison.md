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
   - With `fast_weight_normalized=True`, the model caps the fast matrix at the configured Frobenius norm and bounds the eligibility signal driving molecular gain. The earlier matrix-only cap permitted overflow during sequential training. [Training evidence and limits](theory/gauge_consolidation.md)

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

## 4. Recall after eviction

The November 24 component screen supplies the same normalized key features
and correct value to each writer. The 24 queried bindings are older than
the recent 32-entry window. Each arm is evaluated on eight held-out streams
at both interference levels, after separate calibration of the two
recurrent rules and the sign-aligned diagnostic. The delta comparator uses
a scalar-gated delta update over the same feature space. The cosine cache
fits 73 entries under the synaptic writer's mutable-state ceiling.

| Writer | Mutable state | 32 later writes, clean / noisy | 128 later writes, clean / noisy |
|---|---:|---:|---:|
| Native projected Hebbian | 2,436 bytes | 15 / 11 | 13 / 15 |
| Gated delta rule | 1,024 bytes | 36 / 32 | 13 / 21 |
| Sign-aligned Hebbian, fresh trace per write | 2,436 bytes | 12 / 15 | 16 / 25 |
| Cosine cache | 2,413 bytes | 22 / 20 | 15 / 12 |
| Recent window | 1,060 bytes | 16 / 12 | 12 / 15 |

Each count is out of 192 scored queries. When the probe exposes an exact
key ID, a 256-byte dictionary answers all 192 at each interference level.
The native writer's median sequential CPU write phase was 31.1 ms for
96 writes, versus 1.36 ms for the delta update. The cosine cache's time
includes block admission. These measurements belong to the research driver.

For a fresh trace and one paired event with key $x$ and value $y$, the
projected trace product is proportional to
$(x P_{\mathrm{in}} P_{\mathrm{out}}^\mathsf{T} y^\mathsf{T})x^\mathsf{T}y$.
Normalization preserves that scalar's sign. In an exhaustive 16-by-16
basis-pair check with one fixed pair of projection matrices, native
argmax recall succeeded on all **137** positive-sign cases and failed on all
**119** negative-sign cases. Explicit sign alignment recovered **256/256**.
Its crowded-stream results above show how much recall survives that repair.

### Trained models with online writes at evaluation

The model experiment presents two prefixes containing the same keys and
values with different bindings. The two query suffixes are identical,
and both answers must be correct to count as successful paired recall.
Training uses ordinary in-window supervision. The evaluation writes the
prefix into per-layer fast weights, replaces the original KV cache and
reads the suffix with writes disabled. Frozen reads and explicit clearing
of the fast state are controls. Their logits match within each
counterfactual pair.

In the four-pair pilot, only one of three trained seeds reached the
prespecified 50% within-window floor. That seed answered 96/96 inside
the window, then 7/96 with writes after eviction and 6/96 with frozen
memory. None of its 48 counterfactual pairs had both answers correct
after eviction. In the separate two-pair trial, three fresh seeds reached
288/288 within the window. They answered 14/288 after eviction with or
without active writes, and none of the 144 pairs had both answers correct.
Active writes changed the logits in all 144 pairs.

The position-matched control used the same two-pair setup and filled a
separate cache with public separator tokens before reading the hidden
prefix. This preserved the original query positions while keeping the
hidden keys and values out of the query's attention cache. It reproduced
288/288 in-window answers. After eviction, active writes yielded 24/288
correct answers versus 21/288 for frozen and cleared memory. Both
answers were correct in 0/144 pairs in all three conditions. The mean
margin favoring each pair's correct bindings over the swapped bindings
was negative in all three active-write seeds.

These trials measure writes first enabled at evaluation. The
[experiment archive](../results/post_eviction_pairs_2026-11-24.json)
contains the drivers, per-seed measurements, and DSR receipts.

### Training the memory rules directly

The follow-up removes attention during training as well as evaluation.
Each arm learns 16-dimensional key and value embeddings, a linear
readout, and scalar write and decay rates. It receives the same
decomposed key and value IDs, and the training loss backpropagates through
the episode's independent writes. The projected Hebbian arm uses fixed
production projections and normalized writes under a neutral gate. At
the default rates, its functional recurrence matched the production
fast matrix and eligibility factors across 80 writes, with a maximum
absolute difference of $1.67\times10^{-16}$ in float64.

The five seeds each train for 768 updates on episodes of two or eight
distinct bindings. The two counterfactual episodes share all keys and
swap the first two values. The shared query asks for the first key, and
the score requires both answers to be correct. Cleared memory gives
identical logits on every pair. Thirty-two bindings test length
extrapolation. The delta arm implements a scalar-gated update
$W\leftarrow\alpha W+\beta k^\mathsf{T}(v-k\alpha W)$. It draws on the
[gated delta rule](https://arxiv.org/abs/2412.06464).

| Trained writer | Mutable state | 2 bindings | 8 bindings | 32 bindings |
|---|---:|---:|---:|---:|
| Projected Hebbian | 2,048 bytes | 2,389 | 349 | 0 |
| Scalar-gated delta | 1,024 bytes | 2,560 | 2,304 | 7 |
| Decayed outer product | 1,024 bytes | 2,560 | 2,547 | 1,303 |

Each cell is out of 2,560 conflicting pairs. The projected rule lost to
both controls on all five seeds at eight bindings. Its gap to delta was
−76.4 percentage points, versus the prespecified +5 point gain required
for a positive result. At 32 bindings, the simple outer-product memory
extrapolated better than both rules. Training left the isolated projection
scalar positive for 10,472 of the 20,480 key–value combinations across
the five Hebbian models.

### Separating cross-event interference from sign errors

Expanding the product of the two eligibility factors gives both
same-event terms $k_i^\mathsf{T}v_i$ and cross-event terms
$k_i^\mathsf{T}v_j$ for $i\ne j$. A cross-event term has coefficient
proportional to $(v_iP_{\mathrm{out}})\cdot(k_jP_{\mathrm{in}})$.
It can write an unobserved binding even when the original keys and
values are orthogonal. The same-event intervention keeps just the
diagonal of this expansion. Its trace decays by the square of the
factor decay. A second intervention also takes the absolute value
of each same-event projection scalar. Both keep the normalized write,
fast-matrix cap, and training protocol. The trace recurrence matched
the explicit expansion across 28 steps to $4.34\times10^{-19}$.

| Trained writer | Mutable state | 2 bindings | 8 bindings | 32 bindings |
|---|---:|---:|---:|---:|
| Projected Hebbian | 2,048 bytes | 1,451 | 213 | 0 |
| Same-event signed | 2,048 bytes | 1,516 | 879 | 0 |
| Same-event sign-aligned | 2,048 bytes | 1,533 | 920 | 1 |
| Scalar-gated delta | 1,024 bytes | 1,536 | 1,366 | 4 |

Each cell is out of 1,536 conflicting pairs over three fresh seeds. At
eight bindings, dropping cross-event products improved paired recall by
43.4 percentage points. Sign alignment added 2.7 points pooled, with a
2-pair loss on one seed. The best same-event variant scored below delta
on every seed. This isolates cross-event interference as a material
failure mode of the neutral-gate recurrence. Production training in
`GPTSynaptic` uses detached writes, and this assay has explicit binding
IDs and differentiates through the writes. Its result applies to the
controlled memory component.

The [trained-memory archive](../results/trained_evicted_recall_2026-11-24.json)
contains both experiment drivers, paired outcomes, and terminal receipts.

### Activity-dependent local erasure

The candidate multiplies each row of the fast matrix by
$\alpha(1-\beta k_i^2)$ before adding $\beta k v^\mathsf{T}$. It uses the
diagonal of the delta rule's erasure matrix. This saves the read needed
to calculate a prediction error and makes erasure depend on the choice
of feature coordinates. For $k=(3/5,4/5)$ and an orthogonal old key
$q=(4/5,-3/5)$, full delta erasure preserves the old response at $1$.
At $\alpha=\beta=1$, local erasure leaves $337/625$. The case isolates
interference from erasure alone. Lean checks the concrete integer
numerators after the denominators are cleared. [Gated delta update](https://arxiv.org/abs/2412.06464)

The CPU assay used continuous noisy cues with hidden identities and two
histories that differed only by a swap of binding values. Each arm used
1,024 bytes of mutable state and saw the same observations. The
original screen used 12 held-out seeds and 576 pairs per panel. Local
erasure got zero correct pairs in each panel. The cosine cache got
435/576 in revision, and the strongest retention control got 3/576.
The original result remains inconclusive under its preset 20% control floor.

On 12 new test seeds, the original settings stayed frozen. The added
conventional reservoir cache used the original development split for
calibration. Each count again represents a complete pair, with 576 pairs
per panel.

| Writer or cache | Retention | Revision |
|---|---:|---:|
| Activity-dependent local erasure | 0 | 1 |
| Decayed outer product | 0 | 1 |
| Scalar-gated delta | 1 | 4 |
| Recent cosine cache | 0 | 415 |
| Reservoir cosine cache | 361 | 340 |

The reservoir cleared the retention floor at the same byte budget.
Local erasure missed the accuracy gate and took 1.30 times the latency
of the fastest matrix control in the paired timing summary. This assay
stops this update at the component stage. [Protocols, raw outcomes, and
execution receipts](../results/local_associative_memory_2026-11-26.json)

To reproduce the two runs, extract the archived driver and protocol text,
verify the SHA-256 values, and map the paths in the recorded DSR recipes
to the new checkout and worker. The diagnostic consumes the original
calibration output, and each recount takes its own raw predictions.

### Separate update rates for cues and values

When a noisy cue matches a stored record, this candidate updates its
value immediately and moves the cue prototype more slowly on a conflicting
value. A stable prototype can keep a single corrupted observation from
moving the address. A truly drifting cue can instead become harder to find.
The assay checks both cases with exact two-dimensional witnesses.
[Synaptic consolidation](https://www.nature.com/articles/nn.4401) ·
[Continual prototype learning](https://openaccess.thecvf.com/content/ICCV2021/html/De_Lange_Continual_Prototype_Evolution_Learning_Online_From_Non-Stationary_Data_Streams_ICCV_2021_paper.html)

All arms used 1,024 bytes, the same hidden-identity cue generator, and
the same held-out pair protocol. The original 12-seed run scored 743/768
on revision with conflict gating. The recent cache and both conventional
prototype controls reached 739/768. The four-pair difference was 0.52
percentage points and favored conflict gating on four seeds. The candidate
also recalled only 6/768 retention pairs versus 620/768 for the reservoir.
Its typical batched latency was 1.76 times the fastest control. The
decision required a three-point revision gain over each control, at least
nine seed wins, losses of at most two points on each guard panel, and
latency within 10% of the fastest control.

Calibration selected recent admission for all prototype methods. A
diagnostic on 12 different held-out seeds fixed admission, the match
threshold, and recency to the original reservoir settings. It selected
the update gains from the existing development grid under those settings.
Each panel again used 768 pairs.

| Method with shared reservoir admission | Retention | Revision | Cue drift |
|---|---:|---:|---:|
| Latest cue | 632 | 619 | 343 |
| Fixed-gain centroid | 626 | 650 | 319 |
| Similarity-conditioned gain | 622 | 653 | 329 |
| Conflict-conditioned gain | 618 | 654 | 334 |

Much of the gain over replacing cues with the latest observation was
also available from a conventional prototype update. Using the value
conflict as the signal added one correct revision pair over the
similarity control, with an interval spanning zero. Both runs failed
the prespecified gate. [Protocols, raw predictions, timings and
reproduction commands](../results/conflict_gated_prototypes_2026-11-26.json)
