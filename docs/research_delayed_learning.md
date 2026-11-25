# Delayed feedback and selective consolidation

## Question and hypotheses

Can a frozen pretrained language model adapt from delayed feedback using a small record of its decisions? Separately, can evidence-gated consolidation improve predictions when rules recur, conflict, or change?

| Hypothesis | Observable prediction | Disposition if it fails |
|---|---|---|
| H1. Intermediate features improve an output adapter's supervised capacity. | A matched multi-depth readout approaches Q/V LoRA on two tasks and improves on the final-only readout. | End this readout design before reward training. |
| H2. A local score trace improves delayed-learning economics. | The streamed update agrees with exact replay for the same policy, followed by a held-out reward and full-cost advantage. | Keep exactness and speed as separate claims. Stop if the matched cost frontier fails to improve. |
| H3. Consolidation that requires independent supporting episodes helps under interference. | A small durable store improves prediction in both changing and recurring contexts versus matched robust episodic controls. | Keep the loss and skip public-benchmark escalation. |
| Third alternative. Conventional methods capture the available signal. | Final-only or LoRA closes the capacity comparison, or robust versioned retrieval reaches the same memory frontier. | Prefer the measured incumbent. |

The delayed-credit literature already includes [e-prop](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/), [TreePO](https://proceedings.mlr.press/v306/li26au.html), and [EPIG-Tree](https://arxiv.org/abs/2609.20004). The earlier [exact final-matrix benchmark](theory/neuromodulated_rl_study.md#native-cost-and-gsm8k-checkpoint-screen) demonstrates a cost tradeoff on its workload. [Supersede](https://arxiv.org/abs/2606.27472) and [the consolidation failure study](https://arxiv.org/abs/2605.12978) motivate the separate memory comparison.

## Work and acceptance

The first round prioritizes architecture capacity, exactness, and memory interference. Final-only features, Q/V LoRA, and trust-aware bounded retrieval are competing explanations. Adaptive branch placement, compressed score bases, mixed reward horizons, and selective replay become useful when a measured bottleneck survives the simpler controls. The cost advantage of the earlier final-matrix trace does not establish an advantage for this readout against optimized replay.

The project ledger is `bio_inspired_nanochat-delayed-learning-ixnn`. Completing a test records its disposition and leaves a failed advancement gate in force.

- [x] Map the existing score-trace and memory evidence and distinguish the new hypotheses.
- [x] Record the third alternative and the falsifiers before scoring.
- [x] Create and check the dependency graph for the two research tracks.
- [x] Pin and verify the pretrained weight and tokenizer inputs.
- [x] Generate disjoint supervised splits, check answer-token identities, and freeze the seed and budget.
- [x] Run unadapted, final-only, multi-depth and Q/V LoRA under the same label and update budget.
- [x] Check baseline potency, exact zero-write parity and parameter counts.
- [x] Independently recompute paired decisions and publish the complete cohort.
- [x] Derive the delayed score identity for frozen features and a single on-policy update.
- [x] Check numerical/autograd parity under temperature, truncated support and scored EOS.
- [x] Falsify unweighted reuse for varying token-level feedback and stale trace reuse after policy change.
- [x] Check the finite-sum identity in Lean and record its axioms and scope.
- [ ] Qualify streamed score vs optimized replay end to end. Deferred by the capacity gate.
- [ ] Run matched delayed-reward learning against a live LoRA/branch-RL control. Deferred by the capacity gate.
- [ ] Replicate on another backbone with fresh tasks. Requires an admitted positive primary result.
- [x] Seal the memory stream and budget, including current, returning, revised and unseen contexts.
- [x] Verify causal order, witness and counterexample cases, and the shared logical-record ceiling.
- [x] Score robust episodic and latest-observation controls alongside the consolidation candidate.
- [ ] Test a disjoint public update benchmark with strong retrieval controls. Deferred by the synthetic result.
- [x] Recount each arm, check held-out answer labels and source hashes, and review the scope of each claim.
- [x] Archive the receipts, decisions, counterexamples and remaining gaps.
- [ ] Start the multi-model BrennerBot review after workspace and roster confirmation.

## Supervised capacity gate

Use pinned Qwen2.5-Math-1.5B-Instruct, two procedural four-choice tasks and three data/initialization seeds. Each task supplies half of 1,536 total training examples per seed and 256 new held-out examples. The controls share answer labels, prompts, the frozen base, and the same data order. The final-only and multi-depth arms have the same number of trained scalars as rank-1 Q/V LoRA in the last block. Multi-depth reads four fixed, disjoint channel slices across four depths. All arms score a four-token output support.

Each task qualifies only if frozen accuracy is between 5% and 90% and LoRA improves it by at least 5 percentage points pooled. Advance only if both tasks qualify, the multi-depth readout is within 3 points of LoRA on both, and it exceeds the final-only arm by at least 4 points on one task without trailing it by more than 3 on the other. The winning task must favor multi-depth in at least two of three seeds. These are decision margins, not claims of statistical power. Report raw paired outcomes and seed effects. Invalid or unqualified tasks cannot pass the gate.

### Capacity result

| Task | Frozen | Final-only | Four depths | Q/V LoRA |
|---|---:|---:|---:|---:|
| Fraction ordering | 186 | 307 | 286 | 364 |
| Last digit of a product plus an addend | 184 | 197 | 186 | 216 |

Each count is out of 768 held-out prompts across three seeds. Each trainable arm used 4,864 scalars. The four depths were 7, 14, 21 and 28, with 304 frozen channels from each. The unadapted model and zero-weight LoRA logits agreed bit for bit in the prespecified check.

Fraction ordering qualified. The multi-depth arm trailed LoRA by 10.16 percentage points and the final-only readout by 2.73 points. Last-digit LoRA improved by 4.17 points over frozen, below the 5-point qualification floor. The design failed the capacity gate. It leaves other ways of allocating channels or choosing fixed features open. This study used restricted four-choice logits from one backbone. The registered reward and second-backbone trials remain deferred.

## Delayed-score boundary

For a fixed token prefix and a frozen feature vector, logit offsets of the form `phi @ A @ B.T` give a local score for `A`. Hold the behavior parameters fixed for one rollout. Use the actual sampled support and temperature, including EOS when sampled. A scalar coefficient known after the rollout distributes over the accumulated token scores. Multiple coefficients require the corresponding distinct sums or saved scores. The identity alone does not establish a policy improvement, a stale-policy update, or throughput.

For a positive capacity result, compare end-to-end time and device-hours at matched held-out reward. Charge collection, sampling, updates and model execution. The prespecified scientific target remains a reproducible twofold cost reduction versus live conventional controls, followed by a second-backbone check.

For logit offsets \(\phi_t A B^\mathsf{T}\) and temperature \(\tau\), the score on a fixed sampler support is
\[
 \nabla_A \log \pi(a_t \mid h_t)
 = \tau^{-1}\phi_t^\mathsf{T}\Big(B_{a_t} - \sum_j \pi(j\mid h_t)B_j\Big).
\]
The derivative treats the realized prefix, support and backbone features as fixed. A scalar return or advantage for a completed trajectory multiplies its accumulated score. A change in the policy, a new coefficient for each token, or feature dependence on \(A\) can invalidate reusing that sum. The Lean statement covers scalar distribution over a finite sum over the integers. The separate rational witness and FP64 autograd comparison test the score formula.

The finite run covered 168 positions in 24 fixtures, including a final EOS action in each. The streamed score matched autograd within \(7.11\times 10^{-15}\) absolute error. Stale scores, differentiated feature dependence, and applying the first coefficient to an unweighted sum each produced a nonzero discrepancy in all fixtures. For three exact rational token scores, the correct signed weighted sum was \(-1013/2475\), whereas the first coefficient times the unweighted sum was \(2/75\). Lean 4.36 checked the integer finite-sum lemma with `propext` as its only reported axiom. No learning or cost outcome follows from that lemma.

## Memory gate

Separate true revisions from returns to an old context. Each model receives the same context cues, source reliability and timestamp metadata. Compress only past evidence and count every persistent entry, including the durable store. A streaming evaluator queries before the next observation, and test seeds are separate from any pilot or development seeds. Require a five-point mean accuracy gain over the strongest scored control and no loss over 2 points on true revisions or returning contexts, pooled over held-out seeds. A planted decisive witness and an environment where consolidation should lose must both behave as designed. A synthetic result concerns this constructed stream.

### Memory result

The frozen run queried 24,576 times across 12 seeded streams. Each policy could retain up to 48 logical records. The proposed rule required two independent sources to corroborate a value before promoting it into the durable half of the store.

| Policy | Overall correct / 24,576 | Returning context / 4,205 | Recent revision / 2,371 |
|---|---:|---:|---:|
| Latest observation | 17,181 | 3,057 | 993 |
| Latest trusted observation | 19,107 | 3,391 | 642 |
| Recency-weighted retrieval | 17,136 | 1,623 | 480 |
| Per-context weighted retrieval | 18,563 | 3,160 | 797 |
| Corroborated consolidation | 18,684 | 2,845 | 384 |

Consolidation gained 381 correct routine predictions over the strongest overall control. It lost 546 on returning contexts and 258 after recent revisions. On revisions, it returned the superseded value 1,719 times, versus 1,466 for the latest-trusted control and 916 for unfiltered latest. All five arms abstained on the 432 first visits. A planted sequence showed both sides of the behavior. Corroboration protected an old true value from one unreliable source and then delayed a real correction until a second source confirmed it.

The rule failed the overall gate and both guard margins. The pilot uses exact context keys and a cap on logical records. It leaves semantic retrieval, byte-level cost, and transfer to unseen contexts open. The public benchmark remains deferred for this candidate.

## Evidence boundaries and retry conditions

| Claim | Evidence | Decision |
|---|---|---|
| Fixed-support streamed score equals the same-policy gradient | Algebraic derivation, rational witness, and numerical matches on 24 fixtures | Retain within the stated assumptions. |
| Scalar reward distributes over an integer finite sum | Lean kernel check and axiom report | Checked for the stated integer lemma. |
| The four-depth view is sufficient for delay-friendly adaptation | One qualified task lost to LoRA by 10.16 points. The other task lacked control potency. | This design failed. A different feature map needs a fresh supervised test. |
| Corroborated consolidation wins under recurrent rules | Matched synthetic record budget and 12 paired streams | This design failed. A different promotion rule must handle first changes and returning contexts on independent streams. |
| Lower GPU-hours to the same reward | No admitted learning-cost experiment | Open. |

The [experiment archive](../results/delayed_learning_pilots_2026-11-24.json) contains the four frozen drivers, the per-seed outcomes, the recount script, and terminal DSR receipts. To reproduce each study, extract its driver and use the receipt command with the paths mapped into a fresh working directory. The supervised study downloads only the pinned public Qwen checkpoint and recomputes its published weight hash.

The local adversarial pass checked the original falsifiers. The memory reader's durable record can shadow a correct newer observation while waiting for a second source. The adapter's fixed projection spends three quarters of its coefficients on earlier depths. The experiment establishes a loss for that feature layout. The trace counterexamples distinguish update exactness from feature capacity. NTM capabilities and Agent Mail were checked. Cross-model review is pending the separate BrennerBot workspace and roster selection.

## Exact score storage versus optimized factor replay

The fixed-policy score identity admits a separate engineering question even after the capacity no-go. A delayed scalar reward can weight a retained per-trajectory score matrix, or it can weight saved score factors for a single GEMM at feedback time. The scope is one exact final-readout update on pinned Qwen. The model policy stays frozen.

- [x] Pin the driver and public checkpoint, and qualify a shared live top-32 rollout with fixed uniforms and no early stopping.
- [x] Run live compact BF16-feature replay, FP32 replay, per-token accumulation, and 16-token accumulation in the same invocation.
- [x] Match sampled tokens and support IDs and compare each reward-weighted score to a factor-derived FP64 oracle for every completed round.
- [x] Time prefill, generation, trace collection, and gradient reduction in each completed round. Report post-feedback time and both retained and peak storage, with the primary timeout visible.
- [x] Compare device work on the same shapes in a separate CUDA-graph probe. Its synthetic factors on device isolate collection and reduction from model execution and host scheduling.
- [x] Repeat the 24-by-192 comparison on the second GPU. Keep its timing independent of the registered gate and retain any disagreement.
- [x] Verify the size prediction and archive the timeout and follow-up measurements.
- [ ] Qualify the registered latency gate in a fresh complete cohort, if there is a deployment need for rank-16 rollout storage. Delayed-reward learning still requires a separate capacity success.

The scheduled shapes are (batch, new tokens) of (8, 16), (8, 384), and (24, 192), at feature width 1,536 and readout rank 16. EOS is suppressed until the final draw to make the three horizons exact. A fixed signed coefficient per trajectory arrives after the rollout. Compact replay retains each native BF16 feature and an FP32 score residual, and its feedback path needs one weighted GEMM. FP32 replay trades retention for conversion time. The accumulator uses FP32 and can discard its scratch chunk once the final chunk is reduced.

The size witness predicts `4*D*R` bytes per trajectory for a dense score and `T*(2*D+4*R)` for compact replay. At these dimensions the storage crossover is between 31 and 32 decisions. The 16-token case exercises the unfavorable side. A bounded result advances only when the same streaming arm halves retained bytes relative to compact replay and stays within 5% of the fastest replay median end to end on both long shapes. Maximum score disagreement must stay within the prespecified FP64 comparison. The experiment includes model execution and stops at the gradient. It does not measure learning, optimizer effects, or reward quality.

The comparison uses a rank-16 basis. At rank 128 with the same feature width the crossover moves to 220 decisions. A single sum across all trajectories would discard information for arbitrary delayed coefficients. A coefficient vector that selects just one trajectory must recover that trajectory's score.

### Storage and timing evidence

The first RTX 4090 invocation completed eight of the nine planned live rounds and hit its 900-second DSR limit before the last round reported. Each completed round printed only after within-round token and support hashes matched and all four gradients passed the FP64 tolerance. The compact final JSON was written only after the full cohort, so the partial receipt and its log are the source for these medians. The registered two-shape latency gate remains unmeasured.

| Live primary shape | Complete rounds | Compact replay median | FP32 replay median | Token accumulator median | Chunk accumulator median |
|---|---:|---:|---:|---:|---:|
| 8 by 16 | 3/3 | 1.60 s | 1.81 s | 1.82 s | 1.65 s |
| 8 by 384 | 3/3 | 38.68 s | 41.41 s | 35.65 s | 40.01 s |
| 24 by 192 | 2/3 | — | — | — | — |

The corresponding score-store sizes follow directly from the sealed tensor dimensions. Compact replay and either accumulator require 0.383 and 0.750 MiB at 8 by 16, 9.188 and 0.750 MiB at 8 by 384, and 13.781 and 2.250 MiB at 24 by 192. These figures cover score state retained while waiting for feedback. KV cache and model parameters are separate.

The device-only comparison includes collection and reduction and uses synthetic inputs on a second RTX 4090. It excludes model execution and Python dispatch. The chunk accumulator consumed 1.06 ms versus compact replay's 0.66 ms at 8 by 384, and 0.66 ms versus 0.37 ms at 24 by 192. Per-token accumulation cost 1.36 and 0.93 ms respectively. Every graph output agreed with its FP64 control within the separately specified fixture tolerance. The storage savings trade off against more device work.

The independent second-GPU live run completed all three 24-by-192 rounds and verified the predicted 6.125-fold reduction in retained score state. The four update paths matched their FP64 reference to at most `4.65e-7` relative Frobenius error. Compact replay took a median of 18.56 s end to end. Chunk accumulation took 19.61 s and per-token accumulation took 23.10 s, 5.63% and 24.45% slower respectively. Neither stayed within the 5% latency allowance in this run. Compact replay itself ranged from 18.40 to 23.96 s. The chunk accumulator's 13.781-to-2.250 MiB reduction in retained score state corresponded to a 206.7-to-187.4 MB reduction in incremental peak PyTorch allocations on this workload. The latter includes other rollout allocations and illustrates the smaller system-level effect.

The [score-storage archive](../results/delayed_score_replay_2026-11-24.json) includes the primary timeout log, the device-only and second-GPU DSR receipts, the frozen drivers, and the recount script. The narrow result is a memory tradeoff for one frozen policy and one fixed-rank gradient. Its registered end-to-end latency gate remains open.
