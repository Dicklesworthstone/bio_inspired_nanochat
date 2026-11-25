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

## Replay only the trainable final block

The rank-one Q/V LoRA control had more supervised capacity than the tested four-depth readout. With that adapter placed only in the final Qwen block, the first 27 blocks' BF16 outputs depend on tokens and frozen weights. Capture those activations at generation time and replay the final block when feedback arrives. Its attention still differentiates through the prompt and previous scored positions. The comparison measures a storage-for-recomputation tradeoff for the same adapter.

For a fixed realized token prefix \(x\), let \(F(x)\) be the frozen blocks' output and \(L_\theta(F(x))\) be the suffix objective. Since \(F\) is independent of the adapter parameters \(\theta\), substituting the recorded \(F(x)\) preserves the derivative in exact arithmetic. The final block's internal cache depends on \(\theta\), so its prompt K/V still belongs in the differentiated computation. Related work uses trainable suffixes to cut training cost. This experiment tests the recorded-policy numeric contract and charges live capture. [Suffix adaptation precedent](https://openaccess.thecvf.com/content/CVPR2024/papers/Mercea_Time-_Memory-_and_Parameter-Efficient_Visual_Adaptation_CVPR_2024_paper.pdf)

- [x] Verify the public checkpoint and lock the rank-one adapter, two rollout shapes, and three rotated rounds.
- [x] Generate live, identical top-32 trajectories in all arms and account for the cost of capturing native BF16 activations.
- [x] Replay the full model and the captured final block in the original cached geometry.
- [x] Test parallel teacher-forced full and final-block replay on the identical tokens and recorded support.
- [x] Require exact behavior-support logits and check every trainable gradient, including through the prompt. Save every completed round before beginning another.
- [x] Compare wall time and retained bytes against the fastest full-model arm that qualifies. Recount the verdict independently and archive terminal DSR evidence.

Use batch and scored-token pairs (4, 16) and (8, 32), one pinned Qwen2.5-Math-1.5B-Instruct checkpoint, nonzero Q/V LoRA factors, and a fixed delayed coefficient per trajectory. Use the native causal SDPA backend. Compare all four replay paths under the actual top-32 support and scored actions from the rollout. EOS is eligible in the last draw. The reference is full cached replay in the original sampling geometry. Eligibility requires its behavior logits to agree bit for bit and the arm's retained logits to agree with the reference bit for bit. Require nonzero gradients for all four factors, with relative Frobenius error at most `1e-6` per factor and max absolute error at most `1e-7` for an eligible arm. Report numeric drift and time for other arms separately.

For a positive engineering result, the same eligible final-block arm needs a twofold median post-feedback speedup and at least 20% lower median collected-rollout-plus-replay time at both shapes relative to the fastest eligible full replay at each boundary. This pilot does not perform an optimizer update or measure reward learning.

### Replay result

The cached final-block path reproduced behavior-support logits and all four Q/V LoRA gradients bit for bit in every one of six rotated rounds. Both parallel paths failed eligibility. Full parallel replay changed selected logits by up to 0.375 and had up to 11.65% relative gradient error in a factor. Parallel final-block replay had up to 7.32% relative gradient error. They remain relevant to a separate comparison that allows approximation.

| Batch by decisions | Full cached replay | Final-block cached replay | Median total time, full → final block | Additional retained BF16 features |
|---|---:|---:|---:|---:|
| 4 by 16 | 1,054 ms | 144 ms | 2.02 s → 1.15 s | 0.469 MiB |
| 8 by 32 | 2,274 ms | 396 ms | 5.15 s → 2.53 s | 1.313 MiB |

These total medians include live generation, collection and gradient calculation on one RTX 4090. The exact path improved post-feedback time by 5.73–7.31× and total time by 43.1–50.9%, passing the locked engineering gate. The negative control held prompt logits fixed while detaching its final-block cache. That changed V-factor gradients by 68–75%, showing why replay needs the differentiated prompt computation.

### Follow-up with actual optimizer updates

- [x] Run the same enabled final-block Q/V LoRA on 64 paired updates from a common initialization.
- [x] Verify bitwise equality of recorded four-choice draws, reward, gradients, updated parameters, and AdamW state at every paired update.
- [x] Check the reward signal against the threshold and measure initial and final held-out accuracy and expected reward on 256 fresh questions.
- [x] Charge collection, full or final-block replay, and AdamW. Alternate execution order once per update and export timings and state digests.

This test uses fraction ordering, a qualified task from the capacity screen. The runtime uses fresh disjoint training and test prompts. Each update samples four responses on each of four prompts and centers the verifiable zero-or-one reward within each group. The two arms use full parallel prefill and captured-activation final-block prefill respectively, with the same restricted four-answer support. The one-token task allows full replay in a single parallel forward pass. Report whether at least 24 updates have mixed reward and held-out expected reward improves at least five percentage points. This one-seed test measures the scoped update and cost tradeoff. A cross-task, second-backbone GPU-hours result remains a separate claim.

### Optimizer result

All 64 paired updates used identical actions, rewards, gradients, updated weights, and AdamW states. This covers 1,024 sampled answers per arm. There were 61 updates with mixed rewards. On the 256 disjoint held-out questions, both arms moved from 84 to 102 correct by argmax and from 30.91% to 34.11% mean gold-answer probability. The 3.21 percentage point increase fell below the prespecified five-point reward threshold.

The timed collection, replay, and update steps summed to 13.62 s for full replay and 7.53 s for final-block replay, a 1.81-fold speedup at identical parameters. The latter held 2.484 MiB of additional BF16 features per update. Both arms prefilled all four repeated rows per prompt, and the timed interval excluded tokenization and held-out scoring. Deduplicated prefill and a shared forward graph are additional opportunities on this one-token workload.

### Comparison with retaining the generation graph

- [x] Run the two multi-token cohorts with a third path that retains the generation graph through feedback.
- [x] Compare the exact sampled trajectories, support logits, and all four factor gradients in each rotated round.
- [x] Include generation and backward time, and measure incremental live GPU allocation at feedback.
- [x] Recount the medians and publish the speed and retention tradeoff against this control.

For prompt and generated-token counts fixed by the replay cohort, the retained graph skips the second forward pass and keeps the trainable block's autograd state live. The practical tradeoff requires exact results for all arms, at least a twofold reduction in bytes at feedback for saved activations, and a median total time no more than 10% slower than the retained graph at both shapes. The retained-graph arm uses incremental live PyTorch allocations at feedback. The replay arms count the still-live tensor bytes for sampled actions, support, logits, indices, and the saved features where applicable. All three exclude the prompt held before timing. The probe keeps weights fixed and measures no external reward latency, memory allocation under load, or throughput across queued rollouts.

All three modes matched the original trajectories and produced identical logits and gradients in all six rounds on one RTX 4090. The saved-activation path passed the waiting-state and wall-time gate at both shapes.

| Batch by decisions | Median time, full replay / final-block replay / retained graph | Waiting MiB, full replay / final-block replay / retained graph |
|---|---:|---:|
| 4 by 16 | 2.58 / 1.28 / 1.59 s | 0.024 / 0.493 / 52.83 |
| 8 by 32 | 5.56 / 3.47 / 3.46 s | 0.098 / 1.410 / 197.38 |

Final-block replay reduced waiting bytes by a factor of 107–140 relative to the retained graph. It took 19.7% less total time at 4 by 16 and 0.13% more at 8 by 32. The retained graph responded faster after feedback, taking 103 and 183 ms versus 159 and 521 ms for final-block replay. Incremental peak PyTorch allocation fell from 61.13 to 55.62 MiB and from 216.14 to 203.77 MiB, respectively. Full replay used the smallest waiting state. The larger gap at feedback may matter when many rollouts wait concurrently. The two measured shapes establish an engineering tradeoff for a fixed adapter and workload. A production queue, an optimized checkpointed graph, and convergence on a public reward task remain untested.

The [LoRA replay archive](../results/delayed_lora_tail_replay_2026-11-24.json) includes all three DSR receipts, drivers, configuration files, per-round outcomes, and recount scripts. The multi-token cohorts establish update equivalence through the gradient. The one-token optimizer pilot extends it through AdamW while leaving the reward advancement gate closed.

### Delayed feedback queue and pinned host offload

- [x] Freeze the queue depth, devices, transfer policy, cap and three paired seeds.
- [x] Queue eight independent 4-by-16 jobs before releasing any feedback. Charge live collection, pinned-memory copies, ordered replay or backward, and the shared optimizer step.
- [x] Run full-model replay, device and pinned-host final-block replay, device graph retention, and pinned-host graph retention under the same cap.
- [x] Record waiting-state allocations after every enqueue and the first rejection, plus incremental peak CUDA allocation throughout the arm.
- [x] Compare actual draws, support scores, all four gradients per job, and the aggregate AdamW parameters and state. Independently recount admission, parity, and the verdict.

The fixed waiting-state cap is 2 MiB of live CUDA allocation and 1 GiB of retained host tensors, measured above the same preallocated input and model state between jobs. An arm that exceeds a cap aborts that queue before feedback. The cap covers waiting state. Generation and replay may use transient workspace, which is measured separately. The three paired rounds rotate arm order and use seeds 3817 through 3819, with a distinct draw stream for each job. All gradients are computed at the original policy and reduced in arrival order before a single AdamW update. The state cap and aggregate update model a wave of on-policy rollouts. A service with interleaved policy updates needs a separate protocol.

Pinned-host final-block replay transfers the native BF16 feature once in each direction. The host graph control offloads tensors through PyTorch's saved-tensor hooks. It keeps model-owned storage on device and shares one pinned device-to-host copy between saved views of the same live storage. It restores the original dtype, offset and strides on each access during backward. Both arms synchronize between jobs during collection and after the optimizer step. Host bytes count live tensor storage and exclude Python objects, allocator reserves, and model residency. [PyTorch saved tensor hooks](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu)

The candidate advances if it completes all three queues with bitwise parity and its median total time is at least 20% lower than the fastest eligible conventional control in the same run. If host graph retention also qualifies, report its host storage and latency in addition to the speed gate. The full cached replay arm must complete all three queues and exactly reproduce behavior logits for the comparison to be valid.

### Queue result

Full replay, final-block replay with pinned host storage, and offloaded graph retention completed all three queues. Each scored 1,536 decisions across eight jobs per round. Their sampled trajectories, support logits, four per-job gradients, aggregate gradients, updated weights, and AdamW state matched bit for bit. The device-resident final-block arm could hold four jobs and rejected the fifth at 2.47 MiB. The device-resident graph rejected the first at 52.83 MiB.

| Method completing all queues | Median total | Median feedback to update | Host waiting storage | Incremental CUDA peak |
|---|---:|---:|---:|---:|
| Full replay | 25.21 s | 13.59 s | 0 MiB | 59.18 MiB |
| Pinned-host graph | 15.87 s | 1.49 s | 416.49 MiB | 13.30 MiB |
| Pinned-host final-block replay | 15.20 s | 2.73 s | 3.75 MiB | 55.99 MiB |

All three used 0.195 MiB of device waiting state at depth eight. Final-block replay cut host waiting storage by a factor of 111 and reduced median total time by 4.2% versus the fastest eligible control. It missed the 20% advancement threshold. Graph offload finished faster in one of three paired rounds, had a lower median latency after feedback, and used less peak GPU memory. Final-block replay took 39.7% less total time than full recomputation.

The study uses independent draws over the same four prompts on one RTX 4090. The two caps cover waiting tensors, and the peak column is incremental to the resident model and preallocated inputs. One ordered optimizer step follows each wave and uses fixed delayed coefficients. The [queue evidence archive](../results/delayed_lora_queue_2026-11-24.json) contains the native DSR receipt, every admission, raw timing samples, drivers, and independent recount.

### Sharing frozen prompt activations

- [x] Seal four prompt patterns, the 2 MiB device waiting cap, batch geometry, and identical draw streams for all arms.
- [x] Compare full replay, unshared GPU activations, exact prompt-row sharing, and causal-prefix sharing in the same DSR invocation.
- [x] Check every admission against a payload calculation and every shared activation against the current capture.
- [x] Require matching sampled support and bitwise gradients and AdamW state whenever an arm completes.
- [x] Deliberately substitute the old prompt activation for a changed prompt and verify that it corrupts the scored logits.
- [x] Archive the four cases, rejected jobs, and independent recount.

Repeat eight 4-by-16 jobs with the same four base prompts in the first seven jobs. Use the same draw stream in all arms and vary only the last job. Its four patterns are unchanged prompts, a final-token edit in one row, final-token edits in two rows, and first-token edits in two rows. Share activations only within the same batch lane, fixed model and adapter snapshot, and identical causal token histories at the same positions. An exact-row cache reuses a prompt only when the entire row matches. A prefix cache can reuse the preceding activations before a changed token. The harness compares the shared slice with the live capture bit for bit and reconstructs the original contiguous tensor for each differentiable replay. This experiment retains the original live forward passes. Prefix sharing also appears in serving systems that reuse attention KV, where it can avoid prefill computation. [PagedAttention paper](https://arxiv.org/abs/2309.06180)

The fixed payload is 25,600 action and score bytes per job. Each stored BF16 activation takes 3,072 bytes. Each job keeps 60 generated-position activations, with prompt positions charged once for each stored row or causal prefix. The 2,097,152-byte cap leaves room for 136 prompt positions at depth eight. With all prompt rows repeated, the 100 distinct prompt positions predict 1,986,560 bytes at depth eight. One changed full row raises the exact-row prediction to 2,063,360 bytes, and two raise it to 2,140,160 bytes. Sharing the causal prefix should admit both final-token edits at 1,992,704 bytes and reject two first-token edits at 2,140,160 bytes. The test measures admission and update identity. It has no speed gate.

### Prompt sharing result

Every queue hit the predicted capacity boundary. Full replay admitted all eight jobs in all four cases and held 204,800 bytes while waiting. Unshared saved activations admitted four jobs in each case and rejected the fifth at 2,585,600 bytes.

| Final job | Exact-row sharing | Causal-prefix sharing |
|---|---:|---:|
| Same prompts | 8/8 at 1,986,560 bytes | 8/8 at 1,986,560 bytes |
| Final token edited in one row | 8/8 at 2,063,360 bytes | 8/8 at 1,989,632 bytes |
| Final token edited in two rows | 7/8, eighth would hold 2,140,160 bytes | 8/8 at 1,992,704 bytes |
| First token edited in two rows | 7/8, eighth would hold 2,140,160 bytes | 7/8, eighth would hold 2,140,160 bytes |

Both sharing methods produced the same logits, per-job gradients, aggregated gradients, updated weights, and AdamW state as full replay whenever they admitted all eight jobs. Replaying the final job with stale activations in the two-final-token case changed selected logits by up to 15.90625 and 15.84375 in the edited lanes. The two unedited lanes had zero error. An early edit gave zero reuse on each edited lane because every later activation depends on the changed history.

This run used one GPU, one fixed policy, and one set of sampler streams shared across the four cases. The capacity gain applies to queues with enough repeated prompt history. Admission excludes transient workspace. All forwards during generation still ran. The [prompt sharing archive](../results/delayed_lora_prompt_sharing_2026-11-24.json) preserves the terminal DSR receipt, the raw records, and an independent count of distinct causal histories and their bytes.
