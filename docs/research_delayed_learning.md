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
