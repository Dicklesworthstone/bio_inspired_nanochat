# Statistical comparison: `val_bpb`

- Baseline: `vanilla`
- Direction: lower is better
- Familywise alpha: `0.05` with Holm correction
- Minimum matched seeds for an inferential verdict: `2`
- Support requires a favorable paired-bootstrap 95% CI and both adjusted paired tests.

| Preset | n | Mean ± sample SD (Student-t 95% CI) | Delta vs baseline | Adjusted paired-t p | Adjusted Wilcoxon p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `vanilla` | 2 | 0.956949 ± 0.000325543 [0.954024, 0.959874] | — | — | — | baseline |
| `synaptic_off` | 2 | 0.961802 ± 0.000856848 [0.954103, 0.9695] | +0.00485279 [+0.00401672, +0.00568887] | 1 | 1 | `null` |
| `bio_all` | 2 | 0.961701 ± 0.00113696 [0.951486, 0.971916] | +0.00475193 [+0.00371778, +0.00578607] | 1 | 1 | `null` |
| `bio_no_presyn` | 2 | 0.961895 ± 0.000880151 [0.953987, 0.969803] | +0.00494652 [+0.00409396, +0.00579907] | 1 | 1 | `null` |
| `bio_no_stochastic_release` | 2 | 0.961432 ± 0.00129242 [0.94982, 0.973044] | +0.00448283 [+0.00333876, +0.0056269] | 1 | 1 | `null` |
| `bio_no_doc2` | 2 | 0.961493 ± 0.00125017 [0.950261, 0.972725] | +0.00454425 [+0.00343005, +0.00565844] | 1 | 1 | `null` |
| `bio_no_septin_barrier` | 2 | 0.961619 ± 0.00133658 [0.949611, 0.973628] | +0.0046704 [+0.0034951, +0.0058457] | 1 | 1 | `null` |
| `add_hebbian` | 2 | 0.957915 ± 7.84021e-06 [0.957845, 0.957985] | +0.000966173 [+0.000730436, +0.00120191] | 1 | 1 | `null` |
| `add_bdnf` | 2 | 0.957915 ± 7.62243e-06 [0.957847, 0.957984] | +0.000966327 [+0.000730744, +0.00120191] | 1 | 1 | `null` |
| `add_bistable_latch` | 2 | 0.957918 ± 1.29364e-05 [0.957802, 0.958035] | +0.00096953 [+0.000748484, +0.00119058] | 1 | 1 | `null` |
| `add_stdp` | 2 | 0.961021 ± 0.00124015 [0.949879, 0.972164] | +0.00407246 [+0.00342574, +0.00471919] | 1 | 1 | `null` |
| `add_native_presyn` | 2 | 0.961498 ± 0.00119337 [0.950776, 0.97222] | +0.00454942 [+0.00347539, +0.00562345] | 1 | 1 | `null` |
| `add_learnable_kinetics` | 2 | 0.96152 ± 0.00123701 [0.950406, 0.972634] | +0.00457104 [+0.00346615, +0.00567594] | 1 | 1 | `null` |
| `add_differentiable_recurrence` | 2 | 0.961515 ± 0.00107716 [0.951838, 0.971193] | +0.00456667 [+0.00357481, +0.00555853] | 1 | 1 | `null` |
| `add_cusp_latch` | 2 | 0.957915 ± 2.97928e-05 [0.957648, 0.958183] | +0.000966543 [+0.000757416, +0.00117567] | 1 | 1 | `null` |
| `add_metriplectic_integrator` | 2 | 0.961522 ± 0.00120617 [0.950685, 0.972359] | +0.00457351 [+0.00349042, +0.00565659] | 1 | 1 | `null` |
| `add_neuromod` | 2 | 0.95607 ± 0.00167423 [0.941027, 0.971112] | -0.000879011 [-0.00183268, +7.46574e-05] | 1 | 1 | `null` |

`null` means the preregistered support rule did not pass; it is not evidence of equivalence. `insufficient_evidence` means too few matched seeds were available for the declared minimum.

## Provenance (2026-10-13)

- Recipe: Full 17-column dense screening matrix (`vanilla`, `synaptic_off`, `bio_all`, `bio_no_presyn`, `bio_no_stochastic_release`, `bio_no_doc2`, `bio_no_septin_barrier`, `add_hebbian`, `add_bdnf`, `add_bistable_latch`, `add_stdp`, `add_native_presyn`, `add_learnable_kinetics`, `add_differentiable_recurrence`, `add_cusp_latch`, `add_metriplectic_integrator`, `add_neuromod`) across seeds `1337,1338` (`34` cells total) via `scripts.matrix_launch --stage screening --seeds 1337,1338 --execute` with `--depth=2 --max_seq_len=64 --device_batch_size=4 --total_batch_size=256 --num_iterations=20`, evaluated via `scripts.eval_matrix matrix --eval-bpb --eval-tokens 512` and `bio_inspired_nanochat.eval_stats --min-pairs 2`.
- Scaffold fixes included: `GPTSynapticConfig.final_norm = True` (restoring the pre-lm_head RMSNorm omitted on the synaptic trunk prior to `03bb9bf`), `SynapticConfig.enable_hebbian = False` by default in `bio_all` (online Hebbian trace updates isolated to `add_hebbian` / `add_bdnf` / `add_bistable_latch` / `add_cusp_latch` since `f96747e`), and meta-device initialization fixes for `add_cusp_latch` (`bio_inspired_nanochat/cusp_certificate.py:epsilon_gauge`) and `moe_add_glial_homeostasis` (`bio_inspired_nanochat/glial_homeostasis.py`).
- Companion 3-seed, 300-iteration (614,400-token) anchor comparison on the same fixed `final_norm=True` / `enable_hebbian=False` scaffold (`results/scaffold_diagnosis_2026-10-07.json`): `vanilla` `1.9581 ± 0.0022 bpb`, `synaptic_off (final_norm=True)` `1.9198 ± 0.0040 bpb` (`-0.0383 bpb` vs `vanilla`), `bio_all (final_norm=True, enable_hebbian=False)` `1.9301 ± 0.0046 bpb` (`-0.0280 bpb` vs `vanilla`).
- Reading: All 17 dense columns and all 5 structural MoE columns (`results/structural_smoke_2026-10-13.json`) now train, checkpoint, evaluate, and score end-to-end without meta-device crashes or the earlier `+0.069 bpb` missing-norm penalty (`results/toy_screening_2026-10-07_verdict.md`). Notice that in this 20-step micro-screening sweep `add_neuromod` achieves the lowest mean `val_bpb` (`0.95607 ± 0.00167`, `-0.00088 bpb` vs `vanilla`), while the Hebbian-family columns (`add_hebbian`, `add_bdnf`, `add_bistable_latch`, `add_cusp_latch`) cluster within `+0.00097 bpb` of `vanilla`. Full D1 (`depth=12`) confirmation remains pending GPU execution (`bio_inspired_nanochat-hwxb.10`).
