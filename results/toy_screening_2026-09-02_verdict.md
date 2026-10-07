# Statistical comparison: `val_bpb`

- Baseline: `vanilla`
- Direction: lower is better
- Familywise alpha: `0.05` with Holm correction
- Minimum matched seeds for an inferential verdict: `2`
- Support requires a favorable paired-bootstrap 95% CI and both adjusted paired tests.

| Preset | n | Mean ± sample SD (Student-t 95% CI) | Delta vs baseline | Adjusted paired-t p | Adjusted Wilcoxon p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `vanilla` | 2 | 2.32151 ± 0 [2.32151, 2.32151] | — | — | — | baseline |
| `synaptic_off` | 2 | 2.43751 ± 0 [2.43751, 2.43751] | +0.115997 [+0.115997, +0.115997] | 0 | 1 | `null` |
| `bio_all` | 2 | 6.97834 ± 6.15094 [-48.2857, 62.2424] | +4.65683 [+0.307458, +9.00621] | 0.4783 | 1 | `null` |

`null` means the preregistered support rule did not pass; it is not evidence of equivalence. `insufficient_evidence` means too few matched seeds were available for the declared minimum.

**Superseded 2026-10-07** by `toy_screening_2026-10-07_verdict.md`. This run is invalid: `--init_seed`
never reached weight init, so seeds 1337 and 1338 were the same run (vanilla and synaptic_off losses
equal to 16 digits, std 0), and bio_all seed 1338 diverged because GPTSynaptic's 1-D gains trained at
AdamW LR 0.49 (fixed in 636e955 and 8d1a3d0).
