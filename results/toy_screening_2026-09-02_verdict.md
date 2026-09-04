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
