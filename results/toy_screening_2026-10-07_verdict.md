# Statistical comparison: `val_bpb`

- Baseline: `vanilla`
- Direction: lower is better
- Familywise alpha: `0.05` with Holm correction
- Minimum matched seeds for an inferential verdict: `2`
- Support requires a favorable paired-bootstrap 95% CI and both adjusted paired tests.

| Preset | n | Mean ± sample SD (Student-t 95% CI) | Delta vs baseline | Adjusted paired-t p | Adjusted Wilcoxon p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `vanilla` | 3 | 1.95808 ± 0.00218575 [1.95265, 1.96351] | — | — | — | baseline |
| `synaptic_off` | 3 | 2.02751 ± 0.00898839 [2.00518, 2.04984] | +0.0694309 [+0.0640354, +0.0800564] | 0.009222 | 0.5 | `null` |
| `bio_all` | 3 | 2.0382 ± 0.00918713 [2.01538, 2.06102] | +0.0801263 [+0.0743748, +0.0910406] | 0.009222 | 0.5 | `null` |

`null` means the preregistered support rule did not pass; it is not evidence of equivalence. `insufficient_evidence` means too few matched seeds were available for the declared minimum.

## Provenance (2026-10-07)

- Recipe: `scripts.matrix_launch --columns vanilla,synaptic_off,bio_all --seeds 1337,1338,1339`
  with `--depth=2 --max_seq_len=256 --device_batch_size=8 --total_batch_size=2048 --num_iterations=300`
  (614,400 training tokens), then `scripts.eval_matrix matrix --eval-bpb --eval-tokens 65536` on the
  held-out shard and `bio_inspired_nanochat.eval_stats --min-pairs 2`. Same design as the
  2026-09-02 screening, whose verdict is superseded: its seeds were identical runs and its bio_all
  diverged (fixed in 636e955 and 8d1a3d0).
- Data: WikiText-2 (pytorch/examples copy) as two parquet shards, train / valid+test, with a
  4,096-token tokenizer; FineWeb-Edu was unreachable from this host. CPU, one thread per run.
- Code: 716cacb, i.e. **bio_all still included online Hebbian plasticity** (it became opt-in later
  the same day), scalar LR 0.01, scripted presyn scan.
- Training tok/s (one thread): vanilla 2,871, synaptic_off 2,629, bio_all 1,382.
- Reading: the synaptic substrate with every mechanism off costs +0.069 bpb against vanilla on all
  three seeds, and the default mechanism stack adds +0.011 more. With n = 3 the exact Wilcoxon cannot
  pass the Holm-adjusted 0.05, so the pre-registered rule reports `null` even though every paired
  difference has the same sign. 2 layers x 128 dims says nothing about D1.
