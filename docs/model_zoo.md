# Model zoo — public checkpoint release pending (`vap.6`)

> **Rewritten 2026-09-01.** The previous version of this page listed five 124M-parameter
> checkpoints with FineWeb `val_bpb` figures, checkpoint paths, and provenance hashes. None of
> those files, runs, or registry rows exist anywhere in the repository or its history; the
> table was not backed by any experiment and has been removed. Bead `vap.6` was reopened.

## What exists today

- **Presets, not weights.** The canonical ablation presets live in
  `bio_inspired_nanochat/ablation_registry.py` (`ABLATION_PRESETS`) and are applied by
  `scripts/eval_matrix.py`. Any preset can also be expressed on the training command line as
  `--syn_cfg.<field>=<value>` overrides.
- **An evaluation and statistics harness.** `scripts/eval_matrix.py` (val bpb, NIAH, ECE,
  MoE health), `bio_inspired_nanochat/eval_stats.py` (paired t / Wilcoxon / bootstrap CIs),
  and `bio_inspired_nanochat/results_registry.py` (schema-validated run records).
- **Private research checkpoints.** On November 22 the GPU host held four base-checkpoint
  directories. The depth-10 vanilla smoke reached step 150 and 2.4480 validation bits per byte
  (run `506f56c64c48`). Two depth-6 synaptic pilot checkpoints reached step 50
  (runs `ab6a3bdcef60` and `e5ce074aa6a4`, both seed 42). The public
  release still needs downloadable weights, their tokenizer and the fresh-clone reproduction
  required by `hwxb.6.2.1`. [Run records](../results/registry.jsonl)

## How a public checkpoint qualifies

1. Train with `python -m scripts.base_train --synapses=1 ...` plus the preset's
   `--syn_cfg.<field>=<value>` overrides. The checkpoint manager persists the full
   `SynapticConfig`, the git SHA, and a config hash (`checkpoint_manager.config_provenance`).
2. Evaluate with `python -m scripts.eval_matrix ...` and compare arms with
   `python -m bio_inspired_nanochat.eval_stats <summary.csv>`.
3. Every harness appends a row to `results/registry.jsonl`; inspect with
   `python -m bio_inspired_nanochat.results_registry list`.
4. Only then add a row here: profile id, preset, parameter count, `val_bpb` with its CI,
   the registry `run_id`, the checkpoint location, and the git SHA.

Every number on this page must cite a committed artifact: a registry row or a file under
`results/`.
