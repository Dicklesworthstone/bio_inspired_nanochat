# Model Zoo (bead `vap.6`) — status: no checkpoints exist yet

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
- **No trained model.** The largest model trained under version control is 2 layers × 64
  dims on synthetic tokens on CPU, and `results/registry.jsonl` has no `val_bpb` row. The
  pre-registered experiment that produces the first real checkpoints is
  `docs/ablation_matrix.md` + `docs/scale_up_phase0_decisions.md` (beads `hwxb.3`–`hwxb.6`),
  gated on the dual-RTX-4090 host.

## How a zoo entry gets created (once a GPU run exists)

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
