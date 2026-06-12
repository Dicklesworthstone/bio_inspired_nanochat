"""Ablation-harness dry-run (bead hwxb.7.4).

Runs the FULL pre-registered ablation matrix (``ablation_matrix``, hwxb.5.1) end-to-end on TINY
models / short budgets, through the real reduced-training harness (``e2e_harness``, hwxb.7.3), and
pushes the results through the real statistics + verdict pipeline (``eval_stats``, 74f.3). The point
is to PROVE the whole experiment machinery works — matrix → per-cell runs → summary rows (in the
``eval_matrix`` schema) → multi-seed aggregation → paired verdict table → go/no-go gate — **before**
any real 4090 time is spent on ``hwxb.5.2``. The only thing the real run changes is the scale.

This validates the *machinery*, not the *science*: the tiny harness trains a fixed memorizable pool,
so the per-config metric is a loss proxy and the cross-config differences are noise. The assertions
therefore check that the pipeline is well-formed (every cell ran, rows are schema-valid, every
non-baseline config gets a paired test, the decomposition contrasts compute, the gate fires) — never
that a mechanism "helps".
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from bio_inspired_nanochat import ablation_matrix as am
from bio_inspired_nanochat.e2e_harness import E2EConfig, run_e2e
from bio_inspired_nanochat.eval_stats import compare_matrix, paired_comparison

log = logging.getLogger("bio_inspired_nanochat.ablation_dryrun")

# Summary-row schema (the subset of eval_matrix.SUMMARY_FIELDS the dry-run populates). Keeping the
# names identical means load_matrix_csv / compare_matrix read our rows exactly as the real ones.
SUMMARY_FIELDS: tuple[str, ...] = (
    "status", "preset", "seed", "role", "val_bpb", "final_loss", "e2e_passed", "n_failed_invariants",
)

# In the dry-run the e2e final loss stands in for the primary metric so the real stats path exercises.
PROXY_METRIC: str = "val_bpb"


@dataclass
class DryRunConfig:
    """How big the dry-run is. Defaults are CI-fast; the real matrix only changes the scale."""

    columns: list[am.AblationConfig] = field(default_factory=am.screening_columns)
    seeds: tuple[int, ...] = am.SCREENING_SEEDS
    e2e_steps: int = 40
    n_layer: int = 2
    n_embd: int = 64
    seq_len: int = 32
    tok_per_sec: float = 20_000.0  # planning throughput for the GPU-hour estimate / gate
    run_dir: Optional[Path] = None


def _e2e_config_for(column: am.AblationConfig, seed: int, cfg: DryRunConfig) -> E2EConfig:
    """Map a matrix column to a reduced E2E run config (the override semantics of the three bases)."""
    if column.base is am.Base.VANILLA:
        synapses, overrides = False, {}
    elif column.base is am.Base.SYNAPTIC_OFF:
        synapses = True
        overrides = {**am.SYNAPTIC_OFF_OVERRIDES, **column.overrides}
    else:  # BIO_ALL
        synapses, overrides = True, dict(column.overrides)
    return E2EConfig(
        synapses=synapses,
        n_layer=cfg.n_layer,
        n_embd=cfg.n_embd,
        seq_len=cfg.seq_len,
        steps=cfg.e2e_steps,
        seed=seed,
        syn_overrides=overrides,
        # The dry-run only proves orchestration; don't fail a cell on the tiny-scale "loss must drop"
        # heuristic (a 40-step toy run is noisy). Health (finite grads/params) is still asserted.
        loss_decrease_required=False,
    )


def _run_cell(column: am.AblationConfig, seed: int, cfg: DryRunConfig) -> dict[str, Any]:
    """Run one (config, seed) cell and return a summary row in the eval_matrix schema."""
    try:
        report = run_e2e(_e2e_config_for(column, seed, cfg), verbose=False)
        final = report.summary.get("final_loss")
        return {
            "status": "ok",
            "preset": column.config_id,
            "seed": seed,
            "role": column.role,
            "val_bpb": final,
            "final_loss": final,
            "e2e_passed": report.passed,
            "n_failed_invariants": len(report.failures()),
        }
    except Exception as e:  # a crashed cell is a row with status=error, never a crashed sweep
        log.warning("dry-run cell %s seed=%d crashed: %s", column.config_id, seed, e)
        return {
            "status": "error", "preset": column.config_id, "seed": seed, "role": column.role,
            "val_bpb": None, "final_loss": None, "e2e_passed": False, "n_failed_invariants": -1,
        }


def _write_rows(out_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, jsonl_path = out_dir / "summary.csv", out_dir / "summary.jsonl"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in SUMMARY_FIELDS})
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return csv_path, jsonl_path


@dataclass
class DryRunResult:
    rows: list[dict[str, Any]]
    csv_path: Path
    jsonl_path: Path
    verdict: dict                       # eval_stats.compare_matrix report (baseline = vanilla)
    decomposition: dict[str, Any]       # the three architecture-vs-mechanism contrasts
    gate: am.GoNoGo
    n_cells: int
    n_ok: int


def _metric_by_seed(rows: list[dict[str, Any]], preset: str) -> dict[int, float]:
    return {
        int(r["seed"]): float(r["val_bpb"])
        for r in rows
        if r["preset"] == preset and r["status"] == "ok" and r["val_bpb"] is not None
    }


def _decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The architecture-vs-mechanism contrasts the spec is built around (paired across seeds)."""
    out: dict[str, Any] = {}
    van = _metric_by_seed(rows, "vanilla")
    off = _metric_by_seed(rows, "synaptic_off")
    allc = _metric_by_seed(rows, "bio_all")

    def _pc(treat: dict[int, float], base: dict[int, float]):
        res = paired_comparison(treat, base, lower_is_better=True)
        return None if res is None else {"mean_delta": res.mean_delta, "t_p_value": res.t_p_value,
                                         "wilcoxon_p_value": res.wilcoxon_p_value, "n_pairs": res.n_pairs}

    out["architecture_effect (synaptic_off - vanilla)"] = _pc(off, van)
    out["total_bio_effect (bio_all - vanilla)"] = _pc(allc, van)
    # Each mechanism's CLEAN isolated effect, taken against synaptic_off where the column is add-one-in.
    for r_preset in sorted({r["preset"] for r in rows if r["role"] == "add_one_in"}):
        out[f"mechanism ({r_preset} - synaptic_off)"] = _pc(_metric_by_seed(rows, r_preset), off)
    return out


def run_ablation_dryrun(cfg: Optional[DryRunConfig] = None) -> DryRunResult:
    """Run the full matrix on tiny models and drive the real stats + verdict + gate pipeline."""
    cfg = cfg or DryRunConfig()
    out_dir = Path(cfg.run_dir) if cfg.run_dir else (Path("runs") / "ablation_dryrun")
    rows = [_run_cell(col, seed, cfg) for col in cfg.columns for seed in cfg.seeds]
    csv_path, jsonl_path = _write_rows(out_dir, rows)

    data = {p: _metric_by_seed(rows, p) for p in {r["preset"] for r in rows}}
    data = {p: v for p, v in data.items() if len(v) >= 1}
    verdict = compare_matrix(data, baseline="vanilla", metric=PROXY_METRIC, lower_is_better=True)
    decomposition = _decomposition(rows)

    # The gate would run after a real screening pass; here we exercise it with every non-anchor
    # column as a notional survivor so the path is covered end-to-end.
    survivors = [c.config_id for c in cfg.columns if c.role != "anchor"]
    gate = am.go_no_go(survivors, tok_per_sec=cfg.tok_per_sec)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    log.info("ablation dry-run: %d/%d cells ok; verdict over %d presets; gate proceed=%s",
             n_ok, len(rows), len(verdict["presets"]), gate.proceed)
    return DryRunResult(rows, csv_path, jsonl_path, verdict, decomposition, gate, len(rows), n_ok)


def render_verdict_table(result: DryRunResult) -> str:
    """A compact human-readable table of the dry-run verdict + decomposition + gate."""
    lines = [
        f"ablation dry-run — {result.n_ok}/{result.n_cells} cells ok, "
        f"metric={PROXY_METRIC} (loss proxy at tiny scale; validates MACHINERY not science)",
        "",
        f"{'preset':<26}{'role':<14}{'n':>2}  {'mean':>9}  {'Δ vs vanilla':>13}  verdict",
    ]
    role_by_preset = {r["preset"]: r["role"] for r in result.rows}
    for preset, e in result.verdict["presets"].items():
        a = e["aggregate"]
        role = role_by_preset.get(preset, "")
        delta = f"{e['paired_vs_baseline']['mean_delta']:+.4g}" if "paired_vs_baseline" in e else "—"
        verdict = ""
        if "better" in e:
            verdict = ("better" if e["better"] else "worse") + (" *" if e.get("significant") else "")
        lines.append(f"{preset:<26}{role:<14}{a['n']:>2}  {a['mean']:>9.4g}  {delta:>13}  {verdict}")
    lines += ["", "decomposition (paired across seeds):"]
    for name, pc in result.decomposition.items():
        if pc is None:
            lines.append(f"  {name}: (need >=2 shared seeds)")
        else:
            lines.append(f"  {name}: Δ={pc['mean_delta']:+.4g}  t_p={pc['t_p_value']:.3g}  "
                         f"W_p={pc['wilcoxon_p_value']:.3g}  (n={pc['n_pairs']})")
    g = result.gate
    lines += ["", f"go/no-go: proceed={g.proceed}  est={g.estimated_gpu_hours:.1f} GPU-h "
              f"(cap {g.cap_gpu_hours:.0f})  survivors={g.n_survivors}  — {g.reason}"]
    return "\n".join(lines)
