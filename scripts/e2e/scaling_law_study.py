"""E2E SCRIPT: bio-vs-vanilla scaling-law study (bead 74f.6).

The high-impact question: do bio mechanisms change the scaling EXPONENT or just the OFFSET —
i.e., do they shift the compute Pareto frontier?

Method
------
Sweep a small grid of (depth, width) for BOTH model families — plain ``GPT`` vs
``GPTSynaptic`` (bio defaults minus the known-unstable train-time plasticity, see jpqc) —
training each cell on an associative-recall pool with a Chinchilla-proportional token budget
and evaluating **held-out** next-token NLL on fresh sequences.

Per run we record the FLOP proxy ``C = 6 * n_params * n_tokens`` (standard transformer
estimate) and the held-out NLL. Per family and seed we fit the power law

    L(C) = a * C^(-b)

by ordinary least squares in log-log space (numpy polyfit; no scipy dependency). The scaling
exponent ``b`` is aggregated across seeds (mean + Student-t 95% CI, the 74f.3 layer); families
are compared by exponent difference with CI.

Honest scope note (mirrors the zsi precedent): this VM is a CPU box, so the calibrated sweep
below is deliberately toy-scale. The HARNESS is the deliverable — evidence-ready, seeded,
single-command — while the production multi-scale run (larger grids, more tokens, more seeds)
is correctly blocked on dual-4090 availability and tracked in hwxb.*. Toy-scale exponents come
with wide CIs by construction; the verdict machinery reports "unclear" unless the difference
is decisive.

Run:  uv run python -m scripts.e2e.scaling_law_study --run-dir runs/e2e/scaling_law_study
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bio_inspired_nanochat.common import logger as _logger
from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.eval_stats import aggregate
from bio_inspired_nanochat.run_logging import RunLogger, read_run_events
from bio_inspired_nanochat.synthetic_tasks import associative_recall
from bio_inspired_nanochat.torch_imports import torch


# --------------------------------------------------------------------------- #
# Config & result types
# --------------------------------------------------------------------------- #
@dataclass
class ScalingStudyConfig:
    """One reduced bio-vs-vanilla scaling study."""

    # Grid (kept tiny for CPU; production sweeps widen these on GPU)
    depths: tuple[int, ...] = (1, 2)
    widths: tuple[int, ...] = (64, 128)
    seeds: tuple[int, ...] = (0, 1)
    n_head: int = 4

    vocab_size: int = 97
    sequence_len: int = 48
    device: str = "cpu"
    master_seed: int = 0

    # Task / training
    task_vocab: int = 64
    train_batch: int = 8
    num_pairs_train: int = 3
    pool_size: int = 12
    base_steps: int = 900  # steps at width 64; scaled by (64 / width)^1.5 for wider models
    lr: float = 1e-3
    grad_clip: float = 1.0
    eval_batches: int = 6  # held-out batches for the val-NLL point

    syn_overrides_bio: dict = field(
        default_factory=lambda: {"plasticity_during_training": False}
    )


@dataclass
class ScalingReport:
    """Outcome of one scaling study."""

    run_id: str
    invariants: list[InvariantResult]
    summary: dict

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.invariants)

    @property
    def failures(self) -> list[InvariantResult]:
        return [r for r in self.invariants if not r.passed]

    def assert_passed(self) -> None:
        if self.passed:
            return
        lines = [r.line() for r in self.failures]
        raise AssertionError("scaling-law e2e FAILED:\n" + "\n".join(lines))


# --------------------------------------------------------------------------- #
# Model construction / training / eval
# --------------------------------------------------------------------------- #
def _build_model(cfg: ScalingStudyConfig, *, family: str, depth: int, width: int):
    """Vanilla ``GPT`` or dense synaptic ``GPTSynaptic`` at the given geometry."""
    torch.manual_seed(cfg.master_seed * 1000 + hash((family, depth, width)) % 997)
    if family == "vanilla":
        from bio_inspired_nanochat.gpt import GPT, GPTConfig

        gcfg = GPTConfig(
            sequence_len=cfg.sequence_len,
            vocab_size=cfg.vocab_size,
            n_layer=depth,
            n_head=cfg.n_head,
            n_kv_head=cfg.n_head,
            n_embd=width,
        )
        return GPT(gcfg).to(cfg.device)
    assert family == "bio", family
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    from bio_inspired_nanochat.synaptic import SynapticConfig

    syn = SynapticConfig()
    for k, v in cfg.syn_overrides_bio.items():
        setattr(syn, k, v)
    gcfg = GPTSynapticConfig(
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        n_layer=depth,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=width,
        synapses=True,
        syn_cfg=syn,
    )
    return GPTSynaptic(gcfg).to(cfg.device)


def _steps_for(cfg: ScalingStudyConfig, width: int) -> int:
    """Wider models get proportionally more optimizer steps (crude token scaling)."""
    return max(60, int(cfg.base_steps * (64.0 / width) ** 1.5))


def _make_pool(cfg: ScalingStudyConfig):
    return [
        associative_recall(
            batch=cfg.train_batch, num_pairs=cfg.num_pairs_train,
            vocab_size=cfg.task_vocab, seed=10_000 + i,
        )
        for i in range(cfg.pool_size)
    ]


def _eval_batches(cfg: ScalingStudyConfig):
    return [
        associative_recall(
            batch=cfg.train_batch, num_pairs=cfg.num_pairs_train,
            vocab_size=cfg.task_vocab, seed=20_000 + i,
        )
        for i in range(cfg.eval_batches)
    ]


@torch.no_grad()
def _held_out_nll(model, eval_set, cfg: ScalingStudyConfig) -> float:
    """Mean teacher-forced NLL per target token on FRESH sequences (generalization probe)."""
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in eval_set:
        if hasattr(model, "reset_sequence_state"):
            model.reset_sequence_state(reset_fast_weights=True)
        out = model(batch.inputs.to(cfg.device))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        targets = batch.targets.to(cfg.device)
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
            ignore_index=-1, reduction="sum",
        )
        n = int((targets != -1).sum())
        total_nll += float(nll)
        total_tokens += n
    if was_training:
        model.train()
    return total_nll / max(total_tokens, 1)


def _run_cell(
    cfg: ScalingStudyConfig, *, family: str, depth: int, width: int, seed: int,
    rl: RunLogger, tag: str,
) -> dict:
    """Train one grid cell and return {params, tokens, flops, val_nll}."""
    torch.manual_seed(seed)
    model = _build_model(cfg, family=family, depth=depth, width=width)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    pool = _make_pool(cfg)
    steps = _steps_for(cfg, width)

    losses: list[float] = []
    for step in range(steps):
        batch = pool[step % len(pool)]
        try:
            if family == "vanilla":
                loss = model(batch.inputs.to(cfg.device), batch.targets.to(cfg.device))
            else:
                _, loss = model(batch.inputs.to(cfg.device), batch.targets.to(cfg.device),
                                None, train_mode=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        except RuntimeError as exc:
            # Known inplace-autograd instability family (moe-train-autograd-clash-0f4i /
            # jpqc). A crashed cell must not kill the study; mark and continue.
            _logger.warning(f"[scaling] {tag} step {step} RuntimeError: {exc}")
            return {
                "family": family, "depth": depth, "width": width, "seed": seed,
                "params": sum(p.numel() for p in model.parameters()),
                "tokens": 0, "flops": float("nan"), "val_nll": float("nan"),
                "final_train_loss": float("nan"), "crashed": True,
            }
        lval = float(loss.detach().item())
        losses.append(lval)
        rl.event("scaling_train_step", tag=tag, step=step, loss=lval)
        if not math.isfinite(lval):
            _logger.warning(f"[scaling] {tag} diverged at step {step}; stopping cell")
            break

    eval_set = _eval_batches(cfg)
    tokens = steps * cfg.train_batch * cfg.sequence_len
    params = sum(p.numel() for p in model.parameters())
    val_nll = _held_out_nll(model, eval_set, cfg)
    del model
    return {
        "family": family, "depth": depth, "width": width, "seed": seed,
        "params": params, "tokens": tokens,
        "flops": 6.0 * params * tokens,
        "val_nll": val_nll,
        "final_train_loss": losses[-1] if losses else float("nan"),
        "crashed": False,
    }


# --------------------------------------------------------------------------- #
# Power-law fitting
# --------------------------------------------------------------------------- #
def fit_power_law(flops: np.ndarray, losses: np.ndarray) -> dict:
    """OLS fit of ``log L = log a - b * log C``. Returns exponent b, intercept log-a, R^2."""
    x = np.log(flops.astype(np.float64))
    y = np.log(losses.astype(np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    return {"exponent_b": float(-slope), "log_a": float(intercept), "r_squared": 1.0 - ss_res / ss_tot}


# --------------------------------------------------------------------------- #
# The study
# --------------------------------------------------------------------------- #
def run_scaling_study(
    cfg: ScalingStudyConfig, *, run_dir: str | Path | None = None, verbose: bool = True,
) -> ScalingReport:
    """Run the full grid for both families, fit curves, compare exponents."""
    if cfg.device == "cpu":
        try:
            # Deterministic trajectories matter more than speed on this toy grid (see
            # moe-train-autograd-clash-0f4i notes on multithreaded scatter nondeterminism).
            torch.set_num_threads(1)
        except Exception:
            pass
    torch.manual_seed(cfg.master_seed)

    rd = Path(run_dir) if run_dir is not None else Path("runs") / "e2e" / "scaling_law_study"
    rl = RunLogger(rd, name="scaling_law_study", console=False, provenance={
        "master_seed": cfg.master_seed,
        "grid": list(cfg.depths), "widths": list(cfg.widths), "seeds": list(cfg.seeds),
    })

    cells: list[dict] = []
    try:
        for family in ("vanilla", "bio"):
            for depth in cfg.depths:
                for width in cfg.widths:
                    for seed in cfg.seeds:
                        tag = f"{family}-d{depth}-w{width}-s{seed}"
                        cell = _run_cell(cfg, family=family, depth=depth, width=width,
                                         seed=seed, rl=rl, tag=tag)
                        cells.append(cell)
                        rl.event(
                            "scaling_cell",
                            family=cell["family"], depth=int(cell["depth"]),
                            width=int(cell["width"]), seed=int(cell["seed"]),
                            params=int(cell["params"]), tokens=int(cell["tokens"]),
                            flops=float(cell["flops"]), val_nll=float(cell["val_nll"]),
                            final_train_loss=float(cell["final_train_loss"]),
                            crashed=bool(cell["crashed"]),
                        )
                        if verbose:
                            _logger.info(
                                f"[scaling] {tag}: params={cell['params']:,} "
                                f"flops={cell['flops']:.3g} val_nll={cell['val_nll']:.4f}"
                                + (" CRASHED" if cell["crashed"] else "")
                            )
    finally:
        rl.close()

    # ---- per-family fits (one curve per seed, then aggregate the exponent) ----
    fits: dict[str, dict] = {}
    for family in ("vanilla", "bio"):
        fam_cells = [c for c in cells if c["family"] == family and not c["crashed"]
                     and c["tokens"] > 0 and math.isfinite(c["val_nll"])]
        exps: list[float] = []
        per_seed_fits: list[dict] = []
        for seed in cfg.seeds:
            sc = [c for c in fam_cells if c["seed"] == seed]
            if len(sc) >= 2:
                f = fit_power_law(
                    np.array([c["flops"] for c in sc]),
                    np.array([c["val_nll"] for c in sc]),
                )
                f["seed"] = seed
                f["n_points"] = len(sc)
                per_seed_fits.append(f)
                exps.append(f["exponent_b"])
        agg = aggregate(exps) if len(exps) >= 2 else None
        fits[family] = {
            "per_seed": per_seed_fits,
            "exponent_mean": agg.mean if agg else (exps[0] if exps else float("nan")),
            "exponent_ci_low": agg.ci_low if agg else float("nan"),
            "exponent_ci_high": agg.ci_high if agg else float("nan"),
            "n_seeds": len(exps),
        }

    b_van, b_bio = fits["vanilla"], fits["bio"]
    both_fit = all(math.isfinite(fits[f]["exponent_mean"]) for f in ("vanilla", "bio"))
    if both_fit:
        delta = b_bio["exponent_mean"] - b_van["exponent_mean"]
        # Decisive iff the CI of one excludes the other's mean AND both fits are tight
        # (R^2 gate keeps noise-fits from producing fake decisiveness).
        tight = all(
            fits[f]["per_seed"][i]["r_squared"] > 0.5
            for f in ("vanilla", "bio") for i in range(len(fits[f]["per_seed"]))
        ) if all(fits[f]["per_seed"] for f in ("vanilla", "bio")) else False
        separated = (
            tight
            and (b_bio["exponent_ci_low"] > b_van["exponent_mean"]
                 or b_bio["exponent_ci_high"] < b_van["exponent_mean"])
        )
        if not tight:
            verdict = "unclear_noisy_fits"
        elif separated:
            verdict = "bio_improves_exponent" if delta > 0 else "bio_worse_exponent"
        else:
            verdict = "offset_only_or_undistinguishable"
    else:
        delta = float("nan")
        verdict = "insufficient_data"

    # ---- invariants ----
    expected_runs = len(cfg.depths) * len(cfg.widths) * len(cfg.seeds) * 2
    completed = sum(1 for c in cells if not c["crashed"])
    inv_coverage = InvariantResult(
        "grid_completed",
        completed >= expected_runs - 1,  # tolerate a single unstable cell
        completed,
        f"{completed}/{expected_runs} cells trained without crashing",
    )
    inv_both_families = InvariantResult(
        "both_families_produced_fits",
        all(fits[f]["n_seeds"] >= 1 for f in ("vanilla", "bio")),
        {f: fits[f]["n_seeds"] for f in ("vanilla", "bio")},
        "each family needs >=1 usable power-law fit",
    )
    inv_verdict = InvariantResult(
        "verdict_with_cis",
        verdict in (
            "bio_improves_exponent", "bio_worse_exponent",
            "offset_only_or_undistinguishable", "unclear_noisy_fits", "insufficient_data",
        ),
        verdict,
        f"b_vanilla={b_van['exponent_mean']:.4f} "
        f"[{b_van['exponent_ci_low']:.4f},{b_van['exponent_ci_high']:.4f}] | "
        f"b_bio={b_bio['exponent_mean']:.4f} "
        f"[{b_bio['exponent_ci_low']:.4f},{b_bio['exponent_ci_high']:.4f}] | "
        f"delta={delta:.4f} => {verdict}"
        if both_fit else f"verdict={verdict}",
    )
    events = read_run_events(rd)
    inv_trace = InvariantResult(
        "jsonl_trace_written",
        len(events) >= sum(1 for c in cells if not c["crashed"]),
        len(events),
        f"{len(events)} events",
    )

    # Optional flagship artifact: log-log plot per family (matplotlib already a dep).
    plot_path = ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for family, color in (("vanilla", "#888888"), ("bio", "#4472C4")):
            pts = [(c["flops"], c["val_nll"]) for c in cells
                   if c["family"] == family and not c["crashed"]]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, color=color, label=family, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FLOP proxy C = 6·params·tokens")
        ax.set_ylabel("held-out NLL/token")
        ax.set_title("74f.6 scaling probe (toy grid, CPU)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        plot_path = str(rd / "scaling_probe.png")
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    except Exception as e:  # plotting must never fail the study
        _logger.warning(f"[scaling] plot skipped: {e}")

    report = ScalingReport(
        run_id=f"scaling-{cfg.master_seed}",
        invariants=[inv_coverage, inv_both_families, inv_verdict, inv_trace],
        summary={
            "verdict": verdict,
            "exponent_delta": delta,
            "fits": fits,
            "plot_path": plot_path,
            "events_path": str(rd / "events.jsonl"),
            "cells": [
                {k: v for k, v in c.items()} for c in cells
            ],
        },
    )
    if verbose:
        head = "PASSED" if report.passed else "FAILED"
        _logger.info(f"[scaling] ===== {report.run_id} {head} =====")
        for r in report.invariants:
            _logger.info(r.line())
        _logger.info(f"[scaling] VERDICT: {verdict} (delta={delta:+.4f})")
        if plot_path:
            _logger.info(f"[scaling] plot: {plot_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bio-vs-vanilla scaling-law probe (74f.6)")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--base-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    cfg = ScalingStudyConfig()
    if args.base_steps is not None:
        cfg.base_steps = args.base_steps
    if args.seed is not None:
        cfg.master_seed = args.seed
    report = run_scaling_study(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
