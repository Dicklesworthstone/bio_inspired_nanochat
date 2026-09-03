"""
CPU pilot of the structural pair: does the expert lifecycle fire during ordinary training, and
what does it do to the loss? (bead uta.8 / bridge plan G4; sx1m for the health-signal measurement)

Three arms per seed on a 2-layer / 64-dim GPTSynaptic with SynapticMoE blocks (8 experts, top-2),
trained on a routing-pressure task (sequences drawn from one of several token-transition
"dialects", so experts have something to specialise on):

* ``none``     - no lifecycle controller (the control);
* ``product``  - the default ``SplitMergeConfig`` health (utilization x energy, absolute thresholds);
* ``relative`` - ``health_mode="relative"`` with the fair-share thresholds the D1 structural arm uses
                 (split above 1.5x fair share, merge below 0.35x, reset below 0.05x).

Recorded per (seed, arm): split / merge / reset counts, final loss (mean of the last 20 steps), the
loss delta against the ``none`` arm of the same seed, and the utilization spread across experts at
the end. Chance-level or no-event outcomes are results, not failures: the pre-registered question is
whether the lifecycle fires at all under a healthy training signal and whether firing costs loss.

Run:  uv run --no-sync python -m scripts.e2e.structural_pair_pilot --seeds 0 1 2 --steps 300
Writes results/structural_pair_pilot_<date>.json (stamped with measurement_regime).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.neuroscore import NeuroScore, NeuroScoreConfig
from bio_inspired_nanochat.results_registry import measurement_regime
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import SplitMergeConfig, SplitMergeController

logger = logging.getLogger("bio_inspired_nanochat.structural_pair_pilot")
console = Console()

VOCAB = 97
DIALECTS = 4
ARMS = ("none", "product", "relative", "credit")


class _CountingLogger:
    """Receives the controller's lifecycle callbacks and counts them."""

    def __init__(self) -> None:
        self.counts = {"split": 0, "merge": 0, "reset": 0}

    def on_split(self, *args: Any, **kwargs: Any) -> None:
        self.counts["split"] += 1

    def on_merge(self, *args: Any, **kwargs: Any) -> None:
        self.counts["merge"] += 1

    def on_reset(self, *args: Any, **kwargs: Any) -> None:
        self.counts["reset"] += 1


def _dialect_transitions(seed: int) -> torch.Tensor:
    """(DIALECTS, VOCAB, VOCAB) row-stochastic matrices, each peaked on a few successors."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(DIALECTS, VOCAB, VOCAB, generator=g) * 3.0
    return torch.softmax(logits, dim=-1)


def _batch(trans: torch.Tensor, g: torch.Generator, *, batch: int, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    dialect = torch.randint(0, DIALECTS, (batch,), generator=g)
    tokens = torch.empty(batch, seq + 1, dtype=torch.long)
    tokens[:, 0] = torch.randint(0, VOCAB, (batch,), generator=g)
    for t in range(seq):
        probs = trans[dialect, tokens[:, t]]  # (BATCH, VOCAB)
        tokens[:, t + 1] = torch.multinomial(probs, 1, generator=g).squeeze(1)
    return tokens[:, :-1], tokens[:, 1:]


def _model(seed: int, *, seq: int, balance_loss: float) -> GPTSynaptic:
    torch.manual_seed(seed)
    cfg = GPTSynapticConfig(
        sequence_len=seq, vocab_size=VOCAB, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
        synapses=True, syn_cfg=SynapticConfig(), use_moe=True, num_experts=8, moe_top_k=2,
        moe_balance_loss=balance_loss,
    )
    model = GPTSynaptic(cfg)
    model.train()
    return model


def _controller(arm: str, model: GPTSynaptic, every: int, warmup: int, log: _CountingLogger):
    if arm == "none":
        return None
    if arm == "product":
        cfg = SplitMergeConfig(
            enabled=True, function_preserving=True, min_step_interval=every, warmup_steps=warmup, ddp_broadcast=False
        )
    else:  # relative or credit: the same thresholds in their own units (1.0 = an average expert)
        cfg = SplitMergeConfig(
            enabled=True, function_preserving=True, min_step_interval=every, warmup_steps=warmup, ddp_broadcast=False,
            health_mode=arm, split_health_min=1.5, merge_health_max=0.35, reset_health_max=0.05,
        )
    return SplitMergeController(model, cfg, logger=log)


def _util_spread(model: GPTSynaptic) -> list[float]:
    return [float(m.fatigue.detach().std()) for m in model.modules() if isinstance(m, SynapticMoE)]


def run_arm(
    seed: int, arm: str, *, steps: int, every: int, warmup: int, lr: float, batch: int, seq: int, balance_loss: float
) -> dict[str, Any]:
    model = _model(seed, seq=seq, balance_loss=balance_loss)
    log = _CountingLogger()
    controller = _controller(arm, model, every, warmup, log)
    # The credit arm needs NeuroScore's gradient credit published after every backward.
    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1)) if arm == "credit" else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    trans = _dialect_transitions(seed)
    g = torch.Generator().manual_seed(10_000 + seed)
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        x, y = _batch(trans, g, batch=batch, seq=seq)
        _, loss = model(x, y, train_mode=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"seed={seed} arm={arm}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if score is not None:
            score.step(model, loss.detach(), step)
        if controller is not None:
            controller.step(global_step=step, optimizer=opt)
    experts = [int(m.num_experts) for m in model.modules() if isinstance(m, SynapticMoE)]
    out = {
        "seed": seed,
        "arm": arm,
        "events": dict(log.counts),
        "final_loss": statistics.fmean(losses[-20:]),
        "first_loss": statistics.fmean(losses[:20]),
        "util_spread_per_layer": _util_spread(model),
        "experts_per_layer": experts,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    logger.info("[pilot] seed=%d arm=%s events=%s final_loss=%.4f wall=%.0fs", seed, arm, out["events"], out["final_loss"], out["wall_s"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--every", type=int, default=25, help="controller interval in optimizer steps")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument(
        "--balance-loss", type=float, default=0.01,
        help="GPTSynapticConfig.moe_balance_loss for every arm (default 0.01 = the model default; the 2026-09-02 "
        "pilot showed it keeps utilization within ±0.03 of the fair share so no threshold ever fires)",
    )
    parser.add_argument("--out", default=None, help="JSON path (default results/structural_pair_pilot_<date>.json)")
    parser.add_argument("--arms", default=",".join(ARMS), help=f"Comma-separated subset of {ARMS}")
    args = parser.parse_args(argv)
    out_path = Path(args.out or f"results/structural_pair_pilot_{_dt.date.today().isoformat()}.json")
    arms = tuple(a for a in args.arms.split(",") if a)
    unknown = [a for a in arms if a not in ARMS]
    if unknown or "none" not in arms:
        raise SystemExit(f"--arms must be a subset of {ARMS} and include 'none' (the control); got {arms}")

    rows = [
        run_arm(
            s, arm, steps=args.steps, every=args.every, warmup=args.warmup, lr=args.lr, batch=args.batch, seq=args.seq,
            balance_loss=args.balance_loss,
        )
        for s in args.seeds
        for arm in arms
    ]
    by_seed = {s: {r["arm"]: r for r in rows if r["seed"] == s} for s in args.seeds}
    for s in args.seeds:
        base = by_seed[s]["none"]["final_loss"]
        for arm in arms:
            by_seed[s][arm]["loss_delta_vs_none"] = by_seed[s][arm]["final_loss"] - base
    summary = {
        arm: {
            "events_total": {k: sum(by_seed[s][arm]["events"][k] for s in args.seeds) for k in ("split", "merge", "reset")},
            "mean_loss_delta_vs_none": statistics.fmean(by_seed[s][arm]["loss_delta_vs_none"] for s in args.seeds),
            "seeds_with_any_event": sum(1 for s in args.seeds if sum(by_seed[s][arm]["events"].values()) > 0),
        }
        for arm in arms
    }
    payload = {
        "protocol": {
            "model": "GPTSynaptic 2L/64d, SynapticMoE 8 experts top-2, default SynapticConfig",
            "task": f"{DIALECTS}-dialect token-transition sequences, vocab {VOCAB}, seq {args.seq}, batch {args.batch}",
            "steps": args.steps, "controller_every": args.every, "warmup": args.warmup, "lr": args.lr,
            "moe_balance_loss": args.balance_loss,
            "arms": {"none": "no controller", "product": "SplitMergeConfig defaults (util x energy)",
                     "relative": "health_mode=relative, split 1.5 / merge 0.35 / reset 0.05 fair-share units",
                     "credit": "health_mode=credit (uta.9): NeuroScore gradient credit relative to the mean, same thresholds; NeuroScore stepped every step"},
            "question": "does the lifecycle fire under ordinary training, and what does firing cost in loss?",
        },
        "measurement_regime": measurement_regime(),
        "runs": rows,
        "summary": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table = Table(title=f"structural pair pilot: {len(args.seeds)} seeds x {args.steps} steps")
    for col in ("seed", "arm", "splits", "merges", "resets", "final loss", "delta vs none", "util spread"):
        table.add_column(col, justify="right")
    for r in rows:
        table.add_row(str(r["seed"]), r["arm"], str(r["events"]["split"]), str(r["events"]["merge"]), str(r["events"]["reset"]),
                      f"{r['final_loss']:.4f}", f"{r['loss_delta_vs_none']:+.4f}", " ".join(f"{u:.3f}" for u in r["util_spread_per_layer"]))
    console.print(table)
    console.print(f"[bold]written[/bold] {out_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
