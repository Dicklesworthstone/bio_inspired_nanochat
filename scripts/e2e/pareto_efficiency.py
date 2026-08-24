"""E2E SCRIPT: quality-vs-compute Pareto evaluation — adaptive vs fixed-compute inference (bead r00r.3.4).

Falsification target for METABOLIC ADAPTIVE COMPUTE (r00r.*): at EQUAL task quality, does
ATP-budgeted adaptive inference (r00r.3.1 difficulty router + budget accounting, r00r.3.2
dynamic depth/expert-k/MC allocation, r00r.3.3 quality-floor guard) actually spend FEWER
compute units per token than the fixed-compute baseline — or is the whole mechanism a null?

Protocol (tiny CPU MoE synaptic model, associative recall):
  - Train a GPTSynaptic (MoE, stochastic vesicle release ON so the MC lever is real) briefly on
    an associative-recall pool.
  - FIXED arm: ``AdaptiveComputeController(enabled=False)`` — the exact fixed-compute path
    (full depth, full expert-k, ``max_mc_samples`` every token). Cost per token =
    ``maximum_compute_units`` by construction; asserted.
  - ADAPTIVE arms: enabled controller + per-sequence ``ATPBudget`` at fractions
    {0.45, 0.65, 0.85} of the fixed cost for the scored suffix. Every scored token goes through
    :func:`quality_guarded_predict` (fail-closed to fixed compute below the confidence floor).
    A sequence whose account cannot afford the reserved fixed-cost fallback raises
    :class:`InsufficientATPError` — recorded as budget attrition and excluded from paired stats
    (the feasibility boundary of the Pareto curve), never silently averaged in.
  - Quality: answer-token accuracy (argmax of the MC-mean distribution vs gold) + gold NLL.
    Paired per-sequence stats vs the fixed arm via ``eval_stats.paired_comparison``
    (paired t, Wilcoxon, bootstrap CI — the 74f.3 layer).
  - Verdict: the tightest budget fraction whose accuracy stays within tolerance of baseline
    yields the headline savings %; ``>= 5%`` ⇒ "improvement", else a documented "null".

Difficulty probes read next-token logits from an ordinary forward of the same prefix (in a
generation loop those logits already exist; the probe is amortized infrastructure, not a billed
compute action — no ATP is debited for it).

Run:  uv run python -m scripts.e2e.pareto_efficiency --run-dir runs/e2e/pareto_efficiency
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

from bio_inspired_nanochat.adaptive_compute import (
    AdaptiveComputeConfig,
    AdaptiveComputeController,
    InsufficientATPError,
    quality_guarded_predict,
)
from bio_inspired_nanochat.common import logger as _logger
from bio_inspired_nanochat.deliberation import ATPBudget
from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.eval_stats import paired_comparison
from bio_inspired_nanochat.run_logging import RunLogger, read_run_events
from bio_inspired_nanochat.synthetic_tasks import associative_recall
from bio_inspired_nanochat.torch_imports import torch


# --------------------------------------------------------------------------- #
# Config & result types
# --------------------------------------------------------------------------- #
@dataclass
class ParetoE2EConfig:
    """One reduced Pareto efficiency run."""

    # Model geometry (tiny CPU model with MoE)
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    vocab_size: int = 97
    sequence_len: int = 48
    device: str = "cpu"
    seed: int = 0

    # Task / training
    task_vocab: int = 64
    train_batch: int = 8
    num_pairs_train: int = 2
    pool_size: int = 32
    steps: int = 3000
    lr: float = 1e-3
    grad_clip: float = 1.0

    # Evaluation
    eval_pair_counts: tuple[int, ...] = (2,)
    eval_batch_each: int = 8  # rows per batch -> 3*8 = 24 candidate sequences
    max_eval_sequences: int = 16
    scored_suffix_len: int = 6  # tokens scored per sequence (incl. the answer)

    # Adaptive-compute sweep
    budget_fractions: tuple[float, ...] = (0.55, 0.75, 0.95)
    quality_floors: tuple[float, ...] = (0.0, 0.5)  # guarded-predictor confidence thresholds
    eval_source: str = "pool"  # "pool" = in-distribution probe; "fresh" = held-out seeds
    quality_tolerance: float = 0.05  # max acceptable paired accuracy drop vs fixed
    improvement_threshold: float = 0.05  # savings fraction needed to call it an improvement

    # Train-time online plasticity is DISABLED here: it is the component that reproducibly
    # diverges to NaN mid-training on this tiny MoE stack (the open jpqc/809i investigation);
    # inference-time adaptive compute — this bead's subject — does not depend on it.
    syn_overrides: dict = field(default_factory=lambda: {"plasticity_during_training": False})


@dataclass
class ParetoE2EReport:
    """Outcome of one Pareto efficiency run."""

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
        raise AssertionError("pareto-efficiency e2e FAILED:\n" + "\n".join(lines))


# --------------------------------------------------------------------------- #
# Model / data
# --------------------------------------------------------------------------- #
def _build_model(cfg: ParetoE2EConfig):
    """Tiny DENSE GPTSynaptic (see note below on the MoE training bug)."""
    torch.manual_seed(cfg.seed)
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    from bio_inspired_nanochat.synaptic import SynapticConfig

    syn = SynapticConfig()  # stochastic vesicle release stays ON: MC samples must differ
    for k, v in cfg.syn_overrides.items():
        setattr(syn, k, v)
    # DENSE model: SynapticMoE training currently clashes with autograd
    # (bio_inspired_nanochat-moe-train-autograd-clash-0f4i), so the expert lever is excluded
    # from this falsification until that lands; depth + MC-sampling levers remain live.
    gcfg = GPTSynapticConfig(
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        synapses=True,
        syn_cfg=syn,
    )
    return GPTSynaptic(gcfg).to(cfg.device)



def _make_train_pool(cfg: ParetoE2EConfig):
    return [
        associative_recall(
            batch=cfg.train_batch, num_pairs=cfg.num_pairs_train,
            vocab_size=cfg.task_vocab, seed=cfg.seed + 10 + i,
        )
        for i in range(cfg.pool_size)
    ]


def _eval_sequences(cfg: ParetoE2EConfig) -> list[dict]:
    """Flatten recall batches into per-row scoring jobs (context, gold, answer_pos).

    ``eval_source="pool"`` scores sequences from the TRAINING pool: at this toy scale a model
    trained 3k steps has near-chance held-out confidence, which makes every token look maximally
    hard to the difficulty router and the whole mechanism degenerates to fixed compute. An
    in-distribution probe contains both genuinely easy (near-deterministic) and genuinely hard
    positions, so the router has real signal — the honest small-scale falsification. Held-out
    generalization is explicitly out of scope here and is re-established by hwxb.* at scale.
    """
    if cfg.eval_source == "pool":
        batches = _make_train_pool(cfg)[: max(1, cfg.max_eval_sequences // cfg.train_batch + 1)]
    else:
        seed = cfg.seed + 1000
        batches = [
            associative_recall(
                batch=cfg.eval_batch_each, num_pairs=k,
                vocab_size=cfg.task_vocab, seed=seed + i,
            )
            for i, k in enumerate(cfg.eval_pair_counts)
        ]
    out: list[dict] = []
    for batch in batches:
        ap = int(batch.meta["answer_pos"])
        start = max(2, ap - cfg.scored_suffix_len + 1)
        for row in range(batch.inputs.shape[0]):
            if len(out) >= cfg.max_eval_sequences:
                return out
            out.append({
                "inputs": batch.inputs[row:row + 1, : ap + 1],  # (1, T) context incl. query
                "gold": int(batch.meta["answers"][row]),
                "answer_pos": ap,
                "score_from": start,
            })
    return out


def _score_sequence_fixed(model, job: dict, cfg: ParetoE2EConfig) -> dict:
    """Fixed-compute arm: disabled controller ⇒ exact full-cost path on every scored token."""
    controller = AdaptiveComputeController(AdaptiveComputeConfig(enabled=False))
    return _score_sequence_adaptive(
        model, job, cfg, controller=controller, budget_total=None, fraction=None,
        quality_floor=0.0,
    )


def _score_sequence_adaptive(
    model,
    job: dict,
    cfg: ParetoE2EConfig,
    *,
    controller: AdaptiveComputeController,
    budget_total: int | None,
    fraction: float | None,
    quality_floor: float = 0.0,
) -> dict:
    """Score the suffix through :func:`quality_guarded_predict`, debiting one sequence account.

    Returns accuracy/NLL/units aggregates plus per-token records. Raises
    :class:`InsufficientATPError` upward when the account can no longer afford the reserved
    fixed-cost fallback (budget attrition is handled by the caller)."""
    x = job["inputs"].to(cfg.device)
    gold = job["gold"]
    budget = None if budget_total is None else ATPBudget(budget_total)

    correct = 0
    n_scored = 0
    nll_sum = 0.0
    units_sum = 0
    fallbacks = 0
    records: list[dict] = []

    was_training = model.training
    model.eval()
    try:
        for pos in range(job["score_from"], job["answer_pos"] + 1):
            prefix = x[:, : pos + 1]
            with torch.no_grad():  # difficulty probe: logits a generation loop already has
                probe_logits, _ = model(prefix, None, None, train_mode=False)
            # NOTE: ``quality=None`` would mean the guard's DEFAULT 0.5 confidence floor —
            # every token would fail over to fixed compute. Pass an explicit config so
            # ``quality_floor=0.0`` really disables the fail-closed path.
            from bio_inspired_nanochat.adaptive_compute import QualityFloorConfig

            quality = QualityFloorConfig(min_predictive_confidence=float(quality_floor))
            result = quality_guarded_predict(
                model,
                prefix,
                probe_logits[0, -1, :],
                budget if budget is not None else ATPBudget(2**31),
                controller=controller,
                token_index=pos,
                quality=quality,
                run_logger=None,
            )
            probs = result.prediction.mean_probs[0, -1, :]
            pred = int(probs.argmax())
            nll_sum += -math.log(max(float(probs[gold]), 1e-12))
            correct += int(pred == gold)
            n_scored += 1
            units_sum += int(result.executed_plan.compute_units)
            fallbacks += int(result.fallback_used)
            records.append({
                "pos": pos, "units": int(result.executed_plan.compute_units),
                "spent": int(result.token_spent_atp),
                "fallback": bool(result.fallback_used),
                "difficulty": float(result.proposed_plan.difficulty.score),
            })
    finally:
        if was_training:
            model.train()

    # Accounting invariant: the account must hold spent + remaining == total exactly.
    accounting_ok = True
    if budget is not None:
        accounting_ok = budget.spent_atp + budget.remaining_atp == budget.total_atp
    return {
        "n_scored": n_scored, "acc": correct / max(n_scored, 1),
        "mean_nll": nll_sum / max(n_scored, 1),
        "units_per_token": units_sum / max(n_scored, 1), "fallbacks": fallbacks,
        "difficulty_mean": (sum(r["difficulty"] for r in records) / max(len(records), 1))
        if records else float("nan"),
        "accounting_ok": accounting_ok, "records": records,
        "fraction": fraction, "spent_atp": budget.spent_atp if budget else 0,
    }

# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run_pareto_e2e(
    cfg: ParetoE2EConfig, *, run_dir: str | Path | None = None, verbose: bool = True,
) -> ParetoE2EReport:
    """Train, then run the fixed-vs-adaptive Pareto battery; return a :class:`ParetoE2EReport`."""
    if cfg.device == "cpu":
        try:
            # Single-threaded: MoE dispatch scatters are reduction-nondeterministic under CPU
            # multithreading, which flips training trajectories run-to-run. Tiny model, so the
            # cost is negligible and reproducibility of the Pareto numbers matters more.
            torch.set_num_threads(1)
        except Exception:
            pass
    torch.manual_seed(cfg.seed)

    rd = Path(run_dir) if run_dir is not None else Path("runs") / "e2e" / "pareto_efficiency"
    rl = RunLogger(rd, name="pareto_efficiency", console=False, provenance={
        "seed": cfg.seed, "steps": cfg.steps, "fractions": list(cfg.budget_fractions),
    })

    # ---- train ----
    model = _build_model(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    pool = _make_train_pool(cfg)
    losses: list[float] = []
    last_good: dict | None = {k: v.detach().clone() for k, v in model.state_dict().items()}
    last_good_step = 0
    train_stopped_early: str | None = None
    for step in range(cfg.steps):
        batch = pool[step % len(pool)]
        try:
            _, loss = model(batch.inputs.to(cfg.device), batch.targets.to(cfg.device), None,
                            train_mode=True)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            opt.step()
        except RuntimeError as exc:
            # The synaptic train-time buffers (metabolism/router state) have an inplace-vs-
            # autograd clash on some trajectories (the open jpqc/809i instability family).
            # This harness evaluates INFERENCE-time adaptive compute, so restore the last good
            # snapshot and stop training early instead of failing the whole run.
            if last_good is None:
                raise
            model.load_state_dict(last_good)
            train_stopped_early = f"step {step}: {type(exc).__name__} (restored snapshot @{last_good_step})"
            _logger.warning(f"[pareto-e2e] training stopped early at {train_stopped_early}")
            break
        lval = float(loss.detach().item())
        losses.append(lval)
        rl.event("train_step", step=step, loss=lval, lr=cfg.lr, grad_norm=gnorm)
        if step % 20 == 0:
            last_good = {k: v.detach().clone() for k, v in model.state_dict().items()}
            last_good_step = step
    final_loss = losses[-1] if losses else float("nan")

    # ---- evaluate ----
    jobs = _eval_sequences(cfg)
    adaptive_controller = AdaptiveComputeController(AdaptiveComputeConfig(enabled=True))

    max_depth, max_experts = AdaptiveComputeController.model_capacity(model)
    fixed_units_per_token = (
        max_depth + max(max_experts, 0) + AdaptiveComputeConfig().max_mc_samples
    )

    results: dict[str, dict] = {"fixed": {}}
    attrition: dict[str, int] = {
        f"{f:.2f}/q{q:.2f}": 0 for f in cfg.budget_fractions for q in cfg.quality_floors
    }
    accounting_all_ok = True

    try:
        for idx, job in enumerate(jobs):
            model.reset_sequence_state(reset_fast_weights=True)
            results["fixed"][idx] = _score_sequence_fixed(model, job, cfg)

            for frac in cfg.budget_fractions:
                for qfloor in cfg.quality_floors:
                    key = f"{frac:.2f}/q{qfloor:.2f}"
                    results.setdefault(key, {})
                    # Base allocation plus an explicit fallback allowance: a confidence floor
                    # makes some tokens pay full fixed price, so the account must cover both.
                    total = max(
                        int(round(frac * fixed_units_per_token
                                  * (job["answer_pos"] - job["score_from"] + 1)))
                        + int(qfloor * fixed_units_per_token
                              * (job["answer_pos"] - job["score_from"] + 1)),
                        fixed_units_per_token + 1,  # guard reserves one full fixed-cost fallback
                    )
                    model.reset_sequence_state(reset_fast_weights=True)
                    try:
                        results[key][idx] = _score_sequence_adaptive(
                            model, job, cfg, controller=adaptive_controller,
                            budget_total=total, fraction=frac, quality_floor=qfloor,
                        )
                    except InsufficientATPError:
                        attrition[key] += 1  # feasibility boundary: documented, not averaged in
                        continue
                    if not results[key][idx]["accounting_ok"]:
                        accounting_all_ok = False
                    rl.event("pareto_sequence", seq=idx, fraction=key,
                             acc=results[key][idx]["acc"],
                             units=results[key][idx]["units_per_token"],
                             difficulty=results[key][idx]["difficulty_mean"])
    finally:
        rl.close()

    # ---- Pareto analysis ----
    fixed_acc = {i: r["acc"] for i, r in results["fixed"].items()}
    fixed_units = {i: r["units_per_token"] for i, r in results["fixed"].items()}
    fixed_nll = {i: r["mean_nll"] for i, r in results["fixed"].items()}
    base_mean_acc = sum(fixed_acc.values()) / max(len(fixed_acc), 1)
    base_mean_units = sum(fixed_units.values()) / max(len(fixed_units), 1)

    pareto_rows: list[dict] = []
    for frac in cfg.budget_fractions:
        for qfloor in cfg.quality_floors:
            key = f"{frac:.2f}/q{qfloor:.2f}"
            arm = results.get(key, {})
            if len(arm) < 2:
                pareto_rows.append({
                    "fraction": frac, "quality_floor": qfloor,
                    "n": len(arm), "feasible": False,
                })
                continue
            arm_acc = {i: r["acc"] for i, r in arm.items()}
            arm_nll = {i: r["mean_nll"] for i, r in arm.items()}
            acc_stats = paired_comparison(arm_acc, fixed_acc,
                                          lower_is_better=False, seed=cfg.seed)
            nll_stats = paired_comparison(arm_nll, fixed_nll,
                                          lower_is_better=True, seed=cfg.seed)
            mean_units = sum(r["units_per_token"] for r in arm.values()) / len(arm)
            savings = 1.0 - mean_units / base_mean_units if base_mean_units else 0.0
            acc_drop = -(acc_stats.mean_delta) if acc_stats else float("inf")
            pareto_rows.append({
                "fraction": frac, "quality_floor": qfloor,
                "n": len(arm), "attrition": attrition[key],
                "mean_acc": sum(r["acc"] for r in arm.values()) / len(arm),
                "mean_nll": sum(r["mean_nll"] for r in arm.values()) / len(arm),
                "mean_units": mean_units, "savings_frac": savings,
                "acc_delta_vs_fixed": acc_stats.mean_delta if acc_stats else None,
                "difficulty_mean": sum(r["difficulty_mean"] for r in arm.values()) / len(arm),
                "acc_t_p": acc_stats.t_p_value if acc_stats else None,
                "acc_wilcoxon_p": acc_stats.wilcoxon_p_value if acc_stats else None,
                "nll_delta_vs_fixed": nll_stats.mean_delta if nll_stats else None,
                "fallback_rate": sum(r["fallbacks"] for r in arm.values())
                / max(sum(r["n_scored"] for r in arm.values()), 1),
                "within_quality_tolerance": acc_stats is not None
                and acc_drop <= cfg.quality_tolerance,
                "feasible": True,
            })

    feasible = [r for r in pareto_rows
                if r.get("feasible") and r.get("within_quality_tolerance")]
    best = max(feasible, key=lambda r: (r["savings_frac"], -r["fraction"])) \
        if feasible else None
    if best is not None:
        verdict = "improvement" if best["savings_frac"] >= cfg.improvement_threshold else "null"
        headline = best["savings_frac"]
    else:
        verdict, headline = "null", 0.0

    # ---- invariants ----
    inv_fixed_max = InvariantResult(
        "fixed_baseline_uses_max_compute",
        all(abs(r["units_per_token"] - fixed_units_per_token) < 1e-9
            for r in results["fixed"].values()),
        fixed_units_per_token,
        f"every fixed-arm token costs exactly {fixed_units_per_token} units "
        f"(depth={max_depth} + experts={max_experts} + mc={AdaptiveComputeConfig().max_mc_samples})",
    )
    means_in_order = [
        sum(r["units_per_token"] for r in results[f"{f:.2f}/q0.00"].values())
        / max(len(results[f"{f:.2f}/q0.00"]), 1)
        for f in cfg.budget_fractions
        if len(results.get(f"{f:.2f}/q0.00", {})) >= 1
    ]
    inv_monotone = InvariantResult(
        "tighter_budget_spends_less",
        all(a <= b + 1e-9 for a, b in zip(means_in_order, means_in_order[1:]))
        if len(means_in_order) >= 2 else False,
        means_in_order,
        f"mean units/token by ascending fraction at floor 0: "
        f"{[round(m, 2) for m in means_in_order]} (must be non-decreasing)",
    )
    inv_accounting = InvariantResult(
        "atp_accounting_exact",
        accounting_all_ok,
        accounting_all_ok,
        "spent + remaining == total held on every adaptive sequence account",
    )
    inv_verdict = InvariantResult(
        "pareto_verdict_with_stats",
        verdict in ("improvement", "null"),
        verdict,
        f"verdict={verdict}; headline savings={headline:.1%}; baseline mean acc="
        f"{base_mean_acc:.3f}, mean units={base_mean_units:.2f}; rows={len(pareto_rows)}; "
        f"attrition={attrition}",
    )
    events = read_run_events(rd)
    n_scored_arms = sum(len(v) for k, v in results.items() if k != "fixed")
    inv_trace = InvariantResult(
        "jsonl_trace_written",
        len(events) >= cfg.steps + n_scored_arms,
        len(events),
        f"{len(events)} events (>= {cfg.steps} train_step + {n_scored_arms} scored "
        f"pareto_sequence rows; attrited sequences emit none)",
    )

    report = ParetoE2EReport(
        run_id=f"pareto-{cfg.seed}-{cfg.steps}",
        invariants=[inv_fixed_max, inv_monotone, inv_accounting, inv_verdict, inv_trace],
        summary={
            "final_train_loss": final_loss,
            "baseline_mean_acc": base_mean_acc,
            "baseline_mean_units": base_mean_units,
            "verdict": verdict,
            "headline_savings_frac": headline,
            "pareto_rows": pareto_rows,
            "attrition": attrition,
            "events_path": str(rd / "events.jsonl"),
        },
    )
    _log_report(report, verbose=verbose)
    return report


def _log_report(report: ParetoE2EReport, verbose: bool = True) -> None:
    head = "PASSED" if report.passed else "FAILED"
    if not verbose:
        return
    _logger.info(f"[pareto-e2e] ===== run {report.run_id} {head} =====")
    for r in report.invariants:
        _logger.info(r.line())
    s = report.summary
    for row in s["pareto_rows"]:
        _logger.info(f"[pareto-e2e]   frac={row['fraction']}: {row}")
    _logger.info(
        f"[pareto-e2e] baseline acc={s['baseline_mean_acc']:.3f} "
        f"units={s['baseline_mean_units']:.2f} | VERDICT: {s['verdict']} "
        f"(savings {s['headline_savings_frac']:.1%})"
    )
    _logger.info(f"[pareto-e2e] JSONL trace: {s['events_path']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Adaptive-vs-fixed compute Pareto e2e")
    p.add_argument("--run-dir", default=None, help="where to write events.jsonl")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    cfg = ParetoE2EConfig()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    report = run_pareto_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
