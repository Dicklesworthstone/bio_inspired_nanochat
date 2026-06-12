"""Reusable end-to-end training harness + invariant battery (bead hwxb.7.3).

The single trustworthy scaffold every later phase uses to *prove* a real run works:
it runs a real-but-reduced training loop (tiny model, a few hundred CPU steps, the
real model forward/backward/optimizer path), captures the full telemetry through
``TrainingTelemetry``, and asserts a standard battery of health invariants — emitting
a detailed, human-readable log plus a machine-readable summary.

It is reused by:
  - the dual-4090 smoke test (`hwxb.2.5`),
  - the per-mechanism at-scale e2e suite (`hwxb.7.4`),
  - the ablation-harness dry-run,
and self-tested in ``tests/test_scaleup_e2e.py`` on a known-good and a known-bad
(injected-NaN) run so we know the battery actually fires.

Design notes
------------
- Library module (no dependency on ``tests/``) so scripts and tests can both import it.
- Uses a fixed pool of batches as a small, *learnable* next-token task, so a healthy
  run's loss genuinely decreases (memorization) — that is what makes the
  "loss trends down" invariant meaningful rather than vacuous.
- A plain ``AdamW`` over all params keeps the harness robust on CPU/tiny shapes; the
  point is to exercise the real *model* forward (presyn kinetics, Hebbian fast-weights,
  MoE), not to re-test the distributed optimizers (that's ``tests/test_scaleup_ddp.py``).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from bio_inspired_nanochat.common import logger as _logger
from bio_inspired_nanochat.run_logging import TrainingTelemetry
from bio_inspired_nanochat.torch_imports import torch

log = logging.getLogger("bio_inspired_nanochat.e2e")


# --------------------------------------------------------------------------- #
# Config & result types
# --------------------------------------------------------------------------- #
@dataclass
class E2EConfig:
    """Configuration for one reduced end-to-end run."""

    synapses: bool = True
    n_layer: int = 2
    n_embd: int = 64
    n_head: int = 4
    vocab_size: int = 97
    seq_len: int = 32
    batch_size: int = 4
    pool_size: int = 8          # number of distinct fixed batches cycled (learnable task)
    steps: int = 80
    lr: float = 3e-3
    grad_clip: float = 1.0
    seed: int = 1234
    device: str = "cpu"
    syn_overrides: dict[str, Any] = field(default_factory=dict)
    # Fault injection for the known-bad self-test: NaN the loss at this step (None = healthy).
    inject_nan_at: int | None = None
    # Invariant thresholds.
    grad_norm_bound: float = 1e4
    param_absmax_bound: float = 1e4
    loss_decrease_required: bool = True


@dataclass
class InvariantResult:
    name: str
    passed: bool
    observed: Any
    detail: str

    def line(self) -> str:
        mark = "✓ PASS" if self.passed else "✗ FAIL"
        return f"  [{mark}] {self.name}: {self.detail}"


@dataclass
class E2EReport:
    run_id: str
    config: E2EConfig
    passed: bool
    invariants: list[InvariantResult]
    loss_trajectory: list[float]
    grad_norms: list[float]
    summary: dict[str, Any]

    def failures(self) -> list[InvariantResult]:
        return [r for r in self.invariants if not r.passed]

    def assert_passed(self) -> None:
        if not self.passed:
            lines = "\n".join(r.line() for r in self.invariants)
            raise AssertionError(
                f"e2e run {self.run_id} failed {len(self.failures())} invariant(s):\n{lines}"
            )


# --------------------------------------------------------------------------- #
# Model / data helpers
# --------------------------------------------------------------------------- #
def _build_model(cfg: E2EConfig) -> torch.nn.Module:
    torch.manual_seed(cfg.seed)
    common = dict(
        sequence_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
    )
    if cfg.synapses:
        from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
        from bio_inspired_nanochat.synaptic import SynapticConfig

        syn_cfg = SynapticConfig()
        for k, v in cfg.syn_overrides.items():
            setattr(syn_cfg, k, v)
        # ty cannot match the **dict unpacking to the dataclass fields (see _bio_testkit.py).
        model = GPTSynaptic(GPTSynapticConfig(synapses=True, syn_cfg=syn_cfg, **common))  # ty: ignore[invalid-argument-type]
    else:
        from bio_inspired_nanochat.gpt import GPT, GPTConfig

        model = GPT(GPTConfig(**common))  # ty: ignore[invalid-argument-type]
    return model.to(cfg.device)


def _make_pool(cfg: E2EConfig) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fixed pool of (x, y) next-token batches — a small, learnable LM task."""
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    pool = []
    for _ in range(cfg.pool_size):
        toks = torch.randint(
            0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len + 1), generator=gen, dtype=torch.long
        ).to(cfg.device)
        pool.append((toks[:, :-1].contiguous(), toks[:, 1:].contiguous()))
    return pool


def _forward_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, synapses: bool) -> torch.Tensor:
    if synapses:
        _, loss = model(x, y, None, train_mode=True)
    else:
        loss = model(x, y)
    return loss


@torch.no_grad()
def _forward_logits(model: torch.nn.Module, x: torch.Tensor, synapses: bool) -> torch.Tensor:
    if synapses:
        logits, _ = model(x, None, None, train_mode=False)
    else:
        logits = model(x)
    return logits


@torch.no_grad()
def _eval_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, synapses: bool) -> float:
    """Deterministic eval-mode loss (no stochastic vesicle release, no train-time state writes)."""
    was_training = model.training
    model.eval()
    if synapses:
        _, loss = model(x, y, None, train_mode=False)
    else:
        loss = model(x, y)
    if was_training:
        model.train()
    return float(loss.detach().item())


@torch.no_grad()
def _sample_continuation(model: torch.nn.Module, cfg: E2EConfig, n_new: int = 16) -> list[int]:
    """Sampled greedy-free decode from a fixed prompt; returns the new token ids."""
    model.eval()
    gen = torch.Generator().manual_seed(cfg.seed + 7)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), generator=gen, dtype=torch.long).to(cfg.device)
    out: list[int] = []
    for _ in range(n_new):
        ctx = ids[:, -cfg.seq_len:]
        logits = _forward_logits(model, ctx, cfg.synapses)
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1, generator=gen).item())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=cfg.device)], dim=1)
    model.train()
    return out


def _fast_weight_fingerprint(model: torch.nn.Module) -> float:
    """Sum of L2 norms of fast-weight params (the online-adapting Hebbian weights)."""
    total = 0.0
    for n, p in model.named_parameters():
        if "fast" in n.lower():
            total += float(p.detach().float().norm().item())
    return total


def _max_abs_param(model: torch.nn.Module) -> float:
    m = 0.0
    for p in model.parameters():
        if p.numel():
            v = float(p.detach().abs().max().item())
            # Propagate non-finite immediately: ``max(m, nan)`` returns m in Python
            # (nan > m is False), which would hide a NaN/Inf runaway from mechanism_stable.
            if not math.isfinite(v):
                return v
            m = max(m, v)
    return m


def _any_nonfinite_param(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p.detach()).all():
            return True
    return False


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #
def run_e2e(
    cfg: E2EConfig,
    *,
    run_dir: str | Path | None = None,
    verbose: bool = True,
    on_step: Callable[[int, float, float], None] | None = None,
) -> E2EReport:
    """Run one reduced e2e training + the invariant battery; return an E2EReport."""
    # The model is tiny; spinning all cores on 64x64 matmuls is pure thread-launch
    # overhead. Cap threads so the reduced run is genuinely fast (CI-friendly) and
    # not dominated by oversubscription on a busy box.
    if cfg.device == "cpu":
        try:
            torch.set_num_threads(min(4, os.cpu_count() or 4))
        except Exception:
            pass
    torch.manual_seed(cfg.seed)
    model = _build_model(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    pool = _make_pool(cfg)

    telemetry = TrainingTelemetry(
        run_dir or (Path("runs") / "e2e" / ("syn" if cfg.synapses else "vanilla")),
        name="e2e",
        is_master=True,
        tensorboard=False,
        provenance={"synapses": cfg.synapses, "n_layer": cfg.n_layer, "seed": cfg.seed},
    )
    fp_start = _fast_weight_fingerprint(model)

    losses: list[float] = []
    grad_norms: list[float] = []
    try:
        for step in range(cfg.steps):
            x, y = pool[step % len(pool)]
            loss = _forward_loss(model, x, y, cfg.synapses)
            if cfg.inject_nan_at is not None and step == cfg.inject_nan_at:
                loss = loss * float("nan")  # fault injection for the known-bad self-test
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            opt.step()
            lval = float(loss.detach().item())
            losses.append(lval)
            grad_norms.append(gnorm)
            telemetry.log_train_step(step, loss=lval, lr=cfg.lr, grad_norm=gnorm)
            if on_step is not None:
                on_step(step, lval, gnorm)
            if verbose and (step % max(1, cfg.steps // 8) == 0 or step == cfg.steps - 1):
                _logger.info(f"[e2e] step {step:03d}/{cfg.steps} loss={lval:.4f} |grad|={gnorm:.3f}")

        # --- post-run probes -------------------------------------------------
        # A degenerate/diverged model (e.g. the NaN self-test) must yield FAILED
        # invariants, never crash the harness — catching is part of the job.
        try:
            # len(losses) == number of completed steps (== cfg.steps), and is defined
            # even if cfg.steps == 0 (unlike the loop variable).
            ckpt_ok, ckpt_detail = _checkpoint_roundtrip(model, opt, cfg, pool, len(losses))
        except Exception as e:
            ckpt_ok, ckpt_detail = False, f"checkpoint probe raised: {type(e).__name__}: {e}"
        try:
            sample = _sample_continuation(model, cfg)
        except Exception as e:
            sample = []  # empty -> generation_nondegenerate fails (0 distinct tokens)
            _logger.warning(f"[e2e] generation probe raised (treated as degenerate): {e}")
        fp_end = _fast_weight_fingerprint(model)
        max_abs = _max_abs_param(model)
        nonfinite = _any_nonfinite_param(model)
    finally:
        telemetry.close()

    invariants = _invariant_battery(
        cfg, losses, grad_norms, ckpt_ok, ckpt_detail, sample, fp_start, fp_end, max_abs, nonfinite
    )
    passed = all(r.passed for r in invariants)
    report = E2EReport(
        run_id=telemetry.run_id or "e2e",
        config=cfg,
        passed=passed,
        invariants=invariants,
        loss_trajectory=losses,
        grad_norms=grad_norms,
        summary={
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "max_grad_norm": max(grad_norms) if grad_norms else None,
            "fast_weight_delta": fp_end - fp_start,
            "max_abs_param": max_abs,
            "n_invariants": len(invariants),
            "n_failed": sum(1 for r in invariants if not r.passed),
        },
    )
    if verbose:
        _log_report(report)
    return report


def _checkpoint_roundtrip(
    model: torch.nn.Module, opt: torch.optim.Optimizer, cfg: E2EConfig,
    pool: list[tuple[torch.Tensor, torch.Tensor]], next_step: int,
) -> tuple[bool, str]:
    """save → build fresh model+opt → load → identical eval output (the resume contract).

    The synaptic model carries *per-sequence transient* state (online fast-weight /
    eligibility adaptation, stochastic vesicle draws) that is intentionally NOT
    checkpointed — it is rebuilt/reset at each sequence boundary. So the checkpoint
    contract is: after resetting that transient state, the *persistent* parameters +
    buffers round-trip exactly, verified by an identical **eval-mode** forward (which
    is deterministic — no stochastic release). The at-scale checkpointer (hwxb.2.6)
    additionally persists RNG/optimizer/step state for bit-comparable *training* resume.
    """
    import copy

    # Clear per-sequence transient state so we compare what a real checkpoint saves.
    if cfg.synapses and hasattr(model, "reset_sequence_state"):
        model.reset_sequence_state()
    snap = {"model": copy.deepcopy(model.state_dict()), "opt": copy.deepcopy(opt.state_dict())}
    x, y = pool[next_step % len(pool)]
    la = _eval_loss(model, x, y, cfg.synapses)

    # Restore into a fresh model+opt and verify it reproduces the eval output.
    model_b = _build_model(cfg)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=cfg.lr)
    model_b.load_state_dict(snap["model"])
    opt_b.load_state_dict(snap["opt"])
    if cfg.synapses and hasattr(model_b, "reset_sequence_state"):
        model_b.reset_sequence_state()
    lb = _eval_loss(model_b, x, y, cfg.synapses)

    ok = bool(abs(la - lb) <= 1e-5 + 1e-4 * abs(la))
    return ok, f"reload eval loss={lb:.6f} vs original={la:.6f} (Δ={abs(la - lb):.2e})"


# --------------------------------------------------------------------------- #
# Invariant battery
# --------------------------------------------------------------------------- #
def _invariant_battery(
    cfg: E2EConfig,
    losses: list[float],
    grad_norms: list[float],
    ckpt_ok: bool,
    ckpt_detail: str,
    sample: list[int],
    fp_start: float,
    fp_end: float,
    max_abs: float,
    nonfinite: bool,
) -> list[InvariantResult]:
    out: list[InvariantResult] = []

    finite_losses = all(math.isfinite(x) for x in losses)
    out.append(InvariantResult(
        "loss_finite", finite_losses, finite_losses,
        "all step losses finite" if finite_losses
        else f"{sum(1 for x in losses if not math.isfinite(x))} non-finite loss(es)",
    ))

    # loss trends down: mean of last quartile < mean of first quartile
    q = max(1, len(losses) // 4)
    first = sum(losses[:q]) / q if losses else math.nan
    last = sum(losses[-q:]) / q if losses else math.nan
    decreased = finite_losses and (last < first)
    out.append(InvariantResult(
        "loss_decreases", (decreased or not cfg.loss_decrease_required), (first, last),
        f"first-quartile mean={first:.4f} -> last-quartile mean={last:.4f}"
        + ("" if decreased else "  (NOT decreasing)"),
    ))

    gn_ok = all(math.isfinite(g) and g <= cfg.grad_norm_bound for g in grad_norms)
    out.append(InvariantResult(
        "grad_norm_finite_bounded", gn_ok, max(grad_norms) if grad_norms else None,
        f"max grad norm={max(grad_norms):.4f} (bound {cfg.grad_norm_bound:g})" if grad_norms else "no grads",
    ))

    params_ok = not nonfinite
    out.append(InvariantResult(
        "params_finite", params_ok, params_ok,
        "all final params finite (no unrecovered divergence)" if params_ok
        else "final params contain NaN/Inf",
    ))

    out.append(InvariantResult("checkpoint_roundtrip", ckpt_ok, ckpt_ok, ckpt_detail))

    uniq = len(set(sample))
    gen_ok = uniq >= 2
    out.append(InvariantResult(
        "generation_nondegenerate", gen_ok, uniq,
        f"{uniq} distinct tokens in a {len(sample)}-token sample"
        + ("" if gen_ok else "  (degenerate: collapsed to one token)"),
    ))

    if cfg.synapses:
        delta = abs(fp_end - fp_start)
        engaged = delta > 1e-9
        out.append(InvariantResult(
            "mechanism_engaged", engaged, delta,
            f"fast-weight fingerprint moved by {delta:.3e}" if engaged
            else "fast-weights did not adapt (mechanism may be off)",
        ))
        stable = math.isfinite(max_abs) and max_abs <= cfg.param_absmax_bound
        out.append(InvariantResult(
            "mechanism_stable", stable, max_abs,
            f"max|param|={max_abs:.4g} within bound {cfg.param_absmax_bound:g}" if stable
            else f"max|param|={max_abs:.4g} EXCEEDS bound {cfg.param_absmax_bound:g} (runaway)",
        ))

    return out


def _log_report(report: E2EReport) -> None:
    head = "PASSED" if report.passed else "FAILED"
    _logger.info(f"[e2e] ===== run {report.run_id} {head} =====")
    for r in report.invariants:
        _logger.info(r.line())
    s = report.summary

    def _fmt(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else "n/a"

    _logger.info(
        f"[e2e] loss {_fmt(s['initial_loss'], '.4f')} -> {_fmt(s['final_loss'], '.4f')} | "
        f"max|grad|={_fmt(s['max_grad_norm'], '.3f')} | fast-weight Δ={_fmt(s['fast_weight_delta'], '.3e')}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: run a reduced e2e and exit nonzero if any invariant fails.

    Used as the invariant-checked smoke entry point by later phases (hwxb.2.5 smoke
    train, hwxb.7.4 per-mechanism e2e, the ablation dry-run):
        python -m bio_inspired_nanochat.e2e_harness --synapses --steps 80
    """
    import argparse

    p = argparse.ArgumentParser(description="Reduced e2e training + invariant battery")
    p.add_argument("--synapses", action="store_true", help="use the synaptic model (default: vanilla)")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-dir", default=None)
    args = p.parse_args(argv)

    cfg = E2EConfig(
        synapses=args.synapses, steps=args.steps, n_layer=args.n_layer,
        n_embd=args.n_embd, seed=args.seed, device=args.device,
    )
    report = run_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
