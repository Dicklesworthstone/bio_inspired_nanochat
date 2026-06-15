"""E2E SCRIPT: full bio training run (tiny) with assertions + detailed logs (bead eqyk.4).

The single most important "is the whole thing wired correctly?" check. It trains a tiny
``GPTSynaptic`` with the per-synapse bio stack ON — presynaptic calcium/RRP/energy kinetics and
postsynaptic Hebbian fast-weights + CaMKII/PP1/BDNF consolidation — for a handful of steps on a
small *learnable* synthetic next-token task, and asserts the invariants that together mean the
integration across synaptic / optimizer / checkpoint / logging is sound:

(The structural Synaptic-MoE lifecycle is exercised end-to-end by its own safety net,
``tests/test_e2e_structural_lifecycle.py`` / bead ``eqyk.8``; it stays opt-in here via ``--moe``.)

  * **health**: per-step loss is finite and trends DOWN; grad norms finite+bounded; final params
    finite and bounded (no silent divergence); a sampled continuation is non-degenerate.
  * **mechanism engaged**: the *online* Hebbian state (eligibility traces) actually grows.
  * **checkpoint round-trip**: save → rebuild-from-identical-config → load reproduces the eval
    output exactly (the resume contract).
  * **bio buffers in range AND changing** (the eqyk.4 value-add over the generic harness):
    presynaptic calcium/RRP/energy, postsynaptic CaMKII/BDNF/PP1, and the MoE per-expert energy
    & fatigue all stay inside their biologically/numerically expected ranges and demonstrably
    *move* (a dead constant would pass a range check but means the mechanism is inert).

It emits a human-readable report plus a machine-readable ``events.jsonl`` trace (per-step
bio-state via the ``eqyk.2`` :class:`RunLogger` stream) for post-hoc inspection.

This reuses the canonical, self-tested primitives in ``bio_inspired_nanochat.e2e_harness``
(``_hebbian_state_fingerprint`` / ``_max_abs_param`` / ``_any_nonfinite_param`` / ``_eval_loss``)
and the ``InvariantResult`` record type, adding the bio-buffer instrumentation the generic
harness does not do.

Run:
    python -m scripts.e2e.train_bio --steps 80
    python -m scripts.e2e.train_bio --steps 80 --run-dir runs/e2e/bio   # keep the JSONL trace
Exits non-zero if any invariant fails (usable as an invariant-checked smoke entry point).
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bio_inspired_nanochat.common import logger as _logger
from bio_inspired_nanochat.e2e_harness import (
    InvariantResult,
    _any_nonfinite_param,
    _eval_loss,
    _hebbian_state_fingerprint,
    _max_abs_param,
)
from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import PostsynapticHebb, SynapticMoE
from bio_inspired_nanochat.torch_imports import torch


# --------------------------------------------------------------------------- #
# Config & result types
# --------------------------------------------------------------------------- #
@dataclass
class BioE2EConfig:
    """One tiny, all-mechanisms-ON bio training run."""

    n_layer: int = 2
    n_embd: int = 64
    n_head: int = 4
    vocab_size: int = 97
    seq_len: int = 32
    batch_size: int = 4
    pool_size: int = 8            # distinct fixed batches cycled (a small learnable task)
    steps: int = 80
    lr: float = 3e-3              # matches the dense-bio recipe proven stable in e2e_harness
    grad_clip: float = 1.0
    seed: int = 1234
    device: str = "cpu"
    # The structural Synaptic-MoE lifecycle (energy/fatigue metabolism, split/merge) is exercised
    # end-to-end by its OWN safety net (bead eqyk.8 / tests/test_e2e_structural_lifecycle.py); this
    # script targets the per-synapse bio stack (presyn kinetics + postsynaptic plasticity), whose
    # five buffers the bead names. MoE stays opt-in (and is less stable at this tiny scale).
    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    syn_overrides: dict[str, Any] = field(default_factory=dict)
    # Invariant thresholds.
    grad_norm_bound: float = 1e4
    param_absmax_bound: float = 1e4
    change_eps: float = 1e-5      # a buffer must move by more than this to count as "changing"
    presyn_change_eps: float = 1e-3  # presyn integrator spread across positions


@dataclass
class BioE2EReport:
    run_id: str
    config: BioE2EConfig
    passed: bool
    invariants: list[InvariantResult]
    loss_trajectory: list[float]
    summary: dict[str, Any]

    def failures(self) -> list[InvariantResult]:
        return [r for r in self.invariants if not r.passed]

    def assert_passed(self) -> None:
        if not self.passed:
            lines = "\n".join(r.line() for r in self.invariants)
            raise AssertionError(
                f"bio e2e run {self.run_id} failed {len(self.failures())} invariant(s):\n{lines}"
            )


# --------------------------------------------------------------------------- #
# Model / data
# --------------------------------------------------------------------------- #
def _build_bio_model(cfg: BioE2EConfig):
    """A tiny GPTSynaptic with all bio mechanisms ON (+ MoE)."""
    torch.manual_seed(cfg.seed)
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    from bio_inspired_nanochat.synaptic import SynapticConfig

    syn = SynapticConfig()  # enable_presyn / enable_hebbian / enable_metabolism default ON
    for k, v in cfg.syn_overrides.items():
        setattr(syn, k, v)
    gcfg = GPTSynapticConfig(
        sequence_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        synapses=True,
        syn_cfg=syn,
        use_moe=cfg.use_moe,
        num_experts=cfg.num_experts,
        moe_top_k=cfg.moe_top_k,
    )
    return GPTSynaptic(gcfg).to(cfg.device)


def _make_pool(cfg: BioE2EConfig) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fixed pool of (x, y) next-token batches — a small, *learnable* LM task (memorization),
    so a healthy run's loss genuinely decreases rather than the invariant being vacuous."""
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    pool = []
    for _ in range(cfg.pool_size):
        toks = torch.randint(
            0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len + 1), generator=gen, dtype=torch.long
        ).to(cfg.device)
        pool.append((toks[:, :-1].contiguous(), toks[:, 1:].contiguous()))
    return pool


# --------------------------------------------------------------------------- #
# Bio-state collection
# --------------------------------------------------------------------------- #
def _post_modules(model) -> list[PostsynapticHebb]:
    return [m for m in model.modules() if isinstance(m, PostsynapticHebb)]


def _moe_modules(model) -> list[SynapticMoE]:
    return [m for m in model.modules() if isinstance(m, SynapticMoE)]


@torch.no_grad()
def collect_persistent_bio(model) -> dict[str, float]:
    """Mean of the *persistent* bio buffers across the model — readable directly after a real
    training step with NO extra forward (so it never perturbs the run).

    Covers the postsynaptic (CaMKII / BDNF / PP1) and structural-MoE (energy / fatigue) state.
    The presynaptic calcium/RRP live in the per-forward KV-cache state, captured separately by
    :func:`presyn_position_trace` (a kv-cache forward) to avoid running an extra plasticity step
    in the training loop.
    """
    out: dict[str, float] = {}
    post = _post_modules(model)
    if post:
        out["camkii"] = sum(float(m.camkii.mean()) for m in post) / len(post)
        out["bdnf"] = sum(float(m.bdnf.mean()) for m in post) / len(post)
        out["pp1"] = sum(float(m.pp1.mean()) for m in post) / len(post)
    moe = _moe_modules(model)
    if moe:
        out["moe_energy"] = sum(float(m.energy.mean()) for m in moe) / len(moe)
        out["fatigue"] = sum(float(m.fatigue.mean()) for m in moe) / len(moe)
    return out


def presyn_position_trace(model, cfg: BioE2EConfig, batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """Presynaptic calcium/RRP/energy as a per-position curve from a *non-mutating* eval forward.

    Calcium is a leaky integrator of the attention drive and RRP a depleting vesicle pool, so a
    *live* presyn mechanism produces a curve that varies across key positions (rather than a dead
    constant). Presyn state is only exposed through a KV-cache forward.

    NB: this deliberately runs in eval mode WITH grad enabled (no ``torch.no_grad``) and never calls
    ``backward``. Under ``torch.no_grad`` the plasticity gate (``not grad_on or ...``) would RUN
    Hebbian consolidation / metabolism on the probe batch and mutate CaMKII/BDNF (the "inference runs
    plasticity" path); with grad on + ``training=False`` that gate stays OFF, so the probe runs NO new
    plasticity (CaMKII/BDNF/metabolism are left as trained). Like any forward it does flush the
    slow-weight write that vg9.2 *defers* from the previous training step — a legitimate, bounded
    application of training's own update, harmless to the post-run probes. We never backward, so
    KVCache's in-place writes don't trip autograd; only detached summaries escape.
    """
    was_training = model.training
    model.eval()
    B, T = batch.shape
    kv = KVCache(
        batch_size=B,
        num_heads=cfg.n_head,
        seq_len=T,
        head_dim=cfg.n_embd // cfg.n_head,
        num_layers=cfg.n_layer,
    )
    trace: dict[str, torch.Tensor] = {}
    try:
        model(batch, None, kv_cache=kv, train_mode=False)  # grad on, no backward → plasticity gated OFF
        states = kv.presyn_state if isinstance(kv.presyn_state, list) else []
        if states:
            last = states[-1]  # deepest layer's presyn state, shape (B, H, T)
            for key in ("C", "RRP", "E", "BUF"):
                v = last.get(key)
                if torch.is_tensor(v) and v.dim() >= 1:
                    trace[key] = v.detach().float().reshape(-1, v.shape[-1]).mean(dim=0)  # (T,)
    finally:
        if was_training:
            model.train()
    return trace


# --------------------------------------------------------------------------- #
# Generation probe (non-degeneracy)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _sample_continuation(model, cfg: BioE2EConfig, n_new: int = 16) -> list[int]:
    model.eval()
    gen = torch.Generator().manual_seed(cfg.seed + 7)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), generator=gen, dtype=torch.long).to(cfg.device)
    out: list[int] = []
    for _ in range(n_new):
        ctx = ids[:, -cfg.seq_len:]
        logits, _ = model(ctx, None, None, train_mode=False)
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1, generator=gen).item())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=cfg.device)], dim=1)
    model.train()
    return out


def _checkpoint_roundtrip(
    model, cfg: BioE2EConfig, probe: tuple[torch.Tensor, torch.Tensor]
) -> tuple[bool, str]:
    """save → fresh model (identical config) → load → identical eval-mode loss (resume contract).

    The synaptic model's *per-sequence transient* state (fast-weights / eligibility / presyn) is
    intentionally not checkpointed; it is reset at sequence boundaries. So the contract is: after
    resetting transient state, the persistent params+buffers round-trip exactly, verified by an
    identical deterministic eval forward.
    """
    x, y = probe
    if hasattr(model, "reset_sequence_state"):
        model.reset_sequence_state()
    state = copy.deepcopy(model.state_dict())
    la = _eval_loss(model, x, y, True)

    model_b = _build_bio_model(cfg)
    model_b.load_state_dict(state)
    if hasattr(model_b, "reset_sequence_state"):
        model_b.reset_sequence_state()
    lb = _eval_loss(model_b, x, y, True)
    ok = bool(abs(la - lb) <= 1e-5 + 1e-4 * abs(la))
    return ok, f"reload eval loss={lb:.6f} vs original={la:.6f} (Δ={abs(la - lb):.2e})"


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run_bio_e2e(cfg: BioE2EConfig, *, run_dir: str | Path | None = None, verbose: bool = True) -> BioE2EReport:
    """Train a tiny all-bio model + run the health/bio invariant battery; return a report."""
    if cfg.device == "cpu":
        try:
            torch.set_num_threads(min(4, os.cpu_count() or 4))
        except Exception:
            pass
    torch.manual_seed(cfg.seed)
    model = _build_bio_model(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    pool = _make_pool(cfg)

    rl = RunLogger(
        run_dir or (Path("runs") / "e2e" / "bio"),
        name="train_bio",
        console=False,
        provenance={"seed": cfg.seed, "n_layer": cfg.n_layer, "use_moe": cfg.use_moe},
    )

    fp_start = _hebbian_state_fingerprint(model)
    bio0 = collect_persistent_bio(model)

    losses: list[float] = []
    grad_norms: list[float] = []
    # Per-buffer trajectories (means) for the "in range AND changing" invariants.
    bio_traj: dict[str, list[float]] = {k: [] for k in bio0}
    try:
        for step in range(cfg.steps):
            x, y = pool[step % len(pool)]
            _, loss = model(x, y, None, train_mode=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            opt.step()

            lval = float(loss.detach().item())
            losses.append(lval)
            grad_norms.append(gnorm)
            bio = collect_persistent_bio(model)  # persistent buffers, no extra forward
            for k, v in bio.items():
                bio_traj[k].append(v)
            rl.event("train_step", step=step, loss=lval, lr=cfg.lr, grad_norm=gnorm)
            rl.log_bio_state(step=step, **{k: torch.tensor(v) for k, v in bio.items()})
            if verbose and (step % max(1, cfg.steps // 8) == 0 or step == cfg.steps - 1):
                _logger.info(
                    f"[bio-e2e] step {step:03d}/{cfg.steps} loss={lval:.4f} |grad|={gnorm:.3f} "
                    f"camkii={bio.get('camkii', float('nan')):.4f} bdnf={bio.get('bdnf', float('nan')):.4f} "
                    f"pp1={bio.get('pp1', float('nan')):.4f}"
                )

        # Hebbian fingerprint BEFORE the checkpoint probe (which resets the eligibility traces).
        fp_end = _hebbian_state_fingerprint(model)

        # Presynaptic calcium/RRP/energy as a per-position curve (post-training, no perturbation).
        # A degenerate model must FAIL the bio invariants (empty trace), not crash the run.
        try:
            presyn = presyn_position_trace(model, cfg, pool[0][0])
        except Exception as e:
            presyn = {}
            _logger.warning(f"[bio-e2e] presyn probe raised (treated as missing): {e}")
        rl.log_bio_state(
            step=cfg.steps,
            **{f"presyn_{k}": v for k, v in presyn.items()},
        )

        try:
            ckpt_ok, ckpt_detail = _checkpoint_roundtrip(model, cfg, pool[0])
        except Exception as e:  # a degenerate model must FAIL the invariant, not crash the run
            ckpt_ok, ckpt_detail = False, f"checkpoint probe raised: {type(e).__name__}: {e}"
        try:
            sample = _sample_continuation(model, cfg)
        except Exception as e:
            sample = []
            _logger.warning(f"[bio-e2e] generation probe raised (treated as degenerate): {e}")
        max_abs = _max_abs_param(model)
        nonfinite = _any_nonfinite_param(model)
    finally:
        rl.close()

    invariants = _invariant_battery(
        cfg, losses, grad_norms, bio_traj, presyn, fp_start, fp_end,
        max_abs, nonfinite, ckpt_ok, ckpt_detail, sample,
    )
    passed = all(r.passed for r in invariants)
    report = BioE2EReport(
        run_id=rl.run_id,
        config=cfg,
        passed=passed,
        invariants=invariants,
        loss_trajectory=losses,
        summary={
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "max_grad_norm": max(grad_norms) if grad_norms else None,
            "hebbian_delta": fp_end - fp_start,
            "max_abs_param": max_abs,
            "bio_final": {k: (v[-1] if v else None) for k, v in bio_traj.items()},
            "events_path": str(rl.events_path),
            "n_failed": sum(1 for r in invariants if not r.passed),
        },
    )
    if verbose:
        _log_report(report)
    return report


# --------------------------------------------------------------------------- #
# Invariant battery
# --------------------------------------------------------------------------- #
def _expected_ranges(cfg: BioE2EConfig) -> dict[str, tuple[float, float]]:
    """Biologically/numerically expected ranges. The clamped quantities (CaMKII/PP1∈[0,1],
    BDNF∈[0,bdnf_max]) are tight; the rest assert non-negativity + boundedness (catches NaN/Inf
    and sign/runaway bugs)."""
    from bio_inspired_nanochat.synaptic import SynapticConfig

    syn = SynapticConfig()
    for k, v in cfg.syn_overrides.items():
        setattr(syn, k, v)
    eps = 1e-4
    return {
        "camkii": (0.0, 1.0 + eps),
        "pp1": (0.0, 1.0 + eps),
        "bdnf": (0.0, float(syn.bdnf_max) + eps),
        "moe_energy": (0.0, 2.0),       # filled toward ~1, capped; >1 would be suspicious
        "fatigue": (0.0, 1e3),          # non-negative, bounded
        "presyn_C": (0.0, 1e3),         # softplus integrator: non-negative, bounded
        "presyn_RRP": (0.0, float(syn.init_rrp) * 2.0 + eps),  # pool bounded by refill
        "presyn_E": (0.0, 5.0),         # energy gate: non-negative, bounded
        "presyn_BUF": (-eps, 1.0 + eps),
    }


def _invariant_battery(
    cfg: BioE2EConfig,
    losses: list[float],
    grad_norms: list[float],
    bio_traj: dict[str, list[float]],
    presyn: dict[str, torch.Tensor],
    fp_start: float,
    fp_end: float,
    max_abs: float,
    nonfinite: bool,
    ckpt_ok: bool,
    ckpt_detail: str,
    sample: list[int],
) -> list[InvariantResult]:
    out: list[InvariantResult] = []
    ranges = _expected_ranges(cfg)

    # -- health ---------------------------------------------------------------
    finite_losses = bool(losses) and all(math.isfinite(x) for x in losses)
    out.append(InvariantResult(
        "loss_finite", finite_losses, finite_losses,
        "all step losses finite" if finite_losses
        else f"{sum(1 for x in losses if not math.isfinite(x))} non-finite loss(es)",
    ))

    q = max(1, len(losses) // 4)
    first = sum(losses[:q]) / q if losses else math.nan
    last = sum(losses[-q:]) / q if losses else math.nan
    decreased = finite_losses and (last < first)
    out.append(InvariantResult(
        "loss_decreases", decreased, (first, last),
        f"first-quartile mean={first:.4f} -> last-quartile mean={last:.4f}"
        + ("" if decreased else "  (NOT decreasing)"),
    ))

    gn_ok = bool(grad_norms) and all(math.isfinite(g) and g <= cfg.grad_norm_bound for g in grad_norms)
    out.append(InvariantResult(
        "grad_norm_finite_bounded", gn_ok, max(grad_norms) if grad_norms else None,
        f"max grad norm={max(grad_norms):.4f} (bound {cfg.grad_norm_bound:g})" if grad_norms else "no grads",
    ))

    params_ok = not nonfinite
    out.append(InvariantResult(
        "params_finite", params_ok, params_ok,
        "all final params finite" if params_ok else "final params contain NaN/Inf",
    ))

    stable = math.isfinite(max_abs) and max_abs <= cfg.param_absmax_bound
    out.append(InvariantResult(
        "mechanism_stable", stable, max_abs,
        f"max|param|={max_abs:.4g} within bound {cfg.param_absmax_bound:g}" if stable
        else f"max|param|={max_abs:.4g} EXCEEDS bound {cfg.param_absmax_bound:g} (runaway)",
    ))

    delta = abs(fp_end - fp_start)
    engaged = delta > 1e-9
    out.append(InvariantResult(
        "mechanism_engaged", engaged, delta,
        f"online Hebbian state grew by {delta:.3e}" if engaged
        else "online Hebbian state did not move (mechanism inert)",
    ))

    out.append(InvariantResult("checkpoint_roundtrip", ckpt_ok, ckpt_ok, ckpt_detail))

    uniq = len(set(sample))
    gen_ok = uniq >= 2
    out.append(InvariantResult(
        "generation_nondegenerate", gen_ok, uniq,
        f"{uniq} distinct tokens in a {len(sample)}-token sample"
        + ("" if gen_ok else "  (degenerate)"),
    ))

    # -- bio buffers: in range -----------------------------------------------
    range_problems: list[str] = []
    # persistent buffers (per-step trajectories)
    for name, traj in bio_traj.items():
        lo, hi = ranges.get(name, (-math.inf, math.inf))
        bad = [v for v in traj if not (math.isfinite(v) and lo <= v <= hi)]
        if bad:
            range_problems.append(f"{name}∉[{lo:g},{hi:g}] (e.g. {bad[0]:.4g})")
    # presyn position curves
    for key, vec in presyn.items():
        name = f"presyn_{key}"
        lo, hi = ranges.get(name, (-math.inf, math.inf))
        vmin, vmax = float(vec.min()), float(vec.max())
        if not (math.isfinite(vmin) and math.isfinite(vmax) and lo <= vmin and vmax <= hi):
            range_problems.append(f"{name}∉[{lo:g},{hi:g}] (min={vmin:.4g},max={vmax:.4g})")
    range_ok = not range_problems
    out.append(InvariantResult(
        "bio_buffers_in_range", range_ok, len(range_problems),
        "all bio buffers within expected ranges" if range_ok
        else "out-of-range: " + "; ".join(range_problems),
    ))

    # -- bio buffers: actually change ----------------------------------------
    change_problems: list[str] = []

    def _span(vals: list[float]) -> float:
        return (max(vals) - min(vals)) if vals else 0.0

    # The five bead-named buffers must actually move (a dead constant passes a range check but
    # means the mechanism is inert): postsynaptic CaMKII/BDNF accumulate over training, and the
    # presynaptic calcium(C)/RRP/energy(E) vary across key positions (live leaky integrator +
    # vesicle depletion + energy gate). MoE energy is an optional bonus when --moe is set.
    for name in ("camkii", "bdnf"):
        sp = _span(bio_traj.get(name, []))
        if name not in bio_traj or sp <= cfg.change_eps:
            change_problems.append(f"{name} static (span={sp:.2e})")
    if "moe_energy" in bio_traj and _span(bio_traj["moe_energy"]) <= cfg.change_eps:
        change_problems.append(f"moe_energy static (span={_span(bio_traj['moe_energy']):.2e})")
    for key in ("C", "RRP", "E"):
        if key not in presyn:
            change_problems.append(f"presyn_{key} missing")
        else:
            sp = float(presyn[key].max() - presyn[key].min())
            if sp <= cfg.presyn_change_eps:
                change_problems.append(f"presyn_{key} flat across positions (span={sp:.2e})")
    change_ok = not change_problems
    out.append(InvariantResult(
        "bio_buffers_change", change_ok, len(change_problems),
        "all tracked bio buffers move (mechanisms live)" if change_ok
        else "static buffers: " + "; ".join(change_problems),
    ))

    return out


def _log_report(report: BioE2EReport) -> None:
    head = "PASSED" if report.passed else "FAILED"
    _logger.info(f"[bio-e2e] ===== run {report.run_id} {head} =====")
    for r in report.invariants:
        _logger.info(r.line())
    s = report.summary

    def _fmt(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else "n/a"

    _logger.info(
        f"[bio-e2e] loss {_fmt(s['initial_loss'], '.4f')} -> {_fmt(s['final_loss'], '.4f')} | "
        f"max|grad|={_fmt(s['max_grad_norm'], '.3f')} | Hebbian Δ={_fmt(s['hebbian_delta'], '.3e')}"
    )
    _logger.info(f"[bio-e2e] bio final: {s['bio_final']}")
    _logger.info(f"[bio-e2e] JSONL trace: {s['events_path']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tiny all-bio e2e training run + invariant battery")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--moe", action="store_true", help="also exercise the structural MoE path (opt-in)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-dir", default=None, help="where to write events.jsonl (default: runs/e2e/bio)")
    args = p.parse_args(argv)

    cfg = BioE2EConfig(
        steps=args.steps, n_layer=args.n_layer, n_embd=args.n_embd, lr=args.lr,
        seed=args.seed, use_moe=args.moe, device=args.device,
    )
    report = run_bio_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
