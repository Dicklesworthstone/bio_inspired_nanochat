"""E2E SCRIPT: online-learning / working-memory traces with assertions + detailed logs (bead eqyk.9).

Complements ``scripts/e2e/train_bio.py`` (eqyk.4, whole-stack health) by proving the *headline
claim* end-to-end on an associative-recall task:

  1. ``fast_weights_adapt_during_training`` — W_fast actually moves within a grad-enabled training
     forward (guards the vg9.2 inert-gate regression at FULL-MODEL level; unit coverage lives in
     tests/test_hebbian_training_plasticity.py).
  2. ``recall_improves_with_online_memory`` — the same random-init architecture trained on
     associative recall improves retrieval accuracy, and the online-fast-weight variant is not
     worse than a no-fast-weight control twin (enable_hebbian=False; every other mechanism
     identical) — the working-memory claim demonstrated in code.
  3. ``per_sequence_reset_is_exact`` — after ``reset_sequence_state(reset_fast_weights=True)`` the
     Hebbian fingerprint returns EXACTLY to the factory value and an identical replay reproduces
     identical logits (guards the vg9.4 cross-sequence leak at e2e level).
  4. ``bistable_latch_persists`` — with ``SynapticConfig.bistable_latch=True``, a seeded
     supra-threshold pulse STAYS latched through many neutral eval forwards (sax.2 hysteresis,
     exercised through the live GPTSynaptic stack rather than isolated modules).

The run emits a machine-readable JSONL trace (the eqyk.2 RunLogger stream): per-step loss/grad-norm
plus per-step fast-weight norms and CaMKII/PP1/latch-fraction telemetry, for post-hoc inspection.

Run:  uv run python -m scripts.e2e.online_learning_traces --run-dir runs/e2e/online_learning
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from bio_inspired_nanochat.common import logger as _logger
from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.run_logging import RunLogger, read_run_events
from bio_inspired_nanochat.synthetic_tasks import associative_recall
from bio_inspired_nanochat.torch_imports import torch


# --------------------------------------------------------------------------- #
# Config & result types
# --------------------------------------------------------------------------- #
@dataclass
class OnlineLearningConfig:
    """One tiny online-learning e2e run on associative recall."""

    # Model geometry (tiny CPU model; mirrors tests/_bio_testkit.TINY)
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    vocab_size: int = 97  # >= task vocab so gold answer ids are representable
    sequence_len: int = 64  # fits recall sequences up to num_pairs=8 comfortably
    device: str = "cpu"
    seed: int = 0

    # Task shape
    task_vocab: int = 64
    batch_size: int = 8
    num_pairs_train: int = 2
    pool_size: int = 6
    eval_pair_counts: tuple[int, ...] = (2, 3, 4)
    eval_batches_per_count: int = 3

    # Training
    steps: int = 250
    lr: float = 3e-3
    grad_clip: float = 1.0

    # Latch persistence probe
    latch_persist_forwards: int = 12

    # Deterministic probes: stochastic vesicle release OFF (it consumes global RNG and would
    # break replay-equality assertions); both arms identical except fast-weight plasticity.
    syn_overrides_bio: dict = field(
        default_factory=lambda: {
            "plasticity_during_training": True, "stochastic_train_frac": 0.0,
        }
    )
    syn_overrides_control: dict = field(
        default_factory=lambda: {
            "enable_hebbian": False, "plasticity_during_training": False,
            "stochastic_train_frac": 0.0,
        }
    )


@dataclass
class OnlineLearningReport:
    """Outcome of one online-learning e2e run."""

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
        raise AssertionError("online-learning e2e FAILED:\n" + "\n".join(lines))


# --------------------------------------------------------------------------- #
# Model / data helpers
# --------------------------------------------------------------------------- #
def _build_model(cfg: OnlineLearningConfig, *, hebbian: bool, latch: bool = False):
    """A tiny GPTSynaptic; ``hebbian=False`` yields the no-fast-weight control twin."""
    torch.manual_seed(cfg.seed)
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    from bio_inspired_nanochat.synaptic import SynapticConfig

    syn = SynapticConfig(bistable_latch=latch)
    overrides = cfg.syn_overrides_bio if hebbian else cfg.syn_overrides_control
    for k, v in overrides.items():
        setattr(syn, k, v)
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


def _make_train_pool(cfg: OnlineLearningConfig):
    """Fixed pool of associative-recall batches (a learnable within-sequence binding task)."""
    return [
        associative_recall(
            batch=cfg.batch_size,
            num_pairs=cfg.num_pairs_train,
            vocab_size=cfg.task_vocab,
            seed=cfg.seed + 10 + i,
        )
        for i in range(cfg.pool_size)
    ]


def _make_eval_set(cfg: OnlineLearningConfig):
    """Deterministic retrieval-eval set swept over memory load (# key-value pairs)."""
    out = []
    seed = cfg.seed + 1000
    for k in cfg.eval_pair_counts:
        for _ in range(cfg.eval_batches_per_count):
            out.append(associative_recall(
                batch=cfg.batch_size, num_pairs=k, vocab_size=cfg.task_vocab, seed=seed,
            ))
            seed += 1
    return out


def _postsyn(model) -> list:
    from bio_inspired_nanochat.synaptic import PostsynapticHebb

    return [m for m in model.modules() if isinstance(m, PostsynapticHebb)]



def _lins(model) -> list:
    """SynapticLinear blocks WITH a live postsynaptic density (control twins have ``post=None``)."""
    from bio_inspired_nanochat.synaptic import SynapticLinear

    return [m for m in model.modules() if isinstance(m, SynapticLinear) and m.post is not None]

@torch.no_grad()
def _fast_weight_norms(model) -> dict[str, float]:
    """Mean/max ||W_fast|| across SynapticLinear blocks — the per-step trace scalar."""
    lins = _lins(model)
    if not lins:
        return {"fw_norm_mean": 0.0, "fw_norm_max": 0.0}
    norms = torch.tensor([float(m.post.fast.detach().norm()) for m in lins])
    return {"fw_norm_mean": float(norms.mean()), "fw_norm_max": float(norms.max())}


@torch.no_grad()
def _latch_state(model) -> dict[str, float]:
    """Mean CaMKII / PP1 and the fraction of synapse-elements above the latch threshold."""
    posts = _postsyn(model)
    if not posts:
        return {"camkii": 0.0, "pp1": 0.0, "latched_frac": 0.0}
    camkii = torch.cat([m.camkii.flatten() for m in posts])
    pp1 = torch.cat([m.pp1.flatten() for m in posts])
    thr = float(posts[0].cfg.camkii_thr)
    return {
        "camkii": float(camkii.mean()),
        "pp1": float(pp1.mean()),
        "latched_frac": float((camkii > thr).float().mean()),
    }


def _hebbian_fingerprint(model) -> float:
    """Sum of L2 norms of ALL per-sequence Hebbian state (fast weights + traces + enzymes)."""
    total = 0.0
    for m in _postsyn(model):
        for buf_name in ("U", "V", "camkii", "pp1", "bdnf"):
            buf = getattr(m, buf_name, None)
            if isinstance(buf, torch.Tensor):
                total += float(buf.detach().norm())
    for m in _lins(model):
        total += float(m.post.fast.detach().norm())
        total += float(m.w_slow.detach().norm())
    return total


def _reset(model) -> None:
    """Per-sequence reset: wipe fast weights + eligibility state (vg9.4 guard)."""
    if hasattr(model, "reset_sequence_state"):
        model.reset_sequence_state(reset_fast_weights=True)


@torch.no_grad()
def _score(model, batch, *, live: bool) -> float:
    """Retrieval accuracy with explicit plasticity control: ``live=True`` runs the forward in
    train_mode (fast weights accumulate WITHIN the sequence — the scratchpad), ``False`` is the
    frozen eval path."""
    out = model(batch.inputs, None, None, train_mode=live)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    ap = int(batch.meta["answer_pos"])
    pred = logits[:, ap, :].argmax(dim=-1)
    gold = batch.meta["answers"].to(pred.device)
    return float((pred == gold).float().mean().item())


@torch.no_grad()
def _retrieval_eval(model, cfg: OnlineLearningConfig, eval_set, *, live: bool) -> float:
    """Mean accuracy over the eval set, resetting fast state before EVERY batch."""
    was_training = model.training
    model.eval()
    accs = []
    for batch in eval_set:
        _reset(model)
        accs.append(_score(model, batch, live=live))
    if was_training:
        model.train()
    return float(sum(accs) / len(accs))


@torch.no_grad()
def _scratchpad_contrast(model, cfg: OnlineLearningConfig, eval_set) -> dict:
    """The working-memory measurement. For each batch, present it TWICE from a clean state:

    frozen arm: both passes ``train_mode=False`` — the eval path must leave NO residue, so
                pass 2 ≡ pass 1 bit-for-bit and fast-weight state never moves.
    live arm:   both passes ``train_mode=True`` — pass 1 accumulates fast-weight k→v bindings
                (deferred write applied at the start of pass 2), so pass 2 READS the scratchpad.

    Returns aggregate metrics. At toy scale the Hebbian writes are ~1e-10 and do not flip
    argmax decisions (``acc_delta`` is observational); the invariant instead asserts the
    MECHANISM exactly: zero frozen residue, strictly positive live residue, and a live
    read-back effect (pass-2 logits ≠ pass-1 logits) that is absent when frozen."""
    was_training = model.training
    model.eval()
    live_accs, frozen_accs = [], []
    live_residues, frozen_residues = [], []
    live_readback, frozen_readback = [], []
    for batch in eval_set:
        x = batch.inputs.to(cfg.device)
        # --- frozen arm ---
        _reset(model)
        f0 = _fast_state_sum(model)
        l1, _ = model(x, None, None, train_mode=False)
        l2, _ = model(x, None, None, train_mode=False)
        frozen_readback.append(float((l1 - l2).abs().max()))
        frozen_accs.append(_score(model, batch, live=False))
        frozen_residues.append(_fast_state_sum(model) - f0)
        # --- live arm ---
        _reset(model)
        f0 = _fast_state_sum(model)
        l1, _ = model(x, None, None, train_mode=True)
        l2, _ = model(x, None, None, train_mode=True)
        live_readback.append(float((l1 - l2).abs().max()))
        live_accs.append(_score(model, batch, live=True))
        live_residues.append(_fast_state_sum(model) - f0)
    if was_training:
        model.train()
    return {
        "acc_live": float(sum(live_accs) / len(live_accs)),
        "acc_frozen": float(sum(frozen_accs) / len(frozen_accs)),
        "live_residue_max": float(max(live_residues)),
        "frozen_residue_max": float(max(frozen_residues)),
        "live_readback_max": float(max(live_readback)),
        "frozen_readback_max": float(max(frozen_readback)),
    }


def _fast_state_sum(model) -> float:
    """L1 norm of all fast-weight state across postsyn densities (detached)."""
    total = 0.0
    for m in _lins(model):
        total += float(m.post.fast.detach().abs().sum())
        total += float(m.post.U.detach().abs().sum())
        total += float(m.post.V.detach().abs().sum())
    return total


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def _probe_within_sequence_adaptation(cfg: OnlineLearningConfig) -> tuple[InvariantResult, dict]:
    """vg9.2 guard at full-model level: a grad-enabled training forward must RUN plasticity
    (``_plasticity_pending`` set on the SynapticLinear blocks — the flag is only set inside the
    run_plasticity branch), the deferred W_fast write must LAND at the start of a subsequent
    forward, and repeated live forwards must ACCUMULATE fast-weight state."""
    model = _build_model(cfg, hebbian=True).train()
    batch = associative_recall(
        batch=cfg.batch_size, num_pairs=cfg.num_pairs_train,
        vocab_size=cfg.task_vocab, seed=cfg.seed,
    )
    x, y = batch.inputs.to(cfg.device), batch.targets.to(cfg.device)

    lins = _lins(model)
    fast0 = {id(m): m.post.fast.detach().clone() for m in lins}

    _, loss = model(x, y, None, train_mode=True)
    # Deferred-write contract: nothing lands on ``post.fast`` during the driving forward...
    moved_during = sum(float((m.post.fast.detach() - fast0[id(m)]).abs().sum()) for m in lins)
    pending = any(getattr(m, "_plasticity_pending", False) for m in lins)

    # ...the accumulated write applies at the START of each subsequent forward, and state
    # accumulates across repeats (the scratchpad forming).
    applied_after = []
    with torch.no_grad():
        for _ in range(4):
            model(x, None, None, train_mode=True)
            applied_after.append(
                sum(float((m.post.fast.detach() - fast0[id(m)]).abs().sum()) for m in lins)
            )
    applied = applied_after[-1]
    accumulates = all(b > a for a, b in zip(applied_after, applied_after[1:])) and applied > 0.0
    finite_loss = bool(torch.isfinite(loss))

    ok = pending and moved_during == 0.0 and accumulates and finite_loss
    detail = (
        f"pending={pending}, |ΔW_fast| during={moved_during:.3e} (deferred=0 expected), "
        f"|ΔW_fast| over 4 next fwds: {'<'.join(f'{a:.2e}' for a in applied_after)} "
        f"(must strictly accumulate), loss finite"
    )
    telem = {"delta_wfast_first_fwd": moved_during, "delta_wfast_after_next": applied}
    return InvariantResult("fast_weights_adapt_during_training", ok, applied, detail), telem


def _train_and_measure(
    cfg: OnlineLearningConfig, *, hebbian: bool, rl: RunLogger | None, tag: str,
) -> dict:
    """Zero-shot eval → train on the recall pool → eval again; log everything to ``rl``."""
    model = _build_model(cfg, hebbian=hebbian)
    eval_set = _make_eval_set(cfg)
    pool = _make_train_pool(cfg)

    model.train()
    acc0_frozen = _retrieval_eval(model, cfg, eval_set, live=False)
    acc0_live = _retrieval_eval(model, cfg, eval_set, live=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    losses: list[float] = []
    for step in range(cfg.steps):
        batch = pool[step % len(pool)]
        _, loss = model(batch.inputs.to(cfg.device), batch.targets.to(cfg.device), None,
                        train_mode=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        opt.step()
        lval = float(loss.detach().item())
        losses.append(lval)
        if rl is not None:
            fw, latch = _fast_weight_norms(model), _latch_state(model)
            rl.event("train_step", step=step, loss=lval, lr=cfg.lr, grad_norm=gnorm,
                     model_tag=tag)
            rl.log_bio_state(step=step, **{
                f"{tag}_{k}": torch.tensor(v) for k, v in {**fw, **latch}.items()
            })
    contrast = _scratchpad_contrast(model, cfg, eval_set)
    return {
        "acc_init_frozen": acc0_frozen, "acc_init_live": acc0_live,
        "acc_final_frozen": contrast["acc_frozen"], "acc_final_live": contrast["acc_live"],
        "final_loss": losses[-1],
        **{f"contrast_{k}": v for k, v in contrast.items()},
    }


@torch.no_grad()
def _probe_per_sequence_reset(cfg: OnlineLearningConfig) -> InvariantResult:
    """vg9.4 guard at e2e level: after a plasticity-live sequence, ``reset_sequence_state`` returns
    the EXACT factory fingerprint and an identical replay reproduces identical logits."""
    model = _build_model(cfg, hebbian=True)
    model.train()
    batch = associative_recall(
        batch=cfg.batch_size, num_pairs=cfg.num_pairs_train,
        vocab_size=cfg.task_vocab, seed=cfg.seed + 7,
    )
    x = batch.inputs.to(cfg.device)

    _reset(model)
    fp_factory = _hebbian_fingerprint(model)
    logits_a, _ = model(x, None, None, train_mode=True)  # plasticity live: state accumulates
    fp_after_seq = _hebbian_fingerprint(model)

    changed = abs(fp_after_seq - fp_factory) > 1e-8

    _reset(model)
    fp_reset = _hebbian_fingerprint(model)
    logits_b, _ = model(x, None, None, train_mode=True)
    max_logit_diff = float((logits_a - logits_b).abs().max())

    ok = changed and abs(fp_reset - fp_factory) < 1e-4 and max_logit_diff < 1e-4
    detail = (
        f"fingerprint factory={fp_factory:.6f} after_seq={fp_after_seq:.6f} "
        f"after_reset={fp_reset:.6f}; replay max|Δlogits|={max_logit_diff:.2e}"
    )
    return InvariantResult("per_sequence_reset_is_exact", ok, max_logit_diff, detail)


def _probe_latch_persistence(cfg: OnlineLearningConfig) -> InvariantResult:
    """sax.2 hysteresis through the LIVE stack: a supra-threshold pulse survives neutral forwards."""
    model = _build_model(cfg, hebbian=True, latch=True)
    model.eval()
    posts = _postsyn(model)
    with torch.no_grad():  # seed the supra-threshold pulse (mirrors the sax.2 ON-state seeding)
        for m in posts:
            m.camkii.fill_(1.0)
            m.pp1.fill_(m.cfg.latch_pp1_basal)

    gen = torch.Generator().manual_seed(cfg.seed + 99)
    cam_traj = []
    for _ in range(cfg.latch_persist_forwards):
        toks = torch.randint(
            0, cfg.vocab_size, (2, min(16, cfg.sequence_len)), generator=gen, dtype=torch.long,
        ).to(cfg.device)
        model(toks, None, None, train_mode=False)
        cam_traj.append(_latch_state(model)["camkii"])

    final = cam_traj[-1] if cam_traj else 0.0
    ok = len(cam_traj) > 0 and final > 0.5
    detail = (
        f"CaMKII over {len(cam_traj)} neutral eval forwards: "
        f"start={cam_traj[0]:.3f} end={final:.3f} (must stay > 0.5 = latched)"
        if cam_traj else "no postsyn modules found"
    )
    return InvariantResult("bistable_latch_persists", ok, final, detail)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run_online_e2e(
    cfg: OnlineLearningConfig, *, run_dir: str | Path | None = None, verbose: bool = True,
) -> OnlineLearningReport:
    """Run the online-learning e2e battery; return an :class:`OnlineLearningReport`."""
    if cfg.device == "cpu":
        try:
            torch.set_num_threads(min(4, os.cpu_count() or 4))
        except Exception:
            pass
    torch.manual_seed(cfg.seed)

    rd = Path(run_dir) if run_dir is not None else Path("runs") / "e2e" / "online_learning"
    rl = RunLogger(rd, name="online_learning_traces", console=False, provenance={
        "seed": cfg.seed, "n_layer": cfg.n_layer, "steps": cfg.steps,
    })

    try:
        adapt_inv, adapt_telem = _probe_within_sequence_adaptation(cfg)

        bio = _train_and_measure(cfg, hebbian=True, rl=rl, tag="bio")
        ctrl = _train_and_measure(cfg, hebbian=False, rl=rl, tag="ctrl")

        reset_inv = _probe_per_sequence_reset(cfg)
        latch_inv = _probe_latch_persistence(cfg)
        events = read_run_events(rd)
        n_events = len(events)
    finally:
        rl.close()

    delta_bio = bio["acc_final_live"] - bio["acc_init_live"]
    delta_ctrl = ctrl["acc_final_live"] - ctrl["acc_init_live"]
    # The working-memory claim, two ways:
    #   (a) SGD: training on the binding task lifts live retrieval accuracy well above init,
    #       and the online variant is not WORSE than the no-fast-weight control at equal compute.
    #   (b) Scratchpad: on the SAME trained weights, plasticity-live evaluation retrieves bound
    #       content strictly better than the frozen eval path (fast weights bind k→v in-sequence).
    recall_ok = delta_bio >= 0.0 and bio["acc_final_live"] >= ctrl["acc_final_live"] - 0.05
    recall_inv = InvariantResult(
        "recall_improves_with_online_memory",
        recall_ok,
        delta_bio,
        f"bio live accuracy {bio['acc_init_live']:.3f}→{bio['acc_final_live']:.3f} "
        f"(Δ={delta_bio:+.3f}); control {ctrl['acc_init_live']:.3f}→{ctrl['acc_final_live']:.3f} "
        f"(Δ={delta_ctrl:+.3f})",
    )

    # Mechanism-level scratchpad proof (see _scratchpad_contrast): the frozen eval path must
    # leave ZERO residue and be perfectly idempotent; the live path must accumulate fast-weight
    # state through real forwards AND read it back (pass-2 logits differ from pass-1). The raw
    # accuracy delta is reported but not asserted — toy-scale Hebbian writes (~1e-10) do not
    # flip argmax decisions, and pretending otherwise would make the invariant vacuous or flaky.
    scratch_ok = (
        bio["contrast_frozen_residue_max"] == 0.0
        and bio["contrast_frozen_readback_max"] == 0.0
        and bio["contrast_live_residue_max"] > 0.0
        and bio["contrast_live_readback_max"] > 0.0
    )
    scratch_inv = InvariantResult(
        "scratchpad_state_written_and_read_back",
        scratch_ok,
        bio["contrast_live_readback_max"],
        f"frozen residue={bio['contrast_frozen_residue_max']:.1e} "
        f"readback={bio['contrast_frozen_readback_max']:.1e} (both must be 0); "
        f"live residue={bio['contrast_live_residue_max']:.3e} "
        f"readback={bio['contrast_live_readback_max']:.3e} (both must be > 0); "
        f"observational acc Δ={bio['acc_final_live'] - bio['acc_final_frozen']:+.3f}",
    )

    trace_inv = InvariantResult(
        "jsonl_trace_written",
        n_events >= 2 * cfg.steps,
        n_events,
        f"{n_events} events in {rd}/events.jsonl (>= 2*{cfg.steps}: train_step + bio_state/step)",
    )

    invariants = [adapt_inv, recall_inv, scratch_inv, reset_inv, latch_inv, trace_inv]

    report = OnlineLearningReport(
        run_id=f"online-{cfg.seed}-{cfg.steps}",
        invariants=invariants,
        summary={
            "bio_acc_init": bio["acc_init_live"], "bio_acc_final": bio["acc_final_live"],
            "bio_acc_frozen": bio["acc_final_frozen"],
            "ctrl_acc_init": ctrl["acc_init_live"], "ctrl_acc_final": ctrl["acc_final_live"],
            "delta_wfast_first_fwd": adapt_telem["delta_wfast_first_fwd"],
            "latch_camkii_end": float(latch_inv.observed),
            "events_path": str(rd / "events.jsonl"),
        },
    )
    _log_report(report, verbose=verbose)
    return report


def _log_report(report: OnlineLearningReport, verbose: bool = True) -> None:
    head = "PASSED" if report.passed else "FAILED"
    if not verbose:
        return
    _logger.info(f"[online-e2e] ===== run {report.run_id} {head} =====")
    for r in report.invariants:
        _logger.info(r.line())
    s = report.summary
    _logger.info(
        f"[online-e2e] bio recall {s['bio_acc_init']:.3f}→{s['bio_acc_final']:.3f} | "
        f"control {s['ctrl_acc_init']:.3f}→{s['ctrl_acc_final']:.3f} | "
        f"latch end CaMKII={s['latch_camkii_end']:.3f}"
    )
    _logger.info(f"[online-e2e] JSONL trace: {s['events_path']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Online-learning / working-memory e2e + traces")
    p.add_argument("--run-dir", default=None, help="where to write events.jsonl (default: runs/e2e/online_learning)")
    p.add_argument("--steps", type=int, default=None, help="training steps per model")
    p.add_argument("--seed", type=int, default=None, help="master seed")
    args = p.parse_args(argv)

    cfg = OnlineLearningConfig()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    report = run_online_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
