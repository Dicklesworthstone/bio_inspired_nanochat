"""
Training-loop divergence guards — bead vg9.7.

Covers: non-finite (NaN/Inf) loss and bio-buffer detection -> configured action; loss-spike
-> backoff; bio-norm explosion -> spike action; bio-buffer norm gathering (early-warning
logging); opt-in rollback snapshot/restore; the disabled no-op path. Acceptance: "injecting a
NaN triggers the configured guard; bio-buffer norm logging present."

Run:  pytest tests/test_divergence_guard.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from bio_inspired_nanochat.divergence_guard import (
    DivergenceGuard,
    DivergenceGuardConfig,
    GuardAction,
    _as_list,
    build_divergence_guard,
)

from _bio_testkit import make_tiny_synaptic


# --------------------------------------------------------------------------- #
# 1. Non-finite loss -> configured action (the headline acceptance)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_triggers_configured_action(bad):
    g = DivergenceGuard(DivergenceGuardConfig(on_nonfinite=GuardAction.SKIP, check_bio_buffers=False))
    r = g.check(torch.tensor(bad), torch.nn.Linear(2, 2), step=5)
    assert r.action == GuardAction.SKIP
    assert r.nonfinite is True
    assert "non-finite loss" in r.reason


@pytest.mark.unit
def test_finite_loss_is_ok():
    g = DivergenceGuard(DivergenceGuardConfig(check_bio_buffers=False))
    r = g.check(torch.tensor(2.5), torch.nn.Linear(2, 2), step=5)
    assert r.action == GuardAction.OK and not r.nonfinite


@pytest.mark.unit
def test_on_nonfinite_action_is_configurable():
    g = DivergenceGuard(DivergenceGuardConfig(on_nonfinite=GuardAction.ROLLBACK, check_bio_buffers=False))
    r = g.check(float("nan"), torch.nn.Linear(2, 2))
    assert r.action == GuardAction.ROLLBACK


# --------------------------------------------------------------------------- #
# 2. Loss spike (vs EMA, after warmup) -> backoff
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_loss_spike_after_warmup_triggers_backoff():
    g = DivergenceGuard(DivergenceGuardConfig(
        warmup_steps=3, loss_spike_factor=5.0, on_spike=GuardAction.BACKOFF, check_bio_buffers=False))
    m = torch.nn.Linear(2, 2)
    for step in range(6):
        r = g.check(torch.tensor(2.0), m, step=step)   # build a stable EMA ~2.0
        assert r.action == GuardAction.OK
    spike = g.check(torch.tensor(50.0), m, step=6)      # 25x the EMA -> spike
    assert spike.action == GuardAction.BACKOFF
    assert "loss spike" in spike.reason


@pytest.mark.unit
def test_no_spike_during_warmup():
    g = DivergenceGuard(DivergenceGuardConfig(warmup_steps=10, loss_spike_factor=2.0, check_bio_buffers=False))
    g.check(torch.tensor(1.0), torch.nn.Linear(2, 2), step=0)
    # a big loss before warmup completes must NOT trip the spike guard
    r = g.check(torch.tensor(100.0), torch.nn.Linear(2, 2), step=1)
    assert r.action == GuardAction.OK


# --------------------------------------------------------------------------- #
# 3. Bio-buffer detection on a real tiny model
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_bio_buffer_norms_are_gathered():
    m = make_tiny_synaptic(seed=0, train=True)
    g = DivergenceGuard()
    norms = g.bio_buffer_norms(m)
    assert norms, "expected to find monitored bio buffers/params (camkii/bdnf/w_fast/...)"
    assert all(math.isfinite(v) for v in norms.values())
    # the monitored set should include recognizable bio names
    joined = " ".join(norms).lower()
    assert any(tok in joined for tok in ("camkii", "bdnf", "w_fast", "w_slow", "ema_e"))


@pytest.mark.unit
def test_nonfinite_bio_buffer_is_detected_even_with_finite_loss():
    m = make_tiny_synaptic(seed=0, train=True)
    # Corrupt a CaMKII buffer to NaN; the loss itself stays finite.
    corrupted = False
    for name, buf in m.named_buffers():
        if "camkii" in name.lower():
            with torch.no_grad():
                buf[0] = float("nan")
            corrupted = True
            break
    assert corrupted, "tiny model should have a camkii buffer to corrupt"
    g = DivergenceGuard(DivergenceGuardConfig(on_nonfinite=GuardAction.SKIP))
    r = g.check(torch.tensor(2.0), m, step=1)
    assert r.action == GuardAction.SKIP and r.nonfinite
    assert "non-finite bio buffer" in r.reason


@pytest.mark.unit
def test_bio_norm_explosion_triggers_spike_action():
    m = make_tiny_synaptic(seed=0, train=True)
    g = DivergenceGuard(DivergenceGuardConfig(bio_norm_max=1.0e-6, on_spike=GuardAction.BACKOFF))
    r = g.check(torch.tensor(2.0), m, step=1)  # any real norm exceeds the tiny threshold
    assert r.action == GuardAction.BACKOFF
    assert "explosion" in r.reason


# --------------------------------------------------------------------------- #
# 4. Rollback (opt-in)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_rollback_restores_last_good_snapshot():
    m = make_tiny_synaptic(seed=0, train=True)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    g = DivergenceGuard(DivergenceGuardConfig(enable_rollback=True, snapshot_every=1))
    g.maybe_snapshot(m, opt, step=0)
    assert g.can_rollback()
    # Corrupt a parameter, then roll back.
    p = next(m.parameters())
    good = p.detach().clone()
    with torch.no_grad():
        p.add_(100.0)
    assert not torch.equal(p, good)
    assert g.rollback(m, opt)
    assert torch.equal(next(m.parameters()), good), "rollback must restore the snapshotted weights"


@pytest.mark.unit
def test_rollback_disabled_takes_no_snapshot():
    m = make_tiny_synaptic(seed=0, train=True)
    g = DivergenceGuard(DivergenceGuardConfig(enable_rollback=False))
    g.maybe_snapshot(m, torch.optim.SGD(m.parameters(), lr=0.1), step=0)
    assert not g.can_rollback()
    assert g.rollback(m, None) is False


@pytest.mark.unit
def test_rollback_rejects_optimizer_count_mismatch_before_mutating_model():
    model = torch.nn.Linear(2, 2)
    optimizer_a = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_b = torch.optim.AdamW(model.parameters(), lr=0.01)
    guard = DivergenceGuard(
        DivergenceGuardConfig(enable_rollback=True, snapshot_every=1)
    )
    guard.maybe_snapshot(model, [optimizer_a, optimizer_b], step=0)
    with torch.no_grad():
        model.weight.add_(100.0)
    corrupted_weight = model.weight.detach().clone()

    with pytest.raises(ValueError, match="optimizer count"):
        guard.rollback(model, [optimizer_a])

    torch.testing.assert_close(model.weight, corrupted_weight)


@pytest.mark.unit
def test_checkpoint_state_restores_ema_policy_and_rollback_snapshot():
    model = make_tiny_synaptic(seed=0, train=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original = DivergenceGuard(
        DivergenceGuardConfig(
            enable_rollback=True,
            snapshot_every=1,
            warmup_steps=7,
            on_spike=GuardAction.WARN,
        )
    )
    original.check(torch.tensor(2.0), model, step=0)
    original.maybe_snapshot(model, optimizer, step=3)
    state = original.state_dict()

    restored = DivergenceGuard()
    restored.load_state_dict(state)

    assert restored.cfg.warmup_steps == 7
    assert restored.cfg.on_spike is GuardAction.WARN
    assert restored.state_dict()["loss_ema"] == 2.0
    assert restored.can_rollback()
    good = next(model.parameters()).detach().clone()
    with torch.no_grad():
        next(model.parameters()).add_(100.0)
    assert restored.rollback(model, optimizer)
    torch.testing.assert_close(next(model.parameters()), good)


@pytest.mark.unit
def test_checkpoint_state_rejects_snapshot_step_without_snapshot():
    guard = DivergenceGuard()

    with pytest.raises(ValueError, match="requires a snapshot"):
        guard.load_state_dict(
            {
                "version": 1,
                "config": guard.state_dict()["config"],
                "loss_ema": None,
                "snapshot": None,
                "snapshot_step": 4,
            }
        )


# --------------------------------------------------------------------------- #
# 5. Disabled guard + helpers
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_disabled_guard_is_a_noop():
    g = DivergenceGuard(DivergenceGuardConfig(enabled=False))
    r = g.check(float("nan"), torch.nn.Linear(2, 2))
    assert r.action == GuardAction.OK


@pytest.mark.unit
def test_as_list_helper():
    o = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    assert _as_list(None) == []
    assert _as_list(o) == [o]
    assert _as_list([o, None]) == [o]


@pytest.mark.unit
def test_log_does_not_crash(caplog):
    g = DivergenceGuard()
    r = g.check(float("nan"), make_tiny_synaptic(seed=0, train=True), step=1)
    g.log(r, step=1)  # must not raise; emits a warning for the action


@pytest.mark.unit
def test_builder_returns_guard_with_defaults():
    g = build_divergence_guard(None)
    assert isinstance(g, DivergenceGuard) and g.cfg.enabled
