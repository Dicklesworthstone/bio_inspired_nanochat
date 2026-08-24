"""
Wiring the differentiable synaptic recurrence into the model attention forward (bead hwxb.4.6).

The live attention path always advances presynaptic state causally. With
``differentiable_recurrence=False`` that exact per-query recurrence is detached between queries;
enabling the flag carries gradients through its query blocks so decay kinetics can learn through
time. ``recurrence_block_size>1`` is an explicit training-throughput approximation: queries inside
one block share a state snapshot, while future blocks and future key slots remain causal.

These tests lock the wiring contract:
  1. default-off is byte-identical to the ordinary exact causal path,
  2. the ``differentiable`` flag changes only gradients, never forward values (parity),
  3. a sequence-sized block is observably approximate rather than silently called exact,
  4. through-state decay gradients appear only when differentiable recurrence is on,
  5. all kinetic gradients are finite, including under bf16,
  6. an analytic-vs-numeric gradcheck on a decay parameter through the wired block,
  7. the validator rejects the foot-gun configs.

Run:  pytest tests/test_diff_recurrence_wiring.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bio_testkit import make_tiny_synaptic, random_tokens  # noqa: E402

from bio_inspired_nanochat.ablation_registry import _BY_FIELD, validate_config  # noqa: E402
from bio_inspired_nanochat.synaptic import SynapticConfig  # noqa: E402

pytestmark = pytest.mark.unit

KIN = "h.0.attn.attn.pre.kinetics.theta_"  # prefix of the layer-0 learnable kinetic Parameters


def _model(
    *,
    diff_rec: bool,
    block: int = 8,
    chunk_len: int = 0,
    checkpoint_len: int = 0,
    seq: int = 48,
    seed: int = 3,
    train: bool = False,
    learnable: bool = True,
    metriplectic: bool = False,
):
    syn = SynapticConfig(
        learnable_kinetics=learnable,
        differentiable_recurrence=diff_rec,
        recurrence_block_size=block,
        recurrence_chunk_len=chunk_len,
        recurrence_checkpoint_len=checkpoint_len,
        metriplectic_integrator=metriplectic,
        stochastic_train_frac=0.0 if metriplectic or checkpoint_len > 0 else 0.12,
    )
    return make_tiny_synaptic(seed=seed, train=train, sequence_len=seq, syn_cfg=syn)


def _kinetic_grads(model) -> dict[str, float]:
    out = {}
    for name, p in model.named_parameters():
        if name.startswith(KIN):
            out[name[len(KIN):]] = 0.0 if p.grad is None else p.grad.abs().sum().item()
    return out


def _backward_loss(model, seq=48, seed=5):
    x = random_tokens(2, seq, 97, seed=seed)
    y = random_tokens(2, seq, 97, seed=seed + 1)
    _, loss = model(x, y, train_mode=True)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return loss


# --------------------------------------------------------------------------- #
# 1. DEFAULT-OFF is byte-identical to the ordinary exact causal path.
# --------------------------------------------------------------------------- #
def test_flag_off_is_byte_identical_to_default_causal_path():
    x = random_tokens(2, 48, 97, seed=5)
    # Setting unrelated recurrence-block tuning while differentiability is off must not weaken
    # the exact causal default.
    m_off = _model(diff_rec=False)
    m_legacy = make_tiny_synaptic(seed=3, train=False, sequence_len=48,
                                  syn_cfg=SynapticConfig(learnable_kinetics=True))
    with torch.no_grad():
        a, _ = m_off(x, None, train_mode=False)
        b, _ = m_legacy(x, None, train_mode=False)
    assert torch.equal(a, b), "default-off wiring must be byte-identical to the causal path"


# --------------------------------------------------------------------------- #
# 2. The differentiable flag changes ONLY gradients, not forward values.
#    (Fresh identical models so the in-place ema_e buffer evolves identically.)
# --------------------------------------------------------------------------- #
def test_differentiable_flag_changes_only_gradients_not_values():
    x = random_tokens(2, 48, 97, seed=5)
    m_nograd = _model(diff_rec=True)
    m_grad = _model(diff_rec=True)  # same seed -> identical init + ema_e start
    with torch.no_grad():
        lo_nograd, _ = m_nograd(x, None, train_mode=False)
    lo_grad, _ = m_grad(x, None, train_mode=False)  # grad enabled -> differentiable=True path
    assert torch.equal(lo_nograd, lo_grad), "forward value must not depend on grad-tracking"


def test_enabling_grouped_wiring_changes_the_computation():
    # Sanity that the flag is not a silent no-op: the explicitly grouped recurrence differs from
    # the exact one-query default because queries inside each block share a state snapshot.
    x = random_tokens(2, 48, 97, seed=5)
    m_on = _model(diff_rec=True, block=8)
    m_off = _model(diff_rec=False)
    with torch.no_grad():
        a, _ = m_on(x, None, train_mode=False)
        b, _ = m_off(x, None, train_mode=False)
    assert (a - b).abs().max().item() > 1e-4


# --------------------------------------------------------------------------- #
# 3. A sequence-sized differentiable block is explicitly approximate.
# --------------------------------------------------------------------------- #
def test_block_size_ge_seqlen_differs_from_exact_causal_default():
    x = random_tokens(2, 48, 97, seed=5)
    m_block = _model(diff_rec=True, block=512)  # one block covers the whole sequence
    m_off = _model(diff_rec=False)
    with torch.no_grad():
        a, _ = m_block(x, None, train_mode=False)
        b, _ = m_off(x, None, train_mode=False)
    assert (a - b).abs().max().item() > 1e-4


# --------------------------------------------------------------------------- #
# 4. THE HEADLINE: through-state decay gradients require differentiable recurrence.
# --------------------------------------------------------------------------- #
def test_through_state_decay_kinetics_get_gradient_only_when_wired():
    g_off = _kinetic_grads(_backward_loss_model(diff_rec=False, seq=64))
    g_on = _kinetic_grads(_backward_loss_model(diff_rec=True, block=8, seq=64))
    # The exact detached schedule still trains kinetics from each query's local state snapshot.
    assert g_off["rho_c"] > 0.0
    # Buffer decay only influences later-query state, so it isolates through-state BPTT.
    assert g_off["rho_b"] == 0.0
    assert g_on["rho_b"] > 0.0
    # Wired recurrence also preserves the expected calcium-decay and influx gradients.
    assert g_on["rho_c"] > 0.0, "the wiring must give the calcium decay a nonzero gradient"
    assert g_on["alpha_ca"] > 0.0, "the influx gain must also receive gradient"


def _backward_loss_model(*, diff_rec, block=8, seq=64):
    m = _model(diff_rec=diff_rec, block=block, seq=seq, train=True)
    _backward_loss(m, seq=seq)
    return m


def test_all_kinetic_gradients_are_finite():
    g = _kinetic_grads(_backward_loss_model(diff_rec=True, block=8, seq=64))
    assert g, "expected layer-0 kinetic gradients to be present"
    for name, val in g.items():
        assert val == val and abs(val) < float("inf"), f"kinetic grad {name} not finite: {val}"


# --------------------------------------------------------------------------- #
# 5. bf16 forward+backward through the wired recurrence stays finite.
# --------------------------------------------------------------------------- #
def test_bf16_forward_backward_is_finite():
    m = _model(diff_rec=True, block=8, seq=64, train=True).to(torch.bfloat16)
    x = random_tokens(2, 64, 97, seed=5)
    y = random_tokens(2, 64, 97, seed=6)
    _, loss = m(x, y, train_mode=True)
    assert torch.isfinite(loss), f"bf16 loss must be finite, got {loss}"
    m.zero_grad(set_to_none=True)
    loss.backward()
    grads = _kinetic_grads(m)
    assert grads, "expected kinetic gradients under bf16"
    for name, val in grads.items():
        assert val == val and abs(val) < float("inf"), f"bf16 kinetic grad {name} not finite: {val}"


# --------------------------------------------------------------------------- #
# 6. Analytic-vs-numeric gradcheck on the calcium decay through the wired block.
#    (Central finite difference on theta_rho_c; the through-model autograd grad must match.)
# --------------------------------------------------------------------------- #
def test_calcium_decay_gradient_matches_finite_difference():
    seq = 64

    def loss_at(theta_delta: float) -> float:
        m = _model(diff_rec=True, block=8, seq=seq, train=False)  # train=False -> deterministic
        kin = m.h[0].attn.attn.pre.kinetics
        with torch.no_grad():
            kin.theta_rho_c.add_(theta_delta)
        x = random_tokens(1, seq, 97, seed=5)
        y = random_tokens(1, seq, 97, seed=6)
        _, loss = m(x, y, train_mode=False)
        return float(loss.detach())

    # Analytic gradient through the model.
    m = _model(diff_rec=True, block=8, seq=seq, train=False)
    kin = m.h[0].attn.attn.pre.kinetics
    x = random_tokens(1, seq, 97, seed=5)
    y = random_tokens(1, seq, 97, seed=6)
    _, loss = m(x, y, train_mode=False)
    m.zero_grad(set_to_none=True)
    loss.backward()
    analytic = float(kin.theta_rho_c.grad)

    eps = 1e-3
    numeric = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
    assert abs(analytic) > 0.0, "the calcium decay must have a nonzero gradient through the model"
    # Loose tolerance: tiny CPU fp32 model, finite-difference noise; we are checking sign+magnitude.
    assert abs(analytic - numeric) <= 0.05 * max(1.0, abs(numeric)) + 1e-3, (
        f"analytic {analytic} vs numeric {numeric} disagree beyond tolerance"
    )


# --------------------------------------------------------------------------- #
# 7. The live full-model path preserves values, gradients, and runtime effects.
# --------------------------------------------------------------------------- #
def test_live_model_checkpoint_matches_eager_forward_backward_and_runtime_buffers():
    eager = _model(diff_rec=True, block=4, seq=16, train=True, metriplectic=True)
    replayed = _model(
        diff_rec=True,
        block=4,
        checkpoint_len=2,
        seq=16,
        train=True,
        metriplectic=True,
    )
    x = random_tokens(1, 16, 97, seed=11)
    y = random_tokens(1, 16, 97, seed=12)

    eager_logits, eager_loss = eager(x, y, train_mode=True)
    replayed_logits, replayed_loss = replayed(x, y, train_mode=True)
    assert torch.equal(eager_logits, replayed_logits)
    assert torch.equal(eager_loss, replayed_loss)

    runtime_after_forward = []
    for block in replayed.h:
        pre = block.attn.attn.pre
        runtime_after_forward.append(
            tuple(
                tensor.clone()
                for tensor in (
                    pre.ema_e,
                    pre.metriplectic_steps,
                    pre.metriplectic_fallbacks,
                    pre.metriplectic_last_energy_drift,
                    pre.metriplectic_last_entropy_production,
                    pre.metriplectic_last_free_energy_delta,
                )
            )
        )

    eager_loss.backward()
    replayed_loss.backward()
    for (eager_name, eager_param), (replayed_name, replayed_param) in zip(
        eager.named_parameters(), replayed.named_parameters()
    ):
        assert eager_name == replayed_name
        if eager_param.grad is None or replayed_param.grad is None:
            assert eager_param.grad is None and replayed_param.grad is None
        else:
            assert torch.equal(eager_param.grad, replayed_param.grad), eager_name

    for block, forward_runtime in zip(replayed.h, runtime_after_forward):
        pre = block.attn.attn.pre
        runtime_after_backward = (
            pre.ema_e,
            pre.metriplectic_steps,
            pre.metriplectic_fallbacks,
            pre.metriplectic_last_energy_drift,
            pre.metriplectic_last_entropy_production,
            pre.metriplectic_last_free_energy_delta,
        )
        for before, after in zip(forward_runtime, runtime_after_backward):
            assert torch.equal(before, after)


def test_live_model_checkpoint_runs_under_torch_compile():
    model = _model(
        diff_rec=True,
        block=2,
        checkpoint_len=2,
        seq=8,
        train=True,
        metriplectic=True,
    )
    try:
        compiled = torch.compile(model, backend="eager", dynamic=False)
    except RuntimeError as exc:
        if "torch.compile is not supported on Python 3.14+" in str(exc):
            pytest.skip("installed PyTorch does not support torch.compile on Python 3.14")
        raise
    x = random_tokens(1, 8, 97, seed=13)
    y = random_tokens(1, 8, 97, seed=14)
    logits, loss = compiled(x, y, train_mode=True)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


# --------------------------------------------------------------------------- #
# 8. The validator catches the foot-gun configs.
# --------------------------------------------------------------------------- #
def test_validator_flags_differentiable_recurrence_without_kinetics():
    errors, _ = validate_config(
        SynapticConfig(differentiable_recurrence=True, learnable_kinetics=False)
    )
    assert any("differentiable_recurrence" in e and "learnable_kinetics" in e for e in errors)


def test_validator_range_checks_block_and_chunk():
    assert validate_config(
        SynapticConfig(learnable_kinetics=True, differentiable_recurrence=True,
                       recurrence_block_size=0)
    )[0]
    assert validate_config(
        SynapticConfig(learnable_kinetics=True, differentiable_recurrence=True,
                       recurrence_chunk_len=-1)
    )[0]
    assert validate_config(
        SynapticConfig(learnable_kinetics=True, differentiable_recurrence=True,
                       recurrence_checkpoint_len=-1)
    )[0]


def test_validator_rejects_checkpoint_without_full_differentiable_bptt():
    errors, _ = validate_config(SynapticConfig(recurrence_checkpoint_len=2))
    assert any("differentiable_recurrence" in error for error in errors)
    assert any("metriplectic_integrator" in error for error in errors)
    assert any("stochastic_train_frac=0" in error for error in errors)

    errors, _ = validate_config(
        SynapticConfig(
            learnable_kinetics=True,
            differentiable_recurrence=True,
            recurrence_chunk_len=2,
            recurrence_checkpoint_len=2,
        )
    )
    assert any("mutually exclusive" in error for error in errors)

    errors, _ = validate_config(
        SynapticConfig(
            learnable_kinetics=True,
            differentiable_recurrence=True,
            metriplectic_integrator=True,
            recurrence_checkpoint_len=2,
            stochastic_train_frac=0.0,
            use_flex_attention=True,
        )
    )
    assert any("standard attention path" in error for error in errors)


def test_default_config_does_not_engage_the_recurrence():
    cfg = SynapticConfig()
    assert cfg.differentiable_recurrence is False
    assert cfg.recurrence_checkpoint_len == 0
    assert _BY_FIELD["recurrence_checkpoint_len"].requires == (
        "differentiable_recurrence",
        "metriplectic_integrator",
    )
    errors, warnings = validate_config(cfg)
    assert errors == [] and warnings == []
