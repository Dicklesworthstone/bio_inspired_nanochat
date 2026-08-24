"""Astrocytic homeostasis and live MoE wiring (bead hy8.4)."""

from __future__ import annotations

import math

import pytest
import torch

from bio_inspired_nanochat.ablation_registry import is_mechanism_on, validate_config
from bio_inspired_nanochat.glial_homeostasis import GlialHomeostasis
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE

pytestmark = pytest.mark.unit


def _moe(*, glial: bool, seed: int = 0) -> SynapticMoE:
    torch.manual_seed(seed)
    cfg = SynapticConfig(
        enable_hebbian=False,
        enable_metabolism=False,
        xi_dim=0,
        router_contrastive_lr=0.0,
        router_contrastive_push=0.0,
        glial_homeostasis=glial,
        glial_group_size=2,
        glial_ema_rate=0.2,
        glial_feedback_rate=0.15,
        glial_energy_weight=0.25,
        glial_bias_cap=3.0,
    )
    moe = SynapticMoE(
        n_embd=4,
        num_experts=4,
        top_k=1,
        hidden_mult=1,
        cfg=cfg,
        dropout=0.0,
    )
    moe.eval()
    with torch.no_grad():
        # Remove the input-dependent embedding term so controlled router gaps
        # below are the only source of asymmetry.
        moe.router_probe.weight.zero_()
    return moe


def _routing_entropy(counts: torch.Tensor) -> float:
    probs = counts.to(torch.float64) / counts.sum()
    nz = probs > 0
    return float(-(probs[nz] * probs[nz].log()).sum() / math.log(counts.numel()))


def test_group_pooling_emits_shared_energy_feedback() -> None:
    glia = GlialHomeostasis(
        4,
        group_size=2,
        ema_rate=1.0,
        feedback_rate=1.0,
        energy_weight=1.0,
        bias_cap=2.0,
    )
    glia.observe(torch.ones(4), torch.tensor([0.0, 0.0, 1.0, 1.0]))

    assert torch.equal(glia.group_energy_ema, torch.tensor([0.0, 1.0]))
    assert glia.routing_bias[0] == glia.routing_bias[1]
    assert glia.routing_bias[2] == glia.routing_bias[3]
    assert glia.routing_bias[0] < 0.0 < glia.routing_bias[2]
    assert glia.routing_bias.sum().item() == pytest.approx(0.0, abs=1e-7)


def test_feedback_is_bounded_zero_sum_and_resettable() -> None:
    glia = GlialHomeostasis(
        8,
        group_size=3,
        ema_rate=0.5,
        feedback_rate=0.5,
        energy_weight=0.5,
        bias_cap=0.75,
    )
    for _ in range(100):
        glia.observe(
            torch.tensor([64.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.linspace(0.1, 1.0, 8),
        )

    assert glia.routing_bias.abs().max().item() <= 0.75 + 1e-7
    assert glia.routing_bias.sum().item() == pytest.approx(0.0, abs=1e-6)
    assert glia.steps.item() == 100

    glia.reset_()
    torch.testing.assert_close(glia.activity_ema, torch.full((8,), 1.0 / 8.0))
    assert torch.count_nonzero(glia.routing_bias).item() == 0
    assert glia.steps.item() == 0


def test_glia_recovers_a_reproducibly_collapsed_router() -> None:
    baseline = _moe(glial=False, seed=11)
    protected = _moe(glial=True, seed=11)
    with torch.no_grad():
        for moe in (baseline, protected):
            moe.router.weight.zero_()
            moe.router.weight[0, 0] = 1.5

    x = torch.zeros((1, 32, 4))
    x[..., 0] = 1.0
    recent_baseline = torch.zeros(4)
    recent_protected = torch.zeros(4)
    for step in range(120):
        baseline(x, update_mem=True)
        protected(x, update_mem=True)
        if step >= 80:
            recent_baseline.add_(
                torch.bincount(baseline.last_ctx["indices"].reshape(-1), minlength=4)
            )
            recent_protected.add_(
                torch.bincount(protected.last_ctx["indices"].reshape(-1), minlength=4)
            )

    assert recent_baseline.argmax().item() == 0
    assert recent_baseline.max().item() == recent_baseline.sum().item()
    assert _routing_entropy(recent_protected) >= 0.85
    assert recent_protected.max().item() / recent_protected.sum().item() <= 0.4
    assert protected.glial is not None
    assert protected.glial.routing_bias[0] < 0.0


def test_balanced_training_forward_and_gradients_are_unchanged() -> None:
    baseline = _moe(glial=False, seed=23)
    protected = _moe(glial=True, seed=23)
    protected.load_state_dict(baseline.state_dict(), strict=True)

    with torch.no_grad():
        for moe in (baseline, protected):
            moe.router.weight.copy_(3.0 * torch.eye(4))

    x_base = torch.eye(4).repeat(2, 1).reshape(1, 8, 4).requires_grad_(True)
    x_glia = x_base.detach().clone().requires_grad_(True)
    out_base, aux_base = baseline(x_base, update_mem=True)
    out_glia, aux_glia = protected(x_glia, update_mem=True)

    torch.testing.assert_close(out_glia, out_base, atol=0.0, rtol=0.0)
    torch.testing.assert_close(aux_glia, aux_base, atol=0.0, rtol=0.0)
    (out_base.square().mean() + aux_base).backward()
    (out_glia.square().mean() + aux_glia).backward()
    torch.testing.assert_close(x_glia.grad, x_base.grad, atol=0.0, rtol=0.0)
    for (name_base, param_base), (name_glia, param_glia) in zip(
        baseline.named_parameters(), protected.named_parameters(), strict=True
    ):
        assert name_base == name_glia
        if param_base.grad is None or param_glia.grad is None:
            assert param_base.grad is None and param_glia.grad is None
        else:
            torch.testing.assert_close(param_glia.grad, param_base.grad, atol=0.0, rtol=0.0)

    assert protected.glial is not None
    assert torch.count_nonzero(protected.glial.routing_bias).item() == 0
    assert protected.glial.steps.item() == 1


def test_update_mem_false_freezes_glial_state() -> None:
    moe = _moe(glial=True)
    assert moe.glial is not None
    before = moe.glial.diagnostics()
    moe(torch.randn(2, 5, 4), update_mem=False)
    after = moe.glial.diagnostics()
    for name in before:
        assert torch.equal(after[name], before[name]), name


def test_model_weight_initialization_resets_glial_state() -> None:
    syn_cfg = SynapticConfig(
        enable_hebbian=False,
        glial_homeostasis=True,
        glial_group_size=2,
    )
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            use_moe=True,
            num_experts=4,
            moe_top_k=1,
            moe_hidden_mult=1,
            syn_cfg=syn_cfg,
        )
    )
    moe = model.h[0].mlp
    assert isinstance(moe, SynapticMoE)
    assert moe.glial is not None
    moe.glial.observe(torch.tensor([16.0, 0.0, 0.0, 0.0]), torch.ones(4))
    assert torch.count_nonzero(moe.glial.routing_bias).item() > 0

    model.init_weights()

    assert torch.count_nonzero(moe.glial.routing_bias).item() == 0
    assert moe.glial.steps.item() == 0


def test_toggle_and_glial_ranges_are_validated() -> None:
    default = SynapticConfig()
    assert not default.glial_homeostasis
    assert not is_mechanism_on(default, "glial_homeostasis")
    assert is_mechanism_on(
        SynapticConfig(glial_homeostasis=True), "glial_homeostasis"
    )

    invalid = (
        SynapticConfig(glial_homeostasis=1),  # type: ignore[arg-type]
        SynapticConfig(glial_group_size=0),
        SynapticConfig(glial_ema_rate=0.0),
        SynapticConfig(glial_feedback_rate=1.1),
        SynapticConfig(glial_energy_weight=-0.1),
        SynapticConfig(glial_bias_cap=0.0),
    )
    for cfg in invalid:
        assert validate_config(cfg)[0]


def test_glia_enabled_model_warm_starts_from_pre_glia_checkpoint() -> None:
    old_model = _moe(glial=False, seed=5)
    upgraded = _moe(glial=True, seed=99)
    result = upgraded.load_state_dict(old_model.state_dict(), strict=True)

    assert tuple(result.missing_keys) == ()
    assert tuple(result.unexpected_keys) == ()
    assert upgraded.glial is not None
    assert torch.count_nonzero(upgraded.glial.routing_bias).item() == 0

    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(
        upgraded(x, update_mem=False)[0],
        old_model(x, update_mem=False)[0],
        atol=0.0,
        rtol=0.0,
    )
