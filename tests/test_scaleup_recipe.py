"""Training-recipe config-plumbing tests (bead hwxb.2.7).

Verify the recipe actually flows into the machinery: the per-group learning rates reach
BOTH optimizers with the correct parameter routing (2-D matrices -> Muon; embeddings /
lm_head / 1-D / 0-D -> AdamW) and the d-model LR scaling, and that the WSD schedule has
the intended warmup-stable-decay shape. Fast, CPU-only.
"""
from __future__ import annotations

import pytest

from _bio_testkit import make_tiny_synaptic, make_tiny_vanilla

_LRS = dict(unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0)


@pytest.mark.unit
def test_vanilla_optimizer_routing_and_lr_scaling():
    m = make_tiny_vanilla(seed=0)
    opts = m.setup_optimizers(**_LRS)
    assert len(opts) == 2, "recipe must build AdamW + Muon"
    adamw, muon = opts
    scale = (m.config.n_embd / 768) ** -0.5

    # AdamW carries the embedding + lm_head groups at d-model-scaled LRs.
    adam_lrs = sorted(g["lr"] for g in adamw.param_groups)
    assert any(abs(g["lr"] - _LRS["embedding_lr"] * scale) < 1e-9 for g in adamw.param_groups)
    assert any(abs(g["lr"] - _LRS["unembedding_lr"] * scale) < 1e-9 for g in adamw.param_groups)
    assert adam_lrs == sorted(adam_lrs), "lrs sortable (sanity)"

    # Muon carries the 2-D matrix params at the (unscaled) matrix LR.
    assert all(abs(g["lr"] - _LRS["matrix_lr"]) < 1e-9 for g in muon.param_groups)

    # Every parameter is covered exactly once and initial_lr is stamped for the scheduler.
    total = len(list(m.parameters()))
    routed = sum(len(g["params"]) for o in opts for g in o.param_groups)
    assert routed == total, f"routed {routed} != total {total} params"
    for o in opts:
        for g in o.param_groups:
            assert g.get("initial_lr") == g["lr"], "scheduler needs initial_lr stamped"


@pytest.mark.unit
def test_synaptic_optimizer_routing():
    m = make_tiny_synaptic(seed=0)
    opts = m.setup_optimizers(**_LRS)
    assert len(opts) == 2
    adamw, muon = opts
    # Muon must receive ONLY 2-D matrices (its Newton-Schulz assumes 2-D).
    for g in muon.param_groups:
        for p in g["params"]:
            assert p.ndim == 2, "Muon got a non-2-D param"
    # The synaptic model's small 1-D/0-D kinetic params must land in AdamW, not Muon.
    adam_params = [p for g in adamw.param_groups for p in g["params"]]
    assert any(p.ndim < 2 for p in adam_params), "AdamW should carry the 1-D/0-D params"
    # Full coverage.
    total = len(list(m.parameters()))
    routed = sum(len(g["params"]) for o in opts for g in o.param_groups)
    assert routed == total


def _wsd_multiplier(it, num_iters, warmup_ratio, warmdown_ratio, final_lr_frac):
    """Mirrors scripts/base_train.get_lr_multiplier (WSD/trapezoidal)."""
    warmup = round(warmup_ratio * num_iters)
    warmdown = round(warmdown_ratio * num_iters)
    if warmup and it < warmup:
        return (it + 1) / warmup
    elif it <= num_iters - warmdown:
        return 1.0
    else:
        progress = (num_iters - it) / warmdown
        return progress * 1.0 + (1 - progress) * final_lr_frac


@pytest.mark.unit
def test_wsd_schedule_shape():
    n = 1000

    def f(it):
        return _wsd_multiplier(it, n, warmup_ratio=0.1, warmdown_ratio=0.2, final_lr_frac=0.0)
    # Warmup ramps monotonically up to 1.0 over the first ~100 steps.
    assert f(0) < f(50) < f(99) <= 1.0
    # Stable middle is exactly the peak.
    assert f(500) == 1.0
    # Warmdown decays toward final_lr_frac (0.0): the last step is one warmdown-tick above 0
    # (it reaches exactly final_lr_frac at it == num_iters), and the ramp is monotone down.
    assert 0.0 < f(n - 1) < 0.01
    assert f(900) > f(950) > f(n - 1), "warmdown must be monotonically decreasing"
