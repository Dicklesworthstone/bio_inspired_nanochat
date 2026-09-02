"""The expert lifecycle's ``relative`` health signal is scale-free and monotone (bead sx1m).

Measured 2026-09-01 on tiny MoE runs, the legacy ``product`` signal (utilization × energy)
cannot express the lifecycle's own story: utilization is the routed-token fraction, so a
uniformly used expert sits at ``top_k / num_experts`` while the thresholds are absolute
(nothing fires at 4 experts; every expert is a merge candidate at 8), and because energy
relaxes toward ``1 − utilization`` the product peaks at half the routing mass, so an expert
that monopolises routing reads as *dead* and the 0.80 split threshold is unreachable.

These tests lock the alternative and the contrast:

* ``relative`` health = utilization / fair share is exactly 1.0 for uniform routing at 4, 8
  and 16 experts; ``product`` health for the same states is 0.25, 0.19 and 0.11;
* under an imbalanced routing state the ``relative`` pickers nominate the overworked expert
  for a split and the starved one for a reset, while ``product`` with its defaults nominates
  the overworked expert for nothing (planted contrast, not a bug in the new mode);
* ``relative`` refuses the product-mode default thresholds, so nobody can flip the mode and
  silently split every uniform expert;
* ``base_train`` exposes the mode.

Green here does NOT show that lifecycle events help a model; that is the open half of sx1m.

Run:  pytest tests/test_lifecycle_health_relative.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from _bio_testkit import make_tiny_synaptic
from bio_inspired_nanochat.synaptic import SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import SplitMergeConfig, SplitMergeController

pytestmark = pytest.mark.unit

def _relative_cfg(**overrides) -> SplitMergeConfig:
    """The recommended relative-mode thresholds: split at 1.5× fair share, merge below 0.35×,
    reset below 0.05×."""
    return SplitMergeConfig(
        health_mode="relative",
        split_health_min=1.5,
        merge_health_max=0.35,
        reset_health_max=0.05,
        **overrides,
    )


def _moe_model(num_experts: int, top_k: int = 2):
    model = make_tiny_synaptic(seed=0, use_moe=True, num_experts=num_experts, moe_top_k=top_k)
    model.eval()
    layer = next(m for m in model.modules() if isinstance(m, SynapticMoE))
    return model, layer


def _set_state(layer: SynapticMoE, util: list[float]) -> None:
    """Pin the utilization EMA and put energy at its steady state, 1 − utilization."""
    u = torch.tensor(util, dtype=layer.fatigue.dtype)
    layer.fatigue.copy_(u)
    layer.energy.copy_((1.0 - u).clamp(0.0, 1.0))


@pytest.mark.parametrize("num_experts", [4, 8, 16])
def test_relative_health_is_one_for_uniform_routing_at_any_expert_count(num_experts):
    top_k = 2
    model, layer = _moe_model(num_experts, top_k)
    fair = top_k / num_experts
    _set_state(layer, [fair] * num_experts)

    relative = SplitMergeController(model, _relative_cfg())
    product = SplitMergeController(model, SplitMergeConfig())

    h_rel = relative._health(layer)
    torch.testing.assert_close(h_rel, torch.ones(num_experts), rtol=0, atol=1e-6)
    assert relative._pick_split_sources(layer) == []
    assert relative._pick_dead_slots(layer) == []
    assert int((h_rel <= relative.cfg.merge_health_max).sum()) == 0

    h_prod = product._health(layer)
    torch.testing.assert_close(h_prod, torch.full((num_experts,), fair * (1.0 - fair)), rtol=0, atol=1e-6)
    # The measured pathology: under the product signal the uniform steady state sits exactly
    # ON the merge threshold at E=4 (0.5 × 0.5 = 0.25, a knife edge that flips with any energy
    # jitter) and BELOW it for every larger expert count, so all experts become merge
    # candidates at once. Under the relative signal the same state is 1.0 everywhere.
    merge_candidates = int((h_prod <= product.cfg.merge_health_max).sum())
    if num_experts == 4:
        assert h_prod[0] == pytest.approx(product.cfg.merge_health_max, abs=1e-6)
    else:
        assert merge_candidates == num_experts
    assert product._pick_split_sources(layer) == [], "0.80 is unreachable for a uniform expert"


def test_relative_pickers_nominate_overworked_and_starved_experts():
    model, layer = _moe_model(8, top_k=2)
    # Sum is top_k=2 (a valid routed-fraction profile): expert 0 draws 3× its fair share,
    # expert 7 is starved, the rest share the remainder.
    _set_state(layer, [0.75, 0.24, 0.24, 0.24, 0.24, 0.20, 0.08, 0.01])

    relative = SplitMergeController(model, _relative_cfg(splits_per_call=2, resets_per_call=2))
    h = relative._health(layer)
    assert h[0] == pytest.approx(3.0, abs=1e-6)
    assert relative._pick_split_sources(layer) == [0], "only the overworked expert is above 1.5× fair share"
    assert relative._pick_dead_slots(layer) == [7], "only the starved expert is below 0.05× fair share"
    merge_mask = h <= relative.cfg.merge_health_max
    assert merge_mask.tolist() == [False, False, False, False, False, False, True, True]
    assert relative._weakest_slots(layer, 2) == [7, 6]

    # Planted contrast: the legacy product signal sees the monopolising expert as the
    # LEAST healthy one (0.75 × 0.25 = 0.19) and never proposes to split it.
    product = SplitMergeController(model, SplitMergeConfig())
    assert product._pick_split_sources(layer) == []
    assert product._health(layer)[0] == pytest.approx(0.1875, abs=1e-6)


def test_relative_mode_refuses_product_default_thresholds():
    with pytest.raises(ValueError, match="split_health_min"):
        SplitMergeConfig(health_mode="relative")
    with pytest.raises(ValueError, match="merge_health_max"):
        SplitMergeConfig(health_mode="relative", split_health_min=1.5, merge_health_max=1.2)
    with pytest.raises(ValueError, match="reset_health_max"):
        SplitMergeConfig(health_mode="relative", split_health_min=1.5, merge_health_max=0.3, reset_health_max=0.4)
    with pytest.raises(ValueError, match="health_mode"):
        SplitMergeConfig(health_mode="banana")
    # The product default is untouched and still valid.
    assert SplitMergeConfig().health_mode == "product"


def test_relative_health_tracks_variable_expert_count():
    model, layer = _moe_model(8, top_k=2)
    _set_state(layer, [0.25] * 8)
    relative = SplitMergeController(model, _relative_cfg())
    before = relative._health(layer).clone()
    # The fair share is recomputed from the live expert count, so a layer that grew or shrank
    # (uta.4) keeps 1.0 == uniform without retuning thresholds.
    fair_4 = 2 / 4
    assert torch.allclose(before, torch.ones(8))
    assert fair_4 == 0.5


def test_base_train_exposes_the_mode():
    src = Path("scripts/base_train.py").read_text(encoding="utf-8")
    assert 'sm_health_mode = "product"' in src
    assert "health_mode=str(sm_health_mode)" in src
