"""The lifecycle's ``credit`` health signal (bead uta.9).

Measured 2026-09-02 (results/structural_pair_pilot_2026-09-02*.json): every utilization-based health
signal (product, relative) stays within ±15% of the fair share during healthy training, with or
without the balance loss, so no split/merge/reset ever fires. ``health_mode="credit"`` reads the
per-expert gradient credit NeuroScore publishes (``last_credit``: loss contribution relative to the
layer mean, 1.0 = an average expert), which is loss-derived and per step.

Locked here:

* planted contrast: an expert with 3x the mean credit is the only split candidate, an expert with no
  credit the only reset candidate, and a uniform population yields no candidates at all;
* the mode fails loudly when NeuroScore has not published a signal, or published only the routing
  proxy (which is utilization again);
* the config refuses the product-mode thresholds, like ``relative``;
* NeuroScore really publishes ``last_credit`` with source ``gradient`` once a training backward has
  run, and its mean absolute value is 1.

Green here shows the signal can fire; whether firing helps is the pilot and the NAS re-run.

Run:  pytest tests/test_lifecycle_health_credit.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bio_testkit import make_tiny_synaptic, random_tokens  # noqa: E402

from bio_inspired_nanochat.neuroscore import NeuroScore, NeuroScoreConfig  # noqa: E402
from bio_inspired_nanochat.synaptic import SynapticMoE  # noqa: E402
from bio_inspired_nanochat.synaptic_splitmerge import SplitMergeConfig, SplitMergeController  # noqa: E402

pytestmark = pytest.mark.unit


def _credit_cfg(**overrides) -> SplitMergeConfig:
    return SplitMergeConfig(health_mode="credit", split_health_min=1.5, merge_health_max=0.35, reset_health_max=0.05, **overrides)


def _moe_model(num_experts: int = 8, top_k: int = 2):
    model = make_tiny_synaptic(seed=0, use_moe=True, num_experts=num_experts, moe_top_k=top_k)
    model.eval()
    layer = next(m for m in model.modules() if isinstance(m, SynapticMoE))
    return model, layer


def _publish(layer: SynapticMoE, credit: list[float], source: str = "gradient") -> None:
    c = torch.tensor(credit, dtype=torch.float32)
    object.__setattr__(layer, "last_credit", c / (c.abs().mean() + 1e-12))
    object.__setattr__(layer, "last_credit_source", source)


def test_planted_contrast_split_and_reset_candidates():
    model, layer = _moe_model()
    # Expert 0 carries 3x the average loss contribution, expert 7 none, the rest average.
    _publish(layer, [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    ctl = SplitMergeController(model, _credit_cfg(splits_per_call=2, resets_per_call=2))
    health = ctl._health(layer)
    assert health[0] > 1.5 and health[7] == pytest.approx(0.0, abs=1e-6)
    assert ctl._pick_split_sources(layer) == [0]
    assert ctl._pick_dead_slots(layer) == [7]


def test_uniform_credit_fires_nothing_and_harmful_credit_is_clamped_to_zero():
    model, layer = _moe_model()
    _publish(layer, [1.0] * 8)
    ctl = SplitMergeController(model, _credit_cfg())
    torch.testing.assert_close(ctl._health(layer), torch.ones(8), rtol=0, atol=1e-6)
    assert ctl._pick_split_sources(layer) == [] and ctl._pick_dead_slots(layer) == []
    _publish(layer, [1.0] * 7 + [-2.0])  # a harmful expert (removing it would lower the loss)
    assert float(ctl._health(layer)[7]) == 0.0
    assert ctl._pick_dead_slots(layer) == [7]


def test_missing_or_proxy_credit_fails_loudly_instead_of_silently_no_oping():
    model, layer = _moe_model()
    ctl = SplitMergeController(model, _credit_cfg())
    with pytest.raises(RuntimeError, match="needs NeuroScore stepped"):
        ctl._health(layer)
    _publish(layer, [1.0] * 8, source="proxy")
    with pytest.raises(RuntimeError, match="needs gradient credit"):
        ctl._health(layer)


def test_credit_mode_refuses_product_thresholds():
    with pytest.raises(ValueError, match="split_health_min"):
        SplitMergeConfig(health_mode="credit")
    with pytest.raises(ValueError, match="merge_health_max"):
        SplitMergeConfig(health_mode="credit", split_health_min=1.5, merge_health_max=1.2)
    with pytest.raises(ValueError, match="health_mode"):
        SplitMergeConfig(health_mode="utilization")


def test_neuroscore_publishes_gradient_credit_after_a_training_backward():
    model = make_tiny_synaptic(seed=1, train=True, use_moe=True, num_experts=4, moe_top_k=2)
    layer = next(m for m in model.modules() if isinstance(m, SynapticMoE))
    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = random_tokens(2, 16, model.config.vocab_size, seed=3)
    y = random_tokens(2, 16, model.config.vocab_size, seed=4)
    sources = []
    for step in range(3):
        _, loss = model(x, y, train_mode=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        score.step(model, loss.detach(), step)
        sources.append(getattr(layer, "last_credit_source", None))
    credit = layer.last_credit
    assert tuple(credit.shape) == (4,)
    assert sources[-1] == "gradient", sources  # hooks install on the first step; gradient credit from the next backward on
    assert float(credit.abs().mean()) == pytest.approx(1.0, abs=1e-5)
    ctl = SplitMergeController(model, _credit_cfg())
    assert tuple(ctl._health(layer).shape) == (4,)
