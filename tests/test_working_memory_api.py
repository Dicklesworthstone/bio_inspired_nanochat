"""Unit and E2E tests for the working-memory API (bead ``r00r.9``)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.run_logging import RunLogger, read_run_events
from bio_inspired_nanochat.synaptic import SynapticLinear
from bio_inspired_nanochat.working_memory_api import (
    WORKING_MEMORY_SCHEMA,
    WorkingMemoryPolicy,
    WorkingMemoryScratchpad,
    WorkingMemoryValidationError,
)
from scripts.e2e.working_memory_api_demo import run_demo


def _make_model() -> GPTSynaptic:
    cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    model = GPTSynaptic(cfg)
    model.eval()
    return model


def _first_site(model: GPTSynaptic) -> SynapticLinear:
    return next(module for module in model.modules() if isinstance(module, SynapticLinear))


def test_schema_includes_stable_sites_and_cached_calcium() -> None:
    model = _make_model()
    scratchpad = WorkingMemoryScratchpad(model)
    cache = SimpleNamespace(
        presyn_state=[{"C": torch.tensor([[[0.1, 0.3], [0.2, 0.4]]])}]
    )

    snapshot = scratchpad.read_scratchpad(cache)

    assert snapshot["schema"] == WORKING_MEMORY_SCHEMA
    assert snapshot["num_sites"] == len(snapshot["sites"]) > 0
    assert snapshot["sites"][0]["site_index"] == 0
    assert snapshot["sites"][0]["module"]
    assert snapshot["sites"][0]["shape"] == [16, 64]
    assert snapshot["presynaptic"] == [
        {
            "layer_index": 0,
            "available": True,
            "shape": [1, 2, 2],
            "finite": True,
            "mean": pytest.approx(0.25),
            "minimum": pytest.approx(0.1),
            "maximum": pytest.approx(0.4),
        }
    ]
    json.dumps(snapshot)


def test_nonfinite_snapshot_remains_strict_json() -> None:
    model = _make_model()
    scratchpad = WorkingMemoryScratchpad(model)
    site = _first_site(model)
    assert site.w_fast is not None
    with torch.no_grad():
        site.w_fast[0, 0] = float("nan")

    snapshot = scratchpad.read_scratchpad({"C": torch.tensor([[[float("inf")]]])})

    assert not snapshot["sites"][0]["finite"]
    assert snapshot["sites"][0]["fast_weight_norm"] is None
    assert not snapshot["presynaptic"][0]["finite"]
    assert snapshot["presynaptic"][0]["mean"] is None
    json.dumps(snapshot, allow_nan=False)


def test_clear_resets_all_volatile_state_and_preserves_slow_weights() -> None:
    model = _make_model()
    scratchpad = WorkingMemoryScratchpad(model)
    site = _first_site(model)
    assert site.w_fast is not None
    assert site.u_buf is not None
    assert site.v_buf is not None
    assert site.post is not None
    with torch.no_grad():
        slow_before = site.w_slow.clone()
        site.w_fast.fill_(0.5)
        site.u_buf.fill_(0.6)
        site.v_buf.fill_(0.7)
        site.post.fast.fill_(0.8)
        site.post.camkii.fill_(0.9)
        site.post.pp1.fill_(0.1)
        site.post.bdnf.fill_(0.4)

    receipt = scratchpad.clear_scratchpad(site_index=0)

    assert receipt["schema"] == WORKING_MEMORY_SCHEMA
    assert receipt["cleared_sites"] == 1
    assert torch.count_nonzero(site.w_fast) == 0
    assert torch.count_nonzero(site.u_buf) == 0
    assert torch.count_nonzero(site.v_buf) == 0
    assert torch.count_nonzero(site.post.fast) == 0
    assert torch.count_nonzero(site.post.camkii) == 0
    assert torch.all(site.post.pp1 == 0.5)
    assert torch.count_nonzero(site.post.bdnf) == 0
    assert torch.equal(site.w_slow, slow_before)


def test_write_association_has_exact_rank_one_effect_and_bounded_norm() -> None:
    model = _make_model()
    scratchpad = WorkingMemoryScratchpad(
        model,
        policy=WorkingMemoryPolicy(max_delta_norm=0.25, max_norm_growth=0.25),
    )

    site = _first_site(model)
    assert site.w_fast is not None
    in_dim, out_dim = site.w_fast.shape
    key = torch.ones(in_dim)
    value = torch.linspace(-1.0, 1.0, out_dim)
    query = torch.randn(in_dim)
    before = site.w_fast.detach().clone()

    receipt = scratchpad.write_association(0, key, value, expected_module="h.0.mlp.mlp.fc")

    actual_delta = query @ (site.w_fast.detach() - before)
    predicted_delta = (
        float(receipt["effective_scale"]) * torch.dot(query, key) * value
    )
    assert bool(receipt["clipped"])
    assert float(receipt["applied_delta_norm"]) <= 0.250001
    assert torch.allclose(actual_delta, predicted_delta, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("key", "value", "scale", "match"),
    [
        (torch.full((16,), float("nan")), torch.ones(64), 1.0, "finite"),
        (torch.ones(15), torch.ones(64), 1.0, "dimensions"),
        (torch.ones(16), torch.ones(64), 2.0, "scale"),
        (torch.ones(1, 16), torch.ones(64), 1.0, "one-dimensional"),
    ],
)
def test_invalid_write_is_rejected_atomically(
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    match: str,
) -> None:
    model = _make_model()
    scratchpad = WorkingMemoryScratchpad(model)
    site = _first_site(model)
    assert site.w_fast is not None
    before = site.w_fast.detach().clone()

    with pytest.raises(WorkingMemoryValidationError, match=match):
        scratchpad.write_association(0, key, value, scale=scale)

    assert torch.equal(site.w_fast, before)


def test_write_rejects_training_mode_and_deferred_plasticity() -> None:
    model = _make_model()
    scratchpad = WorkingMemoryScratchpad(model)
    site = _first_site(model)
    assert site.w_fast is not None
    key = torch.ones(site.w_fast.shape[0])
    value = torch.ones(site.w_fast.shape[1])

    model.train()
    with pytest.raises(WorkingMemoryValidationError, match="model.eval"):
        scratchpad.write_association(0, key, value)
    model.eval()
    site._plasticity_pending = True
    with pytest.raises(WorkingMemoryValidationError, match="deferred plasticity"):
        scratchpad.write_association(0, key, value)


def test_multi_site_clear_validates_atomically() -> None:
    model = _make_model()
    scratchpad = WorkingMemoryScratchpad(model)
    sites = [module for module in model.modules() if isinstance(module, SynapticLinear)]
    assert len(sites) >= 2
    assert sites[0].w_fast is not None
    with torch.no_grad():
        sites[0].w_fast.fill_(0.25)
        before = sites[0].w_fast.clone()
    sites[1]._plasticity_pending = True

    with pytest.raises(WorkingMemoryValidationError, match="deferred plasticity"):
        scratchpad.clear_scratchpad()

    assert torch.equal(sites[0].w_fast, before)


def test_structured_jsonl_logs_read_write_and_clear(tmp_path: Path) -> None:
    model = _make_model()
    site = _first_site(model)
    assert site.w_fast is not None
    with RunLogger(tmp_path, name="working-memory-test", console=False) as logger:
        scratchpad = WorkingMemoryScratchpad(model, logger=logger)
        scratchpad.read_scratchpad()
        scratchpad.write_association(
            0,
            torch.zeros(site.w_fast.shape[0]),
            torch.zeros(site.w_fast.shape[1]),
        )
        scratchpad.clear_scratchpad(0)

    events = read_run_events(tmp_path)
    names = [event["event"] for event in events]
    assert "working_memory_read" in names
    assert "working_memory_write" in names
    assert "working_memory_clear" in names
    assert all(
        event["schema"] == WORKING_MEMORY_SCHEMA
        for event in events
        if event["event"].startswith("working_memory_")
    )


def test_e2e_injected_fact_changes_next_token_as_predicted(tmp_path: Path) -> None:
    result = run_demo(tmp_path / "demo", seed=19)

    assert result.passed
    assert result.injected_token == result.target_token
    assert result.baseline_token != result.target_token
    assert result.predicted_margin == pytest.approx(result.observed_margin, abs=1e-4)
    events = read_run_events(tmp_path / "demo")
    assert events[-2]["event"] == "working_memory_demo_result"
    assert bool(events[-2]["passed"])
