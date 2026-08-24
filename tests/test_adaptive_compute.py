"""ATP-gated adaptive depth, expert-k, and MC-sample levers (``r00r.3.2``)."""

from __future__ import annotations

import pytest

from _bio_testkit import make_tiny_synaptic, make_tiny_vanilla, random_tokens
from bio_inspired_nanochat.adaptive_compute import (
    AdaptiveComputeConfig,
    AdaptiveComputeController,
    QualityFloorConfig,
    InsufficientATPError,
    adaptive_forward,
    adaptive_mc_predict,
    quality_guarded_predict,
    temporary_expert_top_k,
)
from bio_inspired_nanochat.deliberation import ATPBudget, DifficultyRouter, DifficultyRouterConfig
from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.torch_imports import torch


def _controller(*, enabled: bool = True) -> AdaptiveComputeController:
    return AdaptiveComputeController(
        AdaptiveComputeConfig(
            enabled=enabled,
            min_depth_layers=1,
            min_experts=1,
            min_mc_samples=1,
            max_mc_samples=4,
        )
    )


@pytest.mark.unit
def test_easy_tokens_receive_less_compute_than_hard_tokens():
    controller = _controller()
    easy_budget = ATPBudget(total_atp=100)
    hard_budget = ATPBudget(total_atp=100)
    easy_plans = [
        controller.plan(
            torch.tensor([margin, 0.0, 0.0, 0.0]),
            easy_budget,
            token_index=token_index,
            max_depth_layers=4,
            max_experts=4,
        )
        for token_index, margin in enumerate((8.0, 9.0, 10.0))
    ]
    hard_plans = [
        controller.plan(
            torch.zeros(4),
            hard_budget,
            token_index=token_index,
            max_depth_layers=4,
            max_experts=4,
        )
        for token_index in range(3)
    ]

    easy_mean = sum(plan.compute_units for plan in easy_plans) / len(easy_plans)
    hard_mean = sum(plan.compute_units for plan in hard_plans) / len(hard_plans)
    assert easy_mean < hard_mean
    assert all(plan.compute_units == plan.maximum_compute_units for plan in hard_plans)
    assert easy_budget.spent_atp < hard_budget.spent_atp
    assert hard_budget.spent_atp + hard_budget.remaining_atp == hard_budget.total_atp
    assert [record.action for record in hard_plans[0].debit_records] == [
        "depth_layer",
        "expert",
        "mc_sample",
        "depth_layer",
        "expert",
        "mc_sample",
    ]


@pytest.mark.unit
def test_free_energy_increases_compute_at_matched_entropy():
    router = DifficultyRouter(DifficultyRouterConfig(entropy_weight=0.5, free_energy_scale=1.0))
    controller = AdaptiveComputeController(_controller().config, router=router)
    logits = torch.tensor([2.0, 0.0, -1.0])
    low = controller.plan(
        logits,
        ATPBudget(100),
        token_index=0,
        max_depth_layers=8,
        max_experts=4,
        free_energy_value=0.0,
    )
    high = controller.plan(
        logits,
        ATPBudget(100),
        token_index=0,
        max_depth_layers=8,
        max_experts=4,
        free_energy_value=20.0,
    )
    assert low.difficulty.normalized_entropy == pytest.approx(high.difficulty.normalized_entropy)
    assert low.compute_units < high.compute_units


@pytest.mark.unit
def test_budget_caps_optional_compute_but_never_removes_minimum_path():
    budget = ATPBudget(total_atp=5)
    plan = _controller().plan(
        torch.zeros(4),
        budget,
        token_index=3,
        max_depth_layers=4,
        max_experts=4,
    )
    assert (plan.depth_layers, plan.expert_top_k, plan.mc_samples) == (3, 1, 1)
    assert budget.spent_atp == 5
    assert budget.remaining_atp == 0
    assert all(value >= 1 for value in (plan.depth_layers, plan.expert_top_k, plan.mc_samples))


@pytest.mark.unit
def test_insufficient_budget_rejects_plan_without_partial_debits():
    budget = ATPBudget(total_atp=2)
    with pytest.raises(InsufficientATPError, match="minimum adaptive path costs 3 ATP"):
        _controller().plan(
            torch.zeros(4),
            budget,
            token_index=0,
            max_depth_layers=4,
            max_experts=4,
        )
    assert budget.spent_atp == 0
    assert budget.records == []


@pytest.mark.unit
def test_disabled_controller_is_fixed_compute_identity_and_does_not_debit():
    budget = ATPBudget(total_atp=0)
    plan = _controller(enabled=False).plan(
        torch.tensor([9.0, 0.0]),
        budget,
        token_index=0,
        max_depth_layers=5,
        max_experts=3,
    )
    assert (plan.depth_layers, plan.expert_top_k, plan.mc_samples) == (5, 3, 4)
    assert plan.debit_records == ()
    assert budget.records == []


@pytest.mark.unit
def test_runtime_depth_executes_only_selected_vanilla_blocks_and_full_depth_is_identity():
    model = make_tiny_vanilla(n_layer=3)
    inputs = random_tokens(batch=1, seq=4)
    controller = _controller()
    shallow = controller.plan(
        torch.tensor([10.0, 0.0]),
        ATPBudget(3),
        token_index=0,
        max_depth_layers=3,
        max_experts=0,
    )
    full = _controller(enabled=False).plan(
        torch.tensor([10.0, 0.0]),
        ATPBudget(0),
        token_index=0,
        max_depth_layers=3,
        max_experts=0,
    )
    calls = [0, 0, 0]
    handles = []
    for layer_index, block in enumerate(model.blocks):
        def count_call(_module, _args, _output, index=layer_index):
            calls[index] += 1

        handles.append(block.register_forward_hook(count_call))
    try:
        shallow_logits = adaptive_forward(model, inputs, shallow)
        full_logits = adaptive_forward(model, inputs, full)
        direct_logits = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [3, 2, 2]
    assert shallow_logits.shape == full_logits.shape == direct_logits.shape
    torch.testing.assert_close(full_logits, direct_logits, rtol=0.0, atol=0.0)
    partial_cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=8,
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=1,
    )
    cached_shallow = model(inputs, kv_cache=partial_cache, max_layers=1)
    assert partial_cache.get_pos() == inputs.shape[1]
    assert cached_shallow.shape == shallow_logits.shape
    torch.testing.assert_close(cached_shallow, shallow_logits, rtol=0.0, atol=0.0)
    model(inputs[:, :1], kv_cache=partial_cache, max_layers=1)
    assert partial_cache.get_pos() == inputs.shape[1] + 1
    mismatched_cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=8,
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=model.config.n_layer,
    )
    with pytest.raises(ValueError, match="KV cache layer count must equal max_layers"):
        model(inputs, kv_cache=mismatched_cache, max_layers=1)
    with pytest.raises(ValueError, match="max_layers must be an integer"):
        model(inputs, max_layers=0)


@pytest.mark.unit
def test_runtime_expert_k_changes_real_moe_routing_and_restores_configuration():
    moe = SynapticMoE(
        n_embd=8,
        num_experts=4,
        top_k=3,
        hidden_mult=1,
        cfg=SynapticConfig(),
        dropout=0.0,
    )
    inputs = torch.randn(1, 3, 8)
    with temporary_expert_top_k(moe, 1) as touched:
        output, auxiliary_loss = moe(inputs, update_mem=False)
        assert touched == 1
        assert moe.last_ctx["indices"].shape[-1] == 1
    assert moe.top_k == 3
    assert output.shape == inputs.shape
    assert bool(torch.isfinite(output).all()) and bool(torch.isfinite(auxiliary_loss))


@pytest.mark.unit
def test_adaptive_depth_and_mc_execute_on_synaptic_model():
    model = make_tiny_synaptic(n_layer=3, use_moe=False)
    inputs = random_tokens(batch=1, seq=3)
    controller = AdaptiveComputeController(
        AdaptiveComputeConfig(
            enabled=True,
            min_depth_layers=1,
            min_experts=1,
            min_mc_samples=1,
            max_mc_samples=4,
            layer_cost_atp=10,
        )
    )
    plan = controller.plan_for_model(
        torch.zeros(model.config.vocab_size),
        ATPBudget(13),
        model=model,
        token_index=0,
    )
    calls = [0, 0, 0]
    handles = []
    for layer_index, block in enumerate(model.h):
        def count_call(_module, _args, _output, index=layer_index):
            calls[index] += 1

        handles.append(block.register_forward_hook(count_call))
    try:
        prediction = adaptive_mc_predict(model, inputs, plan)
    finally:
        for handle in handles:
            handle.remove()

    assert plan.mc_samples == 3 > 1
    assert calls == [plan.mc_samples, 0, 0]
    assert prediction.n_samples == plan.mc_samples
    assert plan.depth_layers == 1 < plan.max_depth_layers
    assert plan.expert_top_k == plan.max_experts == 0
    assert all(record.action != "expert" for record in plan.debit_records)
    assert prediction.mean_probs.shape == (*inputs.shape, model.config.vocab_size)
    assert bool(torch.isfinite(prediction.mean_probs).all())

    direct_logits, _ = model(inputs, train_mode=False)
    full_logits, _ = model(inputs, train_mode=False, max_layers=model.config.n_layer)
    torch.testing.assert_close(full_logits, direct_logits, rtol=0.0, atol=0.0)

    partial_cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=8,
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=1,
    )
    cached_logits, _ = model(inputs, kv_cache=partial_cache, train_mode=False, max_layers=1)
    shallow_logits, _ = model(inputs, train_mode=False, max_layers=1)
    assert partial_cache.get_pos() == inputs.shape[1]
    assert isinstance(partial_cache.presyn_state, list)
    assert len(partial_cache.presyn_state) == 1
    torch.testing.assert_close(cached_logits, shallow_logits, rtol=0.0, atol=0.0)

    mismatched_cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=8,
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=model.config.n_layer,
    )
    with pytest.raises(ValueError, match="KV cache layer count must equal max_layers"):
        model(inputs, kv_cache=mismatched_cache, train_mode=False, max_layers=1)


@pytest.mark.unit
def test_config_and_capacity_validation_reject_invalid_compute_ranges():
    with pytest.raises(ValueError, match="min_depth_layers"):
        AdaptiveComputeConfig(min_depth_layers=0)
    with pytest.raises(ValueError, match="max_mc_samples"):
        AdaptiveComputeConfig(min_mc_samples=3, max_mc_samples=2)
    with pytest.raises(ValueError, match="below min_depth_layers"):
        _controller().plan(
            torch.zeros(2),
            ATPBudget(1),
            token_index=0,
            max_depth_layers=0,
            max_experts=1,
        )


@pytest.mark.unit
def test_quality_floor_config_rejects_invalid_confidence_thresholds():
    for value in (-0.01, 1.01, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="min_predictive_confidence"):
            QualityFloorConfig(min_predictive_confidence=value)


@pytest.mark.unit
def test_quality_guard_accepts_confident_path_without_spending_fallback_atp():
    model = make_tiny_vanilla(n_layer=3)
    inputs = random_tokens(batch=1, seq=4)
    budget = ATPBudget(9)  # exact cheap attempt (2) + reserved full fallback (7)
    result = quality_guarded_predict(
        model,
        inputs,
        torch.tensor([20.0, 0.0, 0.0]),
        budget,
        controller=_controller(),
        token_index=2,
        quality=QualityFloorConfig(min_predictive_confidence=0.0),
    )

    assert result.quality_floor_passed and not result.fallback_used
    assert not result.fallback_reexecuted
    assert result.executed_plan == result.proposed_plan
    assert result.attempted_compute_units == result.proposed_plan.compute_units == 2
    assert result.saved_compute_units > 0
    assert result.token_spent_atp == budget.spent_atp < 7
    assert budget.spent_atp + budget.remaining_atp == budget.total_atp


@pytest.mark.unit
def test_quality_guard_falls_back_to_fixed_compute_and_charges_both_passes():
    model = make_tiny_vanilla(n_layer=3)
    inputs = random_tokens(batch=1, seq=4)
    budget = ATPBudget(9)  # cheap attempt (2) + complete full fallback (7)
    result = quality_guarded_predict(
        model,
        inputs,
        torch.tensor([20.0, 0.0, 0.0]),
        budget,
        controller=_controller(),
        token_index=0,
        quality=QualityFloorConfig(min_predictive_confidence=1.0),
    )

    assert not result.quality_floor_passed and result.fallback_used
    assert result.fallback_reexecuted
    assert result.fallback_reason == "predictive_confidence_below_floor"
    assert result.executed_plan.compute_units == result.executed_plan.maximum_compute_units == 7
    assert result.attempted_compute_units == 9
    assert result.saved_compute_units == -2
    assert result.token_spent_atp == budget.spent_atp == budget.total_atp == 9
    assert [record.action for record in result.executed_plan.debit_records] == [
        "depth_layer",
        "mc_sample",
        "quality_fallback_depth_layer",
        "quality_fallback_mc_sample",
    ]

    fixed = _controller(enabled=False).plan_for_model(
        torch.tensor([20.0, 0.0, 0.0]),
        ATPBudget(0),
        model=model,
        token_index=0,
    )
    fixed_prediction = adaptive_mc_predict(model, inputs, fixed)
    torch.testing.assert_close(
        result.prediction.mean_probs,
        fixed_prediction.mean_probs,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.unit
def test_quality_guard_does_not_rerun_an_already_full_attempt():
    model = make_tiny_vanilla(n_layer=3)
    inputs = random_tokens(batch=1, seq=4)
    calls = 0

    def count_call(_module, _args, _output):
        nonlocal calls
        calls += 1

    handle = model.blocks[0].register_forward_hook(count_call)
    try:
        result = quality_guarded_predict(
            model,
            inputs,
            torch.zeros(3),  # maximum entropy selects the full plan immediately
            ATPBudget(7),
            controller=_controller(),
            token_index=0,
            quality=QualityFloorConfig(min_predictive_confidence=1.0),
        )
    finally:
        handle.remove()

    assert result.fallback_used
    assert not result.fallback_reexecuted
    assert result.proposed_plan == result.executed_plan
    assert result.attempted_compute_units == result.executed_plan.maximum_compute_units == 7
    assert result.saved_compute_units == 0
    assert result.token_spent_atp == 7
    assert calls == 4  # one model forward for each MC draw, never a second full pass


@pytest.mark.unit
def test_quality_fallback_covers_moe_experts_and_replays_fixed_rng_stream():
    model = make_tiny_synaptic(n_layer=2, use_moe=True, num_experts=3, moe_top_k=2)
    inputs = random_tokens(batch=1, seq=3)
    routing_logits = torch.tensor([20.0, 0.0, 0.0])
    budget = ATPBudget(12)  # cheap attempt (3) + full fallback (9)
    torch.manual_seed(77)
    result = quality_guarded_predict(
        model,
        inputs,
        routing_logits,
        budget,
        controller=_controller(),
        token_index=0,
        quality=QualityFloorConfig(min_predictive_confidence=1.0),
    )
    assert result.fallback_used
    assert result.fallback_reexecuted
    assert result.token_spent_atp == 12
    assert result.attempted_compute_units == 12
    assert result.saved_compute_units == -3
    assert result.executed_plan.expert_top_k == result.executed_plan.max_experts == 3
    assert "quality_fallback_expert" in {
        record.action for record in result.executed_plan.debit_records
    }

    fixed = _controller(enabled=False).plan_for_model(
        routing_logits,
        ATPBudget(0),
        model=model,
        token_index=0,
    )
    torch.manual_seed(77)
    fixed_prediction = adaptive_mc_predict(model, inputs, fixed)
    torch.testing.assert_close(
        result.prediction.mean_probs,
        fixed_prediction.mean_probs,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.unit
def test_quality_guard_reserves_fallback_before_any_atp_debit():
    model = make_tiny_vanilla(n_layer=3)
    budget = ATPBudget(8)
    with pytest.raises(InsufficientATPError, match="requires at least 9 ATP"):
        quality_guarded_predict(
            model,
            random_tokens(batch=1, seq=3),
            torch.tensor([20.0, 0.0, 0.0]),
            budget,
            controller=_controller(),
            token_index=0,
        )
    assert budget.spent_atp == 0
    assert budget.records == []


@pytest.mark.unit
def test_quality_guard_caps_optional_attempt_work_behind_fallback_reserve():
    model = make_tiny_vanilla(n_layer=3)
    budget = ATPBudget(10)
    result = quality_guarded_predict(
        model,
        random_tokens(batch=1, seq=3),
        torch.tensor([2.0, 0.0, 0.0]),  # requests five units without the reserve
        budget,
        controller=_controller(),
        token_index=0,
        quality=QualityFloorConfig(min_predictive_confidence=0.0),
    )

    assert result.quality_floor_passed
    assert result.proposed_plan.compute_units == result.attempted_compute_units == 3
    assert result.token_spent_atp == budget.spent_atp == 3
    assert budget.remaining_atp == 7  # one complete fixed path remains available


@pytest.mark.unit
def test_disabled_guarded_runner_is_fixed_compute_identity_without_debits():
    model = make_tiny_vanilla(n_layer=2)
    budget = ATPBudget(0)
    result = quality_guarded_predict(
        model,
        random_tokens(batch=1, seq=3),
        torch.tensor([0.0, 0.0]),
        budget,
        controller=_controller(enabled=False),
        token_index=0,
        quality=QualityFloorConfig(min_predictive_confidence=1.0),
    )
    assert result.quality_floor_passed
    assert not result.fallback_used
    assert not result.fallback_reexecuted
    assert result.executed_plan.compute_units == result.executed_plan.maximum_compute_units
    assert result.attempted_compute_units == result.executed_plan.maximum_compute_units
    assert budget.records == []


@pytest.mark.e2e
def test_quality_guard_logs_detailed_per_token_compute_and_fallback(tmp_path):
    model = make_tiny_vanilla(n_layer=3)
    budget = ATPBudget(18)
    with RunLogger(tmp_path, name="adaptive_quality_guard", console=False) as logger:
        result = quality_guarded_predict(
            model,
            random_tokens(batch=1, seq=5),
            torch.tensor([20.0, 0.0, 0.0]),
            budget,
            controller=_controller(),
            token_index=4,
            quality=QualityFloorConfig(min_predictive_confidence=1.0),
            free_energy_value=0.25,
            run_logger=logger,
        )

    events = logger.read_events()
    records = [event for event in events if event["event"] == "adaptive_compute_token"]
    assert len(records) == 1
    record = records[0]
    assert record["step"] == record["token_index"] == 4
    assert record["fallback_used"] == result.fallback_used
    assert record["fallback_used"]
    assert record["fallback_reexecuted"]
    assert record["fallback_reason"] == "predictive_confidence_below_floor"
    assert record["quality_floor"] == 1.0
    assert not record["quality_floor_passed"]
    assert record["executed"] == record["maximum"]
    assert record["proposed"]["compute_units"] < record["executed"]["compute_units"]
    assert record["attempted_compute_units"] == 9
    assert record["saved_compute_units"] == -2
    assert record["token_spent_atp"] == 9
    assert record["sequence_spent_atp"] == 9
    assert record["sequence_remaining_atp"] == 9
    assert record["difficulty"]["free_energy"] == pytest.approx(0.25)
    assert len(record["debit_records"]) == 4
