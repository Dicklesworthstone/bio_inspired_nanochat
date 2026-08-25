"""Free-energy deliberation controller + engine decode-path wiring (bead `r00r.1.2`).

Covers `bio_inspired_nanochat/deliberation.py` (the `DeliberationController`) and its default-off hook
in `Engine.generate`, against the contract of `docs/theory/free_energy_deliberation.md`:

  - the ponder CONVERGES and is bounded by the compute budget;
  - effort SELF-ALLOCATES — a far-from-equilibrium (active, high-calcium) token uses more iterations
    than a near-equilibrium one ("compute scales with difficulty");
  - the adaptive decode temperature is bounded and commits when self-consistent / explores when not;
  - isolated candidate branches preserve row identity and feed relaxed energy into actual logits;
  - the engine REDUCES TO BASELINE when deliberation is off or has zero budget, and logs bounded
    per-token current-state plus candidate effort when on.

Run:  pytest tests/test_deliberation.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from _bio_testkit import make_tiny_synaptic
from bio_inspired_nanochat import engine as engine_module
from bio_inspired_nanochat.deliberation import (
    ATPBudget,
    CandidateEnergyBatch,
    CandidateEnergyReadout,
    DeliberationConfig,
    DeliberationController,
    DifficultyRouter,
    DifficultyRouterConfig,
    make_controller,
)
from bio_inspired_nanochat.engine import Engine, KVCache
from bio_inspired_nanochat.metriplectic_integrator import boltzmann_weights, run_monitored
from bio_inspired_nanochat.torch_imports import torch


# --------------------------------------------------------------------------- #
# Controller unit tests
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_difficulty_signal_correlates_with_token_loss():
    """As a calibrated correct-token margin shrinks, entropy difficulty tracks cross-entropy loss."""
    router = DifficultyRouter()
    margins = (5.0, 4.0, 3.0, 2.0, 1.0, 0.0)
    difficulties = []
    token_losses = []
    for margin in margins:
        logits = torch.tensor([margin, 0.0, 0.0, 0.0], dtype=torch.float64)
        difficulties.append(router.measure(logits).score)
        token_losses.append(float(-torch.log_softmax(logits, dim=-1)[0]))

    correlation = float(np.corrcoef(difficulties, token_losses)[0, 1])
    assert correlation > 0.9, f"difficulty must correlate strongly with token loss, got r={correlation:.4f}"
    assert all(a < b for a, b in zip(difficulties, difficulties[1:]))


@pytest.mark.unit
def test_difficulty_combines_entropy_and_bounded_free_energy():
    router = DifficultyRouter(DifficultyRouterConfig(entropy_weight=0.5, free_energy_scale=2.0))
    logits = torch.tensor([2.0, 0.0, -1.0])
    low_energy = router.measure(logits, free_energy_value=0.0)
    high_energy = router.measure(logits, free_energy_value=20.0)
    assert 0.0 <= low_energy.score < high_energy.score <= 1.0
    assert low_energy.normalized_entropy == pytest.approx(high_energy.normalized_entropy)
    assert low_energy.normalized_free_energy == 0.0
    assert high_energy.normalized_free_energy == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError, match="finite"):
        router.measure(torch.tensor([0.0, float("nan")]))
    with pytest.raises(ValueError, match="exactly one token distribution"):
        router.measure(torch.zeros(2, 3))


@pytest.mark.unit
def test_atp_budget_respects_exact_hard_limit():
    budget = ATPBudget(total_atp=10)
    first = budget.debit(
        token_index=0,
        action="deliberation_step",
        difficulty_score=0.75,
        requested_units=4,
        unit_cost_atp=3,
    )
    second = budget.debit(
        token_index=1,
        action="mc_sample",
        difficulty_score=1.0,
        requested_units=1,
        unit_cost_atp=2,
    )
    assert (first.granted_units, first.spent_atp, first.remaining_atp) == (3, 9, 1)
    assert (second.granted_units, second.spent_atp, second.remaining_atp) == (0, 0, 1)
    assert budget.spent_atp + budget.remaining_atp == budget.total_atp == 10
    assert budget.spent_atp == sum(record.spent_atp for record in budget.records)
    assert budget.summary() == {
        "total_atp": 10,
        "spent_atp": 9,
        "remaining_atp": 1,
        "exhausted": False,
        "debits": 2,
    }
    payload = json.loads(budget.to_jsonl()[0])
    assert payload["spent_atp"] == 9 and payload["remaining_atp"] == 1
    with pytest.raises(ValueError, match="non-negative integer"):
        ATPBudget(total_atp=-1)
    with pytest.raises(ValueError, match="positive"):
        budget.debit(
            token_index=2,
            action="layer",
            difficulty_score=0.5,
            requested_units=1,
            unit_cost_atp=0,
        )


@pytest.mark.unit
def test_energy_router_allocates_more_to_hard_tokens_without_overspending():
    router = DifficultyRouter()
    easy = router.measure(torch.tensor([6.0, 0.0, 0.0, 0.0]))
    hard = router.measure(torch.zeros(4))
    budget = ATPBudget(total_atp=6)
    easy_debit = router.route(
        budget,
        token_index=0,
        action="expert",
        difficulty=easy,
        min_units=1,
        max_units=5,
        unit_cost_atp=1,
    )
    hard_debit = router.route(
        budget,
        token_index=1,
        action="expert",
        difficulty=hard,
        min_units=1,
        max_units=5,
        unit_cost_atp=1,
    )
    assert hard_debit.requested_units > easy_debit.requested_units
    assert easy_debit.granted_units + hard_debit.granted_units == 6
    assert budget.exhausted and budget.spent_atp == budget.total_atp


@pytest.mark.unit
def test_make_controller_is_none_unless_enabled():
    assert make_controller(None) is None
    assert make_controller(DeliberationConfig(enabled=False)) is None
    assert isinstance(make_controller(DeliberationConfig(enabled=True)), DeliberationController)


@pytest.mark.unit
def test_synaptic_z_aggregates_calcium_and_buffer():
    c = DeliberationController(DeliberationConfig(enabled=True))
    ps = [
        {"C": torch.tensor([2.0, 2.0]), "BUF": torch.tensor([0.3])},
        {"C": torch.tensor([1.0]), "BUF": torch.tensor([0.1])},
    ]
    z = c.synaptic_z(ps)
    assert z is not None and z.shape == (3,)
    assert z[0] == pytest.approx(1.5)          # mean over layers of mean C
    assert z[1] == pytest.approx(0.2, abs=1e-6)
    assert z[2] == 0.0                          # h seeded at 0
    assert c.synaptic_z(None) is None           # no synaptic state ⟹ fall back


@pytest.mark.unit
def test_synaptic_z_rejects_nonfinite_calcium_and_falls_back():
    # An empty or all-NaN calcium tensor must NOT become a NaN z (which would silently drive the
    # explore-ceiling and log NaN); it falls back to None ⟹ single-step decode.
    c = DeliberationController(DeliberationConfig(enabled=True))
    assert c.synaptic_z([{"C": torch.tensor([]), "BUF": torch.tensor([0.5])}]) is None
    assert c.synaptic_z([{"C": torch.tensor([float("nan")]), "BUF": torch.tensor([0.5])}]) is None
    assert c.synaptic_z([{"C": torch.tensor([float("inf")]), "BUF": torch.tensor([0.5])}]) is None
    # ... and the per-token hook therefore returns the base temperature unchanged (no NaN record).
    assert c.effective_temperature([{"C": torch.tensor([float("nan")]), "BUF": torch.tensor([0.5])}], 0.9) == 0.9
    assert c.records == []


@pytest.mark.unit
def test_ponder_converges_within_budget():
    c = DeliberationController(DeliberationConfig(enabled=True, max_iters=64))
    res = c.ponder(np.array([0.5, 0.3, 0.0]))
    assert res.halted_converged, "a typical state must self-consistently halt before the budget"
    assert 1 <= res.iters <= 64
    assert res.F_drop >= -1e-9, "free energy must not increase (Thrust A Lyapunov)"


@pytest.mark.unit
def test_effort_self_allocates_with_difficulty():
    """A far-from-equilibrium (high-calcium) token ponders longer than a near-equilibrium one."""
    c = DeliberationController(DeliberationConfig(enabled=True, max_iters=128))
    easy = c.ponder(np.array([0.05, 0.05, 0.0])).iters
    hard = c.ponder(np.array([3.0, 1.0, 0.0])).iters
    assert hard > easy, f"compute must scale with difficulty (easy={easy}, hard={hard})"


@pytest.mark.unit
def test_adaptive_temperature_is_bounded_and_greedy_safe():
    cfg = DeliberationConfig(enabled=True, max_iters=64, temp_floor=0.7, temp_ceil=1.3)
    c = DeliberationController(cfg)
    easy = c.ponder(np.array([0.05, 0.05, 0.0]))
    hard = c.ponder(np.array([3.0, 1.0, 0.0]))
    t_easy = c.adaptive_temperature(1.0, easy)
    t_hard = c.adaptive_temperature(1.0, hard)
    assert cfg.temp_floor <= t_easy <= t_hard <= cfg.temp_ceil, (t_easy, t_hard)
    assert t_easy < t_hard, "confident (easy) tokens must decode sharper than uncertain (hard) ones"
    assert c.adaptive_temperature(0.0, hard) == 0.0, "greedy decode (base=0) must stay greedy"


@pytest.mark.unit
def test_effective_temperature_falls_back_without_state_or_when_disabled():
    on = DeliberationController(DeliberationConfig(enabled=True))
    assert on.effective_temperature(None, 0.9) == 0.9              # no synaptic state ⟹ base temp
    off = DeliberationController(DeliberationConfig(enabled=False))
    ps = [{"C": torch.tensor([1.0]), "BUF": torch.tensor([0.5])}]
    assert off.effective_temperature(ps, 0.9) == 0.9              # disabled ⟹ base temp
    assert off.records == []                                       # disabled never ponders/logs
    zero_budget = DeliberationController(DeliberationConfig(enabled=True, max_iters=0))
    assert zero_budget.effective_temperature(ps, 0.9) == 0.9
    assert zero_budget.records == []


@pytest.mark.unit
def test_boltzmann_token_weights_equal_temperature_softmax():
    c = DeliberationController(DeliberationConfig(enabled=True))
    logits = torch.tensor([1.0, 2.0, 3.0, 0.5])
    for kT in (0.5, 1.0, 2.0):
        w = c.boltzmann_token_weights(logits, kT=kT)
        sm = torch.softmax(logits.double() / kT, dim=-1)
        assert torch.allclose(w, sm, atol=1e-9), f"energy-based decode must equal kT-softmax (kT={kT})"


@pytest.mark.unit
def test_boltzmann_token_weights_normalize_per_row_for_batches():
    # Each distribution (last axis) must normalize independently — not globally across the batch.
    c = DeliberationController(DeliberationConfig(enabled=True))
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5], [0.0, 0.0, 5.0, 1.0]])
    w = c.boltzmann_token_weights(logits, kT=0.7)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-9), "each row must sum to 1"
    assert torch.allclose(w, torch.softmax(logits.double() / 0.7, dim=-1), atol=1e-9)
    with pytest.raises(ValueError):
        c.boltzmann_token_weights(logits, kT=0.0)


@pytest.mark.unit
def test_candidate_energy_feedback_is_shape_safe_and_changes_selected_logits():
    controller = DeliberationController(
        DeliberationConfig(enabled=True, candidate_top_k=2, candidate_energy_weight=1.0)
    )
    logits = torch.tensor([[5.0, 5.0, 1.0, 0.0]])
    candidate_ids = torch.tensor([[0, 1]])
    scores = CandidateEnergyBatch(
        F_initial=np.array([[3.0, 1.0]]),
        F_final=np.array([[2.0, 0.0]]),
        effort=np.array([[4, 2]]),
        halted_converged=np.array([[True, True]]),
    )
    adjusted = controller.candidate_energy_logits(logits, candidate_ids, scores)
    torch.testing.assert_close(adjusted[0, :2], torch.tensor([3.0, 5.0]))
    assert bool(torch.isneginf(adjusted[0, 2:]).all())
    assert int(adjusted.argmax(dim=-1)) == 1

    wrong_shape = CandidateEnergyBatch(
        F_initial=np.zeros((1, 1)),
        F_final=np.zeros((1, 1)),
        effort=np.ones((1, 1), dtype=np.int64),
        halted_converged=np.ones((1, 1), dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="does not match"):
        controller.candidate_energy_logits(logits, candidate_ids, wrong_shape)


@pytest.mark.unit
def test_candidate_readout_fits_pairwise_energy_on_frozen_calibration_data():
    model_logits = np.zeros((4, 2), dtype=np.float64)
    features = np.array(
        [[[-1.0], [1.0]], [[1.0], [-1.0]], [[-2.0], [2.0]], [[2.0], [-2.0]]],
        dtype=np.float64,
    )
    correct = np.array(
        [[True, False], [False, True], [True, False], [False, True]],
        dtype=np.bool_,
    )
    readout = CandidateEnergyReadout.fit(
        model_logits=model_logits,
        synaptic_features=features,
        correct_mask=correct,
        feature_names=("task_signal",),
    )
    held_out = CandidateEnergyBatch(
        F_initial=np.array([[0.0, 0.0]]),
        # The old aggregate ranks candidate 1 first; the calibrated feature ranks candidate 0.
        F_final=np.array([[2.0, 0.0]]),
        effort=np.ones((1, 2), dtype=np.int64),
        halted_converged=np.ones((1, 2), dtype=np.bool_),
        features=np.array([[[-1.5], [1.5]]]),
        feature_names=("task_signal",),
    )
    energies = readout.energy(np.zeros((1, 2)), held_out)
    assert int(np.argmin(held_out.F_final, axis=1)[0]) == 1
    assert int(np.argmin(energies, axis=1)[0]) == 0
    assert readout.calibration_groups == 4
    assert readout.calibration_pairs == 4

    controller = DeliberationController(
        DeliberationConfig(enabled=True, candidate_top_k=2, candidate_energy_weight=1.0),
        candidate_readout=readout,
    )
    adjusted = controller.candidate_energy_logits(
        torch.zeros((1, 3)),
        torch.tensor([[0, 1]]),
        held_out,
    )
    assert int(adjusted.argmax(dim=-1)) == 0


@pytest.mark.unit
def test_candidate_relaxation_preserves_branch_rows_and_is_bounded():
    controller = DeliberationController(
        DeliberationConfig(enabled=True, max_iters=3, candidate_top_k=2)
    )
    state = [{
        "C": torch.tensor([[[0.1]], [[0.4]], [[0.8]], [[1.2]]]),
        "BUF": torch.zeros(4, 1, 1),
    }]
    scores = controller.relax_candidate_states(state, candidate_shape=(2, 2))
    assert scores is not None
    assert scores.shape == (2, 2)
    assert scores.features is not None
    assert scores.features.shape[:2] == scores.shape
    assert scores.features.shape[-1] == len(scores.feature_names)
    assert scores.feature_names[:5] == (
        "F_initial",
        "F_final",
        "F_drop",
        "effort_fraction",
        "halted_converged",
    )
    assert np.unique(scores.F_final).size > 1
    assert bool(np.all((1 <= scores.effort) & (scores.effort <= 3)))
    assert scores.max_effort_per_row <= 2 * 3


@pytest.mark.unit
def test_f_trajectory_and_summary_are_well_formed():
    c = DeliberationController(DeliberationConfig(enabled=True))
    ps = [{"C": torch.tensor([1.0]), "BUF": torch.tensor([0.4])}]
    for i in range(3):
        c.effective_temperature(ps, 1.0, token_index=i)
    traj = c.f_trajectory()
    assert len(traj) == 3
    assert {"token_index", "effort", "F_initial", "F_final", "F_drop", "effective_temperature"} <= set(traj[0])
    s = c.summary()
    assert s["tokens"] == 3 and s["enabled"] and s["max_budget"] == 9 * 64
    assert s["mean_effort"] > 0


# --------------------------------------------------------------------------- #
# Engine decode-path integration
# --------------------------------------------------------------------------- #
class _FakeTokenizer:
    """Minimal tokenizer: just the special-token API `Engine.generate` consults (no rust build)."""

    _SPECIAL = {
        "<|python_start|>": 91, "<|python_end|>": 92,
        "<|output_start|>": 93, "<|output_end|>": 94, "<|assistant_end|>": 95,
    }

    def encode_special(self, s: str) -> int:
        return self._SPECIAL[s]

    def get_bos_token_id(self) -> int:
        return 96

    def encode(self, s: str):
        return [1, 2]

    def decode(self, toks) -> str:
        return ""


def _engine():
    model = make_tiny_synaptic(seed=1234)
    model.train(False)
    return Engine(model, _FakeTokenizer())


def _decode(engine, **kw):
    return [tuple(tc) for tc, _mask in engine.generate([1, 2, 3, 4], max_tokens=8, seed=7, **kw)]


@pytest.mark.e2e
def test_candidate_cache_fork_is_batch_aligned_and_does_not_mutate_committed_state():
    model = make_tiny_synaptic(seed=1234)
    model.train(False)
    cache = KVCache(
        batch_size=2,
        num_heads=model.config.n_kv_head,
        seq_len=8,
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=model.config.n_layer,
    )
    model(torch.tensor([[1, 2, 3], [4, 5, 6]]), kv_cache=cache, train_mode=False)
    assert cache.kv_cache is not None
    assert isinstance(cache.presyn_state, list)
    committed_kv = cache.kv_cache.clone()
    committed_calcium = cache.presyn_state[0]["C"].clone()
    committed_pos = cache.pos

    forked = cache.fork_batch(2)
    torch.testing.assert_close(
        forked.kv_cache[:, :, 0, :, :committed_pos],
        committed_kv[:, :, 0, :, :committed_pos],
    )
    torch.testing.assert_close(
        forked.kv_cache[:, :, 1, :, :committed_pos],
        committed_kv[:, :, 0, :, :committed_pos],
    )
    torch.testing.assert_close(
        forked.kv_cache[:, :, 2, :, :committed_pos],
        committed_kv[:, :, 1, :, :committed_pos],
    )
    torch.testing.assert_close(
        forked.kv_cache[:, :, 3, :, :committed_pos],
        committed_kv[:, :, 1, :, :committed_pos],
    )
    model(torch.tensor([[7], [8], [9], [10]]), kv_cache=forked, train_mode=False)

    assert forked.kv_cache is not None
    assert isinstance(forked.presyn_state, list)
    assert forked.kv_cache.shape[2] == 4
    assert forked.pos == committed_pos + 1
    assert forked.presyn_state[0]["C"].shape[0] == 4
    assert cache.pos == committed_pos
    # Only the populated prefix is semantically part of the committed cache.
    # The unused capacity comes from torch.empty and may contain NaNs, for which
    # an all-capacity assert_close is spuriously false even when no byte changed.
    torch.testing.assert_close(
        cache.kv_cache[:, :, :, :, :committed_pos],
        committed_kv[:, :, :, :, :committed_pos],
    )
    torch.testing.assert_close(cache.presyn_state[0]["C"], committed_calcium)


@pytest.mark.e2e
def test_generate_deliberation_off_is_byte_identical_baseline():
    """No controller or a zero-step budget must reproduce the default decode exactly."""
    e = _engine()
    base = _decode(e, temperature=0.8)
    off = _decode(e, temperature=0.8, deliberation=None)
    assert base == off, "deliberation=None must not perturb the decode path"
    zero_budget = DeliberationController(DeliberationConfig(enabled=True, max_iters=0))
    assert _decode(e, temperature=0.8, deliberation=zero_budget) == base
    assert zero_budget.records == []


@pytest.mark.e2e
def test_zero_token_request_does_not_create_a_deliberation_record():
    controller = DeliberationController(DeliberationConfig(enabled=True))
    generated = list(
        _engine().generate(
            [1, 2, 3, 4], max_tokens=0, temperature=0.8, seed=7, deliberation=controller
        )
    )
    assert generated == []
    assert controller.records == []


@pytest.mark.e2e
def test_generate_greedy_deliberation_runs_candidate_feedback_and_logs():
    """Greedy sampling still executes the bounded candidate-energy path on every token."""
    controller = DeliberationController(DeliberationConfig(enabled=True))
    on = _decode(_engine(), temperature=0.0, deliberation=controller)
    assert len(controller.records) == len(on), "every generated token must have a controller record"
    assert [record.token_index for record in controller.records] == list(range(len(on)))
    assert all(r.effort >= 1 for r in controller.records)
    assert all(r.F_drop >= -1e-9 for r in controller.records), "free energy must not increase"
    assert all(r.candidate_count == controller.cfg.candidate_top_k for r in controller.records)
    assert all(
        r.total_effort <= (controller.cfg.candidate_top_k + 1) * controller.cfg.max_iters
        for r in controller.records
    )


@pytest.mark.e2e
def test_first_token_candidate_energy_feedback_can_change_greedy_choice(monkeypatch):
    baseline = next(_engine().generate([1, 2, 3, 4], max_tokens=1, temperature=0.0))[0][0]
    controller = DeliberationController(
        DeliberationConfig(enabled=True, max_iters=2, candidate_top_k=2)
    )
    observed_branch_state = []

    def penalize_model_favorite(presyn_state, *, candidate_shape):
        observed_branch_state.append(presyn_state)
        return CandidateEnergyBatch(
            F_initial=np.zeros(candidate_shape),
            F_final=np.array([[100.0, 0.0]]),
            effort=np.ones(candidate_shape, dtype=np.int64),
            halted_converged=np.ones(candidate_shape, dtype=np.bool_),
        )

    monkeypatch.setattr(controller, "relax_candidate_states", penalize_model_favorite)
    changed = next(
        _engine().generate(
            [1, 2, 3, 4],
            max_tokens=1,
            temperature=0.0,
            deliberation=controller,
        )
    )[0][0]
    assert observed_branch_state and changed != baseline
    assert controller.records[0].candidate_count == 2


@pytest.mark.e2e
def test_first_generated_token_uses_deliberation_temperature(monkeypatch):
    observed_temperatures = []
    sample_next_token = engine_module.sample_next_token

    def record_temperature(logits, rng, temperature=1.0, top_k=None):
        observed_temperatures.append(float(temperature))
        return sample_next_token(logits, rng, temperature, top_k)

    monkeypatch.setattr(engine_module, "sample_next_token", record_temperature)
    controller = DeliberationController(
        DeliberationConfig(enabled=True, max_iters=4, temp_floor=0.5, temp_ceil=0.5)
    )
    tokens = _decode(_engine(), temperature=0.8, deliberation=controller)
    assert observed_temperatures == pytest.approx([0.4] * len(tokens))
    assert len(controller.records) == len(tokens)


@pytest.mark.e2e
def test_generate_with_deliberation_runs_and_produces_trajectory():
    e = _engine()
    controller = DeliberationController(DeliberationConfig(enabled=True, max_iters=32))
    toks = _decode(e, temperature=0.9, deliberation=controller)
    assert len(toks) > 0, "generation must produce tokens"
    assert len(controller.records) > 0
    summary = controller.summary()
    assert summary["tokens"] == len(controller.records)
    assert 1 <= summary["max_effort"] <= 9 * 32, "effort must respect the compute budget"


# --------------------------------------------------------------------------- #
# r00r.1.3 — Lyapunov per-step, halting, energy-sampler distribution, JSONL artifact
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_free_energy_is_nonincreasing_across_deliberation_steps():
    """The Lyapunov guarantee that makes "think longer" safe: F must not increase, step over step,
    on the deliberation integrator seeded from a synaptic state."""
    cfg = DeliberationConfig(enabled=True, dt=0.5)
    z0 = np.array([1.0, 0.5, 0.0])  # an active synaptic state
    _traj, monitor = run_monitored(z0, cfg.dt, steps=40, T=cfg.T)
    f_seq = [r.F for r in monitor.records]
    assert all(f_seq[i + 1] <= f_seq[i] + 1e-9 for i in range(len(f_seq) - 1)), "F must be non-increasing"
    assert monitor.free_energy_nonincreasing()


@pytest.mark.unit
def test_halting_distinguishes_convergence_from_budget():
    easy = DeliberationController(DeliberationConfig(enabled=True, max_iters=128, eps=1e-4))
    res_conv = easy.ponder(np.array([0.2, 0.1, 0.0]))
    assert res_conv.halted_converged and res_conv.iters < 128, "an easy state must halt on |ΔF|<eps"
    tight = DeliberationController(DeliberationConfig(enabled=True, max_iters=3, eps=1e-12))
    res_budget = tight.ponder(np.array([3.0, 1.0, 0.0]))
    assert not res_budget.halted_converged and res_budget.iters == 3, "a tiny budget must be hit, not converged"


@pytest.mark.unit
def test_energy_sampler_matches_target_distribution():
    """`boltzmann_weights` must equal the closed-form `exp(−F/kT)/Z`, and sampling reproduces it."""
    free_energies = np.array([0.0, 1.0, 2.0, 0.5])
    kT = 0.8
    w = boltzmann_weights(free_energies, kT=kT)
    # INDEPENDENT closed form — this is what catches a wrong sign or kT-scaling (the sampling check
    # below only verifies torch.multinomial reproduces its own input, which is not a property of w).
    expected = np.exp(-free_energies / kT)
    expected /= expected.sum()
    assert np.allclose(w, expected, atol=1e-12), f"weights {w} must equal exp(−F/kT)/Z {expected}"
    # A wrong kT (e.g. 2·kT) would change the spread; pin it.
    assert not np.allclose(w, boltzmann_weights(free_energies, kT=2 * kT), atol=1e-3)
    probs = torch.as_tensor(w, dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    draws = torch.multinomial(probs, num_samples=40000, replacement=True, generator=gen)
    emp = torch.bincount(draws, minlength=4).double() / 40000.0
    assert torch.allclose(emp, probs, atol=0.02), f"empirical {emp.tolist()} must match target {probs.tolist()}"
    assert int(probs.argmax()) == int(np.argmin(free_energies))  # argmin-F is most probable


@pytest.mark.e2e
def test_generation_writes_f_trajectory_artifact(tmp_path):
    """A generation run must produce a per-token F-trajectory JSONL artifact (the eqyk.2 logging)."""
    e = _engine()
    controller = DeliberationController(DeliberationConfig(enabled=True, max_iters=32))
    _decode(e, temperature=0.0, deliberation=controller)  # greedy: deterministic, still ponders
    assert controller.records, "the run must have pondered per token"
    artifact = tmp_path / "deliberation_trajectory.jsonl"
    controller.write_trajectory(artifact)
    lines = artifact.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(controller.records)
    rec = json.loads(lines[0])
    assert {"token_index", "effort", "F_initial", "F_final", "F_drop", "halted_converged"} <= set(rec)
    assert rec["effort"] >= 1 and rec["F_drop"] >= -1e-9
