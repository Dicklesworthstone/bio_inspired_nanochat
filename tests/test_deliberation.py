"""Free-energy deliberation controller + engine decode-path wiring (bead `r00r.1.2`).

Covers `bio_inspired_nanochat/deliberation.py` (the `DeliberationController`) and its default-off hook
in `Engine.generate`, against the contract of `docs/theory/free_energy_deliberation.md`:

  - the ponder CONVERGES and is bounded by the compute budget;
  - effort SELF-ALLOCATES — a far-from-equilibrium (active, high-calcium) token uses more iterations
    than a near-equilibrium one ("compute scales with difficulty");
  - the adaptive decode temperature is bounded and commits when self-consistent / explores when not;
  - the energy-based sampler is the Boltzmann softmax over logits;
  - the engine REDUCES TO BASELINE when deliberation is off (byte-identical decode), and RUNS in the
    decode path when on (greedy decode is invariant yet the per-token F-trajectory is logged).

Run:  pytest tests/test_deliberation.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from _bio_testkit import make_tiny_synaptic
from bio_inspired_nanochat.deliberation import (
    DeliberationConfig,
    DeliberationController,
    make_controller,
)
from bio_inspired_nanochat.engine import Engine
from bio_inspired_nanochat.metriplectic_integrator import boltzmann_weights, run_monitored
from bio_inspired_nanochat.torch_imports import torch


# --------------------------------------------------------------------------- #
# Controller unit tests
# --------------------------------------------------------------------------- #
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
def test_f_trajectory_and_summary_are_well_formed():
    c = DeliberationController(DeliberationConfig(enabled=True))
    ps = [{"C": torch.tensor([1.0]), "BUF": torch.tensor([0.4])}]
    for i in range(3):
        c.effective_temperature(ps, 1.0, token_index=i)
    traj = c.f_trajectory()
    assert len(traj) == 3
    assert {"token_index", "effort", "F_initial", "F_final", "F_drop", "effective_temperature"} <= set(traj[0])
    s = c.summary()
    assert s["tokens"] == 3 and s["enabled"] and s["max_budget"] == 64
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
    model.eval()
    return Engine(model, _FakeTokenizer())


def _decode(engine, **kw):
    return [tuple(tc) for tc, _mask in engine.generate([1, 2, 3, 4], max_tokens=8, seed=7, **kw)]


@pytest.mark.e2e
def test_generate_deliberation_off_is_byte_identical_baseline():
    """deliberation=None must reproduce the no-deliberation decode exactly (the default-off contract)."""
    e = _engine()
    base = _decode(e, temperature=0.8)
    off = _decode(e, temperature=0.8, deliberation=None)
    assert base == off, "deliberation=None must not perturb the decode path"


@pytest.mark.e2e
def test_generate_greedy_is_invariant_yet_deliberation_runs_and_logs():
    """At temperature 0 the argmax is invariant to the temperature knob, so an ENABLED deliberation
    produces the SAME tokens as baseline — while still running per token (records logged). This is the
    'runs in the engine decode path AND reduces to baseline' acceptance, pinned in one test."""
    e = _engine()
    base = _decode(e, temperature=0.0)
    controller = DeliberationController(DeliberationConfig(enabled=True))
    on = _decode(e, temperature=0.0, deliberation=controller)
    assert on == base, "greedy decode must be invariant to deliberation"
    assert len(controller.records) > 0, "the controller must have pondered in the decode loop"
    assert all(r.effort >= 1 for r in controller.records)
    assert all(r.F_drop >= -1e-9 for r in controller.records), "free energy must not increase"


@pytest.mark.e2e
def test_generate_with_deliberation_runs_and_produces_trajectory():
    e = _engine()
    controller = DeliberationController(DeliberationConfig(enabled=True, max_iters=32))
    toks = _decode(e, temperature=0.9, deliberation=controller)
    assert len(toks) > 0, "generation must produce tokens"
    assert len(controller.records) > 0
    summary = controller.summary()
    assert summary["tokens"] == len(controller.records)
    assert 1 <= summary["max_effort"] <= 32, "effort must respect the compute budget"


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
