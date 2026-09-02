"""The Rust CPU decode kernel is dispatched from ``release_canonical`` (``native_presyn`` on CPU).

Until 2026-09 the README said the Rust kernel was "parity-locked but not live-dispatched":
``rustbpe.presyn_release_canonical_cpu`` was reachable only from tests. These tests lock the
wiring that closes that gap:

* with ``native_presyn=True`` on CPU float32 tensors, an eval-mode one-query
  ``release_canonical`` call runs the Rust kernel (counted through a monkeypatched lookup) and
  returns the same release **and the same advanced state, DELAY queue included** as the PyTorch
  path;
* planted negatives: grad enabled, ``train=True``, or more than one query keep the PyTorch
  path (the kernel is replaced with one that raises);
* when the extension is absent the toggle silently degrades to PyTorch.

Green here does NOT prove a speedup; ``scripts``-level timing is reported separately.

Run:  pytest tests/test_presyn_rust_dispatch.py -v
"""

from __future__ import annotations

import importlib

import pytest
import torch

from bio_inspired_nanochat import synaptic as syn
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn

pytestmark = pytest.mark.unit

_STATE_KEYS = ("C", "BUF", "RRP", "RES", "PR", "CL", "AMP", "E")


def _kernel_or_skip():
    try:
        rustbpe = importlib.import_module("rustbpe")
    except ModuleNotFoundError:
        pytest.skip("rustbpe extension not built (uv sync --extra cpu --dev)")
    kernel = getattr(rustbpe, "presyn_release_canonical_cpu", None)
    if kernel is None:
        pytest.skip("rustbpe lacks presyn_release_canonical_cpu")
    return kernel


def _presyn(native: bool, seed: int = 11) -> SynapticPresyn:
    torch.manual_seed(seed)
    module = SynapticPresyn(d_head=8, cfg=SynapticConfig(native_presyn=native))
    module.eval()
    return module


def _seeded_state(presyn: SynapticPresyn, seed: int, batch: int, heads: int, keys: int):
    g = torch.Generator().manual_seed(seed)
    return {
        "C": torch.rand(batch, heads, keys, generator=g),
        "BUF": torch.rand(batch, heads, keys, generator=g),
        "RRP": torch.rand(batch, heads, keys, generator=g) * 4.0 + 1.0,
        "RES": torch.rand(batch, heads, keys, generator=g),
        "PR": torch.rand(batch, heads, keys, generator=g),
        "CL": torch.rand(batch, heads, keys, generator=g),
        "AMP": torch.ones(batch, heads, keys),
        "E": torch.rand(batch, heads, keys, generator=g) * 0.8 + 0.2,
        "DELAY": [
            torch.rand(batch, heads, keys, generator=g) * 0.1
            for _ in range(presyn.cfg.endo_delay)
        ],
    }


def _inputs(B: int, H: int, T_keys: int, K: int, seed: int = 42):
    torch.manual_seed(seed)
    drive = torch.rand(B, H, 1, K)
    idx = torch.stack(
        [
            torch.randperm(T_keys, generator=torch.Generator().manual_seed(7 + i))[:K]
            for i in range(B * H)
        ]
    ).view(B, H, 1, K)
    valid = torch.ones(B, H, 1, K, dtype=torch.bool)
    return drive, idx, valid


def _assert_states_match(rs_state, py_state):
    for key in _STATE_KEYS:
        torch.testing.assert_close(rs_state[key], py_state[key], rtol=1e-5, atol=1e-6, msg=key)
    assert len(rs_state["DELAY"]) == len(py_state["DELAY"])
    for position, (rs_entry, py_entry) in enumerate(zip(rs_state["DELAY"], py_state["DELAY"])):
        torch.testing.assert_close(
            rs_entry, py_entry, rtol=1e-5, atol=1e-6, msg=f"DELAY[{position}]"
        )


def test_native_presyn_on_cpu_dispatches_to_rust_and_matches_python(monkeypatch):
    kernel = _kernel_or_skip()
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return kernel(*args, **kwargs)

    monkeypatch.setattr(syn, "_rust_presyn_kernel", lambda: counting)
    B, H, T_keys, K = 2, 3, 9, 4
    python = _presyn(native=False)
    rust = _presyn(native=True)
    drive, idx, valid = _inputs(B, H, T_keys, K)
    state_py = _seeded_state(python, 99, B, H, T_keys)
    state_rs = _seeded_state(rust, 99, B, H, T_keys)

    with torch.no_grad():
        e_py = python.release_canonical(state_py, drive, idx, train=False, valid=valid)
        e_rs = rust.release_canonical(state_rs, drive, idx, train=False, valid=valid)

    assert calls["n"] == 1, "the Rust kernel must have served exactly this one decode step"
    assert e_rs.shape == e_py.shape == (B, H, 1, K) and e_rs.dtype == e_py.dtype
    torch.testing.assert_close(e_rs, e_py, rtol=1e-5, atol=1e-6)
    _assert_states_match(state_rs, state_py)


def test_native_presyn_cpu_handles_duplicate_keys_and_invalid_edges(monkeypatch):
    """Duplicate key indices must reduce per key exactly like the scatter path; masked edges
    release nothing and do not touch state."""
    kernel = _kernel_or_skip()
    monkeypatch.setattr(syn, "_rust_presyn_kernel", lambda: kernel)
    B, H, T_keys, K = 1, 2, 5, 4
    python = _presyn(native=False)
    rust = _presyn(native=True)
    torch.manual_seed(3)
    drive = torch.rand(B, H, 1, K)
    idx = torch.tensor([[[[0, 0, 3, 3]], [[1, 4, 4, 2]]]])
    valid = torch.tensor([[[[True, True, True, False]], [[True, False, True, True]]]])
    state_py = _seeded_state(python, 5, B, H, T_keys)
    state_rs = _seeded_state(rust, 5, B, H, T_keys)
    with torch.no_grad():
        e_py = python.release_canonical(state_py, drive, idx, train=False, valid=valid)
        e_rs = rust.release_canonical(state_rs, drive, idx, train=False, valid=valid)
    torch.testing.assert_close(e_rs, e_py, rtol=1e-5, atol=1e-6)
    assert torch.all(e_rs[~valid] == 0)
    _assert_states_match(state_rs, state_py)


def test_python_path_kept_when_grad_train_or_multiquery(monkeypatch):
    _kernel_or_skip()

    def forbidden(*args, **kwargs):
        raise AssertionError("the Rust kernel must not run on this call")

    monkeypatch.setattr(syn, "_rust_presyn_kernel", lambda: forbidden)
    B, H, T_keys, K = 1, 2, 6, 3
    rust = _presyn(native=True)
    drive, idx, valid = _inputs(B, H, T_keys, K)

    # grad enabled -> PyTorch (the release must stay differentiable w.r.t. drive)
    drive_g = drive.clone().requires_grad_(True)
    e = rust.release_canonical(_seeded_state(rust, 1, B, H, T_keys), drive_g, idx, train=False, valid=valid)
    assert e.requires_grad and torch.isfinite(e).all()

    # train=True -> PyTorch (stochastic release + EMA adaptation live there)
    with torch.no_grad():
        e = rust.release_canonical(_seeded_state(rust, 1, B, H, T_keys), drive, idx, train=True, valid=valid)
    assert torch.isfinite(e).all()

    # two queries -> PyTorch (prefill needs the causal chunked recurrence)
    drive2 = torch.rand(B, H, 2, K)
    idx2 = torch.cat([idx, idx], dim=2)
    valid2 = torch.ones(B, H, 2, K, dtype=torch.bool)
    with torch.no_grad():
        e = rust.release_canonical(_seeded_state(rust, 1, B, H, T_keys), drive2, idx2, train=False, valid=valid2)
    assert e.shape == (B, H, 2, K)


def test_toggle_degrades_to_python_without_the_extension(monkeypatch):
    monkeypatch.setattr(syn, "_rust_presyn_kernel", lambda: None)
    B, H, T_keys, K = 1, 1, 4, 2
    python = _presyn(native=False)
    rust = _presyn(native=True)
    drive, idx, valid = _inputs(B, H, T_keys, K)
    state_py = _seeded_state(python, 8, B, H, T_keys)
    state_rs = _seeded_state(rust, 8, B, H, T_keys)
    with torch.no_grad():
        e_py = python.release_canonical(state_py, drive, idx, train=False, valid=valid)
        e_rs = rust.release_canonical(state_rs, drive, idx, train=False, valid=valid)
    torch.testing.assert_close(e_rs, e_py)
    _assert_states_match(state_rs, state_py)


def test_predicate_rejects_non_float32_and_learnable_kinetics():
    _kernel_or_skip()
    rust = _presyn(native=True)
    B, H, T_keys, K = 1, 1, 4, 2
    drive, _idx, _valid = _inputs(B, H, T_keys, K)
    state = _seeded_state(rust, 2, B, H, T_keys)
    common = {
        "train": False,
        "differentiable": False,
        "apply_barrier": False,
        "q_pos": None,
        "active_key_count": None,
        "runtime_buffers": None,
    }
    with torch.no_grad():
        assert rust._can_use_native_presyn_cpu_decode(state, drive, **common)
        assert not rust._can_use_native_presyn_cpu_decode(state, drive.double(), **common)
        assert not rust._can_use_native_presyn_cpu_decode(state, drive, **{**common, "active_key_count": 3})
        assert not rust._can_use_native_presyn_cpu_decode(state, drive, **{**common, "apply_barrier": True})
    learnable = SynapticPresyn(d_head=8, cfg=SynapticConfig(native_presyn=True, learnable_kinetics=True))
    learnable.eval()
    with torch.no_grad():
        assert not learnable._can_use_native_presyn_cpu_decode(state, drive, **common)
