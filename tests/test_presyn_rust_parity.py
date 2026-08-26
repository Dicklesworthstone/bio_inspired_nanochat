"""Rust↔Python presyn decode parity (bead dzvj / README 'parity-locked' claim).

The README states the Rust CPU kernel mirrors ``release_canonical`` exactly, but
until now NO automated test exercised that: a constant or ordering drift between
``synaptic.py`` and ``rust_src/src/presyn.rs`` would ship silently.

These tests run the SAME seeded state/inputs through both implementations and
assert the released activations and every advanced state buffer match within
fp32 tolerance. They SKIP cleanly when the compiled extension is not installed
(build via ``uv run maturin develop --release``).

Run:  pytest tests/test_presyn_rust_parity.py -v
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn

_STATE_KEYS = ("C", "BUF", "RRP", "RES", "PR", "CL", "AMP", "E")


def _rust_kernel():
    try:
        rustbpe = importlib.import_module("rustbpe")
    except ModuleNotFoundError:
        pytest.skip("rustbpe extension not built (uv run maturin develop --release)")
    kernel = getattr(rustbpe, "presyn_release_canonical_cpu", None)
    if kernel is None:
        pytest.skip("rustbpe lacks presyn_release_canonical_cpu")
    return kernel


def _make_presyn(seed: int, d_head: int = 8) -> SynapticPresyn:
    torch.manual_seed(seed)
    cfg = SynapticConfig()
    return SynapticPresyn(d_head=d_head, cfg=cfg)


def _seeded_state(presyn: SynapticPresyn, seed: int, batch: int, heads: int, keys: int):
    g = torch.Generator().manual_seed(seed)
    state = {
        "C": torch.rand(batch, heads, keys, generator=g),
        "BUF": torch.rand(batch, heads, keys, generator=g),
        "RRP": (torch.rand(batch, heads, keys, generator=g) * 4.0 + 1.0),
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
    return state


def _clone_state(state):
    clone = {}
    for key, value in state.items():
        if isinstance(value, list):
            clone[key] = [t.clone() for t in value]
        elif torch.is_tensor(value):
            clone[key] = value.clone()
        else:
            clone[key] = value
    return clone


def _to_numpy_state(state):
    out = {}
    for key, value in state.items():
        if isinstance(value, list):
            out[key] = [t.detach().cpu().numpy() for t in value]
        else:
            out[key] = value.detach().cpu().numpy()
    return out


@pytest.mark.unit
def test_rust_matches_python_canonical_release_and_state():
    B, H, T_keys, Tq, K = 1, 2, 6, 1, 3
    presyn = _make_presyn(seed=11)
    kernel = _rust_kernel()

    torch.manual_seed(42)
    drive = torch.rand(B, H, Tq, K)
    idx = torch.stack(
        [
            torch.randperm(T_keys, generator=torch.Generator().manual_seed(7))[:K]
            for _ in range(B * H)
        ]
    ).view(B, H, Tq, K)
    valid = torch.ones(B, H, Tq, K, dtype=torch.bool)

    # --- Python canonical ---------------------------------------------------
    state_py = _seeded_state(presyn, seed=99, batch=B, heads=H, keys=T_keys)
    with torch.no_grad():
        released_py = presyn.release_canonical(
            state_py, drive, idx, train=False, valid=valid, differentiable=False
        )

    # --- Rust mirror (identical pristine inputs) ----------------------------
    state_rs_torch = _clone_state(_seeded_state(presyn, seed=99, batch=B, heads=H, keys=T_keys))
    state_rs = _to_numpy_state(state_rs_torch)
    released_rs, rs_state = kernel(
        drive.numpy(),
        idx.numpy(),
        valid.numpy(),
        state_rs,
        presyn.cfg,
        float(presyn.ema_e.item()),
    )

    assert np.allclose(
        released_py.detach().cpu().numpy(), released_rs, rtol=1e-5, atol=1e-6
    ), (
        f"release mismatch: max abs diff "
        f"{np.abs(released_py.detach().cpu().numpy() - released_rs).max():.3e}"
    )
    for key in ("C", "BUF", "RRP", "RES", "PR", "CL", "E"):
        assert np.allclose(
            state_py[key].detach().cpu().numpy(),
            np.asarray(rs_state[key]),
            rtol=1e-5,
            atol=1e-6,
        ), f"post-step state buffer {key} diverged from Python canonical"


@pytest.mark.unit
def test_ema_e_accepts_tensor_and_float_identically():
    """Contract fix: live callers hold ema_e as an nn buffer (1-element tensor);
    the kernel previously demanded a bare f32 and forced a .item() device sync."""
    B, H, T_keys, Tq, K = 1, 1, 4, 1, 2
    presyn = _make_presyn(seed=3)
    kernel = _rust_kernel()
    torch.manual_seed(5)
    drive = torch.rand(B, H, Tq, K)
    idx = torch.tensor([[[[0, 2]]]])
    valid = torch.ones(B, H, Tq, K, dtype=torch.bool)

    state_a = _clone_state(_seeded_state(presyn, seed=5, batch=B, heads=H, keys=T_keys))
    state_b = _clone_state(state_a)
    na, nb = _to_numpy_state(state_a), _to_numpy_state(state_b)

    out_tensor = kernel(drive.numpy(), idx.numpy(), valid.numpy(), na, presyn.cfg,
                        torch.tensor([presyn.ema_e.item()]))
    out_float = kernel(drive.numpy(), idx.numpy(), valid.numpy(), nb, presyn.cfg,
                       float(presyn.ema_e.item()))
    assert np.allclose(out_tensor[0], out_float[0])


@pytest.mark.unit
def test_eval_with_stochastic_train_frac_is_deterministic_and_accepted():
    """Contract fix: eval-mode decode is deterministic regardless of
    stochastic_train_frac (the Python canonical gates stochasticity on
    ``train or mc_sampling``), so the kernel must not refuse such configs."""
    B, H, T_keys, Tq, K = 1, 1, 4, 1, 2
    presyn = _make_presyn(seed=21)
    kernel = _rust_kernel()
    torch.manual_seed(9)
    drive = torch.rand(B, H, Tq, K)
    idx = torch.tensor([[[[0, 1]]]])
    valid = torch.ones(B, H, Tq, K, dtype=torch.bool)

    frac_cfg = SynapticConfig(stochastic_train_frac=0.7)
    state_a = _clone_state(_seeded_state(presyn, seed=9, batch=B, heads=H, keys=T_keys))
    state_b = _clone_state(state_a)
    na, nb = _to_numpy_state(state_a), _to_numpy_state(state_b)

    out_a = kernel(drive.numpy(), idx.numpy(), valid.numpy(), na, frac_cfg, 1.0)
    out_b = kernel(drive.numpy(), idx.numpy(), valid.numpy(), nb, frac_cfg, 1.0)
    assert np.allclose(out_a[0], out_b[0]), "eval decode must be deterministic"


@pytest.mark.unit
def test_metriplectic_and_learnable_configs_still_rejected():
    """The genuinely dynamic-altering modes keep their hard rejection."""
    kernel = _rust_kernel()
    for flag in ("metriplectic_integrator", "learnable_kinetics"):
        cfg = (
            SynapticConfig(metriplectic_integrator=True)
            if flag == "metriplectic_integrator"
            else SynapticConfig(learnable_kinetics=True)
        )
        with pytest.raises(ValueError, match="deterministic fixed-kinetics"):
            kernel(
                np.ones((1, 1, 1, 2), dtype=np.float32),
                np.zeros((1, 1, 1, 2), dtype=np.int64),
                np.ones((1, 1, 1, 2), dtype=bool),
                {k: np.zeros((1, 1, 4), dtype=np.float32) for k in _STATE_KEYS}
                | {"DELAY": [np.zeros((1, 1, 4), dtype=np.float32)]},
                cfg,
                1.0,
            )
