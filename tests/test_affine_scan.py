"""
Differentiable associative affine scan for the synaptic leaky integrators (bead yw9.2.1).

The presynaptic kinetics include affine-in-state leaky integrators — canonically the calcium
trace ``C_t = ρc·C_{t-1} + αca·softplus(drive_t)`` (and CL, E, RES, the conditionally-affine
RRP/PR; see docs/differentiable_synaptic_dynamics_design.md). `affine_scan` evaluates such a
recurrence as a Hillis-Steele associative prefix scan: `O(log T)` sequential depth instead of the
`O(T)` Python loop, and fully differentiable so SGD can learn the kinetics (the flagship yw9 epic).

These tests lock the acceptance: the parallel scan MATCHES the sequential reference to tight
tolerance (any T, batched, with/without an initial state) and `torch.autograd.gradcheck` passes.

Run:  pytest tests/test_affine_scan.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from bio_inspired_nanochat.synaptic import affine_scan, affine_scan_sequential


def _rand_decay(shape, lo=0.05, hi=0.95):
    # affine coefficients in (0,1): the leaky-integrator regime (contractive, stable).
    return torch.rand(shape, dtype=torch.float64) * (hi - lo) + lo


# --------------------------------------------------------------------------- #
# 1. EQUIVALENCE to the sequential reference (any T, shape, x0)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("T", [1, 2, 3, 5, 8, 16, 17, 33, 64])
@pytest.mark.parametrize("shape", [(), (4,), (3, 5)])
def test_parallel_scan_matches_sequential(T, shape):
    torch.manual_seed(T + len(shape))
    a = _rand_decay((T, *shape))
    b = torch.randn((T, *shape), dtype=torch.float64)
    x0 = torch.randn(shape, dtype=torch.float64)
    seq = affine_scan_sequential(a, b, x0)
    par = affine_scan(a, b, x0)
    assert torch.allclose(seq, par, atol=1e-11, rtol=1e-9), f"T={T} shape={shape}"


@pytest.mark.unit
def test_scan_without_initial_state_defaults_to_zero():
    torch.manual_seed(1)
    a = _rand_decay((10, 2))
    b = torch.randn((10, 2), dtype=torch.float64)
    no_x0 = affine_scan(a, b)
    zero_x0 = affine_scan(a, b, torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(no_x0, zero_x0, atol=1e-12)
    assert torch.allclose(no_x0, affine_scan_sequential(a, b), atol=1e-11)


# --------------------------------------------------------------------------- #
# 2. DIFFERENTIABILITY (gradcheck, double precision)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_gradcheck_wrt_a_b_x0():
    torch.manual_seed(2)
    a = _rand_decay((7, 2)).requires_grad_(True)
    b = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    x0 = torch.randn(2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b, x0: affine_scan(a, b, x0).sum(), (a, b, x0), eps=1e-6, atol=1e-6
    )


# --------------------------------------------------------------------------- #
# 3. THE CANONICAL USE: calcium leaky integrator, differentiable w.r.t. drive
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_calcium_leaky_integrator_scan_equivalence_and_grad():
    torch.manual_seed(3)
    rho_c = math.exp(-1.0 / 6.0)  # cfg.tau_c default
    alpha_ca = 0.55
    drive = torch.randn(20, 3, dtype=torch.float64, requires_grad=True)
    a = torch.full((20, 3), rho_c, dtype=torch.float64)

    def calcium(drive):
        b = alpha_ca * torch.nn.functional.softplus(drive)
        return affine_scan(a, b)

    seq = affine_scan_sequential(a, alpha_ca * torch.nn.functional.softplus(drive))
    assert torch.allclose(calcium(drive), seq, atol=1e-11), "calcium scan must match the reference"
    assert torch.autograd.gradcheck(calcium, (drive,), eps=1e-6, atol=1e-6), "grad must flow to drive"


# --------------------------------------------------------------------------- #
# 4. STABILITY: contractive decays keep the scan bounded & finite over long T
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_contractive_scan_is_bounded_over_long_sequence():
    torch.manual_seed(4)
    T = 1024
    a = _rand_decay((T, 8), lo=0.1, hi=0.99)  # all |a|<1
    b = torch.randn((T, 8), dtype=torch.float64)
    out = affine_scan(a, b)
    assert torch.isfinite(out).all()
    # bounded by the geometric-series envelope sup|b| / (1 - sup a)
    bound = b.abs().max() / (1.0 - 0.99) + 1.0
    assert out.abs().max() <= bound


# --------------------------------------------------------------------------- #
# 5. float32 parity (the model trains in fp32/bf16) — looser tolerance
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_float32_matches_reference():
    torch.manual_seed(5)
    a = (torch.rand(50, 4) * 0.9 + 0.05)
    b = torch.randn(50, 4)
    x0 = torch.randn(4)
    assert torch.allclose(affine_scan(a, b, x0), affine_scan_sequential(a, b, x0), atol=1e-4, rtol=1e-4)
