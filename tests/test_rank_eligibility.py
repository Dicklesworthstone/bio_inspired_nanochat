"""
Genuine rank-R eligibility traces — bead vg9.9.

The "rank-R" eligibility used to broadcast the mean activity across all R ranks, making every
column identical -> effectively rank 1, so `rank_eligibility` was a no-op knob. The fix uses
FIXED random projections so each rank channel captures a distinct mode of the pre/post
correlation. These tests lock: the trace has genuine rank > 1 for R>1, rank_eligibility actually
controls the rank, the projections are fixed + persisted, and traces update during training.

Run:  pytest tests/test_rank_eligibility.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear

from _bio_testkit import set_seed

IN, OUT, B = 16, 16, 8


def _drive_traces(*, R=4, steps=3, seed=0) -> SynapticLinear:
    """Run a few inference forwards so the eligibility traces accumulate (eval+no_grad takes the
    immediate plasticity path), then return the layer."""
    set_seed(seed)
    cfg = SynapticConfig(enable_hebbian=True, rank_eligibility=R)
    lin = SynapticLinear(IN, OUT, cfg).eval()
    ca, en = torch.ones(B), torch.ones(B)
    with torch.no_grad():
        for _ in range(steps):
            lin(torch.randn(B, IN), ca, en)
    return lin


@pytest.mark.unit
def test_eligibility_trace_has_genuine_rank_greater_than_one():
    lin = _drive_traces(R=4)
    rank_u = torch.linalg.matrix_rank(lin.u_buf).item()
    rank_v = torch.linalg.matrix_rank(lin.v_buf).item()
    assert rank_u > 1, f"u_buf must be genuinely rank-R (>1), got rank {rank_u}"
    assert rank_v > 1, f"v_buf must be genuinely rank-R (>1), got rank {rank_v}"
    # the low-rank Hebbian delta consumed by the weight update is genuinely rank > 1 too
    delta = lin.u_buf @ lin.v_buf
    assert torch.linalg.matrix_rank(delta).item() > 1, "delta = u_buf @ v_buf must have real rank"


@pytest.mark.unit
def test_rank_eligibility_knob_is_meaningful():
    # R=1 -> rank 1; R=4 -> rank > 1. (Before vg9.9 every R gave rank 1.)
    assert torch.linalg.matrix_rank(_drive_traces(R=1).u_buf).item() == 1
    rank4 = torch.linalg.matrix_rank(_drive_traces(R=4).u_buf).item()
    assert rank4 > 1, f"R=4 must give rank > 1, got {rank4}"


@pytest.mark.unit
def test_higher_R_gives_higher_or_equal_trace_rank():
    r2 = torch.linalg.matrix_rank(_drive_traces(R=2).u_buf).item()
    r8 = torch.linalg.matrix_rank(_drive_traces(R=8).u_buf).item()
    assert r8 >= r2, f"more rank channels should not reduce the achievable trace rank ({r8} < {r2})"
    assert r8 > 1


@pytest.mark.unit
def test_projections_are_fixed_across_forwards():
    lin = _drive_traces(R=4)
    p_in, p_out = lin.proj_in.clone(), lin.proj_out.clone()
    with torch.no_grad():
        lin(torch.randn(B, IN), torch.ones(B), torch.ones(B))
    assert torch.equal(lin.proj_in, p_in), "random projections must be FIXED (not updated)"
    assert torch.equal(lin.proj_out, p_out)


@pytest.mark.unit
def test_projections_persist_in_state_dict():
    sd = _drive_traces(R=4).state_dict()
    assert "proj_in" in sd and "proj_out" in sd, "projections must be buffers (persist in checkpoints)"


@pytest.mark.unit
def test_traces_update_during_training_and_stay_finite():
    set_seed(0)
    cfg = SynapticConfig(enable_hebbian=True, rank_eligibility=4)
    lin = SynapticLinear(IN, OUT, cfg).train()
    u0 = lin.u_buf.clone()
    lin(torch.randn(B, IN, requires_grad=True), torch.ones(B), torch.ones(B))  # grad enabled
    assert not torch.equal(lin.u_buf, u0), "eligibility traces must update during training"
    assert torch.isfinite(lin.u_buf).all() and torch.isfinite(lin.v_buf).all()
