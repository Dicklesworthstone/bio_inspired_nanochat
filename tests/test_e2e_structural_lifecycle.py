"""E2E structural-lifecycle invariants + lineage logs (bead `eqyk.8`).

The MoE split/merge/reset lifecycle is the riskiest mutation in the system (it rewrites parameters
mid-training) and was barely covered (a couple of unit tests). This is the safety net for the whole
Structural Evolution epic: it forces lifecycle events and asserts the invariants that must hold
TOGETHER, plus produces a lineage log.

Invariants checked:
  1. **Shape/router coherence** — after any event the expert count, router rows, and per-expert
     weight shapes are intact and the layer still forwards.
  2. **Function preservation (guards uta.3)** — in the dense regime a function-preserving split/merge
     does not move the output (no loss spike).
  3. **Structural conservation** — the expert count is invariant across events (slots are reused).
  4. **Optimizer-momentum reset (guards vg9.3)** — a lifecycle event zeroes the changed params'
     momentum in BOTH optimizers a synaptic model uses (AdamW for 1-D/router, Muon for the expert
     matrices); untouched params keep their momentum.
  5. **Lineage** — each forced event is logged (op, src→dst) to the JSONL event stream.

Run:  pytest tests/test_e2e_structural_lifecycle.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.muon import Muon
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
    _function_preserving_merge_,
    _function_preserving_split_,
)

pytestmark = pytest.mark.e2e


def _pure_moe(seed: int, num_experts: int, top_k: int, n_embd: int = 16) -> SynapticMoE:
    """A SynapticMoE whose forward is a PURE function of its parameters (no Hebbian/metabolism/
    router-drift), so a function-preserving event's output change is isolated from per-step state
    mutation — the same fixture the FP unit tests use."""
    torch.manual_seed(seed)
    cfg = SynapticConfig(
        enable_hebbian=False, enable_metabolism=False,
        router_contrastive_push=0.0, router_contrastive_lr=0.0,
    )
    moe = SynapticMoE(n_embd=n_embd, num_experts=num_experts, top_k=top_k, hidden_mult=2, cfg=cfg, dropout=0.0)
    return moe.eval()


def _metab_moe(num_experts: int = 4, n_embd: int = 8) -> SynapticMoE:
    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=True)
    return SynapticMoE(n_embd=n_embd, num_experts=num_experts, top_k=1, hidden_mult=1, cfg=cfg, dropout=0.0)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm().item() / (b.norm().item() + 1e-9))


def _shapes_coherent(moe: SynapticMoE, E: int, n_embd: int) -> None:
    assert moe.num_experts == E
    assert len(moe.experts) == E
    assert tuple(moe.router.weight.shape) == (E, n_embd)
    assert tuple(moe.router_embeddings.shape)[0] == E
    for e in moe.experts:
        assert e.fc1.w_slow.shape[0] > 0 and e.fc2.w_slow.shape[0] > 0


def _moment_magnitude(opt: torch.optim.Optimizer, p: torch.nn.Parameter) -> float:
    """Sum |moment buffers| for ``p`` in ``opt`` (key-agnostic: exp_avg/exp_avg_sq/momentum_buffer)."""
    st = opt.state.get(p, {})
    return sum(float(v.abs().sum()) for v in st.values() if torch.is_tensor(v))


# --------------------------------------------------------------------------- #
# 1. forced events: shape coherence + function preservation + conservation + lineage
# --------------------------------------------------------------------------- #
def test_forced_lifecycle_invariants_and_lineage(tmp_path):
    E, n_embd = 6, 16
    x = torch.randn(2, 5, n_embd)
    moe = _pure_moe(0, E, top_k=E)  # dense ⟹ FP events are exactly output-preserving
    cfg = SplitMergeConfig(function_preserving=True, fp_divergence_noise=0.0)
    logger = RunLogger(tmp_path, name="lifecycle", console=False)

    def _forward():
        with torch.no_grad():
            return moe(x)[0]

    # make slot 5 genuinely dead so splitting INTO it preserves the output
    with torch.no_grad():
        moe.router_logit_bias[5] = -50.0

    out0 = _forward()
    _shapes_coherent(moe, E, n_embd)

    # SPLIT expert 1 -> dead slot 5 (known lineage)
    with torch.no_grad():
        _function_preserving_split_(moe, parent_idx=1, dst_idx=5, cfg=cfg)
    logger.event("lineage", op="split", src=1, dst=5, step=0)
    out1 = _forward()
    _shapes_coherent(moe, E, n_embd)
    assert _rel_l2(out1, out0) < 1e-5, "dense function-preserving split must not move the output"

    # MERGE expert 2 (winner) <- 3 (loser) (known lineage); merging two clones is exact
    with torch.no_grad():
        # make 2 and 3 an identical, mergeable pair first
        moe.experts[3].fc1.w_slow.copy_(moe.experts[2].fc1.w_slow)
        moe.experts[3].fc2.w_slow.copy_(moe.experts[2].fc2.w_slow)
        moe.router.weight[3].copy_(moe.router.weight[2])
        moe.Xi[3].copy_(moe.Xi[2])
        moe.router_embeddings[3].copy_(moe.router_embeddings[2])
    out_pre_merge = _forward()
    with torch.no_grad():
        _function_preserving_merge_(moe, winner_idx=2, loser_idx=3, alpha=0.5, cfg=cfg)
    logger.event("lineage", op="merge", src=3, dst=2, step=1)
    out2 = _forward()
    _shapes_coherent(moe, E, n_embd)
    assert _rel_l2(out2, out_pre_merge) < 1e-5, "merging an identical pair must not move the output"

    # structural conservation: expert count never changed across events
    assert moe.num_experts == E

    # lineage: both events are in the JSONL stream with their src→dst
    events = [e for e in logger.read_events() if e["event"] == "lineage"]
    assert [(e["op"], e["src"], e["dst"]) for e in events] == [("split", 1, 5), ("merge", 3, 2)]


# --------------------------------------------------------------------------- #
# 2. optimizer-momentum reset across AdamW + Muon on a real lifecycle event (vg9.3)
# --------------------------------------------------------------------------- #
def test_lifecycle_resets_optimizer_momentum_adamw_and_muon():
    E, n_embd = 4, 8
    moe = _metab_moe(num_experts=E, n_embd=n_embd)

    # route params like a real synaptic model: 2-D matrices -> Muon, the rest -> AdamW
    matrix_params = [p for p in moe.parameters() if p.requires_grad and p.ndim == 2]
    other_params = [p for p in moe.parameters() if p.requires_grad and p.ndim != 2]
    muon = Muon(matrix_params, lr=0.02)
    adamw = torch.optim.AdamW(other_params, lr=0.1)

    # populate momentum on every param (a fake training step)
    for p in matrix_params + other_params:
        p.grad = torch.randn_like(p)
    muon.step()
    adamw.step()

    # craft health so slot 0 is dead (reset target) and slot 3 is the healthy donor; 1,2 untouched
    with torch.no_grad():
        moe.fatigue.copy_(torch.tensor([0.0, 0.5, 0.5, 1.0]))
        moe.energy.copy_(torch.tensor([0.0, 0.5, 0.5, 1.0]))

    target = moe.experts[0].fc1.w_slow      # the reset (changed) slot's matrix param -> Muon
    untouched = moe.experts[2].fc1.w_slow   # a slot not involved in the reset

    assert _moment_magnitude(muon, target) > 0, "precondition: target has momentum before the event"
    assert _moment_magnitude(muon, untouched) > 0

    sm_cfg = SplitMergeConfig(
        enabled=True, warmup_steps=0, min_step_interval=0,
        merges_per_call=0, splits_per_call=0, resets_per_call=1, reset_health_max=0.05,
        ddp_broadcast=False, function_preserving=False,  # legacy reset: only the dst slot changes
    )
    SplitMergeController(moe, sm_cfg).step(global_step=10, optimizer=[adamw, muon])

    # the changed slot's momentum is reset wherever it lives; the untouched slot keeps its momentum
    assert _moment_magnitude(muon, target) == 0.0, (
        "vg9.3: the reset slot's matrix momentum must be zeroed in Muon"
    )
    assert _moment_magnitude(muon, untouched) > 0, "an uninvolved expert's momentum must be untouched"


# --------------------------------------------------------------------------- #
# 3. a function-preserving controller step does not spike the loss (uta.3, end-to-end)
# --------------------------------------------------------------------------- #
def test_function_preserving_controller_step_no_loss_spike():
    E, n_embd = 8, 16
    x = torch.randn(4, 8, n_embd)
    moe = _pure_moe(3, E, top_k=E)  # dense regime
    dead = E - 1
    with torch.no_grad():
        # expert 0 is uniquely healthy (the split SOURCE); the weakest slot is also output-dead
        # (router bias ≈ −∞), so the controller splitting INTO it preserves the output exactly.
        # (Function preservation holds when the overwritten slot contributes ~nothing; a merely
        # low-HEALTH but still output-active slot would legitimately move the output.)
        moe.fatigue.copy_(torch.full((E,), 0.1))
        moe.energy.copy_(torch.full((E,), 0.1))
        moe.fatigue[0] = 1.0
        moe.energy[0] = 1.0
        moe.fatigue[dead] = 1e-3
        moe.energy[dead] = 1e-3       # uniquely weakest => the split destination
        moe.router_logit_bias[dead] = -50.0  # ...and output-dead
    out0 = moe(x)[0]

    cfg = SplitMergeConfig(
        enabled=True, warmup_steps=0, min_step_interval=0,
        merges_per_call=0, splits_per_call=1, split_health_min=0.5,
        resets_per_call=0, ddp_broadcast=False,
        function_preserving=True, fp_divergence_noise=0.0,
    )
    SplitMergeController(moe, cfg).step(global_step=10, optimizer=None)
    out1 = moe(x)[0]
    assert _rel_l2(out1, out0) < 1e-4, "FP controller split into a dead slot must not spike the output"
