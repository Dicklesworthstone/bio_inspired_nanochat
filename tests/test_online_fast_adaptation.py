"""
Online fast-weight adaptation — the "fast-weight programmer" inner loop (bead sax.1).

Context. vg9.2 made online Hebbian plasticity RUN during training (autograd-safe deferred
writes); vg9.9 restored genuine rank-R eligibility traces; vg9.4 added per-sequence reset.
sax.1 assembles those pieces into the online-learning inner loop and characterizes it honestly.

Three facts these tests lock:

1. THE HISTORICAL GAP. With `fast_weight_normalized=False`, the raw rank-R Hebbian fast-weight
   write is O(trace²) and normally negligible — but on sparse MoE trajectories it can enter
   positive feedback and diverge. This unsafe legacy path remains only as a negative control.

2. THE FOUNDATIONAL FIX (`fast_weight_normalized=True`, now the default). The write is taken along the
   unit-norm Hebbian direction with an O(1) `fast_weight_eta` and `||w_fast||` is bounded by
   `fast_weight_max_norm`, so the update is BOTH impactful (|Δw_fast| ~ O(eta)) AND stable
   (bounded, finite over many passes). This is the prerequisite for any downstream consolidation
   signal (e.g. three-factor reward-modulated Hebbian, hy8.2) to actually write fast memory.

3. THE PRODUCTION DEFAULT is the normalized path. The raw control caused the all-bio Synaptic-MoE
   training divergence tracked by jpqc; the bounded path is stable on that exact multi-seed repro.

What these tests do NOT claim: that unsupervised adaptation IMPROVES next-token prediction on an
untrained model. It does not — see docs/online_learning_status.md for the measured specificity
(adapting on a pattern does not lower its loss vs a novel pattern). That behavioral payoff needs
a learning signal (hy8.2) and/or a trained-model e2e regime (eqyk.9).

Run:  pytest tests/test_online_fast_adaptation.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear

from _bio_testkit import make_tiny_synaptic, set_seed

IN, OUT, B = 16, 16, 4


def _make_lin(**cfg_overrides) -> SynapticLinear:
    set_seed(0)
    cfg = SynapticConfig(enable_hebbian=True, enable_metabolism=True, **cfg_overrides)
    return SynapticLinear(IN, OUT, cfg)


def _signals():
    return torch.ones(B), torch.ones(B)  # calcium, energy (per-row)


def _adapt_passes(lin: SynapticLinear, x: torch.Tensor, ca, en, n: int) -> None:
    """Run n inference adaptation passes (plasticity applied immediately under no_grad)."""
    lin.eval()
    with torch.no_grad():
        for _ in range(n):
            lin(x, ca, en)


# --------------------------------------------------------------------------- #
# 1. THE GAP: the legacy raw fast-weight write is numerically negligible
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_legacy_fast_weight_write_is_negligible():
    lin = _make_lin(fast_weight_normalized=False)
    lin.reset_sequence_state(reset_fast_weights=True)  # w_fast := 0
    x = torch.randn(B, IN)
    ca, en = _signals()
    _adapt_passes(lin, x, ca, en, 8)
    # The raw rank-R write barely moves w_fast off zero — this is the documented limitation.
    assert lin.w_fast.norm().item() < 1e-3, (
        "legacy un-normalized online write is expected to be numerically negligible; "
        f"got ||w_fast||={lin.w_fast.norm().item():.3e}"
    )


# --------------------------------------------------------------------------- #
# 2. THE FIX: normalized write adapts meaningfully AND stays bounded/finite
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_normalized_write_adapts_meaningfully():
    lin = _make_lin(fast_weight_normalized=True, fast_weight_eta=0.5, fast_weight_max_norm=1.0)
    lin.reset_sequence_state(reset_fast_weights=True)
    x = torch.randn(B, IN)
    ca, en = _signals()
    _adapt_passes(lin, x, ca, en, 6)
    n = lin.w_fast.norm().item()
    # Impactful: w_fast is driven to an O(1) magnitude, not ~0.
    assert n > 0.1, f"normalized online write must move w_fast meaningfully; got ||w_fast||={n:.3e}"


@pytest.mark.unit
def test_normalized_write_is_bounded_and_finite_over_many_passes():
    lin = _make_lin(fast_weight_normalized=True, fast_weight_eta=0.5, fast_weight_max_norm=1.0)
    lin.reset_sequence_state(reset_fast_weights=True)
    x = torch.randn(B, IN)
    ca, en = _signals()
    _adapt_passes(lin, x, ca, en, 200)  # many passes — naive LR boost would NaN here
    assert torch.isfinite(lin.w_fast).all(), "w_fast must stay finite (no positive-feedback blowup)"
    assert lin.w_fast.norm().item() <= 1.0 + 1e-5, "||w_fast|| must respect fast_weight_max_norm"


# --------------------------------------------------------------------------- #
# 3. THE STABLE NORMALIZED WRITE IS THE DEFAULT
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_normalized_write_is_the_default():
    assert SynapticConfig().fast_weight_normalized is True


@pytest.mark.unit
def test_flag_off_preserves_deterministic_legacy_write():
    x = torch.randn(B, IN)
    ca, en = _signals()

    lin_legacy = _make_lin(fast_weight_normalized=False)
    lin_legacy.reset_sequence_state(reset_fast_weights=True)
    _adapt_passes(lin_legacy, x, ca, en, 4)

    # Same seed/inputs, normalization explicitly disabled -> identical w_fast trajectory.
    lin_ref = _make_lin(fast_weight_normalized=False)
    lin_ref.reset_sequence_state(reset_fast_weights=True)
    _adapt_passes(lin_ref, x, ca, en, 4)

    assert torch.equal(lin_legacy.w_fast, lin_ref.w_fast)


# --------------------------------------------------------------------------- #
# 4. PER-SEQUENCE RESET clears the online adaptation (vg9.4 contract, model level)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_reset_sequence_state_clears_online_adaptation():
    lin = _make_lin(fast_weight_normalized=True)
    lin.reset_sequence_state(reset_fast_weights=True)
    x = torch.randn(B, IN)
    ca, en = _signals()
    _adapt_passes(lin, x, ca, en, 5)
    assert lin.w_fast.norm().item() > 0.1, "precondition: adaptation happened"
    assert lin.u_buf.abs().sum().item() > 0, "precondition: eligibility trace accumulated"

    lin.reset_sequence_state(reset_fast_weights=True)
    assert lin.w_fast.norm().item() == 0.0, "per-sequence reset must clear fast weights"
    assert lin.u_buf.abs().sum().item() == 0.0, "per-sequence reset must clear eligibility traces"


# --------------------------------------------------------------------------- #
# 5. GRAD-SAFE: normalized online write does not break training-time autograd
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_normalized_write_is_grad_safe_during_training():
    lin = _make_lin(fast_weight_normalized=True).train()
    ca, en = _signals()
    opt = torch.optim.SGD(lin.parameters(), lr=1e-2)
    for _ in range(4):
        x = torch.randn(B, IN, requires_grad=True)
        y = lin(x, ca, en)
        loss = y.float().pow(2).mean()
        loss.backward()  # MUST NOT raise from in-place fast-weight mutation
        assert torch.isfinite(loss)
    opt.step()
    assert lin._plasticity_pending is True, "online plasticity must have run with grad enabled"


# --------------------------------------------------------------------------- #
# 6. MODEL LEVEL: fast weights adapt within a sequence and reset across sequences
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_model_fast_weights_adapt_within_sequence_and_reset_between():
    model = make_tiny_synaptic(seed=0, train=False)
    for m in model.modules():
        cfg = getattr(m, "cfg", None)
        if cfg is not None and hasattr(cfg, "fast_weight_normalized"):
            cfg.fast_weight_normalized = True
    lins = [m for m in model.modules() if isinstance(m, SynapticLinear) and m.w_fast is not None]
    assert lins, "tiny synaptic model must have fast-weight layers"

    V = model.config.vocab_size
    g = torch.Generator().manual_seed(1)
    seq = torch.randint(0, V, (1, 16), generator=g)

    model.reset_sequence_state(reset_fast_weights=True)
    with torch.no_grad():
        for _ in range(4):
            model(seq)
    adapted = max(lin.w_fast.norm().item() for lin in lins)
    assert adapted > 0.05, "fast weights must adapt within a sequence"

    n = model.reset_sequence_state(reset_fast_weights=True)
    assert n == len(lins)
    assert all(lin.w_fast.norm().item() == 0.0 for lin in lins), "reset must clear all fast weights"
