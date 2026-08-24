"""
Eval-time plasticity isolation — bead 9mxi.

Contract locked here:

1. EVAL NEVER ADAPTS. The validation path (``model.eval()`` + ``torch.no_grad()`` +
   ``train_mode=False`` — exactly what ``evaluate_bpb`` exercises through the
   ``base_train`` syn_forward_wrapper) must leave every plasticity tensor
   bit-identical: ``w_fast``/``w_slow``, the rank-R eligibility traces
   ``u_buf``/``v_buf``, and the whole postsynaptic gate state
   (``post.fast/slow/camkii/pp1/bdnf``). Otherwise validation data adapts the
   model and val_bpb drifts DURING the eval loop itself.
2. EVAL IS IDEMPOTENT. Two consecutive ``evaluate_bpb`` passes over identical
   batches return the identical number.
3. INFERENCE ADAPTATION SURVIVES. A ``no_grad`` forward with the default
   ``update_mem=True`` (the generation path) still writes traces + fast weights.
   That is the intentional "living fluid" working-memory feature — this test
   guards against over-tightening the gate into option-(b)-style semantics.
4. TRAIN-TIME PLASTICITY SURVIVES (vg9.2 contract): a grad-enabled training-mode
   forward still arms the deferred Hebbian Parameter write, threaded through
   Block -> MLP/MoE -> Expert -> SynapticLinear.

Run:  uv run python -m pytest tests/test_eval_plasticity_isolation.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bio_inspired_nanochat.loss_eval import evaluate_bpb
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear

from _bio_testkit import make_tiny_synaptic, random_tokens, set_seed


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _syn_cfg(**overrides) -> SynapticConfig:
    return SynapticConfig(**overrides)

def _plasticity_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone every persistent tensor of the model: all buffers + parameters
    everywhere (covers presyn ``ema_e``, MoE fatigue/energy, postsyn gate state,
    Hebbian params), plus the per-SynapticLinear deferred-write bookkeeping."""
    snap: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, SynapticLinear):
            snap[f"{name}._plasticity_pending"] = torch.tensor(float(m._plasticity_pending))
    for n, v in model.named_buffers():
        snap[f"buf:{n}"] = v.detach().clone()
    for n, p in model.named_parameters():
        snap[f"par:{n}"] = p.detach().clone()
    return snap


def _assert_untouched(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> None:
    assert set(before) == set(after), "snapshot key sets differ"
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert not changed, f"plasticity state mutated during eval: {changed}"


def _eval_batches(vocab: int = 97, n: int = 3):
    xs = [random_tokens(batch=2, seq=16, vocab=vocab, seed=i) for i in range(n)]
    ys = [random_tokens(batch=2, seq=16, vocab=vocab, seed=100 + i) for i in range(n)]
    return list(zip(xs, ys))


def _syn_eval_wrapper(model):
    """Minimal replica of base_train.py's syn_forward_wrapper: accepts the
    vanilla-GPT-style loss_reduction kwarg and forces train_mode=False."""
    orig_forward = model.forward

    def wrapper(idx, targets=None, kv_cache=None, loss_reduction="mean", **kwargs):
        if targets is not None:
            logits, _ = orig_forward(idx, targets, kv_cache, train_mode=False)
            if loss_reduction == "none":
                loss_per_token = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                    ignore_index=-1,
                )
                return loss_per_token.view(targets.shape)
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        logits, _ = orig_forward(idx, None, kv_cache, train_mode=False)
        return logits

    return wrapper


# --------------------------------------------------------------------------- #
# 1. Eval forward mutates nothing (dense AND MoE stacks)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("use_moe", [False, True])
def test_eval_forward_leaves_plasticity_untouched(use_moe):
    set_seed(0)
    model = make_tiny_synaptic(seed=0, syn_cfg=_syn_cfg(), use_moe=use_moe)
    model.eval()
    x = random_tokens(batch=2, seq=16)
    y = random_tokens(batch=2, seq=16, seed=7)

    before = _plasticity_snapshot(model)
    with torch.no_grad():
        model(x, y, None, train_mode=False)
    _assert_untouched(before, _plasticity_snapshot(model))

    # And it is repeatable: same input, same result — eval did not shift the substrate.
    with torch.no_grad():
        logits_a, loss_a = model(x, y, None, train_mode=False)
        logits_b, loss_b = model(x, y, None, train_mode=False)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(loss_a, loss_b)


# --------------------------------------------------------------------------- #
# 2. evaluate_bpb (via the base_train-style wrapper) is non-mutating + idempotent
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_evaluate_bpb_idempotent_and_non_mutating():
    set_seed(0)
    model = make_tiny_synaptic(seed=0, syn_cfg=_syn_cfg())
    model.eval()

    token_bytes = torch.ones(97, dtype=torch.int64)
    batches = _eval_batches()

    model.forward = _syn_eval_wrapper(model)
    try:
        before = _plasticity_snapshot(model)
        bpb1 = evaluate_bpb(model, batches, steps=len(batches), token_bytes=token_bytes)
        mid = _plasticity_snapshot(model)
        bpb2 = evaluate_bpb(model, batches, steps=len(batches), token_bytes=token_bytes)
        after = _plasticity_snapshot(model)
    finally:
        # undo the monkeypatch (model.forward is an instance attribute assignment)
        del model.__dict__["forward"]

    assert bpb1 == bpb2, f"evaluate_bpb drifted across identical passes: {bpb1} vs {bpb2}"
    _assert_untouched(before, mid)
    _assert_untouched(mid, after)


# --------------------------------------------------------------------------- #
# 3. Inference-time adaptation (default update_mem=True) still runs
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("use_moe", [False, True])
def test_inference_adaptation_still_runs(use_moe):
    set_seed(0)
    model = make_tiny_synaptic(seed=0, syn_cfg=_syn_cfg(), use_moe=use_moe)
    model.eval()
    x = random_tokens(batch=2, seq=16)
    y = random_tokens(batch=2, seq=16, seed=7)

    before = _plasticity_snapshot(model)
    with torch.no_grad():
        model(x, y, None, train_mode=True)  # generation-style: plasticity ON by default
    after = _plasticity_snapshot(model)

    moved = [
        k
        for k in before
        if k.endswith(("u_buf", "v_buf")) and not torch.equal(before[k], after[k])
    ]
    assert moved, (
        "no_grad forward with update_mem=True must still adapt eligibility traces "
        "(the working-memory feature); the 9mxi fix over-tightened the gate"
    )


# --------------------------------------------------------------------------- #
# 4. Train-time plasticity still armed through the new plumbing (vg9.2)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("use_moe", [False, True])
def test_train_time_plasticity_still_runs(use_moe):
    set_seed(0)
    model = make_tiny_synaptic(seed=0, syn_cfg=_syn_cfg(), use_moe=use_moe, train=True)
    x = random_tokens(batch=2, seq=16)
    y = random_tokens(batch=2, seq=16, seed=7)

    _, loss = model(x, y, None, train_mode=True)
    loss.backward()

    lins = [m for m in model.modules() if isinstance(m, SynapticLinear)]
    assert any(m._plasticity_pending for m in lins), (
        "grad-enabled training forward must arm the deferred Hebbian write "
        "(update_mem threading must not break vg9.2)"
    )

    # Second forward flushes the deferred Parameter writes at its top.
    before_flush = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        model(x, y, None, train_mode=True)
    flushed = [
        n
        for n, p in model.named_parameters()
        if ("w_fast" in n or "w_slow" in n) and not torch.equal(before_flush[n], p)
    ]
    assert flushed, "deferred plasticity write never landed on the next forward"
