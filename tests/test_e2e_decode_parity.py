"""E2E inference/decode parity — KV-cache + presyn-state (bead `eqyk.5`).

The audit found NO committed synaptic decode-parity test; presyn-state forking is "best-effort".
This locks decode correctness for the **bio path** (not just vanilla):

  1. **Incremental == contiguous** — prefilling a prompt then decoding token-by-token through the
     KV-cache produces the same per-position logits AND the same synaptic-state (calcium/RRP/…)
     evolution as a single contiguous forward over the whole sequence. (The parity gap the audit
     flagged as untested.)
  2. **Fork preserves state** — replicating a batch-1 prefill cache to a batch-N decode cache
     (`KVCache.prefill`, the engine's `generate` fork) broadcasts the presyn state to every row.
  3. **Determinism** — `Engine.generate` with a fixed seed yields identical tokens across runs
     (reproducible sampling: the only nondeterminism in this pure-forward model is the sampler,
     which must be driven by the seeded generator, not global RNG state).
  4. **Per-step logging** — bio-state is logged per decode step (the `eqyk.2` JSONL stream).

Run:  pytest tests/test_e2e_decode_parity.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import torch
import pytest

from bio_inspired_nanochat.engine import Engine, KVCache
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.mc_ensemble import mc_sampling
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    _release_recurrence_group,
    build_presyn_state,
)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _model(
    seed: int = 0,
    *,
    n_layer: int = 2,
    n_head: int = 2,
    n_kv_head: int = 2,
    n_embd: int = 32,
    attn_topk: int = 32,
    stochastic_train_frac: float = 0.12,
    learnable_kinetics: bool = False,
    differentiable_recurrence: bool = False,
    recurrence_checkpoint_len: int = 0,
    metriplectic_integrator: bool = False,
) -> tuple[GPTSynaptic, GPTSynapticConfig]:
    """A GPTSynaptic whose forward is a PURE function of params + the per-sequence presyn state:
    the per-sequence calcium/RRP recurrence (enable_presyn) is ON — that is exactly the KV-cache
    state whose decode parity we test — but the module-state-mutating plasticity (Hebbian
    consolidation, metabolism, router-contrastive EMA) is OFF. With those on, inference mutates
    shared module state DIFFERENTLY under batched vs token-by-token processing, so naive
    incremental==contiguous parity cannot hold (the documented "inference runs plasticity"
    behavior); isolating it here keeps this a clean test of the decode machinery itself.
    """
    torch.manual_seed(seed)
    syn = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=False,
        enable_metabolism=False,
        router_contrastive_push=0.0,
        router_contrastive_lr=0.0,
        attn_topk=attn_topk,
        stochastic_train_frac=stochastic_train_frac,
        learnable_kinetics=learnable_kinetics,
        differentiable_recurrence=differentiable_recurrence,
        recurrence_checkpoint_len=recurrence_checkpoint_len,
        metriplectic_integrator=metriplectic_integrator,
    )
    cfg = GPTSynapticConfig(
        sequence_len=64,
        vocab_size=64,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        syn_cfg=syn,
    )
    return GPTSynaptic(cfg).eval(), cfg


def _kv(cfg: GPTSynapticConfig, batch: int, seqlen: int) -> KVCache:
    return KVCache(
        batch_size=batch,
        num_heads=cfg.n_kv_head,
        seq_len=seqlen,
        head_dim=cfg.n_embd // cfg.n_head,
        num_layers=cfg.n_layer,
    )


def _reset(model: GPTSynaptic) -> None:
    # full per-sequence reset so the two compared paths start from an identical clean scratchpad
    model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=True)


def _presyn_tensors(st: dict) -> dict[str, torch.Tensor]:
    return {k: v for k, v in st.items() if isinstance(v, torch.Tensor)}


def _assert_presyn_state_close(actual: dict, expected: dict, *, context: str) -> None:
    assert actual.keys() == expected.keys(), f"{context}: state schemas differ"
    for key in actual:
        left, right = actual[key], expected[key]
        if key == "DELAY":
            assert len(left) == len(right), f"{context}: DELAY lengths differ"
            for delay_index, (left_delay, right_delay) in enumerate(zip(left, right)):
                assert left_delay.shape == right_delay.shape
                assert torch.allclose(left_delay, right_delay, atol=1e-4, rtol=1e-4), (
                    f"{context}: DELAY[{delay_index}] differs"
                )
            continue
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            assert left.shape == right.shape, f"{context}: {key} shapes differ"
            assert torch.allclose(left, right, atol=1e-4, rtol=1e-4), (
                f"{context}: {key} differs"
            )
            continue
        assert left == right, f"{context}: {key} differs"


# --------------------------------------------------------------------------- #
# 0. baseline — VANILLA GPT decode is causal (the harness + KV-cache are correct)
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
def test_vanilla_decode_parity_is_causal():
    """A vanilla (non-synaptic) GPT: appending future tokens must NOT change earlier positions'
    logits (causality), so prefix-forward == full-forward on the shared positions. This proves the
    KV-cache + test harness are correct, isolating the synaptic-specific gap below."""
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=64, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32)
    model = GPT(cfg).eval()
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.vocab_size, (1, 11), dtype=torch.long)

    def _kvg(seqlen):
        return KVCache(batch_size=1, num_heads=cfg.n_kv_head, seq_len=seqlen,
                       head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer)

    with torch.no_grad():
        l6 = model.forward(tokens[:, :6], kv_cache=_kvg(11))
        l11 = model.forward(tokens[:, :11], kv_cache=_kvg(11))
    assert torch.allclose(l6, l11[:, :6, :], atol=1e-5), "vanilla GPT decode must be causal"


# --------------------------------------------------------------------------- #
# 1. incremental decode == contiguous forward (logits + every presyn-state component)
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
def test_decode_parity_incremental_vs_contiguous():
    model, cfg = _model(0)
    B, prompt_len, cont_len = 1, 6, 5
    total = prompt_len + cont_len
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.vocab_size, (B, total), dtype=torch.long)

    # (A) contiguous: one forward over the whole sequence
    _reset(model)
    kv_full = _kv(cfg, B, total)
    logits_full, _ = model(tokens, kv_cache=kv_full, train_mode=False)

    # (B) incremental: prefill prompt, then feed one token at a time
    _reset(model)
    kv_inc = _kv(cfg, B, total)
    model(tokens[:, :prompt_len], kv_cache=kv_inc, train_mode=False)
    step_logits = []
    for i in range(prompt_len, total):
        li, _ = model(tokens[:, i : i + 1], kv_cache=kv_inc, train_mode=False)
        step_logits.append(li[:, -1, :])

    # logits parity: feeding token i incrementally reproduces the contiguous logits at position i
    for j, i in enumerate(range(prompt_len, total)):
        assert torch.allclose(step_logits[j], logits_full[:, i, :], atol=1e-4, rtol=1e-4), (
            f"incremental vs contiguous logit mismatch at position {i}"
        )

    # presyn-state parity: same calcium/RRP/… evolution per layer after the same tokens
    assert isinstance(kv_full.presyn_state, list) and isinstance(kv_inc.presyn_state, list)
    assert len(kv_full.presyn_state) == len(kv_inc.presyn_state) == cfg.n_layer
    for layer, (sf, si) in enumerate(zip(kv_full.presyn_state, kv_inc.presyn_state)):
        _assert_presyn_state_close(sf, si, context=f"layer {layer}")


@pytest.mark.e2e
@pytest.mark.parametrize("train_mode", [False, True])
def test_appending_future_tokens_does_not_change_prefix_logits(train_mode: bool):
    """Both eval and deterministic adaptation mode must remain causally prefix-invariant."""
    short_model, cfg = _model(0, stochastic_train_frac=0.0)
    long_model, _ = _model(0, stochastic_train_frac=0.0)
    torch.manual_seed(11)
    tokens = torch.randint(0, cfg.vocab_size, (1, 11), dtype=torch.long)

    with torch.no_grad():
        short_logits, _ = short_model(
            tokens[:, :6], kv_cache=_kv(cfg, 1, 11), train_mode=train_mode
        )
        long_logits, _ = long_model(tokens, kv_cache=_kv(cfg, 1, 11), train_mode=train_mode)
    assert torch.allclose(short_logits, long_logits[:, :6], atol=1e-5, rtol=1e-5)


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("n_layer", "n_head", "n_kv_head", "attn_topk"),
    [(1, 4, 1, 1), (3, 4, 2, 64)],
)
def test_contiguous_matches_token_zero_incremental_across_gqa_and_topk_boundaries(
    n_layer: int, n_head: int, n_kv_head: int, attn_topk: int
):
    model, cfg = _model(
        3,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        attn_topk=attn_topk,
    )
    torch.manual_seed(13)
    tokens = torch.randint(0, cfg.vocab_size, (1, 7), dtype=torch.long)

    _reset(model)
    full_cache = _kv(cfg, 1, 7)
    full_logits, _ = model(tokens, kv_cache=full_cache, train_mode=False)

    _reset(model)
    incremental_cache = _kv(cfg, 1, 7)
    incremental_logits = []
    for position in range(tokens.size(1)):
        logits, _ = model(
            tokens[:, position : position + 1],
            kv_cache=incremental_cache,
            train_mode=False,
        )
        incremental_logits.append(logits)
    actual_logits = torch.cat(incremental_logits, dim=1)

    assert torch.allclose(actual_logits, full_logits, atol=1e-4, rtol=1e-4)
    assert isinstance(full_cache.presyn_state, list)
    assert isinstance(incremental_cache.presyn_state, list)
    for layer, (full_state, incremental_state) in enumerate(
        zip(full_cache.presyn_state, incremental_cache.presyn_state)
    ):
        _assert_presyn_state_close(full_state, incremental_state, context=f"layer {layer}")


@pytest.mark.e2e
def test_multi_token_append_matches_one_token_appends():
    model, cfg = _model(5)
    torch.manual_seed(17)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)

    _reset(model)
    chunk_cache = _kv(cfg, 1, 8)
    model(tokens[:, :4], kv_cache=chunk_cache, train_mode=False)
    chunk_logits, _ = model(tokens[:, 4:], kv_cache=chunk_cache, train_mode=False)

    _reset(model)
    step_cache = _kv(cfg, 1, 8)
    model(tokens[:, :4], kv_cache=step_cache, train_mode=False)
    step_logits = []
    for position in range(4, 8):
        logits, _ = model(
            tokens[:, position : position + 1], kv_cache=step_cache, train_mode=False
        )
        step_logits.append(logits)

    assert torch.allclose(torch.cat(step_logits, dim=1), chunk_logits, atol=1e-4, rtol=1e-4)
    assert isinstance(chunk_cache.presyn_state, list)
    assert isinstance(step_cache.presyn_state, list)
    for layer, (chunk_state, step_state) in enumerate(
        zip(chunk_cache.presyn_state, step_cache.presyn_state)
    ):
        _assert_presyn_state_close(chunk_state, step_state, context=f"layer {layer}")


@pytest.mark.unit
def test_inactive_future_state_slots_remain_at_initial_values():
    cfg = SynapticConfig(stochastic_train_frac=0.0)
    presyn = SynapticPresyn(d_head=8, cfg=cfg)
    state = build_presyn_state(1, 5, 2, torch.device("cpu"), torch.float32, cfg)
    before: dict[str, torch.Tensor | list[torch.Tensor]] = {
        key: [item.clone() for item in value] if key == "DELAY" else value.clone()
        for key, value in state.items()
    }
    drive = torch.tensor([[[[0.5, -0.25]], [[0.1, 0.2]]]])
    idx = torch.tensor([[[[0, 1]], [[1, 0]]]])

    presyn.release_canonical(
        state,
        drive,
        idx,
        train=False,
        active_key_count=2,
    )

    for key, value in state.items():
        if key == "DELAY":
            initial_value = before[key]
            assert isinstance(value, list) and isinstance(initial_value, list)
            for current, initial in zip(value, initial_value):
                assert torch.equal(current[..., 2:], initial[..., 2:])
        else:
            initial_value = before[key]
            assert isinstance(value, torch.Tensor) and isinstance(initial_value, torch.Tensor)
            assert torch.equal(value[..., 2:], initial_value[..., 2:]), key


@pytest.mark.unit
def test_scripted_cpu_scan_matches_stochastic_reference_values_state_and_gradients():
    cfg = SynapticConfig(attn_topk=8, stochastic_train_frac=0.12)
    scripted_presyn = SynapticPresyn(d_head=8, cfg=cfg)
    reference_presyn = SynapticPresyn(d_head=8, cfg=cfg)
    batch, heads, tokens, topk = 2, 3, 12, 8
    initial = build_presyn_state(
        batch, tokens, heads, torch.device("cpu"), torch.float32, cfg
    )

    def clone_state():
        return {
            key: [item.clone() for item in value]
            if isinstance(value, list)
            else value.clone()
            for key, value in initial.items()
        }

    scripted_state = clone_state()
    reference_state = clone_state()
    generator = torch.Generator().manual_seed(101)
    drive = torch.randn(batch, heads, tokens, topk, generator=generator)
    idx = torch.randint(0, tokens, (batch, heads, tokens, topk), generator=generator)
    valid = idx <= torch.arange(tokens).view(1, 1, tokens, 1)
    scripted_drive = drive.clone().requires_grad_()
    reference_drive = drive.clone().requires_grad_()

    torch.manual_seed(103)
    with patch.object(
        SynapticPresyn,
        "release_canonical",
        side_effect=AssertionError("scripted scan fell back to the Python query loop"),
    ):
        scripted_output = _release_recurrence_group(
            scripted_presyn,
            scripted_state,
            scripted_drive,
            idx,
            valid,
            train=True,
            differentiable=False,
            active_key_count=tokens,
        )

    torch.manual_seed(103)
    reference_outputs = []
    for position in range(tokens):
        reference_outputs.append(
            reference_presyn.release_canonical(
                reference_state,
                reference_drive[:, :, position : position + 1],
                idx[:, :, position : position + 1],
                train=True,
                valid=valid[:, :, position : position + 1],
                differentiable=False,
                active_key_count=position + 1,
            )
        )
    reference_output = torch.cat(reference_outputs, dim=2)

    assert torch.equal(scripted_output, reference_output)
    assert torch.equal(scripted_presyn.ema_e, reference_presyn.ema_e)
    for key in scripted_state:
        scripted_values = (
            scripted_state[key]
            if isinstance(scripted_state[key], list)
            else [scripted_state[key]]
        )
        reference_values = (
            reference_state[key]
            if isinstance(reference_state[key], list)
            else [reference_state[key]]
        )
        assert all(
            torch.allclose(
                scripted_value, reference_value, atol=1e-7, rtol=1e-7
            )
            for scripted_value, reference_value in zip(
                scripted_values, reference_values
            )
        ), key

    scripted_output.sum().backward()
    reference_output.sum().backward()
    assert scripted_drive.grad is not None and reference_drive.grad is not None
    assert torch.allclose(
        scripted_drive.grad, reference_drive.grad, atol=2e-7, rtol=1e-6
    )


@pytest.mark.unit
def test_scripted_cpu_scan_falls_back_for_differentiable_input_state():
    cfg = SynapticConfig(attn_topk=4, stochastic_train_frac=0.0)
    presyn = SynapticPresyn(d_head=8, cfg=cfg)
    state = build_presyn_state(1, 4, 1, torch.device("cpu"), torch.float32, cfg)
    initial_calcium = state["C"].requires_grad_()
    drive = torch.randn(1, 1, 4, 4, requires_grad=True)
    idx = torch.zeros(1, 1, 4, 4, dtype=torch.long)
    valid = torch.ones_like(idx, dtype=torch.bool)

    output = _release_recurrence_group(
        presyn,
        state,
        drive,
        idx,
        valid,
        train=True,
        differentiable=False,
        active_key_count=4,
    )
    output.sum().backward()

    initial_calcium_grad = initial_calcium.grad
    assert initial_calcium_grad is not None
    assert torch.isfinite(initial_calcium_grad).all()
    assert initial_calcium_grad.abs().sum() > 0


@pytest.mark.unit
def test_private_train_rng_round_trips_through_presyn_state_dict():
    cfg = SynapticConfig(attn_topk=4, stochastic_train_frac=1.0)
    uninterrupted = SynapticPresyn(d_head=8, cfg=cfg)
    resumed = SynapticPresyn(d_head=8, cfg=cfg)
    assert uninterrupted._train_sampling_generator(torch.device("mps")) is None
    tokens = 8
    recurrent_state = build_presyn_state(
        1, tokens, 2, torch.device("cpu"), torch.float32, cfg
    )
    generator = torch.Generator().manual_seed(107)
    drive = torch.randn(1, 2, tokens, 4, generator=generator)
    idx = torch.randint(0, tokens, (1, 2, tokens, 4), generator=generator)
    valid = idx <= torch.arange(tokens).view(1, 1, tokens, 1)

    torch.manual_seed(109)
    _release_recurrence_group(
        uninterrupted,
        recurrent_state,
        drive[:, :, :4],
        idx[:, :, :4],
        valid[:, :, :4],
        train=True,
        differentiable=False,
        active_key_count=4,
    )
    saved_module_state = {
        key: value.clone() for key, value in uninterrupted.state_dict().items()
    }
    saved_recurrent_state = {
        key: [item.clone() for item in value]
        if isinstance(value, list)
        else value.clone()
        for key, value in recurrent_state.items()
    }

    expected = _release_recurrence_group(
        uninterrupted,
        recurrent_state,
        drive[:, :, 4:],
        idx[:, :, 4:],
        valid[:, :, 4:],
        train=True,
        differentiable=False,
        active_key_count=8,
    )

    torch.manual_seed(9_999)
    resumed.load_state_dict(saved_module_state, strict=True)
    actual = _release_recurrence_group(
        resumed,
        saved_recurrent_state,
        drive[:, :, 4:],
        idx[:, :, 4:],
        valid[:, :, 4:],
        train=True,
        differentiable=False,
        active_key_count=8,
    )

    assert torch.equal(actual, expected)
    assert torch.equal(resumed.ema_e, uninterrupted.ema_e)
    for key in recurrent_state:
        expected_values = (
            recurrent_state[key]
            if isinstance(recurrent_state[key], list)
            else [recurrent_state[key]]
        )
        actual_values = (
            saved_recurrent_state[key]
            if isinstance(saved_recurrent_state[key], list)
            else [saved_recurrent_state[key]]
        )
        assert all(
            torch.equal(actual_value, expected_value)
            for actual_value, expected_value in zip(
                actual_values, expected_values
            )
        ), key


@pytest.mark.gpu
@pytest.mark.e2e
def test_private_train_rng_checkpoint_is_backend_safe_on_cuda():
    """CPU and CUDA RNG blobs never get loaded into one another, and CUDA resumes exactly."""
    cfg = SynapticConfig(attn_topk=4, stochastic_train_frac=1.0)
    cpu_source = SynapticPresyn(d_head=8, cfg=cfg)
    cpu_state = build_presyn_state(
        1, 1, 2, torch.device("cpu"), torch.float32, cfg
    )
    cpu_drive = torch.ones(1, 2, 1, 4)
    cpu_idx = torch.zeros(1, 2, 1, 4, dtype=torch.long)
    cpu_valid = torch.ones_like(cpu_idx, dtype=torch.bool)
    torch.manual_seed(113)
    _release_recurrence_group(
        cpu_source,
        cpu_state,
        cpu_drive,
        cpu_idx,
        cpu_valid,
        train=True,
        differentiable=False,
        active_key_count=1,
    )
    assert cpu_source._presyn_train_cpu_rng_state.numel() > 0
    assert cpu_source._presyn_train_cuda_rng_state.numel() == 0

    device = torch.device("cuda")
    uninterrupted = SynapticPresyn(d_head=8, cfg=cfg).to(device)
    uninterrupted.load_state_dict(cpu_source.state_dict(), strict=True)
    assert uninterrupted._presyn_train_cpu_rng_state.numel() > 0
    assert uninterrupted._presyn_train_cuda_rng_state.numel() == 0

    tokens = 8
    recurrent_state = build_presyn_state(1, tokens, 2, device, torch.float32, cfg)
    generator = torch.Generator().manual_seed(127)
    drive = torch.randn(1, 2, tokens, 4, generator=generator).to(device)
    idx = torch.randint(
        0, tokens, (1, 2, tokens, 4), generator=generator
    ).to(device)
    valid = idx <= torch.arange(tokens, device=device).view(1, 1, tokens, 1)
    _release_recurrence_group(
        uninterrupted,
        recurrent_state,
        drive[:, :, :4],
        idx[:, :, :4],
        valid[:, :, :4],
        train=True,
        differentiable=False,
        active_key_count=4,
    )
    assert uninterrupted._presyn_train_cuda_rng_state.numel() > 0
    saved_module_state = {
        key: value.clone() for key, value in uninterrupted.state_dict().items()
    }
    saved_recurrent_state = {
        key: [item.clone() for item in value]
        if isinstance(value, list)
        else value.clone()
        for key, value in recurrent_state.items()
    }
    expected = _release_recurrence_group(
        uninterrupted,
        recurrent_state,
        drive[:, :, 4:],
        idx[:, :, 4:],
        valid[:, :, 4:],
        train=True,
        differentiable=False,
        active_key_count=8,
    )

    resumed = SynapticPresyn(d_head=8, cfg=cfg).to(device)
    resumed.load_state_dict(saved_module_state, strict=True)
    actual = _release_recurrence_group(
        resumed,
        saved_recurrent_state,
        drive[:, :, 4:],
        idx[:, :, 4:],
        valid[:, :, 4:],
        train=True,
        differentiable=False,
        active_key_count=8,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.equal(resumed.ema_e, uninterrupted.ema_e)


@pytest.mark.e2e
def test_train_mode_ema_matches_contiguous_and_incremental_schedules():
    full_model, cfg = _model(7, stochastic_train_frac=0.0)
    incremental_model, _ = _model(7, stochastic_train_frac=0.0)
    torch.manual_seed(19)
    tokens = torch.randint(0, cfg.vocab_size, (1, 7), dtype=torch.long)

    with torch.no_grad():
        full_cache = _kv(cfg, 1, 7)
        full_logits, _ = full_model(tokens, kv_cache=full_cache, train_mode=True)

        incremental_cache = _kv(cfg, 1, 7)
        incremental_logits = []
        for position in range(tokens.size(1)):
            logits, _ = incremental_model(
                tokens[:, position : position + 1],
                kv_cache=incremental_cache,
                train_mode=True,
            )
            incremental_logits.append(logits)

    assert torch.allclose(
        torch.cat(incremental_logits, dim=1), full_logits, atol=1e-4, rtol=1e-4
    )
    for full_block, incremental_block in zip(full_model.h, incremental_model.h):
        assert torch.allclose(
            full_block.attn.attn.pre.ema_e,
            incremental_block.attn.attn.pre.ema_e,
            atol=1e-7,
            rtol=1e-7,
        )


@pytest.mark.e2e
def test_stochastic_train_rng_matches_contiguous_and_incremental_schedules():
    full_model, cfg = _model(29, attn_topk=8, stochastic_train_frac=0.12)
    incremental_model, _ = _model(29, attn_topk=8, stochastic_train_frac=0.12)
    torch.manual_seed(37)
    tokens = torch.randint(0, cfg.vocab_size, (1, 11), dtype=torch.long)

    torch.manual_seed(41)
    full_cache = _kv(cfg, 1, 11)
    full_logits, _ = full_model(tokens, kv_cache=full_cache, train_mode=True)

    torch.manual_seed(41)
    incremental_cache = _kv(cfg, 1, 11)
    incremental_logits = []
    for position in range(tokens.size(1)):
        logits, _ = incremental_model(
            tokens[:, position : position + 1],
            kv_cache=incremental_cache,
            train_mode=True,
        )
        incremental_logits.append(logits)

    assert torch.allclose(
        torch.cat(incremental_logits, dim=1), full_logits, atol=1e-6, rtol=1e-6
    )
    assert isinstance(full_cache.presyn_state, list)
    assert isinstance(incremental_cache.presyn_state, list)
    for layer, (full_state, incremental_state) in enumerate(
        zip(full_cache.presyn_state, incremental_cache.presyn_state)
    ):
        _assert_presyn_state_close(
            full_state, incremental_state, context=f"stochastic layer {layer}"
        )
    for full_block, incremental_block in zip(full_model.h, incremental_model.h):
        assert torch.equal(
            full_block.attn.attn.pre.ema_e,
            incremental_block.attn.attn.pre.ema_e,
        )


@pytest.mark.e2e
def test_mc_release_rng_is_prefix_and_decode_schedule_invariant():
    short_model, cfg = _model(23, attn_topk=8)
    full_model, _ = _model(23, attn_topk=8)
    incremental_model, _ = _model(23, attn_topk=8)
    torch.manual_seed(29)
    tokens = torch.randint(0, cfg.vocab_size, (1, 11), dtype=torch.long)

    torch.manual_seed(31)
    with mc_sampling(short_model):
        short_logits, _ = short_model(
            tokens[:, :6], kv_cache=_kv(cfg, 1, 11), train_mode=False
        )
    torch.manual_seed(31)
    with mc_sampling(full_model):
        full_logits, _ = full_model(
            tokens, kv_cache=_kv(cfg, 1, 11), train_mode=False
        )
    assert torch.allclose(short_logits, full_logits[:, :6], atol=1e-6, rtol=1e-6)

    torch.manual_seed(31)
    incremental_cache = _kv(cfg, 1, 11)
    incremental_logits = []
    with mc_sampling(incremental_model):
        for position in range(tokens.size(1)):
            logits, _ = incremental_model(
                tokens[:, position : position + 1],
                kv_cache=incremental_cache,
                train_mode=False,
            )
            incremental_logits.append(logits)
    assert torch.allclose(
        torch.cat(incremental_logits, dim=1), full_logits, atol=1e-6, rtol=1e-6
    )


@pytest.mark.e2e
def test_metriplectic_telemetry_ignores_unmaterialized_future_slots():
    def make_model():
        return _model(
            37,
            stochastic_train_frac=0.0,
            learnable_kinetics=True,
            differentiable_recurrence=True,
            recurrence_checkpoint_len=2,
            metriplectic_integrator=True,
        )

    full_model, cfg = make_model()
    incremental_model, _ = make_model()
    torch.manual_seed(41)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)

    full_logits, _ = full_model(tokens, kv_cache=_kv(cfg, 1, 8), train_mode=True)
    incremental_cache = _kv(cfg, 1, 8)
    incremental_logits = []
    for position in range(tokens.size(1)):
        logits, _ = incremental_model(
            tokens[:, position : position + 1],
            kv_cache=incremental_cache,
            train_mode=True,
        )
        incremental_logits.append(logits)

    assert torch.allclose(
        torch.cat(incremental_logits, dim=1), full_logits, atol=1e-6, rtol=1e-6
    )
    for full_block, incremental_block in zip(full_model.h, incremental_model.h):
        full_metrics = full_block.attn.attn.pre.get_metriplectic_metrics()
        incremental_metrics = incremental_block.attn.attn.pre.get_metriplectic_metrics()
        assert full_metrics["steps"] == incremental_metrics["steps"]
        assert full_metrics["fallbacks"] == incremental_metrics["fallbacks"]
        for metric in (
            "last_max_energy_drift",
            "last_min_entropy_production",
            "last_max_free_energy_delta",
        ):
            assert full_metrics[metric] == pytest.approx(
                incremental_metrics[metric], abs=1e-8
            )
        assert full_metrics["steps"] == 2 * sum(range(1, tokens.size(1) + 1))


# --------------------------------------------------------------------------- #
# 2. prefill fork broadcasts presyn-state to every decode row
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
def test_prefill_fork_preserves_presyn_state():
    model, cfg = _model(0)
    prompt_len, n_rows = 7, 3
    torch.manual_seed(2)
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), dtype=torch.long)

    _reset(model)
    kv_src = _kv(cfg, 1, prompt_len + 4)
    model(prompt, kv_cache=kv_src, train_mode=False)

    kv_fork = _kv(cfg, n_rows, prompt_len + 4)
    kv_fork.prefill(kv_src)

    assert isinstance(kv_fork.presyn_state, list) and len(kv_fork.presyn_state) == cfg.n_layer
    assert isinstance(kv_src.presyn_state, list)
    for sf, ss in zip(kv_fork.presyn_state, kv_src.presyn_state):
        tf, ts = _presyn_tensors(sf), _presyn_tensors(ss)
        for key in ts:
            if key not in tf or ts[key].shape[0] != 1:
                continue
            assert tf[key].shape[0] == n_rows, f"fork did not expand batch for '{key}'"
            for r in range(n_rows):
                assert torch.allclose(tf[key][r], ts[key][0], atol=0.0, rtol=0.0), (
                    f"forked row {r} presyn '{key}' differs from the source it was cloned from"
                )


# --------------------------------------------------------------------------- #
# 3. sampling is deterministic under a fixed seed
# --------------------------------------------------------------------------- #
class _FakeTok:
    _special = {
        "<|python_start|>": -1, "<|python_end|>": -2,
        "<|output_start|>": -3, "<|output_end|>": -4, "<|assistant_end|>": -5,
    }
    def encode_special(self, s):
        return self._special[s]
    def get_bos_token_id(self):
        return -10
    def decode(self, toks):
        return ""
    def encode(self, s):
        return []


@pytest.mark.e2e
def test_decode_deterministic_under_seed():
    """Two generations with the same seed must produce identical tokens — i.e. sampling is driven by
    the seeded generator, not global RNG state. (The model here is a pure forward, so this isolates
    sampler reproducibility; cross-run state isolation when plasticity is on is a separate concern.)"""
    model, cfg = _model(0)
    eng = Engine(model, _FakeTok())
    prompt = [1, 2, 3, 4]

    def run():
        cols = list(
            eng.generate(prompt, num_samples=1, max_tokens=8, temperature=1.0, top_k=20, seed=123)
        )
        return [c[0][0] for c in cols]  # token id per step (col, masks) -> col[0]

    a, b = run(), run()
    assert a == b, f"decode is not deterministic under a fixed seed: {a} != {b}"
    assert len(a) == 8


# --------------------------------------------------------------------------- #
# 4. per-step bio-state is logged (the eqyk.2 JSONL stream)
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
def test_decode_logs_bio_state_per_step(tmp_path):
    model, cfg = _model(0)
    torch.manual_seed(3)
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), dtype=torch.long)
    n_steps = 6

    logger = RunLogger(tmp_path, name="decode_parity", console=False)
    _reset(model)
    kv = _kv(cfg, 1, 5 + n_steps)
    model(prompt, kv_cache=kv, train_mode=False)
    assert isinstance(kv.presyn_state, list)
    nxt = prompt[:, -1:]
    for step in range(n_steps):
        model(nxt, kv_cache=kv, train_mode=False)
        last = kv.presyn_state[-1]  # deepest layer's presyn state
        logger.log_bio_state(
            step=step,
            calcium=last["C"], rrp=last["RRP"], energy=last["E"], buffer=last["BUF"],
        )
        nxt = torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long)

    events = logger.read_events()
    bio = [e for e in events if e["event"] == "bio_state"]
    assert len(bio) == n_steps, f"expected one bio_state event per decode step, got {len(bio)}"
    assert [e["step"] for e in bio] == list(range(n_steps))
    for e in bio:
        assert {"calcium", "rrp", "energy", "buffer"} <= set(e["tensors"].keys())
