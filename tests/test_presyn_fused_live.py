"""Parity and dispatch guards for jyb.2's live deterministic decode kernel."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
)


def _clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: [item.clone() for item in value]
        if isinstance(value, list)
        else value.clone()
        for name, value in state.items()
    }


def _assert_state_close(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        if isinstance(actual[name], list):
            assert len(actual[name]) == len(expected[name])
            for actual_item, expected_item in zip(actual[name], expected[name]):
                torch.testing.assert_close(
                    actual_item, expected_item, rtol=1e-5, atol=1e-6
                )
        else:
            torch.testing.assert_close(
                actual[name], expected[name], rtol=1e-5, atol=1e-6
            )


def _case(
    device: torch.device,
    *,
    t_key: int = 11,
    topk: int = 4,
) -> tuple[SynapticConfig, dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    batch, heads = 2, 3
    cfg = SynapticConfig(stochastic_train_frac=0.0, attn_topk=topk)
    state = build_presyn_state(batch, t_key, heads, device, torch.float32, cfg)
    state["C"].uniform_(0.0, 1.0)
    state["BUF"].uniform_(0.0, 0.5)
    state["RRP"].uniform_(1.0, 6.0)
    state["RES"].uniform_(2.0, 18.0)
    state["PR"].uniform_(0.2, 1.0)
    state["CL"].uniform_(0.2, 1.0)
    state["E"].uniform_(0.2, 1.0)
    for delay_entry in state["DELAY"]:
        delay_entry.uniform_(0.0, 0.25)
    drive = torch.randn(batch, heads, 1, topk, device=device)
    # Include two valid repeated-key edges plus a separate invalid edge: release_canonical permits
    # both even though the live attention top-k producer normally emits unique indices.
    base_idx = torch.arange(topk, device=device, dtype=torch.int64)
    base_idx[1] = base_idx[0]
    base_idx[-1] = t_key - 1
    idx = base_idx.view(1, 1, 1, topk)
    idx = idx.expand(batch, heads, 1, topk).clone()
    valid = torch.ones_like(drive, dtype=torch.bool)
    valid[..., -1] = False
    return cfg, state, drive, idx, valid


@pytest.mark.unit
def test_live_decode_cpu_interpreter_matches_canonical_multiblock_step():
    program = textwrap.dedent(
        """
        from typing import Any
        import torch
        from bio_inspired_nanochat.kernels.presyn_fused import presyn_live_decode_step
        from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn, build_presyn_state

        def clone(state: dict[str, Any]) -> dict[str, Any]:
            return {name: [item.clone() for item in value] if isinstance(value, list)
                    else value.clone() for name, value in state.items()}

        torch.manual_seed(42)
        cfg = SynapticConfig(stochastic_train_frac=0.0, attn_topk=4)
        initial = build_presyn_state(1, 137, 2, torch.device("cpu"), torch.float32, cfg)
        for entry in initial["DELAY"]:
            entry.uniform_(0.0, 0.25)
        reference, fused = clone(initial), clone(initial)
        drive = torch.randn(1, 2, 1, 4)
        idx = torch.tensor([0, 3, 3, 136]).view(1, 1, 1, 4).expand(1, 2, 1, 4).clone()
        valid = torch.tensor([True, True, True, False]).view(1, 1, 1, 4).expand_as(drive)
        pre = SynapticPresyn(8, cfg)
        expected = pre.release_canonical(reference, drive, idx, False, valid=valid)
        actual = presyn_live_decode_step(
            fused, drive, idx, cfg, ema_e=pre.ema_e, valid=valid, _interpret=True
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for name in reference:
            expected_items = reference[name] if isinstance(reference[name], list) else [reference[name]]
            actual_items = fused[name] if isinstance(fused[name], list) else [fused[name]]
            for actual_item, expected_item in zip(actual_items, expected_items):
                torch.testing.assert_close(actual_item, expected_item, rtol=1e-5, atol=1e-6)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.unit
def test_native_presyn_is_default_off_and_cpu_falls_back():
    cfg, state, drive, _idx, _valid = _case(torch.device("cpu"))
    pre = SynapticPresyn(16, cfg)
    assert not cfg.native_presyn
    assert not pre._can_use_native_presyn_decode(
        state,
        drive,
        train=False,
        differentiable=False,
        apply_barrier=False,
        q_pos=None,
    )

    enabled = SynapticPresyn(
        16, SynapticConfig(native_presyn=True, stochastic_train_frac=0.0)
    )
    assert not enabled._can_use_native_presyn_decode(
        state,
        drive,
        train=False,
        differentiable=False,
        apply_barrier=False,
        q_pos=None,
    )


@pytest.mark.gpu
@pytest.mark.golden
@pytest.mark.parametrize(("t_key", "topk"), ((11, 4), (137, 4), (257, 32)))
def test_live_decode_kernel_matches_canonical_release_state_and_logit_scatter(
    t_key: int, topk: int
):
    device = torch.device("cuda")
    cfg, initial, drive, idx, valid = _case(device, t_key=t_key, topk=topk)
    reference_state = _clone_state(initial)
    fused_state = _clone_state(initial)
    pre = SynapticPresyn(16, cfg).to(device)
    fused_pre = SynapticPresyn(16, replace(cfg, native_presyn=True)).to(device)
    fused_pre.load_state_dict(pre.state_dict())
    with torch.inference_mode():
        for step in range(4):
            step_drive = drive + 0.125 * step
            reference_dots = torch.randn(2, 3, 1, t_key, device=device)
            fused_dots = reference_dots.clone()
            expected_e = pre.release_canonical(
                reference_state, step_drive, idx, train=False, valid=valid
            )
            aug = torch.zeros_like(reference_dots)
            src = cfg.lambda_loge * torch.log(cfg.epsilon + expected_e)
            src = src.clamp(-cfg.loge_bias_clamp, cfg.loge_bias_clamp) * valid
            aug.scatter_add_(-1, idx, src)
            reference_dots.add_(aug)

            actual_e = fused_pre.release_canonical(
                fused_state,
                step_drive,
                idx,
                train=False,
                valid=valid,
                logits=fused_dots,
            )

            torch.testing.assert_close(actual_e, expected_e, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(fused_dots, reference_dots, rtol=1e-5, atol=1e-6)
            _assert_state_close(fused_state, reference_state)


@pytest.mark.gpu
@pytest.mark.golden
def test_live_decode_matches_multilayer_cached_model_trajectory():
    device = torch.device("cuda")
    eager = (
        GPTSynaptic(
            GPTSynapticConfig(
                sequence_len=16,
                vocab_size=97,
                n_layer=2,
                n_head=4,
                n_kv_head=2,
                n_embd=64,
                dropout=0.0,
                syn_cfg=SynapticConfig(
                    native_presyn=False, stochastic_train_frac=0.0, attn_topk=4
                ),
            )
        )
        .to(device)
        .eval()
    )
    fused = (
        GPTSynaptic(
            GPTSynapticConfig(
                sequence_len=16,
                vocab_size=97,
                n_layer=2,
                n_head=4,
                n_kv_head=2,
                n_embd=64,
                dropout=0.0,
                syn_cfg=SynapticConfig(
                    native_presyn=True, stochastic_train_frac=0.0, attn_topk=4
                ),
            )
        )
        .to(device)
        .eval()
    )
    fused.load_state_dict(eager.state_dict())
    eager_cache = KVCache(
        batch_size=1, num_heads=2, seq_len=16, head_dim=16, num_layers=2
    )
    fused_cache = KVCache(
        batch_size=1, num_heads=2, seq_len=16, head_dim=16, num_layers=2
    )
    prefix = torch.tensor([[2, 5, 7, 11, 13]], device=device)
    decode = torch.tensor([[17, 19, 23, 29]], device=device)

    with torch.inference_mode():
        eager(prefix, kv_cache=eager_cache, train_mode=False)
        fused(prefix, kv_cache=fused_cache, train_mode=False)
        for position in range(decode.shape[1]):
            token = decode[:, position : position + 1]
            eager_logits, _ = eager(token, kv_cache=eager_cache, train_mode=False)
            fused_logits, _ = fused(token, kv_cache=fused_cache, train_mode=False)
            torch.testing.assert_close(fused_logits, eager_logits, rtol=1e-5, atol=1e-6)
            assert isinstance(eager_cache.presyn_state, list)
            assert isinstance(fused_cache.presyn_state, list)
            assert len(fused_cache.presyn_state) == len(eager_cache.presyn_state)
            for fused_state, eager_state in zip(
                fused_cache.presyn_state, eager_cache.presyn_state
            ):
                _assert_state_close(fused_state, eager_state)


@pytest.mark.gpu
def test_native_dispatch_falls_back_for_grad_and_prefill(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    cfg = SynapticConfig(native_presyn=True, stochastic_train_frac=0.0, attn_topk=3)
    pre = SynapticPresyn(16, cfg).to(device)
    state = build_presyn_state(1, 5, 1, device, torch.float32, cfg)
    drive = torch.randn(1, 1, 1, 3, device=device, requires_grad=True)
    idx = torch.tensor([[[[0, 2, 4]]]], device=device)

    def fail_if_dispatched(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise AssertionError("unsupported mode reached the native kernel")

    monkeypatch.setattr(
        "bio_inspired_nanochat.kernels.presyn_fused.presyn_live_decode_step",
        fail_if_dispatched,
    )
    out = pre.release_canonical(state, drive, idx, train=False)
    (grad,) = torch.autograd.grad(out.sum(), drive)
    assert torch.isfinite(grad).all()

    prefill_drive = torch.randn(1, 1, 2, 3, device=device)
    prefill_idx = torch.tensor([[[[0, 1, 2], [1, 2, 4]]]], device=device)
    with torch.inference_mode():
        prefill_out = pre.release_canonical(
            build_presyn_state(1, 5, 1, device, torch.float32, cfg),
            prefill_drive,
            prefill_idx,
            train=False,
        )
    assert torch.isfinite(prefill_out).all()
