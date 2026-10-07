"""The multi-query Triton presyn scan (l7c9 / jyb.2) against the scripted recurrence.

``kernels.presyn_fused.presyn_detached_scan`` advances a whole (B, H, T, K) query block of the
exact causal recurrence in one launch and records the per-query edge constants from which
``synaptic.presyn_edge_release`` gives the drive gradient, so training needs no backward kernel.
The oracle is ``synaptic._scripted_detached_presyn_scan`` fed the same stochastic draws.

The CPU tests run the kernel under ``TRITON_INTERPRET=1`` in a subprocess (the interpreter is
selected when the kernel module is imported). They prove the math and the in-place state
bookkeeping, not the CUDA barriers or the speed; the GPU-marked test covers dispatch on a real
device.

Run:  pytest tests/test_presyn_scan_kernel.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

_PARITY_PROGRAM = textwrap.dedent(
    """
    import math
    import torch
    from bio_inspired_nanochat.kernels.presyn_fused import presyn_detached_scan
    from bio_inspired_nanochat.synaptic import (
        SynapticConfig, _scripted_detached_presyn_scan, build_presyn_state, presyn_edge_release,
    )

    def clone(state):
        return {k: [i.clone() for i in v] if isinstance(v, list) else v.clone() for k, v in state.items()}

    def close(actual, expected, what):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6, msg=what)

    def case(train, frac, block):
        torch.manual_seed(7)
        B, H, T, K = 2, 3, 9, 4
        first = 3 if block else 1
        t_key = first + T - 1 + 2  # two never-active suffix keys must stay untouched
        cfg = SynapticConfig(attn_topk=K, stochastic_train_frac=frac)
        state = build_presyn_state(B, t_key, H, torch.device("cpu"), torch.float32, cfg)
        state["C"].uniform_(0, 1); state["BUF"].uniform_(0, .5); state["RRP"].uniform_(1, 6)
        state["RES"].uniform_(0, 3); state["PR"].uniform_(.2, 1); state["CL"].uniform_(.2, 1)
        state["E"].uniform_(.2, 1)
        for entry in state["DELAY"]:
            entry.uniform_(0, .25)
        pos = first + torch.arange(T).view(1, 1, T, 1)
        idx = torch.randint(0, t_key, (B, H, T, K))
        idx[..., 1] = idx[..., 0]  # repeated keys accumulate like scatter_add_
        valid = idx < pos
        drive = torch.where(valid, torch.randn(B, H, T, K), torch.full((B, H, T, K), -math.inf))
        uniform, noise = torch.rand(B, H, T), torch.randn(B, H, T, K)
        ema = torch.tensor([0.7])
        ref_state, ker_state = clone(state), clone(state)
        ref = _scripted_detached_presyn_scan(
            ref_state["C"], ref_state["BUF"], ref_state["RRP"], ref_state["RES"], ref_state["PR"],
            ref_state["CL"], ref_state["E"], ref_state["AMP"], ref_state["DELAY"], drive, idx, valid,
            first, ema, None, train, frac, K, int(cfg.stochastic_count_cap),
            math.exp(-1 / cfg.tau_c), math.exp(-1 / cfg.tau_buf), cfg.alpha_ca, cfg.alpha_buf_on,
            cfg.alpha_buf_off, cfg.syt_fast_kd, cfg.syt_slow_kd, cfg.doc2_gain, cfg.complexin_bias,
            cfg.q_beta, cfg.qmax, cfg.rec_rate, cfg.prime_rate, cfg.unprime_per_release,
            cfg.nsf_recover, cfg.energy_fill, cfg.energy_max, cfg.energy_use, True, uniform, noise,
        )
        out, ema_after, edges = presyn_detached_scan(
            ker_state, drive, idx, valid, cfg, ema_e=ema, train=train, first_active_key_count=first,
            stochastic_frac=frac, uniform=uniform, noise=noise, record_edges=True, _interpret=True,
        )
        tag = f"train={train} frac={frac} first={first}"
        close(out, ref[0], f"{tag}: output")
        close(ema_after, ref[10], f"{tag}: ema")
        for i, name in enumerate(("C", "BUF", "RRP", "RES", "PR", "CL", "E")):
            close(ker_state[name], ref[1 + i], f"{tag}: state {name}")
            assert torch.equal(ker_state[name][..., first + T - 1:], state[name][..., first + T - 1:]), name
        for i, (a, b) in enumerate(zip(ker_state["DELAY"], ref[9])):
            close(a, b, f"{tag}: DELAY[{i}]")
        mask = ref[11][6]
        assert torch.equal(edges[6], mask), f"{tag}: stochastic mask"
        if frac > 0 and train:
            assert mask.any() and not mask.all(), "the case must exercise both release branches"
        for i in (0, 1, 2, 3, 4, 5, 8):
            close(edges[i], ref[11][i], f"{tag}: edge constant {i}")
        close(torch.where(mask, edges[7], 0.0), torch.where(mask, ref[11][7], 0.0), f"{tag}: noise")
        grad_out = torch.randn_like(drive)
        grads = []
        for recorded in (edges, ref[11]):
            leaf = drive.clone().requires_grad_()
            (presyn_edge_release(leaf, recorded, valid, cfg) * grad_out).sum().backward()
            grads.append(torch.nan_to_num(leaf.grad))
        close(grads[0], grads[1], f"{tag}: drive gradient")

    for train, frac in ((False, 0.0), (True, 0.0), (True, 0.5)):
        for block in (False, True):
            case(train, frac, block)
    print("ok")
    """
)


@pytest.mark.unit
def test_scan_kernel_matches_the_scripted_recurrence_under_the_interpreter():
    completed = subprocess.run(
        [sys.executable, "-c", _PARITY_PROGRAM],
        cwd=REPO,
        env={**os.environ, "TRITON_INTERPRET": "1"},
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip().endswith("ok")


def _recurrence_case(native: bool, *, train: bool, frac: float, grad: bool):
    import math
    from dataclasses import replace

    from bio_inspired_nanochat.synaptic import (
        SynapticConfig,
        SynapticPresyn,
        _release_recurrence_group,
        build_presyn_state,
    )

    torch.manual_seed(0)
    batch, heads, queries, topk, first = 2, 3, 40, 8, 5
    cfg = replace(SynapticConfig(attn_topk=topk, stochastic_train_frac=frac), native_presyn=native)
    presyn = SynapticPresyn(8, cfg)
    presyn._presyn_train_rng_seed.fill_(1234)
    t_key = first + queries - 1
    state = build_presyn_state(batch, t_key, heads, torch.device("cpu"), torch.float32, cfg)
    gen = torch.Generator().manual_seed(1)
    dots = torch.randn(batch, heads, queries, t_key, generator=gen)
    future = torch.arange(t_key).view(1, t_key) >= (first + torch.arange(queries)).view(queries, 1)
    vals, idx = torch.topk(dots.masked_fill(future, -math.inf), topk, dim=-1)
    drive = vals.clone().requires_grad_(grad)
    with torch.set_grad_enabled(grad):
        out = _release_recurrence_group(
            presyn, state, drive, idx, torch.isfinite(vals), train=train,
            differentiable=False, active_key_count=t_key,
        )
    if grad:
        (out * torch.linspace(-1, 1, out.numel()).view_as(out)).sum().backward()
    return out.detach(), drive.grad, state, presyn


@pytest.mark.unit
@pytest.mark.parametrize(
    ("train", "frac", "grad"), [(True, 0.3, True), (True, 0.0, True), (False, 0.0, False)]
)
def test_rust_scan_matches_the_scripted_scan(train: bool, frac: float, grad: bool):
    """native_presyn on CPU runs the whole block in rustbpe.presyn_detached_scan_cpu (jyb.9).

    Same RNG stream as the scripted loop (the private generator ends in the same state), values
    and state to float32 rounding, the same one-pass drive gradient, the same EMA.
    """
    from bio_inspired_nanochat.synaptic import _rust_detached_scan_kernel

    if _rust_detached_scan_kernel() is None:
        pytest.skip("rustbpe extension without presyn_detached_scan_cpu (maturin develop)")
    ref_out, ref_grad, ref_state, ref_pre = _recurrence_case(False, train=train, frac=frac, grad=grad)
    out, drive_grad, state, pre = _recurrence_case(True, train=train, frac=frac, grad=grad)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-6)
    for name in ("C", "BUF", "RRP", "RES", "PR", "CL", "E"):
        torch.testing.assert_close(state[name], ref_state[name], rtol=1e-5, atol=1e-6, msg=name)
    for got, want in zip(state["DELAY"], ref_state["DELAY"]):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pre.ema_e, ref_pre.ema_e, rtol=1e-6, atol=1e-7)
    assert torch.equal(pre._presyn_train_cpu_rng_state, ref_pre._presyn_train_cpu_rng_state)
    if grad:
        torch.testing.assert_close(drive_grad, ref_grad, rtol=1e-5, atol=1e-7)


@pytest.mark.gpu
def test_cuda_dispatch_matches_the_scripted_scan_values_and_gradients():
    from dataclasses import replace

    from bio_inspired_nanochat.synaptic import (
        SynapticConfig,
        SynapticPresyn,
        _release_recurrence_group,
        build_presyn_state,
    )

    device = torch.device("cuda")
    torch.manual_seed(3)
    batch, heads, tokens, topk = 2, 4, 160, 32
    base = SynapticConfig(attn_topk=topk, stochastic_train_frac=0.0)
    results = []
    for native in (False, True):
        cfg = replace(base, native_presyn=native)
        presyn = SynapticPresyn(8, cfg).to(device)
        state = build_presyn_state(batch, tokens, heads, device, torch.float32, cfg)
        gen = torch.Generator(device=device).manual_seed(11)
        dots = torch.randn(batch, heads, tokens, tokens, device=device, generator=gen)
        causal = torch.ones(tokens, tokens, device=device, dtype=torch.bool).tril()
        dots = dots.masked_fill(~causal, -torch.inf)
        vals, idx = torch.topk(dots, topk, dim=-1)
        drive = vals.clone().requires_grad_()
        out = _release_recurrence_group(
            presyn, state, drive, idx, torch.isfinite(vals), train=True,
            differentiable=False, active_key_count=tokens,
        )
        out.sum().backward()
        results.append((out.detach(), drive.grad, {k: v for k, v in state.items() if k != "DELAY"}))
    (ref_out, ref_grad, ref_state), (ker_out, ker_grad, ker_state) = results
    torch.testing.assert_close(ker_out, ref_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ker_grad, ref_grad, rtol=1e-4, atol=1e-5)
    for name in ref_state:
        torch.testing.assert_close(ker_state[name], ref_state[name], rtol=1e-4, atol=1e-5)
