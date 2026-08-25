"""Distributed-optimizer correctness tests (bead hwxb.2.1 / hwxb.7.2).

These spawn a real 2-process **gloo/CPU** process group and verify that the
ZeRO-style distributed optimizers produce *exactly* the result of a single-process
optimizer fed the cross-rank-averaged gradients. This is the CPU-runnable core of
the "DDP path is correct" acceptance for hwxb.2.1 — it validates gradient
averaging, optimizer-state sharding + re-replication, and (critically) the
non-shardable fallback for 0-D scalar and odd-length Parameters that the synaptic
model creates (e.g. ``learnable_kinetics`` ``theta_*`` scalars) which would
otherwise crash ``DistAdamW`` under DDP.

No GPU required; runs in a few seconds.
"""
from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Protocol, cast

import pytest

from bio_inspired_nanochat.torch_imports import torch
import torch.distributed as torch_dist
import torch.multiprocessing as mp

from bio_inspired_nanochat.adamw import DistAdamW
from bio_inspired_nanochat.muon import DistMuon, Muon


class _DistributedApi(Protocol):
    """The gloo API exercised by the spawned optimizer workers."""

    def is_available(self) -> bool: ...

    def is_gloo_available(self) -> bool: ...

    def init_process_group(
        self,
        backend: str,
        *,
        rank: int,
        world_size: int,
        timeout: timedelta,
    ) -> None: ...

    def destroy_process_group(self) -> None: ...

    def barrier(self) -> None: ...


dist = cast(_DistributedApi, torch_dist)

pytestmark = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="torch.distributed gloo backend unavailable",
)

# Parameter shapes deliberately mix shardable and non-shardable cases for ws=2:
#   - (4, 3): 2-D, dim0=4 divisible by 2          -> sharded path
#   - (6,)  : 1-D even                            -> sharded path
#   - (3,)  : 1-D ODD (3 % 2 != 0)                -> replicated path
#   - ()    : 0-D scalar (no shape[0])            -> replicated path (the crash case)
_ADAMW_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("mat_even", (4, 3)),
    ("vec_even", (6,)),
    ("vec_odd", (3,)),
    ("scalar", ()),
)
_MUON_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("a", (4, 4)),  # same shape as b -> exercises DistMuon shape-grouping + owner assignment
    ("b", (4, 4)),
    ("c", (6, 2)),  # different shape
)

_ADAMW_LR = 0.1
_ADAMW_BETAS = (0.9, 0.95)
_ADAMW_EPS = 1e-8
_ADAMW_WEIGHT_DECAY = 0.01
_MUON_LR = 0.02
_MUON_MOMENTUM = 0.95
_MUON_NESTEROV = True
_MUON_NS_STEPS = 5
_PROCESS_GROUP_TIMEOUT_SECONDS = 20.0
_SPAWN_TIMEOUT_SECONDS = 30.0
_SPAWN_CLEANUP_SECONDS = 2.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
    )


def _process_statuses(context: mp.ProcessContext) -> str:
    return ", ".join(
        f"pid={process.pid}, alive={process.is_alive()}, exitcode={process.exitcode}"
        for process in context.processes
    )


def _stop_spawned_processes(context: mp.ProcessContext) -> str:
    """Stop only this context's children and report their final status."""
    for process in context.processes:
        if process.is_alive():
            process.terminate()
    for process in context.processes:
        process.join(timeout=_SPAWN_CLEANUP_SECONDS)
    for process in context.processes:
        if process.is_alive():
            process.kill()
    for process in context.processes:
        process.join(timeout=_SPAWN_CLEANUP_SECONDS)
    return _process_statuses(context)


def _run_workers(
    worker: Callable[..., None],
    *,
    args: tuple[object, ...],
    nprocs: int,
    timeout_seconds: float = _SPAWN_TIMEOUT_SECONDS,
) -> None:
    """Run spawned workers with a hard deadline and owned-child cleanup."""
    started_at = time.monotonic()
    worker_name = getattr(worker, "__qualname__", repr(worker))
    context = cast(
        mp.ProcessContext,
        mp.spawn(worker, args=args, nprocs=nprocs, join=False),
    )
    deadline = started_at + timeout_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if context.join(timeout=remaining, grace_period=_SPAWN_CLEANUP_SECONDS):
                return
    except BaseException:
        _stop_spawned_processes(context)
        raise

    timed_out_statuses = _process_statuses(context)
    final_statuses = _stop_spawned_processes(context)
    raise TimeoutError(
        f"worker {worker_name} exceeded {timeout_seconds:.1f}s deadline; "
        f"at deadline: [{timed_out_statuses}]; after cleanup: [{final_statuses}]"
    )


def _init_param(idx: int, shape: tuple[int, ...]) -> torch.Tensor:
    g = torch.Generator().manual_seed(100 + idx)
    return torch.empty(shape).uniform_(-0.5, 0.5, generator=g)


def _grad_for(idx: int, step: int, shape: tuple[int, ...], rank: int) -> torch.Tensor:
    """Deterministic per-(param, step) base gradient, scaled by (1+rank).

    rank 0 -> base, rank 1 -> 2*base, so the cross-rank average is 1.5*base. Both
    worker and oracle build this from the same seed, so the oracle knows the exact
    gradient each rank contributed.
    """
    g = torch.Generator().manual_seed(5000 + idx * 97 + step * 7)
    base = torch.empty(shape).normal_(0.0, 1.0, generator=g)
    return base * (1.0 + rank)


def _avg_grad(idx: int, step: int, shape: tuple[int, ...], world_size: int) -> torch.Tensor:
    acc = torch.zeros(shape)
    for r in range(world_size):
        acc = acc + _grad_for(idx, step, shape, r)
    return acc / world_size


# --------------------------------------------------------------------------- #
# Workers (must be top-level for spawn picklability)
# --------------------------------------------------------------------------- #
def _adamw_worker(rank: int, world_size: int, port: int, tmpdir: str, n_steps: int) -> None:
    _init_gloo(rank, world_size, port)
    try:
        params = [torch.nn.Parameter(_init_param(i, s)) for i, (_, s) in enumerate(_ADAMW_SPECS)]
        opt = DistAdamW(
            [{"params": params}],
            lr=_ADAMW_LR,
            betas=_ADAMW_BETAS,
            eps=_ADAMW_EPS,
            weight_decay=_ADAMW_WEIGHT_DECAY,
        )
        for step in range(n_steps):
            for i, p in enumerate(params):
                p.grad = _grad_for(i, step, tuple(p.shape), rank).clone()
            opt.step()
            for p in params:
                p.grad = None
        if rank == 0:
            torch.save([p.detach().clone() for p in params], os.path.join(tmpdir, "adamw.pt"))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _muon_worker(rank: int, world_size: int, port: int, tmpdir: str, n_steps: int) -> None:
    _init_gloo(rank, world_size, port)
    try:
        params = [torch.nn.Parameter(_init_param(200 + i, s)) for i, (_, s) in enumerate(_MUON_SPECS)]
        opt = DistMuon(
            params,
            lr=_MUON_LR,
            momentum=_MUON_MOMENTUM,
            nesterov=_MUON_NESTEROV,
            ns_steps=_MUON_NS_STEPS,
        )
        for step in range(n_steps):
            for i, p in enumerate(params):
                p.grad = _grad_for(200 + i, step, tuple(p.shape), rank).clone()
            opt.step()
            for p in params:
                p.grad = None
        if rank == 0:
            torch.save([p.detach().clone() for p in params], os.path.join(tmpdir, "muon.pt"))
        # Do not let one rank tear down gloo while a peer is still draining the
        # optimizer's final asynchronous collective. Without this rendezvous,
        # rank 0 can exit cleanly while rank 1 remains stuck during teardown.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _intentional_crash_worker(rank: int, port: int) -> None:
    """Make one child fail while its sibling waits for process-group startup."""
    if rank == 0:
        raise RuntimeError("intentional distributed worker crash")
    _init_gloo(rank, world_size=2, port=port)


def _intentional_hang_worker(_rank: int) -> None:
    time.sleep(60)


# --------------------------------------------------------------------------- #
# Single-process oracles (independently re-derived; NOT calling the prod helper)
# --------------------------------------------------------------------------- #
def _adamw_oracle(n_steps: int, world_size: int) -> list[torch.Tensor]:
    params = [_init_param(i, s).clone() for i, (_, s) in enumerate(_ADAMW_SPECS)]
    state = [dict() for _ in params]
    beta1, beta2 = _ADAMW_BETAS
    eps, lr, wd = _ADAMW_EPS, _ADAMW_LR, _ADAMW_WEIGHT_DECAY
    for step in range(n_steps):
        for i, p in enumerate(params):
            g = _avg_grad(i, step, tuple(p.shape), world_size)
            st = state[i]
            if not st:
                st["t"] = 0
                st["m"] = torch.zeros_like(p)
                st["v"] = torch.zeros_like(p)
            st["t"] += 1
            t = st["t"]
            if wd != 0:
                p.mul_(1 - lr * wd)
            st["m"].mul_(beta1).add_(g, alpha=1 - beta1)
            st["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bias1 = 1 - beta1**t
            bias2 = 1 - beta2**t
            denom = st["v"].sqrt().add_(eps)
            step_size = lr * (bias2**0.5 / bias1)
            p.add_(st["m"].div(denom).mul_(step_size), alpha=-1.0)
    return params


def _muon_oracle(n_steps: int, world_size: int) -> list[torch.Tensor]:
    # Reuse the non-distributed Muon (shares the NS iteration + update formula) fed
    # the same averaged gradients; its per-param result must match DistMuon.
    params = [torch.nn.Parameter(_init_param(200 + i, s)) for i, (_, s) in enumerate(_MUON_SPECS)]
    opt = Muon(
        params,
        lr=_MUON_LR,
        momentum=_MUON_MOMENTUM,
        nesterov=_MUON_NESTEROV,
        ns_steps=_MUON_NS_STEPS,
    )
    for step in range(n_steps):
        for i, p in enumerate(params):
            p.grad = _avg_grad(200 + i, step, tuple(p.shape), world_size)
        opt.step()
        for p in params:
            p.grad = None
    return [p.detach().clone() for p in params]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_distadamw_matches_single_process_on_averaged_grads(tmp_path):
    """DistAdamW(ws=2) == single-process AdamW on averaged grads, incl. 0-D/odd params."""
    n_steps, ws, port = 5, 2, _free_port()
    _run_workers(
        _adamw_worker,
        args=(ws, port, str(tmp_path), n_steps),
        nprocs=ws,
    )
    got = torch.load(os.path.join(tmp_path, "adamw.pt"), weights_only=True)
    expected = _adamw_oracle(n_steps, ws)
    for (name, _), g, e in zip(_ADAMW_SPECS, got, expected):
        max_diff = (g - e).abs().max().item()
        assert torch.allclose(g, e, rtol=1e-5, atol=1e-6), (
            f"DistAdamW param {name!r} diverged from single-process oracle: "
            f"max|Δ|={max_diff:.3e} (shape={tuple(g.shape)})"
        )


@pytest.mark.parametrize("_repeat", range(3))
def test_distmuon_matches_single_process_on_averaged_grads(tmp_path, _repeat):
    """DistMuon(ws=2) == single-process Muon on averaged grads (bf16 NS tolerance)."""
    n_steps, ws, port = 5, 2, _free_port()
    _run_workers(
        _muon_worker,
        args=(ws, port, str(tmp_path), n_steps),
        nprocs=ws,
    )
    got = torch.load(os.path.join(tmp_path, "muon.pt"), weights_only=True)
    expected = _muon_oracle(n_steps, ws)
    for (name, _), g, e in zip(_MUON_SPECS, got, expected):
        max_diff = (g - e).abs().max().item()
        # Muon orthogonalizes in bfloat16, so allow a modest tolerance.
        assert torch.allclose(g, e, rtol=1e-2, atol=1e-2), (
            f"DistMuon param {name!r} diverged from single-process oracle: "
            f"max|Δ|={max_diff:.3e} (shape={tuple(g.shape)})"
        )


def test_spawn_harness_propagates_worker_crash_and_cleans_up() -> None:
    before = {process.pid for process in mp.active_children()}
    with pytest.raises(mp.ProcessRaisedException, match="intentional distributed worker crash"):
        _run_workers(
            _intentional_crash_worker,
            args=(_free_port(),),
            nprocs=2,
            timeout_seconds=5.0,
        )
    leaked = [
        process.pid
        for process in mp.active_children()
        if process.pid not in before and process.is_alive()
    ]
    assert not leaked, f"spawn harness leaked child processes after worker crash: {leaked}"


def test_spawn_harness_times_out_and_cleans_up() -> None:
    before = {process.pid for process in mp.active_children()}
    with pytest.raises(TimeoutError, match=r"exceeded 0\.2s deadline.*after cleanup"):
        _run_workers(
            _intentional_hang_worker,
            args=(),
            nprocs=2,
            timeout_seconds=0.2,
        )
    leaked = [
        process.pid
        for process in mp.active_children()
        if process.pid not in before and process.is_alive()
    ]
    assert not leaked, f"spawn harness leaked child processes after timeout: {leaked}"
