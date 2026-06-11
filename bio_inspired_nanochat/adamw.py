"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.

DDP design (hwxb.2.1)
---------------------
``DistAdamW`` is a ZeRO-2-style optimizer: gradients are averaged across ranks and
optimizer state is sharded, so each rank only stores/updates its slice of every
parameter before the updated parameter is re-replicated with an all-gather.

Sharding is along dim 0, which requires ``param.shape[0] % world_size == 0``. The
bio-inspired model has many *small* learnable Parameters that violate this — most
importantly **0-D scalar kinetics** (``learnable_kinetics=True`` adds per-layer
``theta_*`` scalars) and odd-length 1-D vectors. Sharding those is impossible
(a 0-D tensor has no ``shape[0]``; an odd 1-D vector is not divisible by 2). Such
params take a **replicated** path: their gradient is all-reduced (AVG) and every
rank applies the *full* AdamW update locally, so all ranks stay bit-identical
without any scatter/gather. The per-element math is identical on both paths
(``_adamw_update_`` is the single source of truth), so the sharded and replicated
paths produce the same parameter values — verified by ``tests/test_scaleup_ddp.py``.
"""
from typing import Any, cast

from bio_inspired_nanochat.torch_imports import torch
import torch.distributed as torch_dist
from torch import Tensor

# torch.distributed exposes many collectives dynamically; ty can't see them all and
# emits possibly-missing-attribute false positives. Cast to Any (matching muon.py).
dist = cast(Any, torch_dist)


@torch.no_grad()
def _adamw_update_(
    p: Tensor,
    g: Tensor,
    state: dict,
    *,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    wd: float,
    wd_mul: float = 1.0,
) -> None:
    """In-place AdamW update of ``p`` (or a slice of it) from gradient ``g``.

    The single source of truth for the per-element AdamW math, shared by the
    sharded and replicated code paths in ``DistAdamW`` so they are provably
    equivalent. ``state`` is mutated in place (step counter + moment buffers).
    ``p`` and ``g`` must have the same shape (the caller passes either the full
    param or this rank's shard).
    """
    if not state:
        state["step"] = torch.tensor(0, dtype=torch.int64, device=p.device)
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    state["step"] += 1
    t = state["step"]
    # decoupled weight decay
    if wd != 0:
        p.mul_(1 - lr * wd * wd_mul)
    # running moments
    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    # bias corrections
    bias1 = 1 - beta1**t
    bias2 = 1 - beta2**t
    denom = exp_avg_sq.sqrt().add_(eps)
    step_size = lr * (torch.sqrt(bias2) / bias1)
    update = exp_avg.div(denom).mul_(step_size)
    p.add_(other=update, alpha=-1.0)


def _is_shardable(t: Tensor, world_size: int) -> bool:
    """A tensor is dim-0 shardable iff it is >=1-D and divisible by world_size."""
    return world_size > 1 and t.ndim >= 1 and (t.shape[0] % world_size == 0)


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction.

    Robust to non-shardable parameters (0-D scalars and dim-0 not divisible by
    world_size): those are gradient-all-reduced and updated in full on every rank
    (see module docstring), so the synaptic model's small kinetic Parameters do
    not crash or get silently skipped under DDP.
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    # @torch.compile (not supported on Python 3.14)
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # ---- Phase 1: kick off gradient communication for every param ----
        # For shardable params: reduce_scatter the grad into this rank's slice.
        # For non-shardable params: all_reduce(AVG) the grad in place (full update later).
        # We record per-param work items so Phase 2 can process them in the same order.
        # NOTE: we keep the async ``Work`` handles and ``.wait()`` on them directly rather
        # than ``.get_future()`` — gloo (CPU, used by the tests) does not implement
        # ``Work::getFuture``, while NCCL does; ``.wait()`` is portable to both and has the
        # same launch-async-then-block semantics.
        work: list[tuple] = []  # (group, p, kind, handle, payload)
        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue  # frozen / unused param this step
                if _is_shardable(grad, world_size):
                    rank_size = grad.shape[0] // world_size
                    grad_slice = torch.empty_like(grad[:rank_size])
                    handle = dist.reduce_scatter_tensor(
                        grad_slice, grad.contiguous(), op=dist.ReduceOp.AVG, async_op=True
                    )
                    work.append((group, p, "shard", handle, grad_slice))
                else:
                    handle = (
                        dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
                        if world_size > 1
                        else None
                    )
                    work.append((group, p, "repl", handle, grad))

        # ---- Phase 2: compute updates; re-replicate sharded params ----
        gather_handles = []
        for group, p, kind, handle, payload in work:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"] * getattr(p, "lr_mul", 1.0)
            wd_mul = getattr(p, "wd_mul", 1.0)
            if handle is not None:
                handle.wait()
            if kind == "shard":
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                _adamw_update_(
                    p_slice, payload, self.state[p],
                    beta1=beta1, beta2=beta2, eps=eps, lr=lr, wd=wd, wd_mul=wd_mul,
                )
                gather_handles.append(
                    dist.all_gather_into_tensor(p, p_slice, async_op=True)
                )
            else:  # "repl": grad already averaged across ranks; full local update
                _adamw_update_(
                    p, payload, self.state[p],
                    beta1=beta1, beta2=beta2, eps=eps, lr=lr, wd=wd, wd_mul=wd_mul,
                )
        for h in gather_handles:
            h.wait()
