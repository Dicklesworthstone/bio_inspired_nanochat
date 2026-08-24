"""Distributed-safe deterministic lifecycle surgery — bead uta.5.

Protocol under torch.distributed (UTA path): rank0 plans the round's lifecycle
ops WITHOUT mutating, the serialized decision (kinds, indices, alphas, per-op
RNG seeds) is broadcast, and EVERY rank applies bit-identical surgery locally,
then re-syncs its own optimizer param-groups. No rank is a passive observer, so
survivor moments survive everywhere and shape changes are safe.

The load-bearing acceptance claim: after a lifecycle event, all ranks remain
bit-identical in parameters AND optimizer state, with no RNG/order divergence.
"""

from __future__ import annotations

import hashlib
import multiprocessing
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
)

pytestmark = pytest.mark.unit


def _moe(num_experts: int = 3, n_embd: int = 8, seed: int = 0) -> SynapticMoE:
    torch.manual_seed(seed)
    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=True, stochastic_train_frac=0.0)
    return SynapticMoE(
        n_embd=n_embd, num_experts=num_experts, top_k=num_experts, hidden_mult=1, cfg=cfg
    )


def _digest(module: torch.nn.Module, optimizer=None) -> str:
    h = hashlib.sha256()
    named = sorted(module.named_parameters(), key=lambda np_: np_[0])
    for name, p in named:
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    if optimizer is not None:
        # state must be keyed STABLY: id() ordering differs across processes,
        # which would scramble identical byte multisets into different hashes.
        name_by_id = {id(p): n for n, p in module.named_parameters()}
        state_items = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                st = optimizer.state.get(p)
                if st:
                    blob = b""
                    for k in sorted(st.keys()):
                        v = st[k]
                        if torch.is_tensor(v):
                            blob += k.encode() + v.detach().cpu().contiguous().numpy().tobytes()
                        else:
                            blob += k.encode() + str(v).encode()
                    state_items.append((name_by_id.get(id(p), f"id{id(p)}"), blob))
                else:
                    state_items.append((name_by_id.get(id(p), f"id{id(p)}"), b"<nostate>"))
        for name, blob in sorted(state_items):
            h.update(name.encode())
            h.update(blob)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. Determinism: identical ops+seeds => bit-identical results regardless of
#    the surrounding global-RNG state.
# --------------------------------------------------------------------------- #
def test_apply_uta_ops_is_rng_state_independent():
    summaries = []
    for outer_seed in (0, 999):  # different global-RNG contexts
        torch.manual_seed(outer_seed)
        moe = _moe(num_experts=4, seed=7)
        ctrl = SplitMergeController(
            moe,
            SplitMergeConfig(
                variable_expert_count=True,
                splits_per_call=0,
                resets_per_call=0,
                min_step_interval=0,
                warmup_steps=0,
            ),
        )
        # engineered pressure: strong surplus -> planned grow(+split twins)
        with torch.no_grad():
            moe.energy.fill_(1.0)
            moe.fatigue.fill_(0.9)
        plan = ctrl._plan_resize_layer(moe, lambda kind, cnt: 1000 + cnt[0], [0])
        assert any(op["kind"] == "grow" for op in plan)
        ctrl._apply_uta_ops(moe, plan, optimizer=None, step=0)
        summaries.append(_digest(moe))

    assert summaries[0] == summaries[1], (
        "same ops+seeds must produce bit-identical surgery regardless of global RNG"
    )


def test_planning_is_reproducible():
    moe = _moe(num_experts=4, seed=3)
    ctrl = SplitMergeController(
        moe,
        SplitMergeConfig(
            variable_expert_count=True,
            splits_per_call=2,
            min_step_interval=0,
            warmup_steps=0,
        ),
    )
    with torch.no_grad():
        moe.energy.fill_(1.0)
        moe.fatigue.copy_(torch.tensor([0.9, 0.1, 0.9, 0.1]))
    a = ctrl._plan_uta_layer(moe, step=50, layer_index=0)
    b = ctrl._plan_uta_layer(moe, step=50, layer_index=0)
    assert a == b, "planning must be deterministic for identical inputs"
    kinds = [op["kind"] for op in a]
    assert kinds.count("split") >= 2  # both strong experts get twin-splits


# --------------------------------------------------------------------------- #
# 2. THE multi-rank acceptance test (gloo, CPU): ranks stay bit-identical
#    through a grow event INCLUDING optimizer state.
# --------------------------------------------------------------------------- #
def _distributed_worker(rank: int, world: int, store_path: str, out_path: str) -> None:
    status = "exception"
    detail = ""
    try:
        from datetime import timedelta

        dist.init_process_group(
            backend="gloo",
            store=dist.FileStore(store_path, world),
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=60),
        )
        torch.manual_seed(1234)  # identical replicas on every rank

        class Container(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.moe = _moe(num_experts=3, seed=11)

        model = Container()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

        def synced_step(x: torch.Tensor) -> None:
            loss = model.moe(x)[0].pow(2).mean()
            loss.backward()
            # mimic DDP grad averaging so replicas stay identical pre-event
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
            opt.step()
            opt.zero_grad(set_to_none=False)

        gen = torch.Generator().manual_seed(42)
        for _ in range(2):
            synced_step(torch.randn(2, 6, 8, generator=gen))

        cfg = SplitMergeConfig(
            variable_expert_count=True,
            splits_per_call=0,  # any strong expert => growth demand
            resets_per_call=0,
            min_step_interval=0,
            warmup_steps=0,
            ddp_broadcast=True,
        )
        ctrl = SplitMergeController(model, cfg)
        with torch.no_grad():
            model.moe.energy.fill_(1.0)
            model.moe.fatigue.fill_(0.9)  # strong surplus on BOTH ranks identically
        ctrl.step(global_step=0, optimizer=opt)

        synced_step(torch.randn(2, 6, 8, generator=gen))

        E_after = int(model.moe.num_experts)
        name_hashes = {
            name: hashlib.sha256(
                p.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()[:12]
            for name, p in sorted(model.named_parameters())
        }
        opt_hashes = {}
        for gi, group in enumerate(opt.param_groups):
            for p in group["params"]:
                st = opt.state.get(p)
                if st and "exp_avg" in st:
                    opt_hashes[f"g{gi}:{id(p)}"] = hashlib.sha256(
                        st["exp_avg"].detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()[:12]
                else:
                    opt_hashes[f"g{gi}:{id(p)}"] = "<fresh>"
        local = {
            "E": E_after,
            "digest": _digest(model.moe, opt),
            "names": name_hashes,
            "opt": opt_hashes,
        }
        gathered: list[dict[str, Any]] = [{} for _ in range(world)]
        dist.all_gather_object(gathered, local)
        ok = len({g["digest"] for g in gathered}) == 1
        detail = ""
        if not ok:
            base = gathered[0]
            for g in gathered[1:]:
                diff_n = [k for k in g["names"] if g["names"][k] != base["names"].get(k)]
                common_opt = set(g["opt"]) & set(base["opt"])
                diff_o = [
                    k
                    for k in common_opt
                    if g["opt"][k] != base["opt"][k]
                    and base["opt"][k] != "<fresh>"
                    and g["opt"][k] != "<fresh>"
                ]
                fresh_mismatch = sum(
                    1
                    for k in common_opt
                    if (g["opt"][k] == "<fresh>") != (base["opt"][k] == "<fresh>")
                )
                detail = (
                    f"E0={base['E']}/E1={g['E']} "
                    f"diff_params={diff_n[:6]} diff_opt={diff_o[:6]} "
                    f"fresh_flag_diffs={fresh_mismatch}"
                )
        if rank == 0:
            Path(out_path).write_text(f"{int(ok)}:{E_after}:{detail}")
        status = "ok"
    except Exception as exc:  # noqa: BLE001 - report ANY failure to the parent
        detail = f"{type(exc).__name__}: {exc}"
        if rank == 0:
            Path(out_path).write_text(f"error:{detail}")
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        if rank != 0 and status != "ok":
            # make sure the parent learns about non-zero-rank failures too
            with open(out_path + f".rank{rank}", "w") as fh:
                fh.write(f"{status}:{detail}")


def test_two_ranks_remain_bit_identical_through_grow_event(tmp_path: Path):
    ctx = multiprocessing.get_context("spawn")
    store = str(tmp_path / "rendezvous")
    out = str(tmp_path / "result.txt")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=_distributed_worker, args=(rank, 2, store, out))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)
    assert all(p.exitcode == 0 for p in procs), (
        f"distributed workers failed: exitcodes={[p.exitcode for p in procs]}"
    )
    result = Path(out).read_text()
    assert not result.startswith("error"), result
    parts = result.split(":", 2)
    ok, e_after = parts[0], parts[1]
    detail = parts[2] if len(parts) > 2 else ""
    assert ok == "1", f"ranks diverged: {detail}"
    assert int(e_after) > 3, "the engineered pressure must actually grow the layer"


# --------------------------------------------------------------------------- #
# 3. Shrink under DDP: fold+drop stays consistent across ranks.
# --------------------------------------------------------------------------- #
def test_plan_resize_emits_shrink_for_dead_surplus():
    moe = _moe(num_experts=5, seed=5)
    ctrl = SplitMergeController(
        moe,
        SplitMergeConfig(
            variable_expert_count=True,
            splits_per_call=0,
            resets_per_call=1,
            min_step_interval=0,
            warmup_steps=0,
            min_experts=2,
        )
    )
    with torch.no_grad():
        moe.energy.fill_(1.0)
        moe.fatigue.copy_(torch.tensor([0.9, 0.9, 0.0, 0.0, 0.0]))
    ops = ctrl._plan_resize_layer(moe, lambda kind, cnt: 7 + cnt[0], [0])
    assert ops[0]["victims"] == [2, 3]  # resets_per_call keeps the weakest slot busy
    assert ops[0]["keeper"] in (0, 1)
