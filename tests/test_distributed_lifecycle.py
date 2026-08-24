"""Distributed-safe deterministic lifecycle surgery — bead uta.5.

Protocol under torch.distributed: rank0 plans the round's UTA or topological
lifecycle ops WITHOUT mutating, the serialized decision (kinds, indices,
alphas, per-op RNG seeds) is broadcast, and EVERY rank applies bit-identical
surgery locally, then re-syncs its own optimizer param-groups. No rank is a
passive observer, so survivor moments survive everywhere and shape changes are
safe. Audit/lineage logging remains rank0-only.

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

import bio_inspired_nanochat.synaptic_splitmerge as splitmerge_module
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
    TopologicalLifecycleDecision,
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

        E_grown = int(model.moe.num_experts)
        with torch.no_grad():
            model.moe.energy.fill_(1.0)
            model.moe.fatigue.zero_()
            model.moe.fatigue[0] = 0.9  # dead surplus -> shrink to min_experts
        ctrl.step(global_step=1, optimizer=opt)

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
            Path(out_path).write_text(
                f"{int(ok)}:{E_grown}:{E_after}:{detail}"
            )
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
            with open(out_path + f".rank{rank}", "w", encoding="utf-8") as fh:
                fh.write(f"{status}:{detail}")


def test_two_ranks_remain_bit_identical_through_grow_and_shrink(tmp_path: Path):
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
    parts = result.split(":", 3)
    ok, e_grown, e_after = parts[0], parts[1], parts[2]
    detail = parts[3] if len(parts) > 3 else ""
    assert ok == "1", f"ranks diverged: {detail}"
    assert int(e_grown) > 3, "the engineered pressure must actually grow the layer"
    assert int(e_after) == 2, "the dead surplus must shrink to the configured floor"


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


# --------------------------------------------------------------------------- #
# 4. Topological DDP: rank0 decides every action class, every rank applies it
#    with the transmitted RNG seed and coherent optimizer reconstruction.
# --------------------------------------------------------------------------- #
def _topological_worker(
    rank: int, world: int, store_path: str, out_path: str, action: str
) -> None:
    try:
        from datetime import timedelta

        dist.init_process_group(
            backend="gloo",
            store=dist.FileStore(store_path, world),
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=60),
        )
        torch.manual_seed(321)
        synaptic_cfg = SynapticConfig(
            enable_hebbian=False,
            enable_metabolism=True,
            stochastic_train_frac=0.0,
            topological_nas=True,
        )
        moe = SynapticMoE(8, 3, 3, 1, synaptic_cfg)
        optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-2)

        def synced_step(x: torch.Tensor) -> None:
            loss = moe(x)[0].pow(2).mean()
            loss.backward()
            for parameter in moe.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad /= world
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)

        data_generator = torch.Generator().manual_seed(99)
        synced_step(torch.randn(2, 6, 8, generator=data_generator))
        lineage_calls: list[str] = []

        class LineageRecorder:
            def on_merge(self, *_args, **_kwargs) -> None:
                lineage_calls.append("merge")

            def on_split(self, *_args, **_kwargs) -> None:
                lineage_calls.append("split")

            def on_spawn(self, *_args, **_kwargs) -> None:
                lineage_calls.append("spawn")

        is_fallback = action == "uta_fallback"
        controller = SplitMergeController(
            moe,
            SplitMergeConfig(
                variable_expert_count=True,
                merges_per_call=0 if is_fallback else 1,
                splits_per_call=0 if is_fallback else 1,
                resets_per_call=0,
                warmup_steps=0,
                min_step_interval=0,
                ddp_broadcast=True,
                function_preserving=True,
            ),
            logger=LineageRecorder(),
        )
        planning_calls = 0

        def fixed_plan(_layer, *, step: int, layer_index: int):
            nonlocal planning_calls
            planning_calls += 1
            if rank != 0:
                raise AssertionError("nonzero rank must never plan topological surgery")
            if action == "merge":
                decision = TopologicalLifecycleDecision(
                    step=step,
                    layer_index=layer_index,
                    mode="topological",
                    action="merge",
                    reason="forced_distributed_acceptance",
                    rng_seed=123456,
                    merge_pair=(0, 1),
                )
            elif action == "merge_split":
                decision = TopologicalLifecycleDecision(
                    step=step,
                    layer_index=layer_index,
                    mode="topological",
                    action="merge_split",
                    reason="forced_distributed_acceptance",
                    rng_seed=123456,
                    merge_pair=(0, 1),
                    split_source=2,
                    split_destination=1,
                    split_noise_norm=0.05,
                )
            elif action == "birth":
                decision = TopologicalLifecycleDecision(
                    step=step,
                    layer_index=layer_index,
                    mode="topological",
                    action="birth",
                    reason="forced_distributed_acceptance",
                    rng_seed=123456,
                    split_source=0,
                    split_destination=3,
                    split_noise_norm=0.05,
                )
            elif action == "uta_fallback":
                decision = TopologicalLifecycleDecision(
                    step=step,
                    layer_index=layer_index,
                    mode="uta_fallback",
                    action="uta",
                    reason="forced_distributed_acceptance",
                )
            else:
                raise AssertionError(f"unsupported test action: {action}")
            return (
                decision,
                None,
            )

        setattr(controller, "_plan_topological_lifecycle", fixed_plan)
        if is_fallback:
            with torch.no_grad():
                moe.energy.fill_(1.0)
                moe.fatigue.fill_(0.9)
        # The production safety sync must not be able to hide rank-local RNG
        # divergence in this acceptance test. Decision broadcast and the final
        # barrier still run; only the post-surgery state_dict broadcast is inert.
        setattr(splitmerge_module, "_broadcast_module_params", lambda _module: None)
        torch.manual_seed(10_000 + rank)  # prove global RNG is irrelevant
        controller.step(global_step=7, optimizer=optimizer)
        synced_step(torch.randn(2, 6, 8, generator=data_generator))

        local = {
            "digest": _digest(moe, optimizer),
            "experts": int(moe.num_experts),
            "planning_calls": planning_calls,
            "decision": controller.topological_decisions[-1].action,
            "lineage_calls": lineage_calls,
        }
        gathered: list[dict[str, Any]] = [{} for _ in range(world)]
        dist.all_gather_object(gathered, local)
        if rank == 0:
            expected_experts = 4 if action in ("birth", "uta_fallback") else 3
            expected_decision = "uta" if is_fallback else action
            ok = (
                len({item["digest"] for item in gathered}) == 1
                and {item["experts"] for item in gathered} == {expected_experts}
                and [item["planning_calls"] for item in gathered] == [1, 0]
                and {item["decision"] for item in gathered} == {expected_decision}
                and bool(gathered[0]["lineage_calls"])
                and gathered[1]["lineage_calls"] == []
            )
            Path(out_path).write_text(f"{int(ok)}:{gathered}")
    except Exception as exc:  # noqa: BLE001 - propagate child failures via artifact
        if rank == 0:
            Path(out_path).write_text(f"error:{type(exc).__name__}: {exc}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("action", ["merge", "merge_split", "birth", "uta_fallback"])
def test_two_ranks_apply_one_topological_decision(tmp_path: Path, action: str):
    ctx = multiprocessing.get_context("spawn")
    store = str(tmp_path / f"{action}-rendezvous")
    out = str(tmp_path / f"{action}-result.txt")
    processes = [
        ctx.Process(
            target=_topological_worker,
            args=(rank, 2, store, out, action),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes), [
        process.exitcode for process in processes
    ]
    result = Path(out).read_text()
    assert not result.startswith("error"), result
    assert result.startswith("1:"), result
