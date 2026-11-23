"""Exercise the production two-rank CUDA save, interrupt, and resume path (hwxb.2.6)."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from bio_inspired_nanochat.checkpoint_manager import require_complete_checkpoint

console = Console()
STAGES = ("control", "interrupt", "resume", "incomplete", "compare")
RTOL = 1e-3
ATOL = 1e-4


def checkpoint(arm: Path) -> Path:
    return arm / "base_checkpoints" / "dual"


def copy_boundary(source: Path, destination: Path, step: int, *, omit_rank_one: bool) -> None:
    require_complete_checkpoint(str(source), step)
    tag = f"{step:06d}"
    names = [".checkpoint_commit_regime_v1.json", f"model_{tag}.pt", f"meta_{tag}.json",
             f"optim_{tag}_rank0.pt", f"optim_{tag}_rank1.pt", f"train_{tag}_rank0.pt"]
    if not omit_rank_one:
        names.append(f"train_{tag}_rank1.pt")
    # Preserve the same final commit-marker ordering as the production writer.
    for name in [*names, f"commit_{tag}.json"]:
        shutil.copy2(source / name, destination / name)


def train_stage(stage: str, root: Path, data: Path) -> None:
    arm = root / ("branch" if stage in ("interrupt", "resume") else stage)
    directory = checkpoint(arm)
    if stage != "resume":
        # Starting a stage twice must never replace a previous experiment's split.
        arm.mkdir(parents=True, exist_ok=False)
        directory.mkdir(parents=True)
        for name in ("base_data", "tokenizer"):
            source = (data / name).resolve(strict=True)
            (arm / name).symlink_to(source, target_is_directory=True)
    if stage == "interrupt":
        copy_boundary(checkpoint(root / "control"), directory, 1, omit_rank_one=False)
    elif stage == "resume":
        require_complete_checkpoint(str(directory), 2)
    elif stage == "incomplete":
        copy_boundary(checkpoint(root / "branch"), directory, 2, omit_rank_one=True)

    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
        "-m", "scripts.base_train", "--device_type=cuda", "--model_tag=dual", "--depth=2",
        "--max_seq_len=32", "--device_batch_size=1", "--total_batch_size=64",
        "--num_iterations=3", "--eval_every=1", "--eval_tokens=64",
        "--core_metric_every=-1", "--sample_every=-1", "--save_every=1",
        f"--neuroviz_dir={arm / 'neuroviz'}",
    ]
    if stage == "control":
        command.extend([
            "--synapses=1", "--num_experts=2", "--moe_top_k=1", "--splitmerge_every=1",
            "--sm_warmup_steps=0", "--sm_verbose=1", "--neuromod_enabled=1",
            "--neuromod_log_every=1", "--syn_cfg.enable_hebbian=1",
        ])
    else:
        command.append(f"--resume_from_step={1 if stage == 'interrupt' else 2}")
    env = {
        **os.environ, "NANOCHAT_BASE_DIR": str(arm),
        "BIO_RESULTS_REGISTRY": str(arm / "registry.jsonl"), "NCCL_P2P_DISABLE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    console.print(stage, command)
    started = time.monotonic()
    terminated = False
    log_path = root / f"{stage}.log"
    with log_path.open("x") as stream:
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            if stage == "interrupt":
                marker = directory / "commit_000002.json"
                while proc.poll() is None and time.monotonic() - started < 2700:
                    if marker.exists() and json.loads(marker.read_text()).get("complete") is True:
                        # torchrun forwards SIGTERM to these experiment workers.
                        proc.terminate()
                        terminated = True
                        break
                    time.sleep(0.08)
            rc = proc.wait(timeout=max(1.0, 2700 - (time.monotonic() - started)))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    elapsed = time.monotonic() - started
    timing = {"exit_code": rc, "wall_s": elapsed, "gpu_hours": 2 * elapsed / 3600}
    (root / f"{stage}.timing.json").write_text(json.dumps(timing) + "\n")
    console.print(stage, timing, str(log_path))
    if stage == "interrupt":
        if not terminated or rc == 0 or (directory / "commit_000003.json").exists():
            raise AssertionError(f"Training missed its required step-2 interruption: {log_path}")
        require_complete_checkpoint(str(directory), 2)
    elif stage == "incomplete":
        if rc == 0 or "Checkpoint step 2 is incomplete or uncommitted" not in log_path.read_text():
            raise AssertionError(f"The trainer did not reject the missing rank-1 shard: {log_path}")
        if (directory / "commit_000003.json").exists():
            raise AssertionError("The incomplete checkpoint advanced")
    else:
        if rc:
            raise RuntimeError(f"Training exited {rc}: {log_path}\n{log_path.read_text()[-1600:]}")
        require_complete_checkpoint(str(directory), 3)


def compare_state(a: Any, b: Any, name: str, *, exact: bool, stats: dict[str, Any]) -> None:
    """Collect bitwise differences and retain every tolerance violation by component."""
    if type(a) is not type(b):
        stats["mismatches"].append(f"{name}: type")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            stats["mismatches"].append(f"{name}: keys")
        else:
            for key in a:
                compare_state(a[key], b[key], f"{name}.{key}", exact=exact, stats=stats)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            stats["mismatches"].append(f"{name}: length")
        else:
            for i, (lhs, rhs) in enumerate(zip(a, b, strict=True)):
                compare_state(lhs, rhs, f"{name}.{i}", exact=exact, stats=stats)
    elif isinstance(a, torch.Tensor):
        stats["tensors"] += 1
        if a.dtype != b.dtype or a.shape != b.shape:
            stats["mismatches"].append(f"{name}: dtype or shape")
        elif not torch.equal(a, b):
            # A pretraining run has no external reward. NeuromodulatoryBus uses
            # a scalar NaN for its uninitialized reward EMA; require identical bits.
            if (name.endswith(".neuromod.reward_ema") and a.ndim == 0 and a.is_floating_point()
                    and torch.isnan(a).item()
                    and torch.equal(a.reshape(1).view(torch.uint8), b.reshape(1).view(torch.uint8))):
                return
            stats["non_bitwise_tensors"] += 1
            if a.is_floating_point() and a.numel():
                delta = float((a - b).abs().max())
                stats["max_abs_delta"] = max(stats["max_abs_delta"], delta)
            if exact or not a.is_floating_point() or not torch.allclose(a, b, rtol=RTOL, atol=ATOL):
                stats["mismatches"].append(name)
    elif a != b and (exact or not isinstance(a, float) or not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)):
        stats["mismatches"].append(name)


def compare(root: Path) -> dict[str, Any]:
    directories = [checkpoint(root / arm) for arm in ("control", "branch")]

    def load(filename: str) -> list[Any]:
        if filename.endswith(".json"):
            return [json.loads((d / filename).read_text()) for d in directories]
        return [torch.load(d / filename, map_location="cpu", weights_only=True) for d in directories]

    report: dict[str, Any] = {"rtol": RTOL, "atol": ATOL, "steps": []}
    for step in (1, 2, 3):
        tag = f"{step:06d}"
        for d in directories:
            require_complete_checkpoint(str(d), step)
        m, n = load(f"commit_{tag}.json")
        assert m == n and m["world_size"] == 2 and m["train_shards"] and m["optimizer_shards"]
        a, b = load(f"meta_{tag}.json")
        for meta in (a, b):
            assert meta["synapses"] and meta["model_config"]["use_moe"]
            assert meta["splitmerge"]["every"] == 1 and meta["splitmerge"]["config"]["warmup_steps"] == 0
            assert math.isfinite(meta["val_bpb"])
        result: dict[str, Any] = {"step": step, "val_bpb": [a["val_bpb"], b["val_bpb"]],
                                  "components": {}, "controller_last_steps": []}
        report["steps"].append(result)

        def check(component: str, lhs: Any, rhs: Any, *, exact: bool = False) -> None:
            counts = {"tensors": 0, "non_bitwise_tensors": 0, "max_abs_delta": 0.0, "mismatches": []}
            compare_state(lhs, rhs, component, exact=exact, stats=counts)
            result["components"][component] = counts

        check("val_bpb", a["val_bpb"], b["val_bpb"])
        check("smooth_train_loss", a["loop_state"]["smooth_train_loss"], b["loop_state"]["smooth_train_loss"])
        models = load(f"model_{tag}.pt")
        assert any("post." in key for key in models[0]) and any("experts" in key for key in models[0])
        check("model", *models)
        for rank in (0, 1):
            opt_a, opt_b = load(f"optim_{tag}_rank{rank}.pt")
            assert len(opt_a) == len(opt_b) >= 2 and all(group["param_groups"] for group in opt_a)
            check(f"rank{rank}.optimizers", opt_a, opt_b)
            train_a, train_b = load(f"train_{tag}_rank{rank}.pt")
            for state in (train_a, train_b):
                assert state["step"] == step and state["dataloader_state_dict"]["rank"] == rank
                assert all(key in state["rng"] for key in ("torch", "python", "numpy", "cuda"))
                assert state["splitmerge"]["last_step"] >= step - 1
            result["controller_last_steps"].append([train_a["splitmerge"]["last_step"], train_b["splitmerge"]["last_step"]])
            for key in ("rng", "prefetched_batch", "dataloader_state_dict", "splitmerge",
                        "rank_model_state", "neuromod", "divergence_guard"):
                assert train_a[key] and train_b[key]
                check(f"rank{rank}.{key}", train_a[key], train_b[key],
                      exact=key in ("rng", "prefetched_batch", "dataloader_state_dict", "splitmerge"))
            result.setdefault("deferred_plasticity_present", []).append(
                ["deferred_plasticity" in state for state in (train_a, train_b)]
            )
            if "deferred_plasticity" in train_a or "deferred_plasticity" in train_b:
                for state in (train_a, train_b):
                    assert state["deferred_plasticity"]
                    if step == 1:
                        assert any(entry["pending"] for entry in state["deferred_plasticity"].values())
                check(f"rank{rank}.deferred_plasticity", train_a["deferred_plasticity"],
                      train_b["deferred_plasticity"], exact=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Experiment directory for retained splits and logs.")
    parser.add_argument("--base-dir", type=Path, help="Prepared Nanochat base containing base_data and tokenizer.")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    for stage in STAGES if args.stage == "all" else (args.stage,):
        if stage == "compare":
            report = compare(root)
            path = root / "comparison.json"
            serialized = json.dumps(report, indent=2) + "\n"
            if path.exists():
                if path.read_text() != serialized:
                    raise FileExistsError(f"Existing comparison has different content: {path}")
            else:
                with path.open("x") as stream:
                    stream.write(serialized)
            failed = [name for step in report["steps"] for name, stats in step["components"].items()
                      if stats["mismatches"]]
            console.print(str(path), "failed components", failed)
            if failed:
                raise AssertionError(f"Two-rank trajectories differ; see {path}")
        else:
            if args.base_dir is None:
                parser.error("--base-dir is required to train or resume")
            train_stage(stage, root, args.base_dir)


if __name__ == "__main__":
    main()
