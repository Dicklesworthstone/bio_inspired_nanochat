"""
Launch (or print) the base_train runs behind the pre-registered ablation matrix (hwxb.5.2).

The matrix in ``bio_inspired_nanochat.ablation_matrix`` says WHAT each column is; ``eval_matrix``
loads the finished ``base_train`` checkpoint of each (column, seed) and scores it. Until this script
existed nothing turned a column into a training command, so the headline experiment had a spec and
a scorer but no producer. Every command below is derived from the spec through
``ablation_matrix.base_train_argv`` — the ``--syn_cfg.<field>=<value>`` overrides, the seed as
``--init_seed`` (what eval_matrix checks the checkpoint against) and the model tag eval_matrix
resolves — so the spec, the runs and the scoring cannot drift apart.

Examples
--------
Print the D1 screening roster and recipe without training:

    python -m scripts.matrix_launch --stage screening \
        --nproc 2 --recipe="--depth=10 --max_seq_len=1024 --tie_embeddings=1 --device_type=cuda --device_batch_size=16 --total_batch_size=65536 --num_iterations=152 --eval_tokens=1048576"

(Write ``--recipe=...`` with the ``=``: the value starts with ``--``, and argparse would otherwise
read a single flag such as ``--depth=10`` as an unknown option.)

Supply the slowest measured aggregate training rate with --tok-per-sec and the slowest
single-GPU scoring time with --score-seconds-per-cell. Add --execute
to run the registered screen. Run the printed score command verbatim. For confirmation,
pass the scored screen via --screen-csv. For a small local rehearsal use, for example:

    python -m scripts.matrix_launch --exploratory --columns vanilla,synaptic_off \
        --recipe="--depth=2 --max_seq_len=64 --device_type=cpu --device_batch_size=2 --total_batch_size=128 --num_iterations=2 --eval_tokens=128" --execute

The structural pair (MoE fixed vs split/merge) is ``--stage structural``; it is not part of the
pre-registered screening set.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat import ablation_matrix as am
from bio_inspired_nanochat.common import get_base_dir
from bio_inspired_nanochat import matrix_manifest as mm
from bio_inspired_nanochat.checkpoint_manager import list_checkpoint_steps, load_checkpoint_metadata
from bio_inspired_nanochat import eval_stats

logger = logging.getLogger("bio_inspired_nanochat.matrix_launch")
console = Console()

STAGES = ("screening", "confirmation", "structural", "all")


def columns_for_stage(stage: str, survivors: Sequence[str] = (), columns: Sequence[str] = ()) -> list[am.AblationConfig]:
    if columns:
        available = am.screening_columns() + am.d1_screening_columns() + am.structural_columns() + am.exploratory_columns()
        known = {c.config_id: c for c in available}
        unknown = [c for c in columns if c not in known]
        if unknown:
            raise ValueError(f"unknown column id(s) {unknown}; known: {sorted(known)}")
        return [known[c] for c in columns]
    if stage == "screening":
        return am.screening_columns()
    if stage == "confirmation":
        if not survivors:
            raise ValueError("--stage confirmation needs --survivors (the columns screening kept)")
        known = {c.config_id for c in am.screening_columns() + am.structural_columns()}
        unknown = [s for s in survivors if s not in known]
        if unknown:
            raise ValueError(f"unknown survivor column id(s) {unknown}; known: {sorted(known)}")
        return am.confirmation_columns(list(survivors))
    if stage == "structural":
        return am.structural_columns()
    if stage == "all":
        return am.screening_columns() + am.structural_columns()
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


def launcher_prefix(nproc: int) -> list[str]:
    """``python -m scripts.base_train`` on one process; ``torchrun`` for ``nproc`` GPUs."""
    if nproc <= 1:
        return [sys.executable, "-m", "scripts.base_train"]
    return [
        "torchrun", "--standalone", f"--nproc_per_node={int(nproc)}",
        "-m", "scripts.base_train",
    ]


def build_commands(
    columns: Sequence[am.AblationConfig], seeds: Sequence[int], recipe: Sequence[str], nproc: int
) -> list[tuple[am.AblationConfig, int, list[str]]]:
    out: list[tuple[am.AblationConfig, int, list[str]]] = []
    for column in columns:
        for seed in seeds:
            argv = am.base_train_argv(column, seed=seed, recipe=tuple(recipe))
            out.append((column, seed, launcher_prefix(nproc) + argv))
    return out


def scoring_command(columns, seeds, recipe, template, out_dir, manifest_path=None, *, bpb_only=False):
    argv = [sys.executable, "-m", "scripts.eval_matrix", "matrix", "--presets",
            ",".join(c.config_id for c in columns), "--seeds", ",".join(map(str, seeds)),
            "--checkpoint-dir", str(Path(get_base_dir()) / "base_checkpoints" / template),
            "--eval-bpb", "--data", "fineweb", "--out-dir", str(out_dir), "--fail-fast"]
    for train_key, option in (("num_iterations", "--checkpoint-step"), ("eval_tokens", "--eval-tokens"),
                              ("device_type", "--device-type"), ("device_batch_size", "--device-batch-size")):
        if train_key in recipe:
            argv.extend([option, recipe[train_key]])
    if manifest_path:
        argv.extend(["--matrix-manifest", str(manifest_path)])
    if bpb_only:
        argv.append("--bpb-only")
    return argv


def bind_command(command, column, seed, manifest, path):
    tag = manifest["checkpoint_template"].format(preset=column.config_id, seed=seed)
    return [*command[:-1], f"--model_tag={tag}", f"--matrix_run_id={manifest['run_id']}",
            f"--matrix_manifest={path}"]


def recipe_fields(flags):
    fields = {}
    for flag in flags:
        key, sep, value = flag.removeprefix("--").partition("=")
        if not flag.startswith("--") or not sep or not key or not value or key in fields:
            raise ValueError(f"expected a unique --key=value training flag, got {flag!r}")
        fields[key] = value
    return fields


def train_cell(command, timeout, lock):
    # torchrun forks rank processes. End the whole training group when its budget expires.
    if timeout is not None:
        watchdog = shutil.which("timeout") or shutil.which("gtimeout")
        if not watchdog:
            raise ValueError("a GPU-hour cap requires the coreutils timeout watchdog")
        # The D1 dual-GPU pilot spent 12.1 s on termination and return after
        # TERM. Reserve grace and process-reaping time inside the allocation.
        # The actual elapsed time is always charged, including OS delays.
        if timeout < 1:
            raise ValueError("less than one second remains in the device-hour allocation")
        grace = min(10.0, timeout / 4)
        settle = min(5.0, timeout / 4)
        command = [watchdog, f"--kill-after={grace:.6f}s", f"{timeout - grace - settle:.6f}s", *command]
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(lock.fileno(),))
    try:
        return child.wait()
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise


def run_cells(manifest, path, columns, seeds, recipe, nproc, restart, *, plan_only=False):
    if restart and not path.is_file():
        raise ValueError("restart requires an existing matching matrix manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    # A process crash releases the OS lock. The manifest and checkpoints survive it.
    with (path.parent / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        timings = path.parent / "attempts.jsonl"
        if path.exists():
            if mm.load_manifest(path) != manifest:
                raise ValueError("matrix run manifest differs from requested execution")
            if not plan_only and timings.exists() and not restart:
                raise ValueError("run already exists; use --restart for this exact manifest")
        else:
            staged = path.with_suffix(".pending")
            with staged.open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, sort_keys=True, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staged, path)
        if plan_only:
            return 0
        # The training child inherits the lock. Once we acquire it after a launcher
        # crash, the old child has exited. Charge the full interval, including idle time.
        attempted = mm.attempt_seconds(timings, recover=True)
        scoring = mm.attempt_seconds(path.parent / "scoring_attempts.jsonl", recover=True)
        spent_seconds = sum(attempted.values()) + sum(scoring.values()) / nproc
        cap = manifest["cap_gpu_hours"]
        if cap is not None and spent_seconds > cap * 3600 / nproc:
            raise ValueError("recorded training and scoring time already exceeded the device-hour cap")
        pending = []
        for column, seed, command in build_commands(columns, seeds, recipe, nproc):
            tag = manifest["checkpoint_template"].format(preset=column.config_id, seed=seed)
            checkpoint = Path(get_base_dir()) / "base_checkpoints" / tag
            command = bind_command(command, column, seed, manifest, path)
            if checkpoint.exists():
                steps = list_checkpoint_steps(str(checkpoint))
                if restart and not steps and tag in attempted and not any(checkpoint.iterdir()):
                    pending.append(command)
                    continue
                if not steps or not restart:
                    raise ValueError(f"checkpoint collision or incomplete first save at {checkpoint}")
                step = steps[-1]
                meta = load_checkpoint_metadata(str(checkpoint), step)
                config = meta.get("user_config", {})
                if (config.get("matrix_run_id") != manifest["run_id"] or config.get("model_tag") != tag
                    or config.get("init_seed") != seed or step > manifest["num_iterations"]):
                    raise ValueError(f"checkpoint identity mismatch at {checkpoint}")
                if step == manifest["num_iterations"]:
                    continue
                command = [f for f in command if not f.startswith("--syn_cfg.")]
                command.append(f"--resume_from_step={step}")
            pending.append(command)
        for command in pending:
            console.print(shlex.join(command), markup=False, soft_wrap=True)
            remaining = (cap * 3600 / nproc - spent_seconds) if cap is not None else None
            if remaining is not None and remaining <= 0:
                raise ValueError("device-hour cap exhausted before next matrix cell")
            tag = next(f.split("=", 1)[1] for f in command if f.startswith("--model_tag="))
            mm.append_event(timings, event="start", tag=tag, started_at=time.time())
            started = time.monotonic()
            code = 1
            try:
                code = train_cell(command, remaining, lock)
            finally:
                seconds = time.monotonic() - started
                if code == 0 and remaining is not None and seconds > remaining:
                    code = 124
                mm.append_event(timings, event="end", tag=tag, seconds=seconds, exit=code)
            spent_seconds += seconds
            if code:
                return code
    return 0


def screening_costs(csv_path, base, current_data, source_sha, recipe, columns, seeds, tokens, nproc, *, bpb_only):
    """Project each arm at its slowest observed screen rate and charge past attempts."""
    with csv_path.open(newline="", encoding="utf-8") as stream:
        first = next(csv.DictReader(stream))
    screen_dir = base / "matrix_runs" / first["matrix_manifest_digest"]
    screen = mm.load_manifest(screen_dir / "manifest.json")
    if json.loads((screen_dir / "scored.json").read_text()) != mm.score_fingerprint(csv_path):
        raise ValueError("screen summary differs from the completed scorer receipt")
    if (not screen["registered"] or screen["stage"] != "screening" or
        screen["run_id"] != first["matrix_manifest_digest"] or screen["data"] != current_data
        or screen["source_sha256"] != source_sha or screen["world_size"] != nproc):
        raise ValueError("screen and confirmation inputs do not match")
    if bpb_only is not screen.get("bpb_only", False):
        raise ValueError("confirmation scoring mode differs from the screen")
    for key in set(recipe) | set(screen["recipe"]):
        if key not in {"num_iterations", "save_every", "eval_every"} and recipe.get(key) != screen["recipe"].get(key):
            raise ValueError(f"confirmation changed the screen recipe field {key}")
    wall = mm.attempt_seconds(screen_dir / "attempts.jsonl")
    scoring = mm.attempt_seconds(screen_dir / "scoring_attempts.jsonl")
    per_arm = {}
    for column in columns:
        tags = [screen["checkpoint_template"].format(preset=column.config_id, seed=s) for s in screen["seeds"]]
        if any(tag not in wall or tag not in scoring for tag in tags):
            raise ValueError(f"missing observed training or scoring cost for {column.config_id}")
        train_device_seconds = max(wall[tag] for tag in tags) * tokens / screen["tokens_per_cell"] * nproc
        score_device_seconds = max(scoring[tag] for tag in tags)
        per_arm[column.config_id] = (train_device_seconds + score_device_seconds) * len(seeds) / 3600
    return per_arm, (sum(wall.values()) * nproc + sum(scoring.values())) / 3600


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, default="screening")
    parser.add_argument("--survivors", default="", help="Comma-separated column ids kept after screening")
    parser.add_argument("--columns", default="", help="Comma-separated explicit column ids (overrides --stage; for re-running single cells)")
    parser.add_argument(
        "--seeds", default="",
        help="Comma-separated init seeds (default: the stage's pre-registered seeds)",
    )
    parser.add_argument(
        "--recipe", default="",
        help="Shared base_train flags for every cell, as one shell-quoted string "
        "(depth, batch, tokens, data, eval cadence, ...). Write it as --recipe=\"--depth=10 ...\" "
        "(with the =), since the value itself starts with --",
    )
    parser.add_argument("--nproc", type=int, default=1, help="GPUs per run; >1 uses torchrun")
    parser.add_argument("--execute", action="store_true", help="Run the commands sequentially (default: print only)")
    parser.add_argument("--plan", action="store_true", help="Resolve and validate exact executable commands against local data without starting training")
    parser.add_argument("--exploratory", action="store_true", help="Run a custom recipe outside the registered D1 study")
    parser.add_argument("--restart", action="store_true", help="Continue a matching execution from complete checkpoints")
    parser.add_argument("--screen-csv", type=Path, help="Complete scored D1 screen for confirmation admission")
    parser.add_argument("--tok-per-sec", type=float, help="Conservative aggregate training rate for the initial screen")
    parser.add_argument("--score-seconds-per-cell", type=float, help="Measured conservative single-GPU scoring wall time for one screening cell")
    parser.add_argument("--bpb-only", action="store_true", help="Freeze BPB-only checkpoint scoring in the matrix manifest")
    parser.add_argument("--cap-gpu-hours", type=float, default=am.DEFAULT_GPU_HOUR_CAP)
    parser.add_argument("--out-dir", type=Path, help="Root for the printed scorer's outputs")
    args = parser.parse_args(argv)
    if args.restart and not args.execute:
        parser.error("--restart requires --execute")

    survivors = [s for s in args.survivors.split(",") if s]
    explicit = [c for c in args.columns.split(",") if c]
    if args.nproc < 1 or not 0 < args.cap_gpu_hours <= am.DEFAULT_GPU_HOUR_CAP:
        parser.error("positive world size and a finite cap within the project allocation are required")
    registered = not args.exploratory and args.stage in {"screening", "confirmation"}
    complete_plan = args.execute or args.plan or args.screen_csv is not None
    if args.stage == "confirmation" and args.exploratory:
        parser.error("confirmation always requires an admitted screen; choose an exploratory screening run instead")
    if registered and args.stage == "confirmation":
        if args.screen_csv:
            decision = eval_stats.d1_screen(eval_stats.load_d1_csv(args.screen_csv, stage="screening"))
            if decision["status"] != "go":
                parser.error(f"screen decision is {decision['status']}")
            if survivors and survivors != decision["survivors"]:
                parser.error("explicit survivors disagree with the screen")
            survivors = decision["survivors"]
        elif complete_plan:
            parser.error("confirmation execution requires --screen-csv")
        plan = eval_stats.d1_confirmation_plan(survivors)
        columns = columns_for_stage(args.stage, columns=plan["columns"])
        default_seeds = plan["seeds"]
    elif registered:
        columns = am.d1_screening_columns()
        default_seeds = list(am.SCREENING_SEEDS)
    else:
        columns = columns_for_stage(args.stage, survivors, explicit)
        default_seeds = list(am.SCREENING_SEEDS)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else default_seeds
    if len(set(seeds)) != len(seeds) or len(set(explicit)) != len(explicit):
        parser.error("duplicate seeds or columns")
    if registered and (seeds != default_seeds or (explicit and explicit != [c.config_id for c in columns])):
        parser.error("registered stage cannot override the frozen seeds or columns; use --exploratory for other runs")
    recipe = shlex.split(args.recipe)
    fields = recipe_fields(recipe)
    controlled = {"model_tag", "init_seed", "synapses", "resume_from_step", "matrix_run_id", "matrix_manifest",
                  "load_cmaes_params", "use_moe", "splitmerge_every", "neuromod_enabled", "topological_nas"}
    if any(k in controlled or k.startswith("syn_cfg.") or any(k in c.train_overrides for c in columns) for k in fields):
        parser.error("shared recipe attempts to override a cell-controlled setting")
    base = Path(get_base_dir())
    template = am.MATRIX_CHECKPOINT_TEMPLATE
    path = None
    out_dir = args.out_dir or base / "matrix_scores"
    if complete_plan:
        batch = int(fields.get("total_batch_size", 0))
        steps = int(fields.get("num_iterations", 0))
        if batch < 1 or steps < 1:
            parser.error("execution needs explicit positive --total_batch_size and --num_iterations")
        tokens = steps * batch
        data = mm.dataset_identity()
        revision, source_sha = mm.source_identity()
        per_arm = {}
        screen_spend = 0.0
        if registered:
            if (args.nproc != 2 or fields.get("device_type") != "cuda" or fields.get("depth") != "10"
                or fields.get("max_seq_len") != "1024" or fields.get("tie_embeddings") != "1"
                or int(fields.get("eval_tokens", 0)) < 1_000_000):
                parser.error("D1 requires two CUDA ranks, depth=10, context=1024, tied embeddings and >=1M eval tokens")
            if tokens != mm.registered_tokens(args.stage, batch):
                parser.error("training tokens differ from the registered stage budget")
            if args.stage == "confirmation":
                per_arm, screen_spend = screening_costs(args.screen_csv, base, data, source_sha, fields,
                                                         columns, seeds, tokens, args.nproc, bpb_only=args.bpb_only)
            else:
                if args.tok_per_sec is None:
                    parser.error("screen admission needs its measured conservative --tok-per-sec")
                if (args.score_seconds_per_cell is None or not math.isfinite(args.score_seconds_per_cell)
                    or args.score_seconds_per_cell <= 0):
                    parser.error("screen admission needs a positive measured --score-seconds-per-cell")
                score_hours = len(seeds) * args.score_seconds_per_cell / 3600
                per_arm = {c.config_id: score_hours + am.estimate_gpu_hours([c], tuple(seeds), tokens,
                            args.tok_per_sec, world_size=args.nproc) for c in columns}
            if screen_spend + sum(per_arm.values()) > args.cap_gpu_hours:
                parser.error("registered plan exceeds the aggregate device-hour cap")
        elif not args.exploratory:
            parser.error("structural and all-stage execution require explicit --exploratory")
        column_flags = {c.config_id: recipe_fields(am.base_train_argv(c, seed=0)[:-2]) for c in columns}
        manifest = mm.seal_manifest({
            **({"bpb_only": True} if args.bpb_only else {}),
            "schema": 1, "stage": args.stage, "registered": registered, "git_revision": revision,
            "source_sha256": source_sha, "data": data, "recipe": fields, "seeds": seeds,
            "columns": [c.config_id for c in columns], "column_flags": column_flags, "num_iterations": steps,
            "total_batch_size": batch, "world_size": args.nproc, "tokens_per_cell": tokens,
            "protocol_id": eval_stats.D1_PROTOCOL_VERSION if registered else "exploratory",
            "screen_summary_sha256": mm.file_digest(args.screen_csv) if args.screen_csv else None,
            "screen_rate_tokens_per_second": args.tok_per_sec if registered and args.stage == "screening" else None,
            "screen_scoring_seconds_per_cell": args.score_seconds_per_cell if registered and args.stage == "screening" else None,
            "per_arm_gpu_hours": per_arm, "screen_gpu_hours": screen_spend,
            "cap_gpu_hours": args.cap_gpu_hours - screen_spend if fields.get("device_type") != "cpu" else None,
        })
        template = manifest["checkpoint_template"]
        path = base / "matrix_runs" / manifest["run_id"] / "manifest.json"
        out_dir = out_dir / manifest["run_id"]
        result = run_cells(manifest, path, columns, seeds, recipe, args.nproc, args.restart,
                           plan_only=not args.execute)
        if result:
            return result
    commands = build_commands(columns, seeds, recipe, args.nproc)

    table = Table(title=f"matrix {args.stage}: {len(columns)} columns x {len(seeds)} seeds = {len(commands)} runs")
    table.add_column("column")
    table.add_column("seed", justify="right")
    table.add_column("base_train arguments beyond the recipe")
    for column, seed, cmd in commands:
        table.add_row(column.config_id, str(seed), " ".join(cmd[len(launcher_prefix(args.nproc)) + len(recipe):]))
    console.print(table)
    for column, seed, cmd in ([] if args.execute else commands):
        if complete_plan:
            cmd = bind_command(cmd, column, seed, manifest, path)
        console.print(shlex.join(cmd), markup=False, highlight=False, soft_wrap=True)
    if not complete_plan:
        console.print("Unbound recipe preview. Use --plan with staged data to freeze the run and print its scoring command.")
        return 0
    console.print("\n[bold]score with:[/bold]")
    console.print(shlex.join(scoring_command(columns, seeds, fields, template, out_dir, path, bpb_only=args.bpb_only)),
                  markup=False, highlight=False, soft_wrap=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
