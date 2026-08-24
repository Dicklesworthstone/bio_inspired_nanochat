"""Budgeted, append-only execution of pre-registered experiments.

The runner is intentionally a small orchestration layer over existing project harnesses.  It can
launch only ``scripts.eval_matrix`` confirmatory cells and the odq.2 probe/lesion/stimulation
diagnostic.  It never accepts an executable or free-form command from a hypothesis or the CLI.

Every attempted cell gets an audit ``RunRecord`` in the committed results registry, including
failed and timed-out attempts.  The separate batch ledger is append-only and freezes the exact
commands and budgets before the first process starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.ablation_registry import ABLATION_PRESETS
from bio_inspired_nanochat.hypothesis_generator import (
    DEFAULT_PREREGISTRY,
    PreregisteredHypothesis,
    read_preregistrations,
    results_snapshot_digest,
)
from bio_inspired_nanochat.results_registry import (
    DEFAULT_REGISTRY,
    RunRecord,
    append_record,
    make_record,
    read_records,
)
from bio_inspired_nanochat.synaptic import SynapticConfig

DEFAULT_BATCH_LEDGER = "results/experiment_batches.jsonl"
DEFAULT_OUTPUT_ROOT = "runs/ai_neuroscientist"

_EVAL_ALIASES = frozenset(
    {
        "eval_matrix",
        "working_memory_suite",
        "structural_falsification",
        "uncertainty_calibration",
    }
)
_CAUSAL_MODULE = "scripts.e2e.probe_lesion_stim"
_CAUSAL_SCRIPT = Path("scripts/e2e/probe_lesion_stim.py")
_MAX_CAPTURE_CHARS = 4_000

Arm = Literal["control", "intervention", "diagnostic"]
CellKind = Literal["eval_matrix", "causal_probe"]
CellStatus = Literal["completed", "failed", "timed_out", "invalidated"]


@dataclass(frozen=True)
class HardBudget:
    """Global execution limits; all three are mandatory and fail closed."""

    maximum_runs: int
    maximum_total_tokens: int
    maximum_wall_seconds: float

    def __post_init__(self) -> None:
        if self.maximum_runs < 1:
            raise ValueError("maximum_runs must be positive")
        if self.maximum_total_tokens < 0:
            raise ValueError("maximum_total_tokens cannot be negative")
        if not math.isfinite(self.maximum_wall_seconds) or self.maximum_wall_seconds <= 0:
            raise ValueError("maximum_wall_seconds must be finite and positive")


@dataclass(frozen=True)
class EvalMatrixOptions:
    """Allowlisted eval-matrix knobs; there is deliberately no free-form extra-args field."""

    train_tokens: int
    eval_tokens: int
    checkpoint_dir: str = ""
    inline_smoke_training: bool = False
    device_type: str = "cpu"
    data: str = "synthetic"
    sequence_len: int = 32
    vocab_size: int = 64
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 32
    device_batch_size: int = 1
    total_batch_size_tokens: int = 32

    def __post_init__(self) -> None:
        if self.train_tokens < 1 or self.eval_tokens < 1:
            raise ValueError("train_tokens and eval_tokens must be positive")
        if self.inline_smoke_training == bool(self.checkpoint_dir.strip()):
            raise ValueError(
                "choose exactly one execution source: checkpoint_dir or inline_smoke_training"
            )
        if self.device_type not in {"cpu", "cuda", "mps"}:
            raise ValueError("device_type must be cpu, cuda, or mps")
        if self.data not in {"fineweb", "synthetic"}:
            raise ValueError("data must be fineweb or synthetic")
        dimensions = (
            self.sequence_len,
            self.vocab_size,
            self.n_layer,
            self.n_head,
            self.n_embd,
            self.device_batch_size,
            self.total_batch_size_tokens,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("model and batch dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        micro_tokens = self.device_batch_size * self.sequence_len
        if self.total_batch_size_tokens % micro_tokens:
            raise ValueError(
                "total_batch_size_tokens must be divisible by device_batch_size*sequence_len"
            )

    @property
    def tokens_per_run(self) -> int:
        return self.train_tokens + self.eval_tokens


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    kind: CellKind
    arm: Arm
    seed: int
    token_charge: int
    command: tuple[str, ...]
    output_dir: str
    hypothesis_id: str | None = None
    preregistration_batch_id: str | None = None
    mechanism: str | None = None
    primary_metric: str | None = None
    preset: str | None = None


@dataclass(frozen=True)
class BatchPlan:
    schema_version: int
    execution_batch_id: str
    created_at: str
    git_sha: str | None
    results_snapshot_sha256: str
    budget: HardBudget
    projected_runs: int
    projected_tokens: int
    output_root: str
    cells: tuple[ExperimentCell, ...]


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class CellReceipt:
    cell_id: str
    status: CellStatus
    returncode: int | None
    elapsed_seconds: float
    charged_tokens: int
    source_run_ids: tuple[str, ...]
    audit_run_id: str
    reason: str


@dataclass(frozen=True)
class BatchReport:
    execution_batch_id: str
    status: str
    spent_runs: int
    spent_tokens: int
    elapsed_seconds: float
    receipts: tuple[CellReceipt, ...]


Executor = Callable[[Sequence[str], Path, float], ProcessOutcome]


def _canonical_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _resolve_inside(path: str | Path, *, root: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    if resolved == root_resolved or not resolved.is_relative_to(root_resolved):
        raise ValueError(f"{label} must be a child of the project root")
    return resolved


def _effective_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    defaults = SynapticConfig()
    known = set(defaults.__dataclass_fields__)
    unknown = set(config) - known
    if unknown:
        raise ValueError(f"hypothesis contains unknown SynapticConfig fields: {sorted(unknown)}")
    return {
        field: value
        for field, value in config.items()
        if value != getattr(defaults, field)
    }


def config_to_ablation_preset(config: Mapping[str, Any]) -> str:
    """Map a frozen hypothesis arm to an existing canonical ablation preset."""
    effective = _effective_overrides(config)
    candidates = [
        preset
        for preset, overrides in ABLATION_PRESETS.items()
        if preset != "vanilla" and overrides == effective
    ]
    if len(candidates) != 1:
        raise ValueError(
            "hypothesis arm is not representable by exactly one existing ablation preset: "
            f"effective_overrides={effective!r}"
        )
    return candidates[0]


def _eval_command(
    *,
    preset: str,
    seed: int,
    primary_metric: str,
    output_dir: Path,
    registry_path: Path,
    options: EvalMatrixOptions,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "scripts.eval_matrix",
        "run",
        "--preset",
        preset,
        "--seed",
        str(seed),
        "--device-type",
        options.device_type,
        "--data",
        options.data,
        "--out-dir",
        str(output_dir),
        "--registry-path",
        str(registry_path),
        "--train-tokens",
        str(options.train_tokens),
        "--eval-tokens",
        str(options.eval_tokens),
        "--sequence-len",
        str(options.sequence_len),
        "--vocab-size",
        str(options.vocab_size),
        "--n-layer",
        str(options.n_layer),
        "--n-head",
        str(options.n_head),
        "--n-embd",
        str(options.n_embd),
        "--device-batch-size",
        str(options.device_batch_size),
        "--total-batch-size-tokens",
        str(options.total_batch_size_tokens),
        "--continual-tasks",
        "2",
        "--continual-exposures",
        "1",
        "--niah-lengths",
        str(max(1, min(16, options.sequence_len - 2))),
        "--fail-fast",
    ]
    if options.inline_smoke_training:
        command.append("--inline-smoke-training")
    else:
        command.extend(("--checkpoint-dir", options.checkpoint_dir))
    if primary_metric == "eval_bpb":
        command.append("--eval-bpb")
    if primary_metric in {"dead_expert_frac", "moe_gini"}:
        command.extend(("--use-moe", "--num-experts", "4", "--moe-top-k", "2"))
    return tuple(command)


def build_batch_plan(
    hypotheses: Sequence[PreregisteredHypothesis],
    *,
    budget: HardBudget,
    eval_options: EvalMatrixOptions,
    registry_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    causal_probe_seeds: Sequence[int] = (),
    project_root: str | Path | None = None,
    created_at: str | None = None,
) -> BatchPlan:
    """Build and validate the complete immutable plan before any harness starts."""
    if not hypotheses and not causal_probe_seeds:
        raise ValueError("at least one hypothesis or causal probe seed is required")
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    registry = _resolve_inside(registry_path, root=root, label="registry_path")
    output = _resolve_inside(output_root, root=root, label="output_root")
    snapshot = results_snapshot_digest(registry)

    digests = {item.results_snapshot_sha256 for item in hypotheses}
    if len(digests) > 1:
        raise ValueError("all hypotheses in one execution batch must share an evidence snapshot")
    if digests and digests != {snapshot}:
        raise ValueError(
            "results registry changed after pre-registration; generate a new hypothesis batch"
        )
    prereg_batch_ids = {item.batch_id for item in hypotheses}
    if len(prereg_batch_ids) > 1:
        raise ValueError("all hypotheses in one execution batch must share a preregistration batch")

    identity_payload = {
        "hypotheses": [item.hypothesis_id for item in hypotheses],
        "causal_probe_seeds": list(causal_probe_seeds),
        "snapshot": snapshot,
        "budget": asdict(budget),
        "eval": asdict(eval_options),
    }
    execution_batch_id = _canonical_id("exec", identity_payload)
    batch_output = output / execution_batch_id
    cells: list[ExperimentCell] = []

    for hypothesis in hypotheses:
        if hypothesis.status != "preregistered":
            raise ValueError(f"hypothesis {hypothesis.hypothesis_id} is not preregistered")
        if hypothesis.harness not in _EVAL_ALIASES:
            raise ValueError(
                f"hypothesis harness {hypothesis.harness!r} is not an approved existing runner"
            )
        if eval_options.tokens_per_run > hypothesis.compute_budget.maximum_tokens_per_run:
            raise ValueError(
                f"requested {eval_options.tokens_per_run} tokens exceeds "
                f"{hypothesis.hypothesis_id}'s per-run cap "
                f"{hypothesis.compute_budget.maximum_tokens_per_run}"
            )
        if hypothesis.compute_budget.maximum_runs != 2 * len(hypothesis.paired_seeds):
            raise ValueError(f"{hypothesis.hypothesis_id} does not freeze one pair per seed")
        for arm, arm_config in (
            ("control", hypothesis.control),
            ("intervention", hypothesis.intervention),
        ):
            preset = config_to_ablation_preset(arm_config)
            for seed in hypothesis.paired_seeds:
                cell_id = (
                    f"{execution_batch_id}-{hypothesis.hypothesis_id.removeprefix('hyp-')[:10]}-"
                    f"{arm}-s{seed}"
                )
                cell_output = batch_output / cell_id
                cells.append(
                    ExperimentCell(
                        cell_id=cell_id,
                        kind="eval_matrix",
                        arm=arm,
                        seed=seed,
                        token_charge=eval_options.tokens_per_run,
                        command=_eval_command(
                            preset=preset,
                            seed=seed,
                            primary_metric=hypothesis.primary_metric,
                            output_dir=cell_output,
                            registry_path=registry,
                            options=eval_options,
                        ),
                        output_dir=str(cell_output),
                        hypothesis_id=hypothesis.hypothesis_id,
                        preregistration_batch_id=hypothesis.batch_id,
                        mechanism=hypothesis.mechanism,
                        primary_metric=hypothesis.primary_metric,
                        preset=preset,
                    )
                )

    if causal_probe_seeds:
        causal_script = root / _CAUSAL_SCRIPT
        if not causal_script.is_file():
            raise ValueError(f"approved causal harness is unavailable: {causal_script}")
    if len(causal_probe_seeds) != len(set(causal_probe_seeds)):
        raise ValueError("causal probe seeds must be unique")
    for seed in causal_probe_seeds:
        cell_id = f"{execution_batch_id}-causal-probe-s{seed}"
        cell_output = batch_output / cell_id
        cells.append(
            ExperimentCell(
                cell_id=cell_id,
                kind="causal_probe",
                arm="diagnostic",
                seed=int(seed),
                token_charge=0,
                command=(
                    sys.executable,
                    "-m",
                    _CAUSAL_MODULE,
                    "--run-dir",
                    str(cell_output),
                    "--device",
                    eval_options.device_type,
                    "--seed",
                    str(seed),
                ),
                output_dir=str(cell_output),
            )
        )

    projected_runs = len(cells)
    projected_tokens = sum(cell.token_charge for cell in cells)
    if projected_runs > budget.maximum_runs:
        raise ValueError(
            f"plan requires {projected_runs} runs but hard cap is {budget.maximum_runs}"
        )
    if projected_tokens > budget.maximum_total_tokens:
        raise ValueError(
            f"plan requires {projected_tokens} tokens but hard cap is "
            f"{budget.maximum_total_tokens}"
        )
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("execution plan contains duplicate cell IDs")

    return BatchPlan(
        schema_version=1,
        execution_batch_id=execution_batch_id,
        created_at=created_at or datetime.now(UTC).isoformat(),
        git_sha=_current_git_sha(root),
        results_snapshot_sha256=snapshot,
        budget=budget,
        projected_runs=projected_runs,
        projected_tokens=projected_tokens,
        output_root=str(batch_output),
        cells=tuple(cells),
    )


def _current_git_sha(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.JSONDecoder().decode(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid batch ledger at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"invalid batch ledger row at {path}:{line_number}")
            rows.append(value)
    return rows


def _default_executor(command: Sequence[str], cwd: Path, timeout: float) -> ProcessOutcome:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return ProcessOutcome(completed.returncode, completed.stdout, completed.stderr)


def _single_command_value(command: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"allowlisted command must contain exactly one {flag}")
    return command[positions[0] + 1]


def _validate_runtime_cell(
    cell: ExperimentCell,
    *,
    root: Path,
    batch_output: Path,
    registry: Path,
) -> None:
    """Revalidate an externally constructed plan before trusting any command or path."""
    cell_output = _resolve_inside(cell.output_dir, root=batch_output, label="cell.output_dir")
    if cell_output.parent != batch_output:
        raise ValueError("cell output directories must be direct children of the batch directory")
    if not cell.command or Path(cell.command[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("cell executable is not the allowlisted Python interpreter")
    if cell.kind == "eval_matrix":
        if tuple(cell.command[1:4]) != ("-m", "scripts.eval_matrix", "run"):
            raise ValueError("eval cell command is not the allowlisted eval-matrix harness")
        command_output = Path(_single_command_value(cell.command, "--out-dir")).resolve()
        command_registry = Path(_single_command_value(cell.command, "--registry-path")).resolve()
        if command_output != cell_output or command_registry != registry:
            raise ValueError("eval cell command paths do not match the immutable plan")
    elif cell.kind == "causal_probe":
        if tuple(cell.command[1:3]) != ("-m", _CAUSAL_MODULE):
            raise ValueError("diagnostic command is not the allowlisted causal harness")
        command_output = Path(_single_command_value(cell.command, "--run-dir")).resolve()
        if command_output != cell_output:
            raise ValueError("causal cell command path does not match the immutable plan")
    else:
        raise ValueError(f"unsupported cell kind {cell.kind!r}")


def _audit_record(
    cell: ExperimentCell,
    *,
    status: CellStatus,
    source: RunRecord | None,
    returncode: int | None,
    reason: str,
    elapsed_seconds: float,
) -> RunRecord:
    metrics: dict[str, float] = {}
    if source is not None and cell.primary_metric is not None:
        metrics[cell.primary_metric] = source.metrics[cell.primary_metric]
    notes = json.dumps(
        {
            "orchestrator": "ai_neuroscientist",
            "cell_kind": cell.kind,
            "status": status,
            "hypothesis_id": cell.hypothesis_id,
            "preregistration_batch_id": cell.preregistration_batch_id,
            "mechanism": cell.mechanism,
            "arm": cell.arm,
            "preset": cell.preset,
            "source_run_id": source.run_id if source else None,
            "returncode": returncode,
            "reason": reason,
            "elapsed_seconds": elapsed_seconds,
            "token_charge": cell.token_charge,
            "output_dir": cell.output_dir,
            "command": list(cell.command),
        },
        sort_keys=True,
    )
    invalid = status != "completed"
    return make_record(
        "eval",
        metrics,
        run_id=cell.cell_id,
        config={
            "kind": cell.kind,
            "hypothesis_id": cell.hypothesis_id,
            "arm": cell.arm,
            "seed": cell.seed,
            "token_charge": cell.token_charge,
            "command": list(cell.command),
        },
        seed=cell.seed,
        notes=notes,
        verdict="invalidated" if invalid else None,
        eligible_for_best=False,
    )


def execute_batch_plan(
    plan: BatchPlan,
    *,
    registry_path: str | Path,
    batch_ledger_path: str | Path = DEFAULT_BATCH_LEDGER,
    project_root: str | Path | None = None,
    executor: Executor = _default_executor,
) -> BatchReport:
    """Execute all planned cells without exceeding the preflight run/token/wall caps."""
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    registry = _resolve_inside(registry_path, root=root, label="registry_path")
    ledger = _resolve_inside(batch_ledger_path, root=root, label="batch_ledger_path")
    batch_output = _resolve_inside(plan.output_root, root=root, label="plan.output_root")
    if results_snapshot_digest(registry) != plan.results_snapshot_sha256:
        raise ValueError("registry snapshot changed between planning and execution")
    if batch_output.exists():
        raise ValueError(f"refusing to overwrite existing batch output: {batch_output}")
    if any(
        row.get("execution_batch_id") == plan.execution_batch_id
        for row in _read_jsonl(ledger)
    ):
        raise ValueError(f"execution batch {plan.execution_batch_id} is already in the ledger")
    existing_records = read_records(str(registry))
    existing_ids = {record.run_id for record in existing_records}
    duplicate_ids = existing_ids & {cell.cell_id for cell in plan.cells}
    if duplicate_ids:
        raise ValueError(f"execution cell IDs already exist in registry: {sorted(duplicate_ids)}")
    for cell in plan.cells:
        _validate_runtime_cell(
            cell,
            root=root,
            batch_output=batch_output,
            registry=registry,
        )

    batch_output.mkdir(parents=True, exist_ok=False)
    _append_jsonl(
        ledger,
        {
            "event": "batch_planned",
            "execution_batch_id": plan.execution_batch_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "plan": asdict(plan),
        },
    )

    started = time.monotonic()
    receipts: list[CellReceipt] = []
    spent_tokens = 0
    for run_index, cell in enumerate(plan.cells, start=1):
        elapsed = time.monotonic() - started
        remaining_wall = plan.budget.maximum_wall_seconds - elapsed
        if run_index > plan.budget.maximum_runs:
            raise RuntimeError("internal guard: run budget exhausted")
        if spent_tokens + cell.token_charge > plan.budget.maximum_total_tokens:
            raise RuntimeError("internal guard: token budget exhausted")
        if remaining_wall <= 0:
            raise RuntimeError("hard wall-clock budget exhausted before next cell")
        if Path(cell.output_dir).exists():
            raise ValueError(f"refusing to overwrite existing cell output: {cell.output_dir}")

        before = read_records(str(registry))
        cell_started = time.monotonic()
        outcome: ProcessOutcome | None = None
        status: CellStatus
        reason: str
        try:
            outcome = executor(cell.command, root, remaining_wall)
            status = "completed" if outcome.returncode == 0 else "failed"
            reason = "existing harness exited successfully" if status == "completed" else (
                f"existing harness exited {outcome.returncode}; "
                f"stderr_tail={outcome.stderr[-_MAX_CAPTURE_CHARS:]!r}"
            )
        except subprocess.TimeoutExpired:
            status = "timed_out"
            reason = "hard wall-clock budget expired while the harness was running"
        cell_elapsed = time.monotonic() - cell_started
        after = read_records(str(registry))
        if after[: len(before)] != before:
            raise RuntimeError("results registry changed non-append-only during harness execution")
        source_records = after[len(before) :]
        source: RunRecord | None = None
        if status == "completed" and cell.primary_metric is not None:
            matching = [
                record
                for record in source_records
                if record.seed == cell.seed and cell.primary_metric in record.metrics
            ]
            if len(matching) != 1:
                status = "invalidated"
                reason = (
                    "successful harness must append exactly one seed-matched registry record "
                    f"containing {cell.primary_metric}; found {len(matching)}"
                )
            else:
                source = matching[0]

        audit = _audit_record(
            cell,
            status=status,
            source=source,
            returncode=outcome.returncode if outcome else None,
            reason=reason,
            elapsed_seconds=cell_elapsed,
        )
        append_record(audit, str(registry))
        spent_tokens += cell.token_charge
        receipt = CellReceipt(
            cell_id=cell.cell_id,
            status=status,
            returncode=outcome.returncode if outcome else None,
            elapsed_seconds=cell_elapsed,
            charged_tokens=cell.token_charge,
            source_run_ids=tuple(record.run_id for record in source_records),
            audit_run_id=audit.run_id,
            reason=reason,
        )
        receipts.append(receipt)
        _append_jsonl(
            ledger,
            {
                "event": "cell_finished",
                "execution_batch_id": plan.execution_batch_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "receipt": asdict(receipt),
            },
        )

    elapsed = time.monotonic() - started
    batch_status = (
        "completed"
        if all(item.status == "completed" for item in receipts)
        else "completed_with_failures"
    )
    report = BatchReport(
        execution_batch_id=plan.execution_batch_id,
        status=batch_status,
        spent_runs=len(receipts),
        spent_tokens=spent_tokens,
        elapsed_seconds=elapsed,
        receipts=tuple(receipts),
    )
    _append_jsonl(
        ledger,
        {
            "event": "batch_finished",
            "execution_batch_id": plan.execution_batch_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "report": asdict(report),
        },
    )
    return report


def _render_plan(plan: BatchPlan) -> None:
    table = Table(title=f"AI-neuroscientist batch {plan.execution_batch_id}")
    table.add_column("Cell")
    table.add_column("Kind")
    table.add_column("Arm")
    table.add_column("Seed", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Preset")
    for cell in plan.cells:
        table.add_row(
            cell.cell_id,
            cell.kind,
            cell.arm,
            str(cell.seed),
            str(cell.token_charge),
            cell.preset or "—",
        )
    console = Console()
    console.print(table)
    console.print(
        f"[bold]Projected:[/bold] {plan.projected_runs}/{plan.budget.maximum_runs} runs, "
        f"{plan.projected_tokens}/{plan.budget.maximum_total_tokens} tokens, "
        f"wall cap {plan.budget.maximum_wall_seconds:g}s"
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run preregistered experiments under hard budgets")
    parser.add_argument("--preregistry", default=DEFAULT_PREREGISTRY)
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY)
    parser.add_argument("--batch-ledger", default=DEFAULT_BATCH_LEDGER)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hypothesis", action="append", default=None)
    parser.add_argument("--causal-probe-seed", type=int, action="append", default=[])
    parser.add_argument("--maximum-runs", type=int, required=True)
    parser.add_argument("--maximum-total-tokens", type=int, required=True)
    parser.add_argument("--maximum-wall-seconds", type=float, required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--eval-tokens", type=int, required=True)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--inline-smoke-training", action="store_true")
    parser.add_argument("--device-type", default="cpu")
    parser.add_argument("--data", default="synthetic")
    parser.add_argument("--sequence-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--total-batch-size-tokens", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    hypotheses = read_preregistrations(args.preregistry)
    if args.hypothesis:
        selected = set(args.hypothesis)
        known = {item.hypothesis_id for item in hypotheses}
        if not selected <= known:
            parser.error(f"unknown hypothesis IDs: {sorted(selected - known)}")
        hypotheses = [item for item in hypotheses if item.hypothesis_id in selected]
    plan = build_batch_plan(
        hypotheses,
        budget=HardBudget(
            args.maximum_runs,
            args.maximum_total_tokens,
            args.maximum_wall_seconds,
        ),
        eval_options=EvalMatrixOptions(
            train_tokens=args.train_tokens,
            eval_tokens=args.eval_tokens,
            checkpoint_dir=args.checkpoint_dir,
            inline_smoke_training=args.inline_smoke_training,
            device_type=args.device_type,
            data=args.data,
            sequence_len=args.sequence_len,
            vocab_size=args.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            device_batch_size=args.device_batch_size,
            total_batch_size_tokens=args.total_batch_size_tokens,
        ),
        registry_path=args.registry_path,
        output_root=args.output_root,
        causal_probe_seeds=args.causal_probe_seed,
    )
    _render_plan(plan)
    if args.dry_run:
        Console().print("[yellow]Dry run:[/yellow] no processes or ledgers were changed")
        return 0
    report = execute_batch_plan(
        plan,
        registry_path=args.registry_path,
        batch_ledger_path=args.batch_ledger,
    )
    style = "green" if report.status == "completed" else "red"
    Console().print(
        f"[{style}]{report.status}[/{style}]: {report.spent_runs} runs, "
        f"{report.spent_tokens} tokens, {report.elapsed_seconds:.2f}s"
    )
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(_main())
    except (RuntimeError, TypeError, ValueError) as exc:
        Console().print(f"[bold red]Experiment batch aborted:[/bold red] {exc}")
        raise SystemExit(2) from exc
