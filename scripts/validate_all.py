r"""One-Command Master Validation Report Generator ("Is Everything Working Perfectly?") (beads eqyk.16, eqyk).

Orchestrates and executes the complete verification corpus across:
  1. Foundational E2E Workflows (Training, CMA-ES, Hebbian, Structural, Sleep, Neuromod RL, Interpretability)
  2. Cross-Backend Parity & Mathematical Property Invariants (Triton / Rust / Python reference)
  3. Formal Theory Certificates & Monotone Lyapunov Invariants (0642 Thrusts)
  4. Capability Frontier Batteries (Wave-1 Deliberation/Metabolic & Wave-2 Compositions)
  5. Retrofit, MGR Attention & Uncertainty Calibration Batteries
  6. Performance Regression Throughput Gates against Committed Baselines

Outputs:
  - ``reports/validation_report_latest.md``: Human-readable markdown audit with provenance.
  - ``reports/validation_report_latest.json``: Machine-readable structured validation evidence.
  - Rich interactive summary in console.

Run:
    python -m scripts.validate_all
    python -m scripts.validate_all --suite fast
    pytest tests/test_validate_all.py -v
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.checkpoint_manager import _git_sha
from bio_inspired_nanochat.results_registry import _hardware_string

DEFAULT_REPORTS_DIR = Path("reports")


@dataclass
class SubsystemCheck:
    id: str
    name: str
    category: str
    command: list[str]
    timeout_sec: float = 120.0
    passed: bool = False
    duration_sec: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    invariants_count: int = 0


@dataclass
class MasterValidationReport:
    run_id: str
    timestamp: str
    git_sha: str
    hardware: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    total_duration_sec: float
    all_passed: bool
    verdict: str  # "ALL SYSTEMS PERFECT" | "DEGRADED / FAILURES DETECTED"
    checks: list[SubsystemCheck] = field(default_factory=list)


VALIDATION_SUITES: dict[str, list[SubsystemCheck]] = {
    "unit": [
        SubsystemCheck(
            id="unit_core",
            name="Unit Tests (Ablations, Metrics, Checkpoints, Registry)",
            category="Foundations",
            command=["pytest", "tests/test_ablation_registry.py", "tests/test_results_registry.py", "tests/test_metrics_schema.py", "tests/test_checkpoint_roundtrip.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
    ],
    "e2e_foundations": [
        SubsystemCheck(
            id="e2e_bio_train",
            name="Full Bio Training Loop E2E (eqyk.4)",
            category="Foundations E2E",
            command=["pytest", "tests/test_e2e_train_bio.py", "-v", "--tb=short"],
            timeout_sec=60.0,
        ),
        SubsystemCheck(
            id="e2e_cmaes",
            name="CMA-ES Tune Loop & Resume E2E (eqyk.7)",
            category="Foundations E2E",
            command=["pytest", "tests/test_e2e_cmaes.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
        SubsystemCheck(
            id="e2e_online_hebbian",
            name="Online Hebbian & Bistable Plasticity E2E (eqyk.8)",
            category="Foundations E2E",
            command=["pytest", "tests/test_e2e_online_learning.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
        SubsystemCheck(
            id="e2e_structural_evolution",
            name="Structural Evolution & Neurogenesis E2E (eqyk.9)",
            category="Foundations E2E",
            command=["pytest", "tests/test_e2e_structural_lifecycle.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
        SubsystemCheck(
            id="e2e_wake_sleep",
            name="Wake/Sleep Consolidation & Forgetting E2E (eqyk.10)",
            category="Foundations E2E",
            command=["pytest", "tests/test_e2e_wake_sleep.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
        SubsystemCheck(
            id="e2e_neuromod_rl",
            name="Neuromodulated 3-Factor RL Micro-Run E2E (eqyk.11)",
            category="Foundations E2E",
            command=["pytest", "tests/test_e2e_neuromod_rl.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
        SubsystemCheck(
            id="e2e_interpretability",
            name="Interpretability Probe/Lesion/Stimulation E2E (eqyk.12)",
            category="Foundations E2E",
            command=["pytest", "tests/test_e2e_probe_lesion_stim.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
    ],
    "theory_and_parity": [
        SubsystemCheck(
            id="parity_backends",
            name="Cross-Backend Parity (Triton/Rust/Python Reference) (eqyk.13)",
            category="Parity & Invariants",
            command=["pytest", "tests/test_presyn_backend_parity.py", "-v", "--tb=short"],
            timeout_sec=60.0,
        ),
        SubsystemCheck(
            id="property_invariants",
            name="Property-Based Metamorphic Invariants Suite (eqyk.14)",
            category="Parity & Invariants",
            command=["pytest", "tests/test_property_invariants.py", "-v", "--tb=short"],
            timeout_sec=60.0,
        ),
        SubsystemCheck(
            id="theory_certificates",
            name="Formal Theory Certificates & Lyapunov Invariants (eqyk.18)",
            category="Theory & Proofs",
            command=["pytest", "tests/test_e2e_theory_artifacts.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
    ],
    "capability_frontier": [
        SubsystemCheck(
            id="capability_frontier_1",
            name="Capability Frontier I Batteries (Deliberation/Adaptive/Scientist) (eqyk.19)",
            category="Capability Frontier",
            command=["pytest", "tests/test_e2e_capability_frontier.py", "-v", "--tb=short"],
            timeout_sec=60.0,
        ),
        SubsystemCheck(
            id="uncertainty_calibration",
            name="Uncertainty Calibration & Selective Prediction (eqyk.20)",
            category="Capability Frontier",
            command=["pytest", "tests/test_e2e_uncertainty.py", "-v", "--tb=short"],
            timeout_sec=60.0,
        ),
        SubsystemCheck(
            id="retrofit_mgr_geometry",
            name="Synaptic Retrofit & MGR Attention Variants Battery (eqyk.21)",
            category="Capability Frontier",
            command=["pytest", "tests/test_e2e_retrofit_mgr.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
        SubsystemCheck(
            id="wave2_compositions",
            name="Wave-2 Emergent Compositions (Self-Correct, Metacognition, Search) (eqyk.22)",
            category="Capability Frontier",
            command=["pytest", "tests/test_e2e_wave2_compositions.py", "-v", "--tb=short"],
            timeout_sec=45.0,
        ),
    ],
    "perf_regression": [
        SubsystemCheck(
            id="perf_regression_gates",
            name="Performance Regression Throughput Gates (eqyk.15)",
            category="Performance",
            command=["python", "-m", "scripts.perf_regression_gate", "--mode", "check", "--tolerance", "0.50"],
            timeout_sec=45.0,
        ),
    ],
}


def run_single_check(
    check: SubsystemCheck, *, verbose: bool = True, timeout_scale: float = 1.0
) -> SubsystemCheck:
    """Execute a single subsystem verification command.

    ``timeout_scale`` multiplies the per-check budget; the budgets were tuned on a fast
    dev box and hosted CI runners are several times slower.
    """
    if not math.isfinite(timeout_scale) or timeout_scale <= 0:
        raise ValueError(f"timeout_scale must be a positive finite number, got {timeout_scale!r}")
    cmd = ["uv", "run", "--no-sync"] + check.command
    budget_sec = check.timeout_sec * timeout_scale
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=budget_sec,
            check=False,
        )
        duration = time.perf_counter() - t0
        passed = (proc.returncode == 0)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        error_msg = "" if passed else f"Command exited with code {proc.returncode}"
    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - t0
        passed = False
        stdout = ""
        stderr = ""
        error_msg = f"Command timed out after {budget_sec:.1f}s"
    except Exception as exc:  # noqa: BLE001 — a runner must turn ANY failure into a failed check
        duration = time.perf_counter() - t0
        passed = False
        stdout = ""
        stderr = str(exc)
        error_msg = f"Execution exception: {exc}"

    return SubsystemCheck(
        id=check.id,
        name=check.name,
        category=check.category,
        command=check.command,
        timeout_sec=check.timeout_sec,
        passed=passed,
        duration_sec=duration,
        stdout=stdout,
        stderr=stderr,
        error_message=error_msg,
    )


def generate_validation_report(
    checks: list[SubsystemCheck],
    out_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> MasterValidationReport:
    """Generate Markdown and JSON reports from executed checks."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.datetime.now(datetime.UTC).isoformat()
    run_id = f"val-{int(time.time())}"
    git_sha = _git_sha() or "unknown"
    hardware = _hardware_string()

    total = len(checks)
    passed = sum(1 for c in checks if c.passed)
    failed = total - passed
    total_duration = sum(c.duration_sec for c in checks)
    all_passed = (failed == 0)
    verdict = "ALL SYSTEMS PERFECT (VERIFIED WORKING)" if all_passed else f"DEGRADED ({failed} FAILURE(S))"

    report = MasterValidationReport(
        run_id=run_id,
        timestamp=now_utc,
        git_sha=git_sha,
        hardware=hardware,
        total_checks=total,
        passed_checks=passed,
        failed_checks=failed,
        total_duration_sec=total_duration,
        all_passed=all_passed,
        verdict=verdict,
        checks=checks,
    )

    # 1. Render Markdown Report
    md_lines = [
        f"# System Master Validation Report — {run_id}",
        "",
        f"> **Verdict**: **{verdict}**  ",
        f"> **Timestamp**: `{now_utc}`  ",
        f"> **Git SHA**: `{git_sha}`  ",
        f"> **Hardware**: `{hardware}`  ",
        f"> **Summary**: `{passed}/{total} Passed` ({passed/max(1,total):.1%}) in `{total_duration:.2f}s`",
        "",
        "---",
        "",
        "## Subsystem Verification Matrix",
        "",
        "| Category | Subsystem | Status | Duration | Command |",
        "| :--- | :--- | :---: | :---: | :--- |",
    ]

    for c in checks:
        status_badge = "✅ **PASS**" if c.passed else "❌ **FAIL**"
        cmd_str = f"`{' '.join(c.command)}`"
        md_lines.append(f"| {c.category} | {c.name} | {status_badge} | {c.duration_sec:.2f}s | {cmd_str} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## Detailed Subsystem Findings & Logs",
        "",
    ])

    for c in checks:
        status_title = "PASS" if c.passed else "FAIL"
        md_lines.append(f"### [{status_title}] {c.name} (`{c.id}`)")
        md_lines.append(f"- **Category**: {c.category}")
        md_lines.append(f"- **Duration**: {c.duration_sec:.2f}s")
        if c.error_message:
            md_lines.append(f"- **Error**: `{c.error_message}`")
        if c.stdout:
            tail_stdout = "\n".join(c.stdout.splitlines()[-15:])
            md_lines.append(f"```text\n{tail_stdout}\n```")
        if c.stderr and not c.passed:
            tail_stderr = "\n".join(c.stderr.splitlines()[-15:])
            md_lines.append(f"**Stderr**:\n```text\n{tail_stderr}\n```")
        md_lines.append("")

    md_content = "\n".join(md_lines)

    # Save timestamped and latest markdown report
    ts_compact = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    ts_report_file = out_path / f"validation_report_{ts_compact}.md"
    latest_report_file = out_path / "validation_report_latest.md"
    latest_json_file = out_path / "validation_report_latest.json"

    ts_report_file.write_text(md_content, encoding="utf-8")
    latest_report_file.write_text(md_content, encoding="utf-8")
    latest_json_file.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    return report


def run_validation(
    suite: str = "all",
    *,
    out_dir: Path | str = DEFAULT_REPORTS_DIR,
    fail_fast: bool = False,
    verbose: bool = True,
    timeout_scale: float = 1.0,
) -> MasterValidationReport:
    """Run specified validation suite and generate audit report."""
    console = Console(quiet=not verbose)
    console.print(f"[bold cyan]Starting Master Validation Run (suite='{suite}')...[/bold cyan]")

    selected_checks: list[SubsystemCheck] = []
    if suite == "all":
        for cat_checks in VALIDATION_SUITES.values():
            selected_checks.extend(cat_checks)
    elif suite == "fast":
        selected_checks.extend(VALIDATION_SUITES["unit"])
        selected_checks.extend(VALIDATION_SUITES["theory_and_parity"])
        selected_checks.append(VALIDATION_SUITES["perf_regression"][0])
    elif suite in VALIDATION_SUITES:
        selected_checks.extend(VALIDATION_SUITES[suite])
    else:
        raise ValueError(f"Unknown validation suite '{suite}'. Choose from: all, fast, {list(VALIDATION_SUITES.keys())}")

    executed_checks: list[SubsystemCheck] = []
    for check in selected_checks:
        console.print(f"  [dim]▶ Running: {check.name}...[/dim]")
        res = run_single_check(check, verbose=verbose, timeout_scale=timeout_scale)
        executed_checks.append(res)
        status_text = "[green]✓ PASS[/green]" if res.passed else f"[red]✗ FAIL ({res.error_message})[/red]"
        console.print(f"    {status_text} in {res.duration_sec:.2f}s")
        if fail_fast and not res.passed:
            console.print("[bold red]Fail-fast triggered; stopping remaining checks.[/bold red]")
            break

    report = generate_validation_report(executed_checks, out_dir=out_dir)

    # Render summary table to console
    table = Table(title=f"Master Validation Summary ({report.verdict})")
    table.add_column("Category", style="cyan")
    table.add_column("Subsystem", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")

    for c in report.checks:
        stat = "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]"
        table.add_row(c.category, c.name, stat, f"{c.duration_sec:.2f}s")

    console.print(table)
    console.print(
        f"[bold {'green' if report.all_passed else 'red'}]Report generated at: "
        f"{Path(out_dir) / 'validation_report_latest.md'}[/bold {'green' if report.all_passed else 'red'}]"
    )

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Master Validation Report Generator")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "fast", "unit", "e2e_foundations", "theory_and_parity", "capability_frontier", "perf_regression"],
        help="Validation suite scope (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory to store markdown and json reports",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first check failure",
    )
    parser.add_argument(
        "--timeout-scale",
        type=float,
        default=1.0,
        help="Multiply every per-check timeout (e.g. 4 on slow hosted CI runners)",
    )
    args = parser.parse_args(argv)

    report = run_validation(
        suite=args.suite,
        out_dir=args.out_dir,
        fail_fast=args.fail_fast,
        verbose=True,
        timeout_scale=args.timeout_scale,
    )
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
