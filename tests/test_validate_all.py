"""Tests for the Master Validation Report Generator (beads eqyk.16, eqyk)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_all import (
    SubsystemCheck,
    generate_validation_report,
    main as validate_main,
    run_validation,
)


def test_generate_validation_report_all_passed(tmp_path: Path):
    """Report correctly summarizes passing checks into Markdown and JSON."""
    checks = [
        SubsystemCheck(
            id="test_check_1",
            name="Core Unit Tests",
            category="Foundations",
            command=["pytest", "tests/test_results_registry.py"],
            passed=True,
            duration_sec=1.23,
            stdout="1 passed in 1.23s",
        ),
        SubsystemCheck(
            id="test_check_2",
            name="Theory Certificates",
            category="Theory",
            command=["pytest", "tests/test_e2e_theory.py"],
            passed=True,
            duration_sec=2.45,
            stdout="3 passed in 2.45s",
        ),
    ]

    report = generate_validation_report(checks, out_dir=tmp_path)
    assert report.total_checks == 2
    assert report.passed_checks == 2
    assert report.failed_checks == 0
    assert report.all_passed
    assert "ALL SYSTEMS PERFECT" in report.verdict

    # Check that output files were generated
    latest_md = tmp_path / "validation_report_latest.md"
    latest_json = tmp_path / "validation_report_latest.json"

    assert latest_md.exists()
    assert latest_json.exists()

    md_text = latest_md.read_text(encoding="utf-8")
    assert "System Master Validation Report" in md_text
    assert "Core Unit Tests" in md_text
    assert "Theory Certificates" in md_text

    data = json.loads(latest_json.read_text(encoding="utf-8"))
    assert data["total_checks"] == 2
    assert data["all_passed"] is True


def test_generate_validation_report_with_failure(tmp_path: Path):
    """Report correctly highlights failed checks and records error messages."""
    checks = [
        SubsystemCheck(
            id="check_pass",
            name="Passing Check",
            category="Foundations",
            command=["true"],
            passed=True,
            duration_sec=0.5,
        ),
        SubsystemCheck(
            id="check_fail",
            name="Failing Check",
            category="Foundations",
            command=["false"],
            passed=False,
            duration_sec=0.8,
            error_message="Command exited with code 1",
            stderr="AssertionError: failed",
        ),
    ]

    report = generate_validation_report(checks, out_dir=tmp_path)
    assert not report.all_passed
    assert report.failed_checks == 1
    assert "DEGRADED" in report.verdict

    md_text = (tmp_path / "validation_report_latest.md").read_text(encoding="utf-8")
    assert "❌ **FAIL**" in md_text
    assert "AssertionError: failed" in md_text


def test_validate_all_fast_suite_run(tmp_path: Path):
    """The fast validation suite executes and emits reports."""
    report = run_validation(suite="fast", out_dir=tmp_path, verbose=False)
    assert report.total_checks >= 3
    assert (tmp_path / "validation_report_latest.md").exists()
    assert (tmp_path / "validation_report_latest.json").exists()


def test_validate_all_cli_entrypoint(tmp_path: Path):
    """The CLI entrypoint works with arguments."""
    ret = validate_main([
        "--suite", "unit",
        "--out-dir", str(tmp_path),
    ])
    assert ret == 0
    assert (tmp_path / "validation_report_latest.json").exists()
    assert (tmp_path / "validation_report_latest.md").exists()
