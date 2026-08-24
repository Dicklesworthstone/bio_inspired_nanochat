"""Formal Lean↔Python drift-gate tests (bead r00r.4.3)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.formal_feedback import (
    FormalFeedbackError,
    run_compiled_lean_audit,
    validate_manifest,
)

pytestmark = pytest.mark.unit

_APPROVED_AXIOMS = ["propext", "Classical.choice", "Quot.sound"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    root: Path,
    *,
    cycle_id: str,
    theorem_id: str = "BioInspiredNanochat.Example.certified",
    proof_path: str = "formal/lean/Proof.lean",
    runtime_path: str = "tests/test_runtime.py",
) -> dict[str, object]:
    return {
        "cycle_id": cycle_id,
        "theorem_ids": [theorem_id],
        "artifact_hash": f"sha256:{_sha256(root / proof_path)}",
        "hash_scope": proof_path,
        "sorry_or_admit_present": False,
        "axioms": list(_APPROVED_AXIOMS),
        "runtime_mapping": [
            {"path": runtime_path, "sha256": _sha256(root / runtime_path)}
        ],
    }


def _contract(tmp_path: Path) -> tuple[Path, Path, Path]:
    proof = tmp_path / "formal/lean/Proof.lean"
    runtime = tmp_path / "tests/test_runtime.py"
    manifest = tmp_path / "formal/proof_artifacts.json"
    proof.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    proof.write_text(
        "namespace BioInspiredNanochat.Example\n"
        "theorem certified : True := by trivial\n"
        "#print axioms certified\n"
        "end BioInspiredNanochat.Example\n",
        encoding="utf-8",
    )
    runtime.write_text("def test_runtime():\n    assert True\n", encoding="utf-8")
    return proof, runtime, manifest


def _write_manifest(manifest: Path, artifacts: list[dict[str, object]]) -> None:
    manifest.write_text(
        json.dumps({"schema_version": 1, "artifacts": artifacts}),
        encoding="utf-8",
    )


def test_live_formal_contract_matches_repository_head():
    root = Path(__file__).resolve().parents[1]

    report = validate_manifest(repo_root=root)

    assert report.artifact_count >= 2
    assert "formal/lean/BioInspiredNanochat.lean" in report.proof_paths
    assert (
        "BioInspiredNanochat.StochasticThermodynamics.tur_calibration_inequality"
        in report.theorem_ids
    )
    assert "tests/test_stochastic_thermo.py" in report.test_paths


def test_latest_occurrence_wins_for_proof_and_runtime_paths(tmp_path):
    proof, runtime, manifest = _contract(tmp_path)
    old = _record(tmp_path, cycle_id="old")
    proof.write_text(proof.read_text(encoding="utf-8") + "-- revised\n", encoding="utf-8")
    runtime.write_text(
        runtime.read_text(encoding="utf-8") + "# revised\n", encoding="utf-8"
    )
    current = _record(tmp_path, cycle_id="current")
    _write_manifest(manifest, [old, current])

    report = validate_manifest(manifest, repo_root=tmp_path)

    assert report.artifact_count == 2
    assert report.test_paths == ("tests/test_runtime.py",)


def test_explicit_retirement_allows_absent_historical_paths(tmp_path):
    proof, runtime, manifest = _contract(tmp_path)
    old = {
        "cycle_id": "old",
        "theorem_ids": ["BioInspiredNanochat.Deleted.obsolete"],
        "artifact_hash": f"sha256:{'0' * 64}",
        "hash_scope": "formal/lean/Deleted.lean",
        "sorry_or_admit_present": False,
        "axioms": list(_APPROVED_AXIOMS),
        "runtime_mapping": [
            {"path": "tests/test_deleted.py", "sha256": "0" * 64}
        ],
    }
    current = _record(tmp_path, cycle_id="current")
    current["retired_hash_scopes"] = ["formal/lean/Deleted.lean"]
    current["retired_runtime_paths"] = ["tests/test_deleted.py"]
    current["retired_theorem_ids"] = ["BioInspiredNanochat.Deleted.obsolete"]
    _write_manifest(manifest, [old, current])

    report = validate_manifest(manifest, repo_root=tmp_path)

    assert report.proof_paths == (str(proof.relative_to(tmp_path)),)
    assert report.runtime_paths == (str(runtime.relative_to(tmp_path)),)
    assert report.theorem_ids == ("BioInspiredNanochat.Example.certified",)


@pytest.mark.parametrize("stale_surface", ["proof", "runtime"])
def test_stale_effective_hash_fails_closed(tmp_path, stale_surface):
    proof, runtime, manifest = _contract(tmp_path)
    record = _record(tmp_path, cycle_id="current")
    _write_manifest(manifest, [record])
    target = proof if stale_surface == "proof" else runtime
    target.write_text(target.read_text(encoding="utf-8") + "-- drift\n", encoding="utf-8")

    expected_surface = "Lean" if stale_surface == "proof" else "runtime"
    with pytest.raises(FormalFeedbackError, match=f"stale {expected_surface}"):
        validate_manifest(manifest, repo_root=tmp_path)


@pytest.mark.parametrize("proof_path", ["../escape.lean", "formal/missing.lean"])
def test_invalid_or_missing_proof_path_fails_closed(tmp_path, proof_path):
    _, _, manifest = _contract(tmp_path)
    record = _record(tmp_path, cycle_id="current")
    record["hash_scope"] = proof_path
    _write_manifest(manifest, [record])

    with pytest.raises(FormalFeedbackError, match="repository|does not exist"):
        validate_manifest(manifest, repo_root=tmp_path)


@pytest.mark.parametrize("defect", ["declaration", "axiom_print"])
def test_theorem_and_axiom_print_mapping_is_enforced(tmp_path, defect):
    proof, _, manifest = _contract(tmp_path)
    record = _record(tmp_path, cycle_id="current")
    if defect == "declaration":
        record["theorem_ids"] = ["BioInspiredNanochat.Example.missing"]
    else:
        proof.write_text(
            "namespace BioInspiredNanochat.Example\n"
            "theorem certified : True := by trivial\n"
            "end BioInspiredNanochat.Example\n",
            encoding="utf-8",
        )
        record["artifact_hash"] = f"sha256:{_sha256(proof)}"
    _write_manifest(manifest, [record])

    expected = "no declaration" if defect == "declaration" else "no '#print axioms'"
    with pytest.raises(FormalFeedbackError, match=expected):
        validate_manifest(manifest, repo_root=tmp_path)


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "-- theorem certified : True := by trivial\n#print axioms certified\n",
            "no declaration",
        ),
        (
            (
                "theorem certified : True := by trivial\n"
                "/- outer /- nested -/ #print axioms certified -/\n"
            ),
            "no '#print axioms'",
        ),
    ],
)
def test_commented_lean_evidence_cannot_spoof_mapping(tmp_path, source, expected):
    proof, _, manifest = _contract(tmp_path)
    proof.write_text(source, encoding="utf-8")
    record = _record(tmp_path, cycle_id="current")
    _write_manifest(manifest, [record])

    with pytest.raises(FormalFeedbackError, match=expected):
        validate_manifest(manifest, repo_root=tmp_path)


def test_duplicate_cycles_and_unapproved_axioms_fail_closed(tmp_path):
    _, _, manifest = _contract(tmp_path)
    record = _record(tmp_path, cycle_id="duplicate")
    _write_manifest(manifest, [record, record])
    with pytest.raises(FormalFeedbackError, match="duplicate cycle_id"):
        validate_manifest(manifest, repo_root=tmp_path)

    record["axioms"] = [*_APPROVED_AXIOMS, "sorryAx"]
    _write_manifest(manifest, [record])
    with pytest.raises(FormalFeedbackError, match="unapproved axioms: sorryAx"):
        validate_manifest(manifest, repo_root=tmp_path)


def test_theorem_outside_axiom_audited_namespace_fails_closed(tmp_path):
    _, _, manifest = _contract(tmp_path)
    record = _record(tmp_path, cycle_id="current", theorem_id="Other.certified")
    _write_manifest(manifest, [record])

    with pytest.raises(FormalFeedbackError, match="inside the audited namespace"):
        validate_manifest(manifest, repo_root=tmp_path)


def test_compiled_audit_checks_fully_qualified_theorems(tmp_path, monkeypatch):
    _, _, manifest = _contract(tmp_path)
    _write_manifest(manifest, [_record(tmp_path, cycle_id="current")])
    report = validate_manifest(manifest, repo_root=tmp_path)
    observed: dict[str, object] = {}

    def capture(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.formal_feedback.subprocess.run", capture)

    assert run_compiled_lean_audit(report, repo_root=tmp_path) == 0
    assert observed["command"] == ("lake", "env", "lean", "--stdin")
    assert "#check BioInspiredNanochat.Example.certified" in str(observed["input"])
    assert "#print axioms BioInspiredNanochat.Example.certified" in str(observed["input"])
    assert observed["cwd"] == tmp_path / "formal/lean"
    assert observed["timeout"] == 300.0


def test_compiled_audit_propagates_unresolved_theorem_failure(tmp_path, monkeypatch):
    _, _, manifest = _contract(tmp_path)
    _write_manifest(manifest, [_record(tmp_path, cycle_id="current")])
    report = validate_manifest(manifest, repo_root=tmp_path)

    def unresolved(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr("scripts.formal_feedback.subprocess.run", unresolved)

    assert run_compiled_lean_audit(report, repo_root=tmp_path) == 1
