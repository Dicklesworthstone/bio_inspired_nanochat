"""Formal Lean↔Python drift-gate tests (bead r00r.4.3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.formal_feedback import FormalFeedbackError, validate_manifest

pytestmark = pytest.mark.unit

_APPROVED_AXIOMS = ["propext", "Classical.choice", "Quot.sound"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    root: Path,
    *,
    cycle_id: str,
    theorem_id: str = "certified",
    proof_path: str = "formal/Proof.lean",
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
    proof = tmp_path / "formal/Proof.lean"
    runtime = tmp_path / "tests/test_runtime.py"
    manifest = tmp_path / "formal/proof_artifacts.json"
    proof.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    proof.write_text(
        "theorem certified : True := by trivial\n#print axioms certified\n",
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
    assert "tur_calibration_inequality" in report.theorem_ids
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
        record["theorem_ids"] = ["missing"]
    else:
        proof.write_text("theorem certified : True := by trivial\n", encoding="utf-8")
        record["artifact_hash"] = f"sha256:{_sha256(proof)}"
    _write_manifest(manifest, [record])

    expected = "no declaration" if defect == "declaration" else "no '#print axioms'"
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
