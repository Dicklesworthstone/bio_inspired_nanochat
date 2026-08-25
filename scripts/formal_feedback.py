"""Validate the Lean-to-Python proof contract and optionally run mapped tests.

The manifest is append-only provenance.  For HEAD validation, the newest
occurrence of each Lean ``hash_scope`` and runtime ``path`` is authoritative,
and explicit retirement entries remove obsolete mappings.  Stored command
strings are never executed.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_DEFAULT_MANIFEST = Path("formal/lean/proof_artifacts.json")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_THEOREM_ID_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_']*\.)+[A-Za-z_][A-Za-z0-9_']*\Z"
)
_DECLARATION_RE = re.compile(
    r"^\s*(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)\b", re.MULTILINE
)
_AXIOM_PRINT_RE = re.compile(
    r"^\s*#print\s+axioms\s+([A-Za-z_][A-Za-z0-9_.']*)\b", re.MULTILINE
)
_APPROVED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
_AUDITED_NAMESPACE = "BioInspiredNanochat."


class FormalFeedbackError(ValueError):
    """Raised when the formal proof contract is malformed or stale."""


@dataclass(frozen=True)
class FeedbackReport:
    """Validated effective contract at the current repository state."""

    artifact_count: int
    proof_paths: tuple[str, ...]
    runtime_paths: tuple[str, ...]
    theorem_ids: tuple[str, ...]
    test_paths: tuple[str, ...]


def _as_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FormalFeedbackError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _as_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise FormalFeedbackError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _as_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FormalFeedbackError(f"{label} must be a non-empty string")
    return value


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_repo_path(repo_root: Path, raw_path: str, label: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FormalFeedbackError(f"{label} must stay within the repository: {raw_path}")

    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise FormalFeedbackError(
            f"{label} resolves outside the repository: {raw_path}"
        ) from exc
    return resolved


def _resolve_repo_file(repo_root: Path, raw_path: str, label: str) -> Path:
    resolved = _validate_repo_path(repo_root, raw_path, label)
    if not resolved.is_file():
        raise FormalFeedbackError(f"{label} does not exist as a file: {raw_path}")
    return resolved


def _sha256(value: object, label: str, *, prefixed: bool) -> str:
    raw = _as_string(value, label)
    if prefixed:
        if not raw.startswith("sha256:"):
            raise FormalFeedbackError(f"{label} must start with 'sha256:'")
        raw = raw.removeprefix("sha256:")
    if _SHA256_RE.fullmatch(raw) is None:
        raise FormalFeedbackError(f"{label} must contain a lowercase SHA-256 digest")
    return raw


def _strings(value: object, label: str) -> list[str]:
    values = _as_list(value, label)
    strings: list[str] = []
    for index, item in enumerate(values):
        strings.append(_as_string(item, f"{label}[{index}]"))
    return strings


def _unique_strings(value: object, label: str) -> list[str]:
    strings = _strings(value, label)
    if len(strings) != len(set(strings)):
        raise FormalFeedbackError(f"{label} must not contain duplicates")
    return strings


def _retire[Value](mapping: dict[str, Value], key: str, label: str) -> None:
    if key not in mapping:
        raise FormalFeedbackError(f"{label} cannot retire inactive mapping: {key}")
    del mapping[key]


def _mask_lean_comments_and_strings(source: str) -> str:
    """Mask nested Lean comments and strings for source-level preflight checks."""

    masked: list[str] = []
    index = 0
    block_depth = 0
    in_string = False
    in_line_comment = False

    while index < len(source):
        char = source[index]
        pair = source[index : index + 2]

        if in_line_comment:
            if char == "\n":
                masked.append(char)
                in_line_comment = False
            else:
                masked.append(" ")
            index += 1
            continue

        if block_depth:
            if pair == "/-":
                masked.extend((" ", " "))
                block_depth += 1
                index += 2
            elif pair == "-/":
                masked.extend((" ", " "))
                block_depth -= 1
                index += 2
            else:
                masked.append("\n" if char == "\n" else " ")
                index += 1
            continue

        if in_string:
            if char == "\\" and index + 1 < len(source):
                masked.extend((" ", "\n" if source[index + 1] == "\n" else " "))
                index += 2
            else:
                masked.append("\n" if char == "\n" else " ")
                if char == '"':
                    in_string = False
                index += 1
            continue

        if pair == "--":
            masked.extend((" ", " "))
            in_line_comment = True
            index += 2
        elif pair == "/-":
            masked.extend((" ", " "))
            block_depth = 1
            index += 2
        else:
            masked.append(" " if char == '"' else char)
            in_string = char == '"'
            index += 1

    return "".join(masked)


def validate_manifest(
    manifest_path: Path = _DEFAULT_MANIFEST,
    *,
    repo_root: Path | None = None,
) -> FeedbackReport:
    """Validate the effective proof/runtime mapping against repository HEAD."""

    root = (repo_root or Path.cwd()).resolve()
    manifest = manifest_path if manifest_path.is_absolute() else root / manifest_path
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalFeedbackError(f"cannot read proof manifest {manifest}: {exc}") from exc

    document = _as_object(payload, "manifest")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise FormalFeedbackError("manifest.schema_version must be integer 1")
    artifacts = _as_list(document.get("artifacts"), "manifest.artifacts")
    if not artifacts:
        raise FormalFeedbackError("manifest.artifacts must not be empty")

    cycle_ids: set[str] = set()
    effective_proofs: dict[str, str] = {}
    effective_runtime: dict[str, str] = {}
    theorem_sources: dict[str, str] = {}

    for artifact_index, raw_artifact in enumerate(artifacts):
        label = f"manifest.artifacts[{artifact_index}]"
        artifact = _as_object(raw_artifact, label)
        cycle_id = _as_string(artifact.get("cycle_id"), f"{label}.cycle_id")
        if cycle_id in cycle_ids:
            raise FormalFeedbackError(f"duplicate cycle_id: {cycle_id}")
        cycle_ids.add(cycle_id)

        retired_hash_scopes = _unique_strings(
            artifact.get("retired_hash_scopes", []), f"{label}.retired_hash_scopes"
        )
        retired_runtime_paths = _unique_strings(
            artifact.get("retired_runtime_paths", []), f"{label}.retired_runtime_paths"
        )
        retired_theorem_ids = _unique_strings(
            artifact.get("retired_theorem_ids", []), f"{label}.retired_theorem_ids"
        )
        for retired_path in retired_hash_scopes:
            _validate_repo_path(root, retired_path, f"{label}.retired_hash_scopes")
            _retire(effective_proofs, retired_path, f"{label}.retired_hash_scopes")
        for retired_path in retired_runtime_paths:
            _validate_repo_path(root, retired_path, f"{label}.retired_runtime_paths")
            _retire(effective_runtime, retired_path, f"{label}.retired_runtime_paths")
        for theorem_id in retired_theorem_ids:
            if _THEOREM_ID_RE.fullmatch(theorem_id) is None:
                raise FormalFeedbackError(f"invalid retired theorem id: {theorem_id}")
            if not theorem_id.startswith(_AUDITED_NAMESPACE):
                raise FormalFeedbackError(
                    f"retired theorem must be inside the audited namespace: {theorem_id}"
                )
            _retire(theorem_sources, theorem_id, f"{label}.retired_theorem_ids")

        hash_scope = _as_string(artifact.get("hash_scope"), f"{label}.hash_scope")
        proof_path = _validate_repo_path(root, hash_scope, f"{label}.hash_scope")
        if proof_path.suffix != ".lean":
            raise FormalFeedbackError(f"{label}.hash_scope must name a .lean file")
        if hash_scope in retired_hash_scopes:
            raise FormalFeedbackError(f"{label} cannot retire and replace {hash_scope}")
        artifact_hash = _sha256(
            artifact.get("artifact_hash"), f"{label}.artifact_hash", prefixed=True
        )
        effective_proofs[hash_scope] = artifact_hash

        theorem_ids = _unique_strings(
            artifact.get("theorem_ids"), f"{label}.theorem_ids"
        )
        if not theorem_ids:
            raise FormalFeedbackError(f"{label}.theorem_ids must be non-empty")
        for theorem_id in theorem_ids:
            if _THEOREM_ID_RE.fullmatch(theorem_id) is None:
                raise FormalFeedbackError(f"invalid theorem id: {theorem_id}")
            if not theorem_id.startswith(_AUDITED_NAMESPACE):
                raise FormalFeedbackError(
                    f"theorem must be inside the audited namespace: {theorem_id}"
                )
            if theorem_id in retired_theorem_ids:
                raise FormalFeedbackError(
                    f"{label} cannot retire and replace theorem {theorem_id}"
                )
            theorem_sources[theorem_id] = hash_scope

        sorry_or_admit_present = artifact.get("sorry_or_admit_present")
        if not isinstance(sorry_or_admit_present, bool) or sorry_or_admit_present:
            raise FormalFeedbackError(f"{label}.sorry_or_admit_present must be false")
        axioms = set(_strings(artifact.get("axioms"), f"{label}.axioms"))
        unsupported_axioms = axioms - _APPROVED_AXIOMS
        if unsupported_axioms:
            names = ", ".join(sorted(unsupported_axioms))
            raise FormalFeedbackError(f"{label} declares unapproved axioms: {names}")

        runtime_mapping = _as_list(
            artifact.get("runtime_mapping"), f"{label}.runtime_mapping"
        )
        if not runtime_mapping:
            raise FormalFeedbackError(f"{label}.runtime_mapping must not be empty")
        paths_in_artifact: set[str] = set()
        for mapping_index, raw_mapping in enumerate(runtime_mapping):
            mapping_label = f"{label}.runtime_mapping[{mapping_index}]"
            mapping = _as_object(raw_mapping, mapping_label)
            raw_path = _as_string(mapping.get("path"), f"{mapping_label}.path")
            if raw_path in paths_in_artifact:
                raise FormalFeedbackError(f"duplicate runtime path in {cycle_id}: {raw_path}")
            if raw_path in retired_runtime_paths:
                raise FormalFeedbackError(f"{label} cannot retire and replace {raw_path}")
            paths_in_artifact.add(raw_path)
            _validate_repo_path(root, raw_path, f"{mapping_label}.path")
            expected_hash = _sha256(
                mapping.get("sha256"), f"{mapping_label}.sha256", prefixed=False
            )
            effective_runtime[raw_path] = expected_hash

    resolved_proofs: dict[str, Path] = {}
    for raw_path, expected in effective_proofs.items():
        path = _resolve_repo_file(root, raw_path, f"effective Lean path {raw_path}")
        actual = _digest(path)
        if not hmac.compare_digest(actual, expected):
            raise FormalFeedbackError(
                f"stale Lean hash for {raw_path}: expected {expected}, actual {actual}"
            )
        resolved_proofs[raw_path] = path
    for raw_path, expected in effective_runtime.items():
        path = _resolve_repo_file(root, raw_path, f"effective runtime path {raw_path}")
        actual = _digest(path)
        if not hmac.compare_digest(actual, expected):
            raise FormalFeedbackError(
                f"stale runtime hash for {raw_path}: expected {expected}, actual {actual}"
            )

    lean_symbols: dict[str, tuple[set[str], set[str]]] = {}
    for raw_path, path in resolved_proofs.items():
        source = _mask_lean_comments_and_strings(path.read_text(encoding="utf-8"))
        declarations = set(_DECLARATION_RE.findall(source))
        axiom_prints = {name.rsplit(".", 1)[-1] for name in _AXIOM_PRINT_RE.findall(source)}
        lean_symbols[raw_path] = (declarations, axiom_prints)
    for theorem_id, raw_path in theorem_sources.items():
        if raw_path not in lean_symbols:
            raise FormalFeedbackError(
                f"manifest theorem {theorem_id} maps to retired Lean path {raw_path}"
            )
        declarations, axiom_prints = lean_symbols[raw_path]
        theorem_name = theorem_id.rsplit(".", 1)[-1]
        if theorem_name not in declarations:
            raise FormalFeedbackError(
                f"manifest theorem {theorem_id} has no declaration in {raw_path}"
            )
        if theorem_name not in axiom_prints:
            raise FormalFeedbackError(
                f"manifest theorem {theorem_id} has no '#print axioms' in {raw_path}"
            )

    test_paths = tuple(
        sorted(
            raw_path
            for raw_path in effective_runtime
            if raw_path.startswith("tests/test_") and raw_path.endswith(".py")
        )
    )
    if not test_paths:
        raise FormalFeedbackError("effective runtime mapping contains no pytest modules")

    return FeedbackReport(
        artifact_count=len(artifacts),
        proof_paths=tuple(sorted(effective_proofs)),
        runtime_paths=tuple(sorted(effective_runtime)),
        theorem_ids=tuple(sorted(theorem_sources)),
        test_paths=test_paths,
    )


def run_mapped_tests(
    report: FeedbackReport,
    *,
    repo_root: Path | None = None,
    timeout_seconds: float = 900.0,
) -> int:
    """Run only validated mapped pytest modules without evaluating manifest commands."""

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        *report.test_paths,
    ]
    root = (repo_root or Path.cwd()).resolve()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        Console(stderr=True).print(
            Panel(
                f"mapped regressions exceeded {timeout_seconds:g} seconds",
                title="Formal feedback timed out",
                style="red",
            )
        )
        return 124
    return int(completed.returncode)


def run_compiled_lean_audit(
    report: FeedbackReport,
    *,
    repo_root: Path | None = None,
    timeout_seconds: float = 300.0,
    lean_command: tuple[str, ...] = ("lake", "env", "lean", "--stdin"),
) -> int:
    """Ask Lean's compiled environment to resolve every mapped theorem identity."""

    root = (repo_root or Path.cwd()).resolve()
    package_root = root / "formal/lean"
    module_names: list[str] = []
    for raw_path in report.proof_paths:
        try:
            relative = Path(raw_path).relative_to("formal/lean")
        except ValueError as exc:
            raise FormalFeedbackError(
                f"compiled Lean audit path is outside formal/lean: {raw_path}"
            ) from exc
        module_parts = relative.with_suffix("").parts
        if not module_parts or any(
            _THEOREM_ID_RE.fullmatch(f"Root.{part}") is None for part in module_parts
        ):
            raise FormalFeedbackError(f"invalid Lean module path: {raw_path}")
        module_names.append(".".join(module_parts))

    audit_source = "\n".join(
        [
            *(f"import {module_name}" for module_name in sorted(set(module_names))),
            "",
            *(
                f"#check {theorem_id}\n#print axioms {theorem_id}"
                for theorem_id in report.theorem_ids
            ),
            "",
        ]
    )
    try:
        completed = subprocess.run(
            lean_command,
            cwd=package_root,
            check=False,
            input=audit_source,
            text=True,
            timeout=timeout_seconds,
        )
    except OSError as exc:
        return_code = 127 if isinstance(exc, FileNotFoundError) else 126
        Console(stderr=True).print(
            Panel(
                f"could not launch compiled Lean audit via {lean_command[0]}: {exc}",
                title="Formal feedback unavailable",
                style="red",
            )
        )
        return return_code
    except subprocess.TimeoutExpired:
        Console(stderr=True).print(
            Panel(
                f"compiled Lean audit exceeded {timeout_seconds:g} seconds",
                title="Formal feedback timed out",
                style="red",
            )
        )
        return 124
    return int(completed.returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="run the effective mapped pytest modules after validating hashes",
    )
    parser.add_argument(
        "--test-timeout-seconds",
        type=float,
        default=900.0,
        help="deadline for mapped regressions (default: 900 seconds)",
    )
    parser.add_argument(
        "--run-lean-audit",
        action="store_true",
        help="resolve every mapped theorem in the compiled Lake environment",
    )
    parser.add_argument(
        "--lean-timeout-seconds",
        type=float,
        default=300.0,
        help="deadline for the compiled Lean audit (default: 300 seconds)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    console = Console()
    if args.test_timeout_seconds <= 0 or args.lean_timeout_seconds <= 0:
        Console(stderr=True).print(
            Panel("formal-feedback timeout values must be positive", style="red")
        )
        return 2
    try:
        report = validate_manifest(args.manifest, repo_root=args.repo_root)
    except FormalFeedbackError as exc:
        Console(stderr=True).print(Panel(str(exc), title="Formal feedback failed", style="red"))
        return 1

    table = Table(title="Lean ↔ Python proof contract")
    table.add_column("check")
    table.add_column("count", justify="right")
    table.add_row("artifact records", str(report.artifact_count))
    table.add_row("effective Lean files", str(len(report.proof_paths)))
    table.add_row("effective runtime files", str(len(report.runtime_paths)))
    table.add_row("mapped theorems", str(len(report.theorem_ids)))
    table.add_row("mapped pytest modules", str(len(report.test_paths)))
    console.print(table)

    if args.run_tests:
        console.print("[bold cyan]Running validated mapped regressions[/bold cyan]")
        test_result = run_mapped_tests(
            report,
            repo_root=args.repo_root,
            timeout_seconds=args.test_timeout_seconds,
        )
        if test_result:
            return test_result
    if args.run_lean_audit:
        console.print("[bold cyan]Auditing mapped theorems in compiled Lean[/bold cyan]")
        return run_compiled_lean_audit(
            report,
            repo_root=args.repo_root,
            timeout_seconds=args.lean_timeout_seconds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
