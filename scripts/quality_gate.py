"""
Quality gate for this repo: ruff + ty + UBS.

Designed for:
- Local use (pre-push / pre-commit)
- CI use (GitHub Actions output supported)

Ruff/ty are scoped to *changed files* for speed.
UBS is run in its supported change-detection modes (or whole-repo when needed).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

_HASH_CHUNK_SIZE = 1024 * 1024
_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".ruff_cache",
    "dev",
    "runs",
    "wandb",
    "dist",
    "target",
    "rust_src",
    "dev-ignore",
    "eval_bundle",
}


@dataclass(frozen=True)
class GateResult:
    ok: bool
    exit_code: int


def _run(cmd: list[str], *, cwd: Path) -> int:
    console.print(f"[bold cyan]$[/bold cyan] {shlex.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    return int(proc.returncode)


def _git_lines(cmd: list[str], *, cwd: Path) -> list[str]:
    proc = subprocess.run(
        ["git", *cmd],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(cmd)} failed")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _repo_root(*, cwd: Path) -> Path:
    root = _git_lines(["rev-parse", "--show-toplevel"], cwd=cwd)
    if len(root) != 1:
        raise RuntimeError("Unable to determine git repo root")
    return Path(root[0]).resolve()


def _collect_paths(mode: str, *, repo: Path, base: str, head: str) -> list[Path]:
    diff_filter = "ACMR"

    if mode == "staged":
        rels = _git_lines(
            ["diff", "--name-only", "--cached", f"--diff-filter={diff_filter}"], cwd=repo
        )
    elif mode == "worktree":
        rels = _git_lines(["diff", "--name-only", f"--diff-filter={diff_filter}"], cwd=repo)
        rels.extend(_git_lines(["ls-files", "--others", "--exclude-standard"], cwd=repo))
    elif mode == "branch":
        try:
            rels = _git_lines(
                ["diff", "--name-only", f"{base}...{head}", f"--diff-filter={diff_filter}"],
                cwd=repo,
            )
        except RuntimeError as e:
            console.print(
                f"[yellow]Warning:[/yellow] unable to compute git diff {base}...{head}: {e}\n"
                "Falling back to running on the whole repo."
            )
            return [repo]
    elif mode == "all":
        return [repo]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    out: list[Path] = []
    for rel in rels:
        p = (repo / rel).resolve()
        if not p.exists():
            continue
        try:
            rel_path = p.relative_to(repo)
        except ValueError:
            continue
        if any(part in _SKIP_DIR_NAMES for part in rel_path.parts):
            continue
        out.append(p)
    return out


def _filter_paths(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    py: list[Path] = []
    js: list[Path] = []
    for p in paths:
        if p.is_dir():
            # keep directories; tools can recurse.
            py.append(p)
            js.append(p)
            continue
        suf = p.suffix.lower()
        if suf == ".py":
            py.append(p)
        elif suf in {".js", ".jsx", ".ts", ".tsx"}:
            js.append(p)
    return _dedupe(py), _dedupe(js)


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _hash_file(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _python_files_for_targets(*, repo: Path, targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    for t in targets:
        if t.is_dir():
            for p in t.rglob("*.py"):
                if not p.is_file():
                    continue
                rel = p.relative_to(repo)
                if any(part in _SKIP_DIR_NAMES for part in rel.parts):
                    continue
                files.append(p)
            continue

        if t.is_file() and t.suffix.lower() == ".py":
            files.append(t)
    return _dedupe(files)


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required tool {name!r} not found in PATH. Install it first and re-run."
        )


def run_quality_gate(*, mode: str, base: str, head: str) -> GateResult:
    repo = _repo_root(cwd=Path.cwd())
    paths = _collect_paths(mode, repo=repo, base=base, head=head)
    py_paths, js_paths = _filter_paths(paths)

    if not py_paths and not js_paths:
        console.print("[green]No relevant changed files; skipping quality gate.[/green]")
        return GateResult(ok=True, exit_code=0)

    table = Table(title="Quality Gate Inputs")
    table.add_column("category")
    table.add_column("paths", overflow="fold")
    table.add_row("mode", mode)
    table.add_row("base...head", f"{base}...{head}")
    table.add_row("python", ", ".join(str(p.relative_to(repo)) for p in py_paths) or "(none)")
    table.add_row("js/ts", ", ".join(str(p.relative_to(repo)) for p in js_paths) or "(none)")
    console.print(table)

    _require_tool("uv")
    _require_tool("uvx")

    # Ruff (autofix)
    if py_paths:
        ruff_files = _python_files_for_targets(repo=repo, targets=py_paths)
        before_hash = {p: _hash_file(p) for p in ruff_files}
        rc = _run(
            [
                "uv",
                "run",
                "ruff",
                "check",
                "--fix",
                "--unsafe-fixes",
                *[str(p) for p in py_paths],
            ],
            cwd=repo,
        )
        if rc != 0:
            return GateResult(ok=False, exit_code=rc)

        after_hash = {p: _hash_file(p) for p in ruff_files if p.exists()}
        changed_by_ruff = sorted(
            (p for p in ruff_files if before_hash.get(p) != after_hash.get(p)),
            key=lambda p: str(p),
        )
        if changed_by_ruff:
            msg = (
                "ruff --fix modified files. Please review, stage, and commit the changes:\n"
                + "\n".join(f"- {p.relative_to(repo)}" for p in changed_by_ruff)
            )
            console.print(Panel(msg, title="Uncommitted Autofixes", style="yellow"))
            return GateResult(ok=False, exit_code=2)

    # ty
    if py_paths:
        output_format = "github" if os.environ.get("GITHUB_ACTIONS") == "true" else "full"
        rc = _run(
            [
                "uvx",
                "ty",
                "check",
                "--output-format",
                output_format,
                *[str(p) for p in py_paths],
            ],
            cwd=repo,
        )
        if rc != 0:
            return GateResult(ok=False, exit_code=rc)

    # UBS (changed files only; fast)
    if py_paths or js_paths:
        _require_tool("ubs")
        ubs_cmd = ["ubs", "--only=python", "--category=resource-lifecycle"]
        if os.environ.get("GITHUB_ACTIONS") == "true":
            ubs_cmd.extend(["--ci", "--fail-on-warning"])
        if mode == "staged":
            ubs_cmd.append("--staged")
        elif mode == "worktree":
            ubs_cmd.append("--diff")
        elif mode == "branch":
            console.print(
                "[yellow]Note:[/yellow] UBS does not currently support an arbitrary "
                "base...head range; scanning the whole repo."
            )
        ubs_cmd.append(str(repo))
        rc = _run(ubs_cmd, cwd=repo)
        if rc != 0:
            return GateResult(ok=False, exit_code=rc)

    console.print(Panel("[bold green]All checks passed.[/bold green]", title="Quality Gate"))
    return GateResult(ok=True, exit_code=0)


def main() -> int:
    p = argparse.ArgumentParser(description="Run ruff + ty + ubs on changed files.")
    p.add_argument(
        "--mode",
        default="staged",
        choices=["staged", "worktree", "branch", "all"],
        help="How to select files for checking.",
    )
    p.add_argument(
        "--base",
        default="origin/main",
        help="Base ref/sha for --mode=branch (default: origin/main).",
    )
    p.add_argument(
        "--head",
        default="HEAD",
        help="Head ref/sha for --mode=branch (default: HEAD).",
    )
    args = p.parse_args()

    result = run_quality_gate(mode=args.mode, base=args.base, head=args.head)
    return int(result.exit_code if not result.ok else 0)


if __name__ == "__main__":
    raise SystemExit(main())
