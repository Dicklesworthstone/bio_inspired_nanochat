"""
Committed results corpus + experiment registry — bead hm4.1.

A tracked, schema'd results store so findings accumulate and every claim is verifiable. Every
run (train / eval / tune) emits a provenance-stamped, schema-valid `RunRecord` appended to the
committable `results/registry.jsonl` JSONL registry; a query CLI summarizes past runs.

Provenance reuses checkpoint_manager (git SHA + config hash); metrics are validated against the
canonical schema (metrics_schema / hm4.2). The registry is the audit trail the project was
missing (empty anomaly index, placeholder artifacts, no runs/ dir).

CLI:
    python -m bio_inspired_nanochat.results_registry list [--harness train] [--limit 20]
    python -m bio_inspired_nanochat.results_registry best --metric val_bpb
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import numbers
import os
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from rich.console import Console

from bio_inspired_nanochat.checkpoint_manager import _git_sha, config_hash
from bio_inspired_nanochat.common import decouple_config
from bio_inspired_nanochat.metrics_schema import Direction, get_metric, validate_metrics

logger = logging.getLogger(__name__)

# The tracked corpus is the default. ``BIO_RESULTS_REGISTRY`` (``.env`` or environment)
# redirects every harness that does not pass an explicit path — the test suite points it
# at a throwaway directory so pytest runs can never append to the committed registry
# (the 2026-08 pollution: 41 rows whose artifact paths were pytest temp dirs).
TRACKED_REGISTRY = os.path.join("results", "registry.jsonl")
DEFAULT_REGISTRY = str(decouple_config("BIO_RESULTS_REGISTRY", default=TRACKED_REGISTRY))
_HARNESSES = ("train", "eval", "tune")


@dataclass
class RunRecord:
    run_id: str
    harness: str
    metrics: dict[str, float]
    git_sha: str | None = None
    config_hash: str | None = None
    seed: int | None = None
    hardware: str | None = None
    dataset_shards: list[str] = field(default_factory=list)
    timestamp: float | None = None
    notes: str = ""
    verdict: str | None = None
    eligible_for_best: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Enforce the persisted registry schema on every construction path."""
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if self.harness not in _HARNESSES:
            raise ValueError(
                f"unknown harness {self.harness!r}; expected one of {_HARNESSES}"
            )
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        self.metrics = validate_metrics(self.metrics, strict=True)
        if self.verdict not in (None, "positive", "null", "invalidated"):
            raise ValueError("verdict must be positive, null, invalidated, or None")
        if not isinstance(self.eligible_for_best, bool):
            raise TypeError("eligible_for_best must be a bool")
        if self.verdict in ("null", "invalidated") and self.eligible_for_best:
            raise ValueError("null and invalidated records cannot be eligible for best-result queries")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if self.timestamp is not None and (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, numbers.Real)
            or not math.isfinite(float(self.timestamp))
        ):
            raise ValueError("timestamp must be finite numeric data or None")
        if not isinstance(self.dataset_shards, list) or any(
            not isinstance(shard, str) for shard in self.dataset_shards
        ):
            raise TypeError("dataset_shards must be a list of strings")
        for name, value in (
            ("git_sha", self.git_sha),
            ("config_hash", self.config_hash),
            ("hardware", self.hardware),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")

    def to_json(self) -> dict:
        # Fields remain mutable for ergonomic callers, so revalidate immediately
        # before serialization instead of trusting construction-time validation.
        self._validate()
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> RunRecord:
        if not isinstance(d, Mapping):
            raise TypeError("registry record must be a JSON object")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"registry record contains unknown field(s): {unknown}")
        payload = dict(d)
        if "eligible_for_best" not in payload:
            # Legacy free-text verdicts cannot safely participate in best-result queries.  Their
            # claims remain readable, but require an explicit schema migration before eligibility.
            verdict = payload.get("verdict")
            payload["eligible_for_best"] = (
                verdict == "positive"
                if verdict is not None
                else "verdict=" not in str(payload.get("notes", ""))
            )
        return cls(**payload)


def _hardware_string() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}"
    except Exception as exc:
        logger.debug("CUDA hardware probe failed; recording CPU fallback: %s", exc)
    return f"cpu:{platform.machine()}"


def make_record(
    harness: str,
    metrics: Mapping[str, Any],
    *,
    run_id: str,
    config: Any = None,
    syn_cfg: Any = None,
    seed: int | None = None,
    dataset_shards: list[str] | None = None,
    timestamp: float | None = None,
    notes: str = "",
    verdict: str | None = None,
    eligible_for_best: bool = True,
) -> RunRecord:
    """Build a provenance-stamped, schema-valid RunRecord.

    Metrics are validated against the canonical schema (unknown/non-finite -> error). Provenance
    (git SHA + a stable config hash) is stamped automatically. ``config`` should contain the
    complete harness configuration; ``syn_cfg`` remains available for focused mechanism records.
    Pass `timestamp` for reproducible records (else it is left None for the caller/CLI to fill).
    """
    if harness not in _HARNESSES:
        raise ValueError(f"unknown harness {harness!r}; expected one of {_HARNESSES}")
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if config is not None and syn_cfg is not None:
        raise ValueError("pass either config or syn_cfg, not both")
    if verdict not in (None, "positive", "null", "invalidated"):
        raise ValueError("verdict must be positive, null, invalidated, or None")
    if verdict in ("null", "invalidated") and eligible_for_best:
        raise ValueError("null and invalidated records cannot be eligible for best-result queries")
    valid = validate_metrics(metrics, strict=True)
    cfg_hash = None
    config_value = config if config is not None else syn_cfg
    if config_value is not None:
        if is_dataclass(config_value) and not isinstance(config_value, type):
            config_payload = asdict(config_value)
        elif isinstance(config_value, Mapping):
            config_payload = dict(config_value)
        else:
            raise TypeError("config must be a dataclass instance or mapping")
        cfg_hash = config_hash(config_payload)
    return RunRecord(
        run_id=run_id,
        harness=harness,
        metrics=valid,
        git_sha=_git_sha(),
        config_hash=cfg_hash,
        seed=seed,
        hardware=_hardware_string(),
        dataset_shards=list(dataset_shards or []),
        timestamp=timestamp,
        notes=notes,
        verdict=verdict,
        eligible_for_best=eligible_for_best,
    )


def append_record(record: RunRecord, path: str = DEFAULT_REGISTRY) -> None:
    """Append a record to the committed JSONL registry (creating the dir/file if needed)."""
    if not isinstance(record, RunRecord):
        raise TypeError("record must be a RunRecord")
    encoded = json.dumps(record.to_json(), sort_keys=True, allow_nan=False)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(encoded + "\n")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting ambiguous duplicate field names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject Python's non-standard NaN/Infinity JSON extensions."""
    raise ValueError(f"non-standard JSON constant {value!r}")


def read_records(path: str = DEFAULT_REGISTRY) -> list[RunRecord]:
    if not os.path.exists(path):
        return []
    out: list[RunRecord] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    payload = json.loads(
                        line,
                        object_pairs_hook=_strict_json_object,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid registry JSON at {path}:{line_number}: {exc}"
                    ) from exc
                try:
                    out.append(RunRecord.from_json(payload))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid registry record at {path}:{line_number}: {exc}"
                    ) from exc
    return out


def best_record(records: list[RunRecord], metric: str) -> RunRecord | None:
    """The record optimizing `metric` per its schema direction (lower/higher better)."""
    spec = get_metric(metric)
    if spec is None:
        raise KeyError(f"unknown metric {metric!r}")
    if spec.direction == Direction.NEUTRAL:
        raise ValueError(
            f"metric {metric!r} has neutral direction and cannot define a best record"
        )
    for record in records:
        if not isinstance(record, RunRecord):
            raise TypeError("records must contain only RunRecord instances")
        record._validate()
    have = [r for r in records if r.eligible_for_best and metric in r.metrics]
    if not have:
        return None
    reverse = spec.direction == Direction.HIGHER_BETTER
    return sorted(have, key=lambda r: r.metrics[metric], reverse=reverse)[0]


def summarize(records: list[RunRecord], *, harness: str | None = None, limit: int = 20) -> str:
    for record in records:
        if not isinstance(record, RunRecord):
            raise TypeError("records must contain only RunRecord instances")
        record._validate()
    rows = [r for r in records if harness is None or r.harness == harness]
    rows = rows[-limit:]
    if not rows:
        return "(no runs in the registry)"
    lines = [f"{len(rows)} run(s):"]
    for r in rows:
        m = ", ".join(f"{k}={v:.4g}" for k, v in sorted(r.metrics.items())[:4])
        sha = (r.git_sha or "????????")[:8]
        lines.append(f"  [{r.harness:5}] {r.run_id}  sha={sha} cfg={r.config_hash or '-'}  {m}")
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Query the bio_inspired_nanochat results registry.")
    ap.add_argument("command", choices=["list", "best"])
    ap.add_argument("--path", default=DEFAULT_REGISTRY)
    ap.add_argument("--harness", default=None)
    ap.add_argument("--metric", default="val_bpb")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)
    console = Console()
    records = read_records(args.path)
    if args.command == "list":
        console.print(summarize(records, harness=args.harness, limit=args.limit))
    else:
        b = best_record(records, args.metric)
        console.print(
            f"best by {args.metric}: {b.run_id} = {b.metrics[args.metric]:.4g}"
            if b
            else "(none)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
