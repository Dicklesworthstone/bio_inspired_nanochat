"""CMA-ES parameter-file ingestion (bead c2l).

Loads a JSON file produced by the CMA-ES search (``best_params.json`` written by
``scripts/tune_bio_params``) and overlays it onto a :class:`SynapticConfig`. Kept as a small
importable module because BOTH ``scripts/base_train`` (the ``--load-cmaes-params`` /
``load_cmaes_params=...`` training flag, bead c2l) and the joint 48D search (bead scq) need
programmatic access, and ``scripts/*`` are not importable side-effect-free.

File format (JSON object, one level deep):

    {"tau_rrp_log": -3.12, "camkii_up_log": -1.7}

Every key MUST be a :class:`SynapticConfig` field name and every value a real number
(booleans are rejected — a bool is almost always a copy/paste mistake for 0/1).
Unknown keys are an error that lists the closest valid names, not a silent skip: a
typo'd parameter would otherwise silently train with the default while looking tuned.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from bio_inspired_nanochat.synaptic import SynapticConfig

__all__ = ["parse_cmaes_params", "apply_cmaes_params"]

_SYNAPTIC_FIELDS: frozenset[str] = frozenset(f.name for f in fields(SynapticConfig))


def _closest_matches(name: str, limit: int = 5) -> list[str]:
    """Field names sharing a suffix/prefix token with ``name`` (cheap did-you-mean)."""
    tokens = {s for s in name.replace("_", " ").split() if len(s) >= 3}
    scored: list[tuple[int, str]] = []
    for field in _SYNAPTIC_FIELDS:
        hay = field.replace("_", " ")
        score = sum(1 for t in tokens if t in hay)
        if score:
            scored.append((-score, field))
    return [f for _, f in sorted(scored)[:limit]]


def parse_cmaes_params(path: str | Path) -> dict[str, float]:
    """Parse and validate a CMA-ES params JSON file. Raises ValueError with actionable text."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"CMA-ES params file {str(p)!r} could not be read: {e}") from e
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"CMA-ES params file {str(p)!r} is not valid JSON (line {e.lineno}, col {e.colno}): {e.msg}"
        ) from e
    if not isinstance(doc, dict):
        raise ValueError(
            f"CMA-ES params file {str(p)!r} must contain a JSON object mapping "
            f"SynapticConfig field names to numbers; got top-level {type(doc).__name__}"
        )
    out: dict[str, float] = {}
    for key, value in doc.items():
        if not isinstance(key, str) or key not in _SYNAPTIC_FIELDS:
            hints = _closest_matches(key if isinstance(key, str) else "")
            hint = f"; closest SynapticConfig fields: {hints}" if hints else ""
            known = (
                "unknown SynapticConfig field"
                if isinstance(key, str)
                else "non-string key"
            )
            raise ValueError(
                f"CMA-ES params file {str(p)!r}: {known} {key!r}{hint}. "
                f"The file must map SynapticConfig field names to numbers."
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"CMA-ES params file {str(p)!r}: field {key!r} must be a number, "
                f"got {type(value).__name__} ({value!r}). Booleans are rejected on purpose."
            )
        out[key] = float(value)
    if not out:
        raise ValueError(f"CMA-ES params file {str(p)!r} contains no parameters")
    return out


def apply_cmaes_params(syn_cfg: SynapticConfig, path: str | Path) -> SynapticConfig:
    """Overlay a validated params file onto ``syn_cfg`` IN PLACE and return it."""
    params = parse_cmaes_params(path)
    for key, value in params.items():
        setattr(syn_cfg, key, value)
    return syn_cfg
