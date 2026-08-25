"""CMA-ES parameter-file ingestion (bead c2l, bead 3nx5).

Loads a JSON file produced by the CMA-ES search (``best_params.json`` written by
``scripts/tune_bio_params``) and overlays it onto a :class:`SynapticConfig`. Kept as a small
importable module because BOTH ``scripts/base_train`` (the ``--load-cmaes-params`` /
``load_cmaes_params=...`` training flag, bead c2l) and the joint 48D search (bead scq) need
programmatic access, and ``scripts/*`` are not importable side-effect-free.

File format (JSON object, one level deep):

    {"tau_rrp": 35.0, "camkii_up": 0.08}

Every key MUST be a tunable numeric :class:`SynapticConfig` field name and every value a
finite real number matching the field's schema (booleans, strings, and non-integer values
for integer fields are rejected). Unknown keys are an error that lists the closest valid names,
not a silent skip: a typo'd parameter would otherwise silently train with the default while
looking tuned.
"""

from __future__ import annotations

import json
import math
from dataclasses import Field, fields
from pathlib import Path
from typing import Any

from bio_inspired_nanochat.certificate_bundle import _MAX_SAFE_INTEGER, _synaptic_config_schema_errors
from bio_inspired_nanochat.synaptic import SynapticConfig

__all__ = ["parse_cmaes_params", "apply_cmaes_params"]

_FIELDS_BY_NAME: dict[str, Field[Any]] = {f.name: f for f in fields(SynapticConfig)}
_SYNAPTIC_FIELDS: frozenset[str] = frozenset(_FIELDS_BY_NAME.keys())


def _closest_matches(name: str, limit: int = 5) -> list[str]:
    """Field names sharing a suffix/prefix token with ``name`` (cheap did-you-mean)."""
    tokens = {s for s in name.replace("_", " ").split() if len(s) >= 3}
    scored: list[tuple[int, str]] = []
    for field_name in _SYNAPTIC_FIELDS:
        hay = field_name.replace("_", " ")
        score = sum(1 for t in tokens if t in hay)
        if score:
            scored.append((-score, field_name))
    return [f for _, f in sorted(scored)[:limit]]


def parse_cmaes_params(path: str | Path) -> dict[str, float | int]:
    """Parse and validate a CMA-ES params JSON file against the SynapticConfig schema.

    Raises ValueError with actionable text on malformed JSON, unknown fields, type
    mismatches (e.g. attempting to override bool toggles or string literals with floats),
    non-finite values, or out-of-domain numbers.
    """
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
    out: dict[str, float | int] = {}
    for key, value in doc.items():
        if not isinstance(key, str) or key not in _FIELDS_BY_NAME:
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

        field_info = _FIELDS_BY_NAME[key]
        default = field_info.default

        if type(default) is bool:
            raise ValueError(
                f"CMA-ES params file {str(p)!r}: field {key!r} is a boolean toggle, "
                f"not a tunable numeric CMA-ES parameter. Booleans are rejected on purpose."
            )

        if type(default) is str:
            raise ValueError(
                f"CMA-ES params file {str(p)!r}: field {key!r} is a string mode setting, "
                f"not a tunable numeric CMA-ES parameter."
            )

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"CMA-ES params file {str(p)!r}: field {key!r} must be a number, "
                f"got {type(value).__name__} ({value!r}). Booleans are rejected on purpose."
            )

        float_val = float(value)
        if not math.isfinite(float_val):
            raise ValueError(
                f"CMA-ES params file {str(p)!r}: field {key!r} must be a finite number, got {value!r}"
            )

        if type(default) is int:
            if not float_val.is_integer():
                raise ValueError(
                    f"CMA-ES params file {str(p)!r}: field {key!r} must be an integer, got non-integer float {value!r}"
                )
            int_val = int(value)
            if int_val < 0 or int_val > _MAX_SAFE_INTEGER:
                raise ValueError(
                    f"CMA-ES params file {str(p)!r}: field {key!r} must be a non-negative integer <= {_MAX_SAFE_INTEGER}, got {int_val}"
                )
            out[key] = int_val
        elif type(default) is float:
            out[key] = float_val
        else:
            raise ValueError(
                f"CMA-ES params file {str(p)!r}: field {key!r} has unsupported schema type {type(default).__name__}"
            )

    if not out:
        raise ValueError(f"CMA-ES params file {str(p)!r} contains no parameters")
    return out


def apply_cmaes_params(syn_cfg: SynapticConfig, path: str | Path) -> SynapticConfig:
    """Overlay a validated params file onto ``syn_cfg`` IN PLACE and validate the result."""
    params = parse_cmaes_params(path)
    for key, value in params.items():
        setattr(syn_cfg, key, value)
    schema_errors = _synaptic_config_schema_errors(syn_cfg)
    if schema_errors:
        raise ValueError(
            f"CMA-ES params file {str(path)!r} resulted in an invalid SynapticConfig: "
            + "; ".join(schema_errors)
        )
    return syn_cfg
