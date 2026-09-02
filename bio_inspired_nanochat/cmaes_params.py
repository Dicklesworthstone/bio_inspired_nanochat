"""SynapticConfig overlays from outside the code: CMA-ES JSON files and CLI overrides.

Two front doors write user-supplied values onto a :class:`SynapticConfig`, and both must fail
closed on typos and type mistakes (a mis-spelled parameter would otherwise silently train with
the default while looking tuned):

* :func:`apply_cmaes_params` — overlays a ``best_params.json`` written by
  ``scripts/tune_bio_params`` (bead c2l / 3nx5). Numeric fields only; booleans and string
  settings are rejected on purpose because CMA-ES searches a continuous space.
* :func:`apply_syn_cfg_overrides` — overlays ``--syn_cfg.<field>=<value>`` command-line
  arguments (the "Key Training Flags" the README documents). Any field is allowed, values are
  coerced to the field's declared type, and the result must pass both the schema check and the
  mechanism-prerequisite check in :mod:`ablation_registry`.

Kept as a small importable module because ``scripts/base_train`` (the training flags), the
joint 48D search (bead scq) and the tests need programmatic access, and ``scripts/*`` are not
importable side-effect-free.

JSON file format (one level deep)::

    {"tau_rrp": 35.0, "camkii_up": 0.08}
"""

from __future__ import annotations

import json
import logging
import math
from ast import literal_eval
from collections.abc import Mapping, Sequence
from dataclasses import Field, fields
from pathlib import Path
from typing import Any

from bio_inspired_nanochat.ablation_registry import (
    MAX_SAFE_INTEGER,
    synaptic_config_schema_errors,
    validate_config,
)
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticGranularity

__all__ = [
    "SYN_CFG_CLI_PREFIX",
    "apply_cmaes_params",
    "apply_syn_cfg_overrides",
    "coerce_syn_cfg_override",
    "extract_syn_cfg_cli_overrides",
    "parse_cmaes_params",
]

logger = logging.getLogger(__name__)

_FIELDS_BY_NAME: dict[str, Field[Any]] = {f.name: f for f in fields(SynapticConfig)}
_SYNAPTIC_FIELDS: frozenset[str] = frozenset(_FIELDS_BY_NAME.keys())

#: Prefix of a command-line SynapticConfig override, e.g. ``--syn_cfg.tau_rrp=100.0``.
SYN_CFG_CLI_PREFIX = "--syn_cfg."


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

        raise ValueError(  # noqa: TRY004
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
            raise ValueError(  # noqa: TRY004 — file contract: ValueError for any malformed value
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
            if int_val < 0 or int_val > MAX_SAFE_INTEGER:
                raise ValueError(
                    f"CMA-ES params file {str(p)!r}: field {key!r} must be a non-negative integer <= {MAX_SAFE_INTEGER}, got {int_val}"
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
    schema_errors = synaptic_config_schema_errors(syn_cfg)
    if schema_errors:
        raise ValueError(
            f"CMA-ES params file {str(path)!r} resulted in an invalid SynapticConfig: "
            + "; ".join(schema_errors)
        )
    return syn_cfg


# --------------------------------------------------------------------------- #
# ``--syn_cfg.<field>=<value>`` command-line overrides
# --------------------------------------------------------------------------- #
def extract_syn_cfg_cli_overrides(argv: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Split ``--syn_cfg.<field>=<value>`` arguments out of ``argv``.

    Returns ``(remaining_argv, {field: raw_value})``. Only the *syntax* is checked here; the
    training script's configurator consumes ``remaining_argv`` and would otherwise reject the
    dotted keys as unknown settings. Field names and value types are validated by
    :func:`apply_syn_cfg_overrides`.
    """
    remaining: list[str] = []
    overrides: dict[str, str] = {}
    for arg in argv:
        if not arg.startswith(SYN_CFG_CLI_PREFIX):
            remaining.append(arg)
            continue
        body = arg[len(SYN_CFG_CLI_PREFIX) :]
        if "=" not in body:
            raise ValueError(f"{arg!r}: expected the form --syn_cfg.<field>=<value>")
        key, raw = body.split("=", 1)
        if not key:
            raise ValueError(f"{arg!r}: missing the SynapticConfig field name")
        if key in overrides:
            raise ValueError(
                f"--syn_cfg.{key} was given twice ({overrides[key]!r} and {raw!r}); pass it once"
            )
        overrides[key] = raw
    return remaining, overrides


def coerce_syn_cfg_override(field_name: str, raw: str) -> Any:
    """Convert one command-line string to the declared type of ``SynapticConfig.<field_name>``.

    Booleans accept ``0/1/true/false/yes/no/on/off``; integers must be whole numbers; floats must
    be finite; string and granularity settings are passed through (surrounding quotes stripped)
    and checked against their literal sets by the schema validator afterwards.
    """
    if field_name not in _FIELDS_BY_NAME:
        hints = _closest_matches(field_name)
        hint = f"; closest SynapticConfig fields: {hints}" if hints else ""
        raise ValueError(f"--syn_cfg.{field_name}: unknown SynapticConfig field{hint}")
    default = _FIELDS_BY_NAME[field_name].default
    text = raw.strip()
    if type(default) is bool:
        lowered = text.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(
            f"--syn_cfg.{field_name}: expected a boolean (0/1/true/false), got {raw!r}"
        )
    if type(default) is int:
        try:
            value = literal_eval(text)
        except (SyntaxError, ValueError):
            raise ValueError(f"--syn_cfg.{field_name}: expected an integer, got {raw!r}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"--syn_cfg.{field_name}: expected an integer, got {raw!r}")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"--syn_cfg.{field_name}: expected an integer, got {raw!r}")
        return int(value)
    if type(default) is float:
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"--syn_cfg.{field_name}: expected a number, got {raw!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"--syn_cfg.{field_name}: expected a finite number, got {raw!r}")
        return value
    if type(default) is str or isinstance(default, SynapticGranularity):
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1]
        return text
    raise ValueError(
        f"--syn_cfg.{field_name}: field has unsupported schema type {type(default).__name__}"
    )


def apply_syn_cfg_overrides(
    syn_cfg: SynapticConfig, overrides: Mapping[str, str]
) -> SynapticConfig:
    """Overlay command-line overrides onto ``syn_cfg`` IN PLACE and validate the result.

    Every value is coerced by :func:`coerce_syn_cfg_override`; the combined config must then
    pass :func:`synaptic_config_schema_errors` (types and literal sets) and
    :func:`validate_config` (an opt-in mechanism enabled without its prerequisite is an error,
    not a silent no-op). Legal-but-risky combinations are logged as warnings.
    """
    for key, raw in overrides.items():
        setattr(syn_cfg, key, coerce_syn_cfg_override(key, raw))
    schema_errors = synaptic_config_schema_errors(syn_cfg)
    if schema_errors:
        raise ValueError(
            "--syn_cfg overrides produced an invalid SynapticConfig: " + "; ".join(schema_errors)
        )
    errors, warnings = validate_config(syn_cfg)
    if errors:
        raise ValueError(
            "--syn_cfg overrides produced an inconsistent SynapticConfig: " + "; ".join(errors)
        )
    for message in warnings:
        logger.warning("--syn_cfg override warning: %s", message)
    return syn_cfg
