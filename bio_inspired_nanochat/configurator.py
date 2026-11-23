"""Apply trusted Python config files and typed command-line overrides in order."""

import os
import sys
from ast import literal_eval
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bio_inspired_nanochat.common import print0


def apply_config(
    namespace: dict[str, Any], args: Sequence[str] | None = None
) -> dict[str, Any]:
    """Evaluate positional files in the supplied namespace, then apply each following flag."""
    # These names were already available to trusted recipes before their first statement.
    namespace.update(os=os, sys=sys, literal_eval=literal_eval, print0=print0)
    for arg in sys.argv[1:] if args is None else args:
        if "=" not in arg:
            if arg.startswith("--"):
                raise ValueError(f"Expected --key=value, got {arg!r}")
            code = Path(arg).read_text(encoding="utf-8")
            print0(f"Overriding config with {arg}:")
            print0(code)
            exec(compile(code, arg, "exec"), namespace)
            continue
        if not arg.startswith("--"):
            raise ValueError(f"Expected --key=value, got {arg!r}")
        key, raw = arg[2:].split("=", 1)
        if key not in namespace:
            raise ValueError(f"Unknown config key: {key}")
        try:
            value = literal_eval(raw)
        except (SyntaxError, ValueError):
            value = raw
        current = namespace[key]
        if current is not None and type(value) is not type(current):
            raise AssertionError(f"Type mismatch: {type(value)} != {type(current)}")
        print0(f"Overriding: {key} = {value}")
        namespace[key] = value
    return namespace


def configured_copy(
    namespace: dict[str, Any], *, script_name: str | None = None
) -> dict[str, Any]:
    """Keep isolated recipes in the same module or script context as their former runpy runner."""
    spec = __spec__ if script_name is None else None
    path = Path(__file__)
    if script_name is not None and path.is_relative_to(Path.cwd()):
        path = path.relative_to(Path.cwd())
    return apply_config({
        **namespace,
        "__name__": __name__ if script_name is None else script_name,
        "__file__": str(path),
        "__package__": __package__ if spec is not None else "",
        "__spec__": spec,
        "__loader__": spec.loader if spec is not None else None,
        "__cached__": spec.cached if spec is not None else None,
    })
