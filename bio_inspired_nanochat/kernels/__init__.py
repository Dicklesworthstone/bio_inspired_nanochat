import ast as _ast

for _ast_alias in ("Num", "Str", "Bytes", "NameConstant", "Ellipsis"):
    if not hasattr(_ast, _ast_alias):
        setattr(_ast, _ast_alias, _ast.Constant)

from .genetics_fused import (  # noqa: E402
    accumulate_router_stats as accumulate_router_stats,
    update_metabolism_fused as update_metabolism_fused,
)
from .metrics_fused import update_metrics_fused as update_metrics_fused  # noqa: E402
from .structural_fused import (  # noqa: E402
    mix_and_shift_rows as mix_and_shift_rows,
    mix_and_shift_tensors as mix_and_shift_tensors,
)

__all__ = [
    "accumulate_router_stats",
    "mix_and_shift_rows",
    "mix_and_shift_tensors",
    "update_metabolism_fused",
    "update_metrics_fused",
]

