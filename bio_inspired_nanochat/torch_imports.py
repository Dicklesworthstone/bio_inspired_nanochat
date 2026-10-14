from __future__ import annotations

import ast as _ast
from typing import Any, cast

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

# Python 3.14 removed the deprecated ast.Num / ast.Str / ast.Bytes / ast.NameConstant /
# ast.Ellipsis aliases for ast.Constant, which Triton 3.5's @triton.jit AST code generator
# still calls when lowering 1-arg or 2-arg `for ... in range(...)` loops.
for _ast_alias in ("Num", "Str", "Bytes", "NameConstant", "Ellipsis"):
    if not hasattr(_ast, _ast_alias):
        setattr(_ast, _ast_alias, _ast.Constant)

torch = cast(Any, _torch)
nn = cast(Any, _nn)
F = cast(Any, _F)
Tensor = Any

