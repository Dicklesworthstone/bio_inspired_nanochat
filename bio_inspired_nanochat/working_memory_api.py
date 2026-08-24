"""Inspectable, bounded read/write access to synaptic working memory.

The API treats every :class:`~bio_inspired_nanochat.synaptic.SynapticLinear`
fast-weight matrix as one addressable scratchpad site. Writes are explicit
rank-one key/value associations and are deliberately restricted to evaluation
mode: mutating a parameter while an autograd graph is live is unsafe. Slow
weights are never changed.

Presynaptic calcium lives in a KV cache rather than in the model modules. A
cache (or its ``presyn_state`` payload) can therefore be supplied when reading
the scratchpad; the resulting snapshot remains JSON-safe and schema-versioned.
All mutating operations can emit structured events through the project's
``RunLogger`` interface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor, nn

from bio_inspired_nanochat.synaptic import SynapticLinear


WORKING_MEMORY_SCHEMA = "synaptic-working-memory/1"


class WorkingMemoryValidationError(ValueError):
    """Raised before a write when the proposed mutation violates policy."""


class EventLogger(Protocol):
    """The small portion of :class:`RunLogger` used by this API."""

    def event(self, event: str, **fields: Any) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class WorkingMemoryPolicy:
    """Safety envelope for explicit neural-memory writes.

    ``max_delta_norm`` bounds the Frobenius norm of a single rank-one write.
    Oversized deltas are scaled down, with the effective scale reported in the
    write receipt. ``max_norm_growth`` additionally bounds how much the whole
    fast-weight matrix may grow in one operation.
    """

    require_eval_mode: bool = True
    max_vector_norm: float = 16.0
    max_abs_scale: float = 1.0
    max_delta_norm: float = 1.0
    max_norm_growth: float = 1.0
    sparsity_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        positive = {
            "max_vector_norm": self.max_vector_norm,
            "max_abs_scale": self.max_abs_scale,
            "max_delta_norm": self.max_delta_norm,
            "max_norm_growth": self.max_norm_growth,
            "sparsity_epsilon": self.sparsity_epsilon,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")


@dataclass(frozen=True, slots=True)
class LayerMemoryState:
    """JSON-safe summary of one addressable fast-weight site."""

    site_index: int
    module: str
    shape: tuple[int, int] | None
    writable: bool
    finite: bool
    fast_weight_norm: float | None
    fast_weight_max_abs: float | None
    sparsity: float | None
    eligibility_u_norm: float | None
    eligibility_v_norm: float | None
    postsynaptic_fast_norm: float | None
    camkii_mean: float | None
    pp1_mean: float | None
    bdnf_mean: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable public representation."""
        return {
            "site_index": self.site_index,
            "module": self.module,
            "shape": list(self.shape) if self.shape is not None else None,
            "writable": self.writable,
            "finite": self.finite,
            "fast_weight_norm": self.fast_weight_norm,
            "fast_weight_max_abs": self.fast_weight_max_abs,
            "sparsity": self.sparsity,
            "eligibility_u_norm": self.eligibility_u_norm,
            "eligibility_v_norm": self.eligibility_v_norm,
            "postsynaptic_fast_norm": self.postsynaptic_fast_norm,
            "camkii_mean": self.camkii_mean,
            "pp1_mean": self.pp1_mean,
            "bdnf_mean": self.bdnf_mean,
        }


def _finite_float(value: Tensor) -> float:
    return float(value.detach().float().item())


def _optional_norm(value: Tensor | None) -> float | None:
    if value is None:
        return None
    norm = _finite_float(value.norm())
    return norm if math.isfinite(norm) else None


def _optional_float(value: Tensor) -> float | None:
    result = _finite_float(value)
    return result if math.isfinite(result) else None


class WorkingMemoryScratchpad:
    """Structured controller for the model's volatile synaptic state.

    Site indices are indices into ``model.named_modules()`` filtered to
    ``SynapticLinear`` instances. Every snapshot includes the stable module
    path as well, so callers can audit exactly what they addressed.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        policy: WorkingMemoryPolicy | None = None,
        logger: EventLogger | None = None,
    ) -> None:
        self.model = model
        self.policy = policy or WorkingMemoryPolicy()
        self.logger = logger

    def _sites(self) -> list[tuple[str, SynapticLinear]]:
        return [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, SynapticLinear)
        ]

    def _site(self, site_index: int) -> tuple[str, SynapticLinear]:
        if isinstance(site_index, bool) or not isinstance(site_index, int):
            raise WorkingMemoryValidationError("site_index must be an integer")
        sites = self._sites()
        if not 0 <= site_index < len(sites):
            raise WorkingMemoryValidationError(
                f"site_index {site_index} is outside [0, {len(sites)})"
            )
        return sites[site_index]

    def _emit(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.event(event, schema=WORKING_MEMORY_SCHEMA, **fields)

    def _layer_state(
        self,
        site_index: int,
        module_name: str,
        module: SynapticLinear,
    ) -> LayerMemoryState:
        weight = module.w_fast
        post = module.post
        if weight is None:
            return LayerMemoryState(
                site_index=site_index,
                module=module_name,
                shape=None,
                writable=False,
                finite=True,
                fast_weight_norm=0.0,
                fast_weight_max_abs=0.0,
                sparsity=1.0,
                eligibility_u_norm=_optional_norm(module.u_buf),
                eligibility_v_norm=_optional_norm(module.v_buf),
                postsynaptic_fast_norm=_optional_norm(post.fast) if post is not None else None,
                camkii_mean=_optional_float(post.camkii.mean()) if post is not None else None,
                pp1_mean=_optional_float(post.pp1.mean()) if post is not None else None,
                bdnf_mean=_optional_float(post.bdnf.mean()) if post is not None else None,
            )

        detached = weight.detach()
        finite = bool(torch.isfinite(detached).all().item())
        return LayerMemoryState(
            site_index=site_index,
            module=module_name,
            shape=(int(detached.shape[0]), int(detached.shape[1])),
            writable=True,
            finite=finite,
            fast_weight_norm=_finite_float(detached.norm()) if finite else None,
            fast_weight_max_abs=_finite_float(detached.abs().max()) if finite else None,
            sparsity=(
                _finite_float((detached.abs() < self.policy.sparsity_epsilon).float().mean())
                if finite
                else None
            ),
            eligibility_u_norm=_optional_norm(module.u_buf),
            eligibility_v_norm=_optional_norm(module.v_buf),
            postsynaptic_fast_norm=_optional_norm(post.fast) if post is not None else None,
            camkii_mean=_optional_float(post.camkii.mean()) if post is not None else None,
            pp1_mean=_optional_float(post.pp1.mean()) if post is not None else None,
            bdnf_mean=_optional_float(post.bdnf.mean()) if post is not None else None,
        )

    @staticmethod
    def _presyn_states(source: Any | None) -> list[Any | None]:
        if source is None:
            return []
        if not isinstance(source, (Mapping, Sequence)) and hasattr(source, "presyn_state"):
            source = source.presyn_state
        if source is None:
            return []
        if isinstance(source, Mapping):
            return [source]
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            return list(source)
        raise TypeError("presyn_state must be a cache, mapping, sequence, or None")

    @classmethod
    def _presynaptic_snapshot(cls, source: Any | None) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for layer_index, state in enumerate(cls._presyn_states(source)):
            calcium = state.get("C") if isinstance(state, Mapping) else None
            if not torch.is_tensor(calcium):
                result.append({"layer_index": layer_index, "available": False})
                continue
            values = calcium.detach().float()
            if values.numel() == 0:
                result.append(
                    {
                        "layer_index": layer_index,
                        "available": True,
                        "shape": list(values.shape),
                        "finite": True,
                        "mean": 0.0,
                        "minimum": 0.0,
                        "maximum": 0.0,
                    }
                )
                continue
            finite = bool(torch.isfinite(values).all().item())
            result.append(
                {
                    "layer_index": layer_index,
                    "available": True,
                    "shape": list(values.shape),
                    "finite": finite,
                    "mean": _finite_float(values.mean()) if finite else None,
                    "minimum": _finite_float(values.min()) if finite else None,
                    "maximum": _finite_float(values.max()) if finite else None,
                }
            )
        return result

    def read_scratchpad(self, presyn_state: Any | None = None) -> dict[str, Any]:
        """Return a schema-versioned, JSON-safe working-memory snapshot."""
        layers = [
            self._layer_state(index, name, module).to_dict()
            for index, (name, module) in enumerate(self._sites())
        ]
        presynaptic = self._presynaptic_snapshot(presyn_state)
        snapshot = {
            "schema": WORKING_MEMORY_SCHEMA,
            "num_sites": len(layers),
            "sites": layers,
            "presynaptic": presynaptic,
        }
        self._emit(
            "working_memory_read",
            num_sites=len(layers),
            writable_sites=sum(bool(layer["writable"]) for layer in layers),
            presynaptic_layers=len(presynaptic),
        )
        return snapshot

    def _validate_write_mode(self, module: SynapticLinear) -> None:
        if self.policy.require_eval_mode and self.model.training:
            raise WorkingMemoryValidationError(
                "working-memory writes require model.eval(); training-mode mutation is unsafe"
            )
        if module._plasticity_pending:
            raise WorkingMemoryValidationError(
                "target site has a deferred plasticity write; finish the training step first"
            )

    @torch.no_grad()
    def write_association(
        self,
        site_index: int,
        key_vector: Tensor,
        value_vector: Tensor,
        *,
        scale: float = 1.0,
        expected_module: str | None = None,
    ) -> dict[str, Any]:
        """Write the bounded association ``delta_W = scale * key outer value``.

        The operation validates everything before the parameter is touched. If
        the requested delta exceeds the configured per-write norm, the delta is
        scaled down and the exact effective scale is returned. Shape mismatch,
        non-finite input, stale module identity, training mode, and excessive
        whole-matrix growth are rejected atomically.
        """
        module_name, module = self._site(site_index)
        if expected_module is not None and expected_module != module_name:
            raise WorkingMemoryValidationError(
                f"site {site_index} is {module_name!r}, not expected module {expected_module!r}"
            )
        self._validate_write_mode(module)
        weight = module.w_fast
        if weight is None:
            raise WorkingMemoryValidationError(f"site {module_name!r} has no fast weights")
        if not torch.is_tensor(key_vector) or not torch.is_tensor(value_vector):
            raise WorkingMemoryValidationError("key_vector and value_vector must be tensors")
        if key_vector.ndim != 1 or value_vector.ndim != 1:
            raise WorkingMemoryValidationError("key_vector and value_vector must be one-dimensional")
        expected_key, expected_value = int(weight.shape[0]), int(weight.shape[1])
        if key_vector.shape[0] != expected_key or value_vector.shape[0] != expected_value:
            raise WorkingMemoryValidationError(
                "association dimensions do not match target site: "
                f"expected ({expected_key}, {expected_value}), got "
                f"({key_vector.shape[0]}, {value_vector.shape[0]})"
            )
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise WorkingMemoryValidationError("scale must be a real number")
        requested_scale = float(scale)
        if not math.isfinite(requested_scale) or abs(requested_scale) > self.policy.max_abs_scale:
            raise WorkingMemoryValidationError(
                f"abs(scale) must be finite and <= {self.policy.max_abs_scale}"
            )

        key = key_vector.detach().to(device=weight.device, dtype=weight.dtype)
        value = value_vector.detach().to(device=weight.device, dtype=weight.dtype)
        if not bool(torch.isfinite(key).all().item()) or not bool(torch.isfinite(value).all().item()):
            raise WorkingMemoryValidationError("association vectors must contain only finite values")
        key_norm = _finite_float(key.norm())
        value_norm = _finite_float(value.norm())
        if key_norm > self.policy.max_vector_norm or value_norm > self.policy.max_vector_norm:
            raise WorkingMemoryValidationError(
                f"vector norms must be <= {self.policy.max_vector_norm}; "
                f"got key={key_norm:.6g}, value={value_norm:.6g}"
            )

        delta = torch.outer(key, value) * requested_scale
        if not bool(torch.isfinite(delta).all().item()):
            raise WorkingMemoryValidationError("association delta became non-finite in target dtype")
        requested_delta_norm = _finite_float(delta.norm())
        clip_factor = 1.0
        if requested_delta_norm > self.policy.max_delta_norm:
            clip_factor = self.policy.max_delta_norm / requested_delta_norm
            delta = delta * clip_factor
        effective_scale = requested_scale * clip_factor
        applied_delta_norm = _finite_float(delta.norm())

        before = weight.detach().clone()
        before_norm = _finite_float(before.norm())
        proposed = before + delta
        if not bool(torch.isfinite(proposed).all().item()):
            raise WorkingMemoryValidationError("write would make the target fast weights non-finite")
        after_norm = _finite_float(proposed.norm())
        if after_norm > before_norm + self.policy.max_norm_growth + 1e-6:
            raise WorkingMemoryValidationError(
                "write would exceed the allowed fast-weight norm growth: "
                f"before={before_norm:.6g}, after={after_norm:.6g}, "
                f"limit={self.policy.max_norm_growth:.6g}"
            )

        weight.copy_(proposed)
        receipt = {
            "schema": WORKING_MEMORY_SCHEMA,
            "operation": "write_association",
            "site_index": site_index,
            "module": module_name,
            "requested_scale": requested_scale,
            "effective_scale": effective_scale,
            "clipped": clip_factor < 1.0,
            "key_norm": key_norm,
            "value_norm": value_norm,
            "requested_delta_norm": requested_delta_norm,
            "applied_delta_norm": applied_delta_norm,
            "fast_weight_norm_before": before_norm,
            "fast_weight_norm_after": after_norm,
        }
        self._emit(
            "working_memory_write",
            **{key: value for key, value in receipt.items() if key != "schema"},
        )
        return receipt

    @torch.no_grad()
    def clear_scratchpad(self, site_index: int | None = None) -> dict[str, Any]:
        """Clear volatile state at one site or all sites, preserving slow memory."""
        if self.policy.require_eval_mode and self.model.training:
            raise WorkingMemoryValidationError(
                "working-memory clears require model.eval(); training-mode mutation is unsafe"
            )
        if site_index is None:
            targets = list(enumerate(self._sites()))
        else:
            module_name, module = self._site(site_index)
            targets = [(site_index, (module_name, module))]

        for _, (_, module) in targets:
            self._validate_write_mode(module)

        cleared: list[dict[str, Any]] = []
        for index, (module_name, module) in targets:
            before_norm = _optional_norm(module.w_fast)
            module.reset_sequence_state(reset_fast_weights=True, reset_consolidation=True)
            cleared.append(
                {
                    "site_index": index,
                    "module": module_name,
                    "fast_weight_norm_before": before_norm,
                }
            )

        receipt = {
            "schema": WORKING_MEMORY_SCHEMA,
            "operation": "clear_scratchpad",
            "cleared_sites": len(cleared),
            "sites": cleared,
        }
        self._emit(
            "working_memory_clear",
            **{key: value for key, value in receipt.items() if key != "schema"},
        )
        return receipt

    def log_scratchpad_state(
        self,
        console: Console | None = None,
        *,
        presyn_state: Any | None = None,
    ) -> None:
        """Render a compact Rich table; machine-readable logging remains authoritative."""
        snapshot = self.read_scratchpad(presyn_state)
        output = console or Console()
        table = Table(title="Synaptic working-memory scratchpad")
        table.add_column("Site", justify="right")
        table.add_column("Module")
        table.add_column("Writable", justify="center")
        table.add_column("Fast norm", justify="right")
        table.add_column("Max |w|", justify="right")
        table.add_column("Sparse", justify="right")
        for site in snapshot["sites"]:
            fast_norm = site["fast_weight_norm"]
            max_abs = site["fast_weight_max_abs"]
            sparsity = site["sparsity"]
            table.add_row(
                str(site["site_index"]),
                str(site["module"]),
                "yes" if site["writable"] else "no",
                f"{fast_norm:.4f}" if isinstance(fast_norm, (int, float)) else "non-finite",
                f"{max_abs:.4f}" if isinstance(max_abs, (int, float)) else "non-finite",
                f"{100.0 * sparsity:.1f}%" if isinstance(sparsity, (int, float)) else "n/a",
            )
        output.print(table)
