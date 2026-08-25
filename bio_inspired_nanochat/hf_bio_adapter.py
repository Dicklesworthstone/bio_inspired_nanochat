"""Function-preserving bio adapters for Hugging Face transformer feed-forward layers.

The adapter deliberately targets the architecture seam shared by most transformer families:
feed-forward projections.  It converts PyTorch ``Linear`` and Hugging Face ``Conv1D`` modules
into the project's existing :class:`~bio_inspired_nanochat.synaptic.SynapticLinear`, while
copying the pretrained affine map exactly.  Attention projections and language-model heads are
never selected by the default patterns.

Injection is fail-closed: a model with no recognized feed-forward projections raises instead of
silently claiming success.  Explicit target patterns let downstream users support unusual model
families without adding architecture-specific compatibility shims here.
"""

from __future__ import annotations

import fnmatch
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from transformers.pytorch_utils import Conv1D

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear
from bio_inspired_nanochat.torch_imports import Tensor, nn, torch

ADAPTER_MANIFEST = "bio_adapter.json"
ADAPTER_WEIGHTS = "bio_adapter.safetensors"

# Family examples covered by these structural names include GPT-2, GPT-NeoX, Llama/Mistral,
# OPT/BART, and BERT-like encoder blocks.  The ``mlp``/``ffn``/``intermediate`` ancestors keep
# attention output projections and task heads outside the default edit surface.
DEFAULT_TARGET_PATTERNS: tuple[str, ...] = (
    "*.mlp.c_fc",
    "*.mlp.c_proj",
    "*.mlp.gate_proj",
    "*.mlp.up_proj",
    "*.mlp.down_proj",
    "*.mlp.fc1",
    "*.mlp.fc2",
    "*.mlp.dense_h_to_4h",
    "*.mlp.dense_4h_to_h",
    "*.feed_forward.*",
    "*.ffn.*",
    "*.intermediate.dense",
    "*.layer.*.output.dense",
)


@dataclass(frozen=True)
class BioAdapterReport:
    """Auditable result of one in-place adapter injection."""

    model_class: str
    model_type: str | None
    adapter_names: tuple[str, ...]
    source_kinds: tuple[str, ...]
    adapted_parameters: int
    target_patterns: tuple[str, ...]

    @property
    def adapter_count(self) -> int:
        return len(self.adapter_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_class": self.model_class,
            "model_type": self.model_type,
            "adapter_names": list(self.adapter_names),
            "source_kinds": list(self.source_kinds),
            "adapter_count": self.adapter_count,
            "adapted_parameters": self.adapted_parameters,
            "target_patterns": list(self.target_patterns),
        }


class HFBioLinearAdapter(nn.Module):
    """A Hugging Face affine projection backed by ``SynapticLinear``.

    The scalar calcium and energy states summarize each projection's recent activation stream.
    They gate the existing fast-weight path; all actual eligibility, CaMKII/PP1, BDNF, and
    fast/slow weight updates remain owned by ``SynapticLinear``.
    """

    def __init__(self, source: nn.Module, synaptic_config: SynapticConfig) -> None:
        super().__init__()
        if isinstance(source, nn.Linear):
            in_features = int(source.in_features)
            out_features = int(source.out_features)
            source_weight = source.weight.detach().transpose(0, 1)
            source_bias = source.bias
            source_kind = "torch.nn.Linear"
        elif isinstance(source, Conv1D):
            in_features = int(source.weight.shape[0])
            out_features = int(source.weight.shape[1])
            source_weight = source.weight.detach()
            source_bias = source.bias
            source_kind = "transformers.Conv1D"
        else:
            raise TypeError(
                "HFBioLinearAdapter supports torch.nn.Linear and transformers.Conv1D; "
                f"got {type(source).__module__}.{type(source).__qualname__}"
            )

        object.__setattr__(self, "synaptic_config", synaptic_config)
        self.in_features = in_features
        self.out_features = out_features
        self.source_kind = source_kind
        # Injection is function-preserving and behavior-neutral by default. Callers must opt in
        # explicitly with set_bio_adaptation(..., True) before training or adaptive inference.
        self.adaptation_enabled = False

        core = SynapticLinear(
            in_features,
            out_features,
            synaptic_config,
            bias=source_bias is not None,
            use_input_ln=False,
        ).to(device=source_weight.device, dtype=source_weight.dtype)
        with torch.no_grad():
            core.w_slow.copy_(source_weight)
            if core.w_fast is not None:
                core.w_fast.zero_()
            if core.bias is not None and source_bias is not None:
                core.bias.copy_(source_bias.detach())
            # U=0 makes the postsynaptic residual exactly zero at injection.  V deliberately
            # remains random so U receives a gradient immediately instead of creating a dead
            # zero-times-zero low-rank branch.
            if core.post is not None:
                core.post.fast.zero_()
                core.post.slow.zero_()
                core.post.U.zero_()
        self.core = core

        state_device = source_weight.device
        self.register_buffer(
            "calcium_state",
            torch.zeros((), device=state_device, dtype=torch.float32),
        )
        initial_energy = synaptic_config.init_energy if synaptic_config.enable_metabolism else 1.0
        self.register_buffer(
            "energy_state",
            torch.tensor(initial_energy, device=state_device, dtype=torch.float32),
        )
        self.register_buffer(
            "adaptation_steps",
            torch.zeros((), device=state_device, dtype=torch.int64),
        )

    @torch.no_grad()
    def _signals(self, x: Tensor, *, advance: bool) -> tuple[Tensor, Tensor]:
        """Advance bounded presynaptic/metabolic state and return per-row gates."""
        rows = int(x.shape[0])
        if advance:
            drive = torch.tanh(x.detach().float().square().mean().sqrt())
            cfg = self.synaptic_config
            if cfg.enable_presyn:
                retention = math.exp(-1.0 / max(float(cfg.tau_c), 1e-6))
                self.calcium_state.mul_(retention).add_(float(cfg.alpha_ca) * drive)
                self.calcium_state.clamp_(0.0, 1.6)
            if cfg.enable_metabolism:
                refill = float(cfg.energy_fill) * (
                    float(cfg.energy_max) - self.energy_state
                )
                self.energy_state.add_(refill - float(cfg.energy_use) * drive)
                self.energy_state.clamp_(0.0, max(float(cfg.energy_max), 1e-6))
            self.adaptation_steps.add_(1)

        if self.synaptic_config.enable_presyn:
            calcium = self.calcium_state.to(dtype=x.dtype).expand(rows)
        else:
            calcium = torch.ones(rows, device=x.device, dtype=x.dtype)
        if self.synaptic_config.enable_metabolism:
            energy = self.energy_state.to(dtype=x.dtype).expand(rows)
        else:
            energy = torch.ones(rows, device=x.device, dtype=x.dtype)
        return calcium, energy

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or int(x.shape[-1]) != self.in_features:
            raise ValueError(
                f"adapter expected final dimension {self.in_features}, got {tuple(x.shape)}"
            )
        output_shape = (*x.shape[:-1], self.out_features)
        flat = x.reshape(-1, self.in_features)
        grad_on = torch.is_grad_enabled()
        advance = self.adaptation_enabled and (not grad_on or self.training)
        calcium, energy = self._signals(flat, advance=advance)
        output = self.core(
            flat,
            calcium,
            energy,
            update_mem=advance,
        )
        return output.reshape(output_shape)

    @torch.no_grad()
    def reset_bio_state(
        self,
        *,
        reset_fast_weights: bool = False,
        reset_consolidation: bool = True,
    ) -> None:
        """Reset per-sequence dynamics without replacing pretrained slow weights."""
        self.calcium_state.zero_()
        initial_energy = (
            self.synaptic_config.init_energy
            if self.synaptic_config.enable_metabolism
            else 1.0
        )
        self.energy_state.fill_(initial_energy)
        self.adaptation_steps.zero_()
        self.core.reset_sequence_state(
            reset_fast_weights=reset_fast_weights,
            reset_consolidation=reset_consolidation,
        )

    def metrics(self) -> dict[str, float | int | str | bool]:
        """Return small scalar evidence suitable for structured logging."""
        fast_norm = 0.0
        eligibility_norm = 0.0
        camkii_mean = 0.0
        if self.core.w_fast is not None:
            fast_norm = float(self.core.w_fast.detach().float().norm().item())
        if self.core.u_buf is not None and self.core.v_buf is not None:
            eligibility_norm = float(
                (self.core.u_buf.detach().float().norm() + self.core.v_buf.detach().float().norm())
                .item()
            )
        if self.core.post is not None:
            camkii_mean = float(self.core.post.camkii.detach().float().mean().item())
        return {
            "source_kind": self.source_kind,
            "adaptation_enabled": self.adaptation_enabled,
            "adaptation_steps": int(self.adaptation_steps.item()),
            "calcium": float(self.calcium_state.item()),
            "energy": float(self.energy_state.item()),
            "fast_weight_norm": fast_norm,
            "eligibility_norm": eligibility_norm,
            "camkii_mean": camkii_mean,
        }


def iter_bio_adapters(model: nn.Module) -> Iterator[tuple[str, HFBioLinearAdapter]]:
    """Yield adapter name/module pairs in model traversal order."""
    for name, module in model.named_modules():
        if isinstance(module, HFBioLinearAdapter):
            yield name, module


def _model_type(model: nn.Module) -> str | None:
    value = getattr(getattr(model, "config", None), "model_type", None)
    return str(value) if value is not None else None


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def inject_bio_adapters(
    model: nn.Module,
    synaptic_config: SynapticConfig | None = None,
    *,
    target_patterns: Sequence[str] = DEFAULT_TARGET_PATTERNS,
) -> BioAdapterReport:
    """Replace matching feed-forward projections in ``model`` in place.

    ``target_patterns`` are shell-style globs over ``named_modules()`` paths.  Exact module names
    are therefore valid patterns and are used by :func:`load_bio_adapter` for strict restoration.
    """
    patterns = tuple(str(pattern) for pattern in target_patterns)
    if not patterns or any(not pattern.strip() for pattern in patterns):
        raise ValueError("target_patterns must contain at least one non-blank pattern")
    existing = tuple(iter_bio_adapters(model))
    if existing:
        names = ", ".join(name for name, _ in existing[:5])
        raise ValueError(f"model already contains bio adapters: {names}")

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name
        and isinstance(module, (nn.Linear, Conv1D))
        and _matches(name, patterns)
    ]
    if not candidates:
        available = [
            name
            for name, module in model.named_modules()
            if name and isinstance(module, (nn.Linear, Conv1D))
        ]
        sample = ", ".join(available[:8]) or "<none>"
        raise ValueError(
            "no supported feed-forward projections matched target_patterns; "
            f"available affine modules include: {sample}"
        )

    cfg = synaptic_config or SynapticConfig()
    adapter_names: list[str] = []
    source_kinds: list[str] = []
    adapted_parameters = 0
    for name, source in candidates:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        adapter = HFBioLinearAdapter(source, cfg)
        setattr(parent, child_name, adapter)
        adapter_names.append(name)
        source_kinds.append(adapter.source_kind)
        adapted_parameters += sum(parameter.numel() for parameter in adapter.parameters())

    return BioAdapterReport(
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        model_type=_model_type(model),
        adapter_names=tuple(adapter_names),
        source_kinds=tuple(source_kinds),
        adapted_parameters=adapted_parameters,
        target_patterns=patterns,
    )


def set_bio_adaptation(model: nn.Module, enabled: bool) -> int:
    """Explicitly enable or pause training/adaptive-inference updates on every adapter."""
    count = 0
    for _, adapter in iter_bio_adapters(model):
        adapter.adaptation_enabled = bool(enabled)
        count += 1
    if count == 0:
        raise ValueError("model contains no bio adapters")
    return count


def reset_bio_adapters(
    model: nn.Module,
    *,
    reset_fast_weights: bool = False,
    reset_consolidation: bool = True,
) -> int:
    """Reset per-sequence state for every adapter and return the adapter count."""
    count = 0
    for _, adapter in iter_bio_adapters(model):
        adapter.reset_bio_state(
            reset_fast_weights=reset_fast_weights,
            reset_consolidation=reset_consolidation,
        )
        count += 1
    if count == 0:
        raise ValueError("model contains no bio adapters")
    return count


def bio_adapter_metrics(
    model: nn.Module,
) -> dict[str, dict[str, float | int | str | bool]]:
    """Collect scalar runtime evidence for all adapters."""
    metrics = {name: adapter.metrics() for name, adapter in iter_bio_adapters(model)}
    if not metrics:
        raise ValueError("model contains no bio adapters")
    return metrics


def bio_adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return unique adapter parameters for a light adapter-only optimizer."""
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for _, adapter in iter_bio_adapters(model):
        for parameter in adapter.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                parameters.append(parameter)
    if not parameters:
        raise ValueError("model contains no bio adapters")
    return parameters


def save_bio_adapter(model: nn.Module, directory: str | Path) -> dict[str, Any]:
    """Package injected adapter state as safetensors plus a strict JSON manifest."""
    adapters = tuple(iter_bio_adapters(model))
    if not adapters:
        raise ValueError("model contains no bio adapters")
    output_dir = Path(directory)
    manifest_path = output_dir / ADAPTER_MANIFEST
    weights_path = output_dir / ADAPTER_WEIGHTS
    existing = [path for path in (manifest_path, weights_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing adapter artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    config_values = [asdict(adapter.synaptic_config) for _, adapter in adapters]
    config_fingerprints = {json.dumps(config, sort_keys=True) for config in config_values}
    if len(config_fingerprints) != 1:
        raise ValueError("all adapters must share one SynapticConfig to create a portable bundle")
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, Tensor] = {}
    for name, adapter in adapters:
        for key, value in adapter.state_dict().items():
            tensors[f"{name}::{key}"] = value.detach().cpu().contiguous()
    save_file(tensors, weights_path)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "format": "bio_inspired_nanochat.hf_bio_adapter",
        "base_model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "base_model_type": _model_type(model),
        "adapter_names": [name for name, _ in adapters],
        "source_kinds": [adapter.source_kind for _, adapter in adapters],
        "synaptic_config": config_values[0],
        "weights": ADAPTER_WEIGHTS,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_bio_adapter(
    model: nn.Module,
    directory: str | Path,
    *,
    adaptation_enabled: bool = False,
) -> BioAdapterReport:
    """Inject and strictly restore a packaged adapter into a fresh base model."""
    input_dir = Path(directory)
    try:
        manifest_raw = json.loads(
            (input_dir / ADAPTER_MANIFEST).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read bio adapter manifest: {error}") from error
    if not isinstance(manifest_raw, dict) or manifest_raw.get("schema_version") != 1:
        raise ValueError("unsupported or malformed bio adapter manifest")
    names_raw = manifest_raw.get("adapter_names")
    config_raw = manifest_raw.get("synaptic_config")
    if (
        not isinstance(names_raw, list)
        or not names_raw
        or not all(isinstance(name, str) and name for name in names_raw)
        or not isinstance(config_raw, dict)
    ):
        raise ValueError("bio adapter manifest is missing adapter_names or synaptic_config")
    expected_model_type = manifest_raw.get("base_model_type")
    if expected_model_type != _model_type(model):
        raise ValueError(
            "base model type does not match adapter bundle: "
            f"expected {expected_model_type!r}, got {_model_type(model)!r}"
        )

    names = tuple(names_raw)
    report = inject_bio_adapters(
        model,
        SynapticConfig(**config_raw),
        target_patterns=names,
    )
    if report.adapter_names != names:
        raise ValueError(
            f"adapter topology mismatch: expected {names!r}, got {report.adapter_names!r}"
        )

    weights_name = manifest_raw.get("weights")
    if weights_name != ADAPTER_WEIGHTS:
        raise ValueError(f"unsupported adapter weights file {weights_name!r}")
    flat_state = load_file(input_dir / ADAPTER_WEIGHTS)
    expected_keys: set[str] = set()
    adapters = dict(iter_bio_adapters(model))
    for name, adapter in adapters.items():
        state = adapter.state_dict()
        restored: dict[str, Tensor] = {}
        for key in state:
            flat_key = f"{name}::{key}"
            expected_keys.add(flat_key)
            if flat_key not in flat_state:
                raise ValueError(f"adapter bundle is missing tensor {flat_key!r}")
            restored[key] = flat_state[flat_key]
        adapter.load_state_dict(restored, strict=True)
        adapter.adaptation_enabled = bool(adaptation_enabled)
    extras = sorted(set(flat_state) - expected_keys)
    if extras:
        raise ValueError(f"adapter bundle contains unexpected tensors: {extras[:5]}")
    return report
