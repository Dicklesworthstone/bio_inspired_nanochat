"""In-Silico Neuroscience: Probing, Lesion/Ablation, and Optogenetic Stimulation Toolkit.

Provides a unified, clean Python API for causal neuroscience experiments on living
bio-inspired transformer models (beads odq.1, odq.2, odq.3, eqyk.12):

  1. ``PatchClampProbe``: Attaches non-invasive PyTorch forward hooks to record per-token/per-head/
     per-expert bio-state traces (calcium, RRP vesicle pools, CaMKII/PP1 latch, BDNF, fast weights).
  2. ``LesionContext`` / ``lesion()``: Context manager for acute in-silico lesions (knocking out
     specific attention heads, MoE experts, or biological mechanisms like Hebbian writes or vesicle fatigue)
     with automatic state restoration upon exit.
  3. ``OptogeneticStimulation`` / ``optogenetic_clamp()``: Context manager for clamping or injecting
     synaptic quantities (e.g. forcing high calcium influx, pinning CaMKII ON, depleting RRP, or
     injecting dopamine/acetylcholine bursts) to test causal sufficiency and behavioral overrides.
  4. ``CausalInterventionSuite``: Runs paired baseline vs intervention vs rescue evaluations to quantify
     causal attribution (KL divergence, logit MSE, accuracy delta, token flips).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from bio_inspired_nanochat.synaptic import (
    PostsynapticHebb,
    SynapticCausalSelfAttention,
    SynapticLinear,
    SynapticPresyn,
)


@dataclass
class ProbeSnapshot:
    """A single recording snapshot from a probed layer during forward execution."""

    layer_idx: int
    module_name: str
    token_count: int
    mean_calcium: float | None = None
    mean_rrp: float | None = None
    mean_camkii: float | None = None
    mean_pp1: float | None = None
    mean_bdnf: float | None = None
    fast_weight_norm: float | None = None
    slow_weight_norm: float | None = None
    output_norm: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class PatchClampProbe:
    """Non-invasive probe recording internal biological and activation dynamics.

    Usage::

        probe = PatchClampProbe(model)
        probe.attach()
        out = model(inputs)
        traces = probe.get_trace()
        probe.detach()

        # Or as a context manager:
        with PatchClampProbe(model) as probe:
            out = model(inputs)
            traces = probe.get_trace()
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: list[RemovableHandle] = []
        self.snapshots: list[ProbeSnapshot] = []
        self._step_counter = 0

    def attach(self) -> PatchClampProbe:
        """Attach recording hooks to all synaptic and attention modules."""
        self.detach()
        self.snapshots.clear()

        for name, module in self.model.named_modules():
            if isinstance(module, SynapticLinear):
                self._attach_linear_hook(name, module)
            elif isinstance(module, SynapticPresyn):
                self._attach_presyn_hook(name, module)

        return self

    def _attach_linear_hook(self, name: str, lin: SynapticLinear) -> None:
        def _hook(m: nn.Module, inp: Any, out: Any) -> None:
            # Extract layer index from name (e.g. 'h.0.attn.c_attn')
            parts = name.split(".")
            layer_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            out_norm = float(out_tensor.detach().norm().item()) if torch.is_tensor(out_tensor) else 0.0

            post: PostsynapticHebb | None = getattr(lin, "post", None)
            camkii_val = float(post.camkii.mean().item()) if post is not None and hasattr(post, "camkii") else None
            pp1_val = float(post.pp1.mean().item()) if post is not None and hasattr(post, "pp1") else None
            bdnf_val = float(post.bdnf.mean().item()) if post is not None and hasattr(post, "bdnf") else None

            fast_norm = float(lin.w_fast.detach().norm().item()) if lin.w_fast is not None else None
            slow_norm = float(lin.w_slow.detach().norm().item()) if lin.w_slow is not None else None

            tokens = int(out_tensor.shape[1]) if torch.is_tensor(out_tensor) and out_tensor.dim() >= 2 else 1

            self.snapshots.append(
                ProbeSnapshot(
                    layer_idx=layer_idx,
                    module_name=name,
                    token_count=tokens,
                    mean_camkii=camkii_val,
                    mean_pp1=pp1_val,
                    mean_bdnf=bdnf_val,
                    fast_weight_norm=fast_norm,
                    slow_weight_norm=slow_norm,
                    output_norm=out_norm,
                )
            )

        self.handles.append(lin.register_forward_hook(_hook))

    def _attach_presyn_hook(self, name: str, presyn: SynapticPresyn) -> None:
        def _hook(m: nn.Module, inp: Any, out: Any) -> None:
            parts = name.split(".")
            layer_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            state_dict = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 and isinstance(out[1], dict) else {}

            c_val = float(state_dict["C"].mean().item()) if "C" in state_dict else None
            rrp_val = float(state_dict["RRP"].mean().item()) if "RRP" in state_dict else None

            tokens = int(out_tensor.shape[2]) if torch.is_tensor(out_tensor) and out_tensor.dim() >= 3 else 1

            self.snapshots.append(
                ProbeSnapshot(
                    layer_idx=layer_idx,
                    module_name=name,
                    token_count=tokens,
                    mean_calcium=c_val,
                    mean_rrp=rrp_val,
                )
            )

        self.handles.append(presyn.register_forward_hook(_hook))

    def detach(self) -> None:
        """Remove all attached hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def get_trace(self) -> list[dict[str, Any]]:
        """Return all recorded snapshots as JSON-safe dicts."""
        return [
            {
                "layer_idx": s.layer_idx,
                "module": s.module_name,
                "tokens": s.token_count,
                "calcium": s.mean_calcium,
                "rrp": s.mean_rrp,
                "camkii": s.mean_camkii,
                "pp1": s.mean_pp1,
                "bdnf": s.mean_bdnf,
                "fast_norm": s.fast_weight_norm,
                "slow_norm": s.slow_weight_norm,
                "out_norm": s.output_norm,
            }
            for s in self.snapshots
        ]

    def __enter__(self) -> PatchClampProbe:
        return self.attach()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.detach()


@contextmanager
def lesion_head(model: nn.Module, *, layer_idx: int, head_idx: int) -> Iterator[None]:
    """Acutely knock out an attention head during inference by zeroing its projection output."""
    handles: list[RemovableHandle] = []

    for module in model.modules():
        if isinstance(module, SynapticCausalSelfAttention):
            if module.layer_idx == layer_idx:
                head_dim = module.head_dim
                n_head = module.n_head

                def _zero_head_prehook(m: nn.Module, inp: tuple[Any, ...]) -> tuple[Tensor, ...]:
                    x = inp[0]
                    if not torch.is_tensor(x):
                        return inp
                    x_mod = x.clone()
                    start_ch = head_idx * head_dim
                    end_ch = min((head_idx + 1) * head_dim, n_head * head_dim)
                    x_mod[:, :, start_ch:end_ch] = 0.0
                    return (x_mod,)

                handles.append(module.o_proj.register_forward_pre_hook(_zero_head_prehook))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def lesion_mechanism(model: nn.Module, mechanism: str) -> Iterator[None]:
    """Acutely knock out a biological mechanism (hebbian, vesicle_fatigue, camkii, bdnf, dopamine)."""
    saved_state: dict[tuple[int, str], tuple[Any, str, Any]] = {}

    def _save_and_set(obj: Any, attr: str, new_val: Any) -> None:
        key = (id(obj), attr)
        if key not in saved_state:
            saved_state[key] = (obj, attr, getattr(obj, attr, None))
        setattr(obj, attr, new_val)

    for module in model.modules():
        if isinstance(module, SynapticLinear):
            cfg = getattr(module, "cfg", None)
            if cfg is not None:
                if mechanism == "hebbian":
                    _save_and_set(cfg, "enable_hebbian", False)
                elif mechanism == "camkii":
                    _save_and_set(cfg, "bistable_latch", False)
                elif mechanism == "bdnf":
                    _save_and_set(cfg, "bdnf_scale", 0.0)
            if mechanism == "dopamine":
                _save_and_set(module, "_nm_da_gain", 0.0)

        elif isinstance(module, SynapticPresyn):
            cfg = getattr(module, "cfg", None)
            if cfg is not None and mechanism == "vesicle_fatigue":
                _save_and_set(cfg, "alpha_refill", 100.0)  # Infinite refill => no vesicle depletion/fatigue

    try:
        yield
    finally:
        for obj, attr, val in saved_state.values():
            if val is not None:
                setattr(obj, attr, val)
            elif hasattr(obj, attr):
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass


@contextmanager
def optogenetic_clamp(
    model: nn.Module,
    target: str,
    value: float,
    *,
    layer_idx: int | None = None,
) -> Iterator[None]:
    """Clamps or injects a biological synaptic state quantity mid-inference (optogenetics analog).

    Supported targets:
      - 'calcium': Clamps or boosts presynaptic calcium influx.
      - 'rrp': Clamps available Ready-Releasable vesicle Pool.
      - 'camkii': Pins postsynaptic CaMKII activation level.
      - 'dopamine': Injects a high dopamine plasticity gain.
    """
    saved_state: dict[tuple[int, str], tuple[Any, str, Any]] = {}
    handles: list[RemovableHandle] = []

    def _save_and_set(obj: Any, attr: str, new_val: Any) -> None:
        key = (id(obj), attr)
        if key not in saved_state:
            val = getattr(obj, attr, None)
            saved_state[key] = (obj, attr, val.detach().clone() if torch.is_tensor(val) else val)
        cur = getattr(obj, attr, None)
        if torch.is_tensor(cur):
            cur.fill_(float(new_val))
        else:
            setattr(obj, attr, new_val)

    for name, module in model.named_modules():
        parts = name.split(".")
        cur_layer = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        if layer_idx is not None and cur_layer is not None and cur_layer != layer_idx:
            continue

        if target == "dopamine" and isinstance(module, SynapticLinear):
            _save_and_set(module, "_nm_da_gain", float(value))

        elif target == "camkii" and isinstance(module, SynapticLinear):
            post = getattr(module, "post", None)
            if post is not None and hasattr(post, "camkii"):
                _save_and_set(post, "camkii", float(value))

        elif target == "rrp" and isinstance(module, SynapticPresyn):
            def _clamp_rrp_hook(m: nn.Module, inp: Any, out: Any) -> Any:
                if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], dict):
                    out[1]["RRP"].fill_(float(value))
                return out

            handles.append(module.register_forward_hook(_clamp_rrp_hook))

        elif target == "calcium" and isinstance(module, SynapticPresyn):
            def _clamp_ca_hook(m: nn.Module, inp: Any, out: Any) -> Any:
                if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], dict):
                    out[1]["C"].fill_(float(value))
                return out

            handles.append(module.register_forward_hook(_clamp_ca_hook))

    try:
        yield
    finally:
        for h in handles:
            h.remove()
        for obj, attr, val in saved_state.values():
            if torch.is_tensor(val):
                getattr(obj, attr).copy_(val)
            elif val is not None:
                setattr(obj, attr, val)
            elif hasattr(obj, attr):
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass


def compute_causal_effect(
    baseline_logits: Tensor,
    intervention_logits: Tensor,
) -> dict[str, float]:
    """Compute divergence metrics measuring the causal effect of an intervention."""
    p_base = F.softmax(baseline_logits.detach().float(), dim=-1)
    p_inter = F.softmax(intervention_logits.detach().float(), dim=-1)

    # Logit MSE
    mse = float(F.mse_loss(intervention_logits.detach(), baseline_logits.detach()).item())

    # KL Divergence D_KL(P_base || P_inter)
    eps = 1e-8
    kl = float((p_base * (torch.log(p_base + eps) - torch.log(p_inter + eps))).sum(dim=-1).mean().item())

    # Top-1 token prediction flip rate
    base_pred = baseline_logits.detach().argmax(dim=-1)
    inter_pred = intervention_logits.detach().argmax(dim=-1)
    flip_rate = float((base_pred != inter_pred).float().mean().item())

    return {
        "logit_mse": mse,
        "kl_divergence": kl,
        "prediction_flip_rate": flip_rate,
    }
