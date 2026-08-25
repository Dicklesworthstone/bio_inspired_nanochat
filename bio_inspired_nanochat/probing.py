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
from bio_inspired_nanochat import synaptic as _synaptic_module


class _MethodPatch:
    """Instance-level bound-method patch with a hook-handle-compatible removal.

    Forward hooks only fire on ``Module.__call__`` — but the live attention path
    invokes ``self.pre.release_canonical(...)`` as a PLAIN BOUND METHOD
    (synaptic.py), so hooks registered on SynapticPresyn never execute and any
    instrument relying on them silently records nothing / clamps nothing.
    Wrapping the bound method on the instance is the interception point that
    runs in every live path. ``remove()`` deletes the instance attribute,
    restoring the class method.
    """

    def __init__(self, owner: nn.Module, attr: str, wrapper) -> None:
        self._owner = owner
        self._attr = attr
        # Save the CURRENT value (bound method OR module-level function) and
        # restore exactly that on remove() — a blind pop would leave module
        # globals undefined (NameError on the next caller).
        self._had_original = hasattr(owner, attr)
        self._original = getattr(owner, attr, None)
        setattr(owner, attr, wrapper)

    def remove(self) -> None:
        original = self._original
        # If the saved original was a method BOUND TO THIS OWNER, restoring via
        # setattr would leave it as a permanent instance attribute shadowing the
        # class method — delete the instance attr instead so normal lookup
        # resumes. Module globals and foreign objects restore by assignment.
        bound_to_owner = getattr(original, "__self__", None) is self._owner
        if self._had_original and not bound_to_owner:
            setattr(self._owner, self._attr, original)
        else:
            try:
                delattr(self._owner, self._attr)
            except AttributeError:
                pass



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

        # Presynaptic calcium/RRP are recorded from the model's public
        # bio_telemetry() read API via a ROOT forward hook. The previous design
        # hooked SynapticPresyn modules, but the live attention path invokes
        # release_canonical as a plain bound method — Module.__call__ (and any
        # hook on it) never executes on ANY path (canonical, chunked, or fused
        # CPU scan), so those snapshots silently stayed None forever.
        presyn_layer_names = [
            name for name, module in self.model.named_modules()
            if isinstance(module, SynapticPresyn)
        ]
        if presyn_layer_names:
            self._attach_presyn_telemetry_hook(presyn_layer_names)

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

    def _attach_presyn_telemetry_hook(self, layer_names: list[str]) -> None:
        """Append per-layer calcium/RRP snapshots after every root forward.

        Reads the public ``bio_telemetry()`` schema ({layers[i].attention.{C,RRP}})
        — valid on EVERY execution path (canonical, chunked recurrence, and the
        fused CPU scan), unlike module hooks which never fire on SynapticPresyn.
        """

        def _root_hook(_module: nn.Module, _inp: Any, _out: Any) -> None:
            try:
                telem = self.model.bio_telemetry()
            except Exception:  # pragma: no cover - telemetry must never break probing
                return
            for entry in telem.get("layers", []):
                attn = entry.get("attention") or {}
                if "C" not in attn or "RRP" not in attn:
                    continue

                def _flat_mean(node) -> float | None:
                    vals: list[float] = []

                    def _walk(n):
                        if isinstance(n, list):
                            for item in n:
                                _walk(item)
                        elif isinstance(n, (int, float)):
                            vals.append(float(n))

                    _walk(node)
                    return sum(vals) / len(vals) if vals else None

                c_val = _flat_mean(attn.get("C"))
                rrp_val = _flat_mean(attn.get("RRP"))
                self.snapshots.append(
                    ProbeSnapshot(
                        layer_idx=int(entry.get("index", 0)),
                        module_name=f"h.{entry.get('index', 0)}.pre",
                        token_count=1,
                        mean_calcium=c_val,
                        mean_rrp=rrp_val,
                    )
                )

        self.handles.append(self.model.register_forward_hook(_root_hook))

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
            # Wrap release_canonical on the instance: forward hooks on
            # SynapticPresyn never fire (the attention path calls the bound
            # method directly), so the old hook-based clamp silently no-opped
            # and optogenetics experiments reported false null effects. Clamp
            # BEFORE each call so the dynamics actually see the pinned value.
            original_release = module.release_canonical

            def _clamped_release(state, *args, _orig=original_release, _key="RRP", **kwargs):
                if isinstance(state, dict) and _key in state:
                    state[_key].fill_(float(value))
                return _orig(state, *args, **kwargs)

            handles.append(_MethodPatch(module, "release_canonical", _clamped_release))

        elif target == "calcium" and isinstance(module, SynapticPresyn):
            original_release_ca = module.release_canonical

            def _clamped_release_ca(state, *args, _orig=original_release_ca, **kwargs):
                if isinstance(state, dict) and "C" in state:
                    state["C"].fill_(float(value))
                return _orig(state, *args, **kwargs)

            handles.append(_MethodPatch(module, "release_canonical", _clamped_release_ca))

    if target in ("rrp", "calcium") and hasattr(
        _synaptic_module, "_scripted_detached_presyn_scan_cpu"
    ):
        # The fused CPU scan bypasses release_canonical entirely; clamp its
        # returned state too so EVERY execution path stays pinned. Return tuple
        # order: (out, C, BUF, RRP, RES, PR, CL, E, AMP, DELAY, ema_e).
        state_index = 3 if target == "rrp" else 1

        def _scan_clamp(*args, _orig=_synaptic_module._scripted_detached_presyn_scan_cpu,
                        _idx=state_index, _v=float(value), **kwargs):
            result = _orig(*args, **kwargs)
            tensor = result[_idx]
            if torch.is_tensor(tensor):
                tensor.fill_(_v)
            return result

        handles.append(
            _MethodPatch(_synaptic_module, "_scripted_detached_presyn_scan_cpu", _scan_clamp)
        )

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
