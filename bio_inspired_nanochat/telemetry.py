"""Model-provided bio-telemetry API (hm4.6).

A single, schema'd entry point (:func:`collect_bio_telemetry`, exposed on
``GPTSynaptic.bio_telemetry``) that returns every synaptic bio-state signal —
presynaptic calcium/vesicle/energy state, postsynaptic CaMKII/PP1/BDNF
plasticity, MoE energy/fatigue metabolism and routing snapshots — as plain,
JSON-safe dicts instead of the brittle ``hasattr``-chain introspection the
engine and neuroviz used before.

Schema ``bio-telemetry/1``
-------------------------
::

    {
      "schema": "bio-telemetry/1",
      "num_layers": int,
      "layers": [
        {
          "index": int,
          "attention": {"C": [[...]], "RRP": [[...]], ...} | None,   # per-head last-step values
          "mlp":
            {"type": "moe", "num_experts": E, "energy": [E], "fatigue": [E],
             "router_logit_bias": [E], "experts": [{"fc1": site, "fc2": site}],
             "routing": {"gates": ..., "indices": ...}}             # when include_routing
            | {"type": "dense", "fc": site, "proj": site}
            | {"type": "vanilla"},
        }
      ],
    }

where a *site* describes one :class:`SynapticLinear`::

    {"camkii": f, "pp1": f, "bdnf": f, "bdnf_max": f, "hebb_accum": f,
     "last_delta_mag": f, "slow_norm": f, "fast_norm": f | absent,
     "u_buf_norm": f | absent, "v_buf_norm": f | absent}

All values are floats/lists (``.item()`` / ``tolist()``), safe to log as JSON.
The synaptic classes in :mod:`bio_inspired_nanochat.synaptic` are the stable
contract; collection uses ``isinstance`` checks against them, never attribute
name probes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union, cast

import torch
from torch import nn

from .synaptic import (
    PostsynapticHebb,
    SynapticLinear,
    SynapticMLP,
    SynapticMoE,
)

BIO_TELEMETRY_SCHEMA = "bio-telemetry/1"

#: Canonical presynaptic state keys (see ``build_presyn_state``). The DELAY
#: endocytosis queue holds tensors-in-lists, not scalar telemetry, so it is
#: intentionally excluded.
PRESYN_STATE_KEYS = ("C", "BUF", "RRP", "RES", "PR", "CL", "E", "AMP")


def _f(x: Any) -> float:
    """Tensor-or-number -> Python float."""
    return float(x.item()) if torch.is_tensor(x) else float(x)


def _norm(x: Any) -> float:
    return _f(x.float().norm()) if torch.is_tensor(x) else 0.0


def linear_site_telemetry(lin: Any) -> Dict[str, float]:
    """Postsynaptic plasticity snapshot of one ``SynapticLinear`` site."""
    post = getattr(lin, "post", None)
    if not isinstance(post, PostsynapticHebb):
        return {}
    out: Dict[str, float] = {
        "camkii": _f(post.camkii.mean()),
        "pp1": _f(post.pp1.mean()),
        "bdnf": _f(post.bdnf.mean()),
        "bdnf_max": _f(post.bdnf.max()),
        "hebb_accum": _f(post.bdnf_hebb_accum.mean()),
        "last_delta_mag": _f(post._last_hebb_delta_mag),
        "slow_norm": _norm(post.slow),
    }
    if torch.is_tensor(post.fast):
        out["fast_norm"] = _norm(post.fast)
    if torch.is_tensor(getattr(lin, "u_buf", None)):
        out["u_buf_norm"] = _norm(lin.u_buf)
    if torch.is_tensor(getattr(lin, "v_buf", None)):
        out["v_buf_norm"] = _norm(lin.v_buf)
    return out


def expert_telemetry(expert: Any) -> Dict[str, Dict[str, float]]:
    """Both projection sites of one ``SynapticExpert``."""
    return {
        "fc1": linear_site_telemetry(expert.fc1),
        "fc2": linear_site_telemetry(expert.fc2),
    }


def moe_telemetry(moe: Any, *, include_routing: bool = False) -> Dict[str, Any]:
    """Metabolism + lifecycle + (optionally) routing snapshot of a MoE layer."""
    out: Dict[str, Any] = {
        "type": "moe",
        "num_experts": int(moe.num_experts),
        "energy": [_f(e) for e in moe.energy],
        "fatigue": [_f(f) for f in moe.fatigue],
        "router_logit_bias": [_f(b) for b in moe.router_logit_bias],
        "experts": [expert_telemetry(e) for e in moe.experts],
    }
    ctx = getattr(moe, "last_ctx", None)
    if include_routing and ctx:
        gates, indices = ctx.get("gates"), ctx.get("indices")
        if torch.is_tensor(gates) and torch.is_tensor(indices):
            out["routing"] = {
                "gates": gates.detach().float().cpu().tolist(),
                "indices": indices.detach().cpu().tolist(),
            }
    return out


def dense_mlp_telemetry(mlp: Any) -> Dict[str, Any]:
    """Snapshot of a dense synaptic MLP (fc + proj sites)."""
    fc = getattr(mlp, "fc", None)
    proj = getattr(mlp, "proj", None)
    return {
        "type": "dense",
        "fc": linear_site_telemetry(fc) if isinstance(fc, SynapticLinear) else {},
        "proj": linear_site_telemetry(proj) if isinstance(proj, SynapticLinear) else {},
    }


def unwrap_mlp_module(mod: Optional[nn.Module]) -> Optional[nn.Module]:
    """Resolve a block's ``mlp`` attribute to its synaptic implementation.

    Handles the ``Block.mlp`` -> wrapper ``MLP`` -> ``SynapticMLP`` indirection;
    returns ``None`` for non-synaptic (vanilla) MLPs.
    """
    for _ in range(5):  # defensive: never recurse on pathological nesting
        if mod is None or isinstance(mod, (SynapticMoE, SynapticMLP)):
            return mod
        inner = getattr(mod, "mlp", None)
        mod = inner if isinstance(inner, nn.Module) and inner is not mod else None
    return None


def presyn_telemetry(
    state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, List[List[float]]]]:
    """Per-head presynaptic state at the most recent sequence step.

    ``state`` is one layer's presyn dict (tensors shaped ``(B, H, T)``).
    Returns ``{key: [[head...] batch]}`` or ``None`` when absent.
    """
    if not state:
        return None
    out: Dict[str, List[List[float]]] = {}
    for key in PRESYN_STATE_KEYS:
        t = state.get(key)
        if torch.is_tensor(t) and t.ndim == 3 and t.shape[-1] > 0:
            out[key] = t[..., -1].detach().float().cpu().tolist()
    return out or None


def layer_attention_telemetry(
    presyn_states: Optional[Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]]],
    layer_index: int,
) -> Optional[Dict[str, List[List[float]]]]:
    """Aligned per-layer presyn snapshot from a model-level state container.

    Accepts either a list of per-layer states (the KV-cache layout) or a single
    dict broadcast to every layer (legacy decode path).
    """
    if presyn_states is None:
        return None
    if isinstance(presyn_states, (list, tuple)):
        if layer_index >= len(presyn_states):
            return None
        return presyn_telemetry(presyn_states[layer_index])
    if isinstance(presyn_states, dict):
        return presyn_telemetry(cast(Optional[Dict[str, Any]], presyn_states))
    return None


def collect_bio_telemetry(
    model: nn.Module,
    *,
    presyn_state: Optional[Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]]] = None,
    include_routing: bool = False,
) -> Dict[str, Any]:
    """Collect the full bio-state snapshot of a synaptic model.

    Args:
        model: typically :class:`~bio_inspired_nanochat.gpt_synaptic.GPTSynaptic`
            (uses its ``h`` block stack); any object exposing an iterable ``h``
            of blocks with an ``mlp`` attribute works.
        presyn_state: optional KV-cache presyn container — per-layer list of
            state dicts, or a single state dict broadcast to all layers.
        include_routing: include the latest router ``gates``/``indices``
            snapshot from each MoE layer's ``last_ctx``.
    """
    blocks = getattr(model, "h", None)
    layers: List[Dict[str, Any]] = []
    if blocks is not None:
        for i, block in enumerate(blocks):
            entry: Dict[str, Any] = {"index": i}
            entry["attention"] = layer_attention_telemetry(presyn_state, i)
            raw_mlp = getattr(block, "mlp", None)
            mlp = unwrap_mlp_module(raw_mlp if isinstance(raw_mlp, nn.Module) else None)
            if isinstance(mlp, SynapticMoE):
                entry["mlp"] = moe_telemetry(mlp, include_routing=include_routing)
            elif isinstance(mlp, SynapticMLP):
                entry["mlp"] = dense_mlp_telemetry(mlp)
            else:
                entry["mlp"] = {"type": "vanilla"}
            layers.append(entry)
    return {
        "schema": BIO_TELEMETRY_SCHEMA,
        "num_layers": len(layers),
        "layers": layers,
    }


def layer_camkii_mean(layer: Dict[str, Any]) -> float:
    """Mean CaMKII across all postsynaptic sites of one telemetry layer.

    Convenience aggregate matching the legacy engine ``memory`` metric
    semantics (mean over sites; 0.0 when the layer exposes no sites).
    """
    vals: List[float] = []
    mlp = layer.get("mlp") or {}
    if mlp.get("type") == "moe":
        for expert in mlp.get("experts", []):
            for site in ("fc1", "fc2"):
                s = expert.get(site) or {}
                if "camkii" in s:
                    vals.append(s["camkii"])
    elif mlp.get("type") == "dense":
        for site in ("fc", "proj"):
            s = mlp.get(site) or {}
            if "camkii" in s:
                vals.append(s["camkii"])
    return sum(vals) / len(vals) if vals else 0.0
