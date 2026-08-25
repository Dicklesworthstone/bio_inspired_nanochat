"""In-Silico Optogenetic Stimulation & Synaptic Clamping Engine (bead `odq.3`).

Provides causal interventions on living transformer synaptic state:
Allows clamping, injecting, or pinning postsynaptic state variables (CaMKII latch,
PP1 phosphatase, BDNF metaplasticity, and fast weights) during generation.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional, Tuple

import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear


class ClampMode(str, Enum):
    PIN_VALUE = "pin_value"
    ADD_DELTA = "add_delta"
    SCALE_GAIN = "scale_gain"


@dataclass(frozen=True)
class SynapticClamp:
    """Specification of an optogenetic intervention at a target synaptic site."""

    layer_idx: Optional[int] = None  # None = apply to all layers
    site_type: str = "dense_fc"  # "dense_fc", "moe_expert", "all"
    variable_name: str = "camkii"  # "camkii", "pp1", "bdnf", "w_fast"
    mode: ClampMode = ClampMode.PIN_VALUE
    value: float = 1.0

    def __post_init__(self) -> None:
        if self.layer_idx is not None and (
            isinstance(self.layer_idx, bool)
            or not isinstance(self.layer_idx, int)
            or self.layer_idx < 0
        ):
            raise ValueError("layer_idx must be a non-negative integer or None")
        if self.site_type not in {"dense_fc", "moe_expert", "all"}:
            raise ValueError("site_type must be 'dense_fc', 'moe_expert', or 'all'")
        if self.variable_name not in {"camkii", "pp1", "bdnf", "w_fast"}:
            raise ValueError("unsupported synaptic variable_name")
        if not isinstance(self.mode, ClampMode):
            raise ValueError("mode must be a ClampMode")
        if not math.isfinite(self.value):
            raise ValueError("clamp value must be finite")


class OptogeneticStimulator:
    """Applies and manages causal synaptic interventions across GPTSynaptic models."""

    def __init__(self, model: GPTSynaptic):
        self.model = model

    def _targets_for_clamp(
        self, clamp: SynapticClamp
    ) -> List[Tuple[SynapticLinear, str, Tensor]]:
        targets: List[Tuple[SynapticLinear, str, Tensor]] = []
        for name, mod in self.model.named_modules():
            if not isinstance(mod, SynapticLinear):
                continue

            mod_layer = None
            parts = name.split(".")
            for idx, part in enumerate(parts):
                if part == "h" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    mod_layer = int(parts[idx + 1])
                    break

            if clamp.layer_idx is not None and mod_layer != clamp.layer_idx:
                continue
            if clamp.site_type == "dense_fc" and ("experts" in name or "moe" in name):
                continue
            if clamp.site_type == "moe_expert" and not (
                "experts" in name or "moe" in name
            ):
                continue

            if clamp.variable_name == "w_fast" and mod.w_fast is not None:
                targets.append((mod, "w_fast", mod.w_fast))
            elif mod.post is not None and hasattr(mod.post, clamp.variable_name):
                parameter = getattr(mod.post, clamp.variable_name)
                if isinstance(parameter, (torch.Tensor, torch.nn.Parameter)):
                    targets.append((mod, clamp.variable_name, parameter))
        return targets

    @staticmethod
    def _apply_to_tensor(target: Tensor, clamp: SynapticClamp) -> None:
        if clamp.mode == ClampMode.PIN_VALUE:
            target.data.fill_(clamp.value)
        elif clamp.mode == ClampMode.ADD_DELTA:
            target.data.add_(clamp.value)
        elif clamp.mode == ClampMode.SCALE_GAIN:
            target.data.mul_(clamp.value)

    def apply_clamps(self, clamps: List[SynapticClamp]) -> List[Tuple[SynapticLinear, str, Tensor]]:
        """Apply a set of synaptic clamps, returning saved state for rollbacks."""
        saved_states: List[Tuple[SynapticLinear, str, Tensor]] = []
        try:
            for clamp_idx, clamp in enumerate(clamps):
                targets = self._targets_for_clamp(clamp)
                if not targets:
                    raise ValueError(
                        f"clamp specification matched no synaptic sites: index {clamp_idx}"
                    )
                for mod, variable_name, target in targets:
                    saved_states.append((mod, variable_name, target.data.clone()))
                    self._apply_to_tensor(target, clamp)
        except Exception:
            self.restore_states(saved_states)
            raise
        return saved_states

    def restore_states(self, saved_states: List[Tuple[SynapticLinear, str, Tensor]]) -> None:
        """Restore model parameters/buffers to pre-clamp values."""
        # A variable can be targeted by multiple clamps. Each saved value is the state before
        # that clamp, so unwind in stack order to recover the true pre-intervention value.
        for mod, var_name, orig_val in reversed(saved_states):
            if var_name == "w_fast" and mod.w_fast is not None:
                mod.w_fast.data.copy_(orig_val)
            elif mod.post is not None and hasattr(mod.post, var_name):
                param = getattr(mod.post, var_name)
                param.data.copy_(orig_val)

    @contextmanager
    def stimulate(self, clamps: List[SynapticClamp]) -> Iterator[None]:
        """Context manager applying optogenetic clamps during execution and restoring state on exit."""
        target_modes: dict[int, set[ClampMode]] = {}
        for clamp in clamps:
            for _mod, _variable_name, target in self._targets_for_clamp(clamp):
                target_modes.setdefault(id(target), set()).add(clamp.mode)
        if any(
            ClampMode.PIN_VALUE in modes and len(modes) > 1
            for modes in target_modes.values()
        ):
            raise ValueError(
                "PIN_VALUE cannot overlap ADD_DELTA or SCALE_GAIN on the same target"
            )
        saved = self.apply_clamps(clamps)
        hook_handles = []

        def repin_pre(_module, _args, *, target: Tensor, value: float) -> None:
            target.data.fill_(value)

        def repin_post(_module, _args, _output, *, target: Tensor, value: float) -> None:
            target.data.fill_(value)

        try:
            for clamp in clamps:
                if clamp.mode != ClampMode.PIN_VALUE:
                    continue
                for mod, _variable_name, target in self._targets_for_clamp(clamp):
                    hook_handles.append(
                        mod.register_forward_pre_hook(
                            lambda module, args, target=target, value=clamp.value: repin_pre(
                                module, args, target=target, value=value
                            )
                        )
                    )
                    hook_handles.append(
                        mod.register_forward_hook(
                            lambda module, args, output, target=target, value=clamp.value: repin_post(
                                module, args, output, target=target, value=value
                            )
                        )
                    )
            yield
        finally:
            for handle in hook_handles:
                handle.remove()
            self.restore_states(saved)

    def log_stimulation(self, clamps: List[SynapticClamp], console: Optional[Console] = None) -> None:
        """Render Rich summary of active optogenetic clamps."""
        c = console or Console()
        c.rule("[bold cyan]Active Optogenetic Synaptic Clamps[/bold cyan]")
        table = Table(title="Intervention Targets")
        table.add_column("Layer Target", justify="center")
        table.add_column("Site Type", justify="center")
        table.add_column("Variable", justify="center", style="bold yellow")
        table.add_column("Clamp Mode", justify="center")
        table.add_column("Value / Gain", justify="right", style="bold green")

        for cl in clamps:
            table.add_row(
                f"L{cl.layer_idx}" if cl.layer_idx is not None else "ALL",
                cl.site_type,
                cl.variable_name,
                cl.mode.value,
                f"{cl.value:.4f}",
            )
        c.print(table)
