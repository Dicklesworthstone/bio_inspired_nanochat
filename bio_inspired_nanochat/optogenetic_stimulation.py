"""In-Silico Optogenetic Stimulation & Synaptic Clamping Engine (bead `odq.3`).

Provides causal interventions on living transformer synaptic state:
Allows clamping, injecting, or pinning bio-state variables (CaMKII latch, PP1 phosphatase,
BDNF metaplasticity, vesicle depletion, and fast weights) to test causal hypotheses during generation.
"""

from __future__ import annotations

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


@dataclass
class SynapticClamp:
    """Specification of an optogenetic intervention at a target synaptic site."""

    layer_idx: Optional[int] = None  # None = apply to all layers
    site_type: str = "dense_fc"  # "dense_fc", "moe_expert", "all"
    variable_name: str = "camkii"  # "camkii", "pp1", "bdnf", "w_fast"
    mode: ClampMode = ClampMode.PIN_VALUE
    value: float = 1.0


class OptogeneticStimulator:
    """Applies and manages causal synaptic interventions across GPTSynaptic models."""

    def __init__(self, model: GPTSynaptic):
        self.model = model

    def apply_clamps(self, clamps: List[SynapticClamp]) -> List[Tuple[SynapticLinear, str, Tensor]]:
        """Apply a set of synaptic clamps, returning saved state for rollbacks."""
        saved_states: List[Tuple[SynapticLinear, str, Tensor]] = []

        for mod in self.model.modules():
            if isinstance(mod, SynapticLinear):
                for clamp in clamps:
                    if clamp.variable_name == "w_fast" and mod.w_fast is not None:
                        saved_states.append((mod, "w_fast", mod.w_fast.data.clone()))
                        if clamp.mode == ClampMode.PIN_VALUE:
                            mod.w_fast.data.fill_(clamp.value)
                        elif clamp.mode == ClampMode.ADD_DELTA:
                            mod.w_fast.data.add_(clamp.value)
                        elif clamp.mode == ClampMode.SCALE_GAIN:
                            mod.w_fast.data.mul_(clamp.value)

                    elif mod.post is not None:
                        post = mod.post
                        if hasattr(post, clamp.variable_name):
                            param = getattr(post, clamp.variable_name)
                            if isinstance(param, (torch.Tensor, torch.nn.Parameter)):
                                saved_states.append((mod, clamp.variable_name, param.data.clone()))
                                if clamp.mode == ClampMode.PIN_VALUE:
                                    param.data.fill_(clamp.value)
                                elif clamp.mode == ClampMode.ADD_DELTA:
                                    param.data.add_(clamp.value)
                                elif clamp.mode == ClampMode.SCALE_GAIN:
                                    param.data.mul_(clamp.value)

        return saved_states

    def restore_states(self, saved_states: List[Tuple[SynapticLinear, str, Tensor]]) -> None:
        """Restore model parameters/buffers to pre-clamp values."""
        for mod, var_name, orig_val in saved_states:
            if var_name == "w_fast" and mod.w_fast is not None:
                mod.w_fast.data.copy_(orig_val)
            elif mod.post is not None and hasattr(mod.post, var_name):
                param = getattr(mod.post, var_name)
                param.data.copy_(orig_val)

    @contextmanager
    def stimulate(self, clamps: List[SynapticClamp]) -> Iterator[None]:
        """Context manager applying optogenetic clamps during execution and restoring state on exit."""
        saved = self.apply_clamps(clamps)
        try:
            yield
        finally:
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
