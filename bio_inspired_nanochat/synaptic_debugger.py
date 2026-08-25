"""Living-Model Synaptic Debugger & Cognitive IDE (bead `re4e.14`).

Enables interactive step-through debugging of transformer cognition:
1. `BioBreakpoint`: Conditional breakpoint triggered by synaptic bio-state thresholds (e.g. CaMKII > 0.8, high free energy, vesicle depletion).
2. `SynapticDebugger`: Interactive runtime controller allowing step-over generation, cognitive call-stack
   inspection (reasoning trace + bio-state), live synaptic state editing, and execution resumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from torch import Tensor

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.optogenetic_stimulation import (
    ClampMode,
    OptogeneticStimulator,
    SynapticClamp,
)
from bio_inspired_nanochat.reasoning_trace_decoder import (
    ReasoningTraceDecoder,
)
from bio_inspired_nanochat.working_memory_api import WorkingMemoryScratchpad


def _mean_bio_energy(telem: Dict[str, Any]) -> Optional[float]:
    """Mean bio-energy across layers from REAL telemetry channels.

    ``collect_bio_telemetry`` exposes per-layer presynaptic state under
    ``layers[i]["attention"]`` (keys C/BUF/RRP/.../E as nested per-head lists) and
    metabolism under ``layers[i]["mlp"]["energy"]`` for MoE blocks. The previous
    implementation read nonexistent top-level keys and silently reported a constant
    1.0, so the energy-trajectory feature never reflected model state. Returns
    ``None`` when no energy channel is populated rather than fabricating one.
    """
    layer_means: List[float] = []
    for lyr in telem.get("layers", []):
        attn = lyr.get("attention") or {}
        e_channel = attn.get("E")
        if isinstance(e_channel, list):
            flat: List[float] = []

            def _collect(node: Any) -> None:
                if isinstance(node, list):
                    for item in node:
                        _collect(item)
                elif isinstance(node, (int, float)):
                    flat.append(float(node))

            _collect(e_channel)
            if flat:
                layer_means.append(sum(flat) / len(flat))
                continue
        mlp = lyr.get("mlp") or {}
        mlp_energy = mlp.get("energy")
        if isinstance(mlp_energy, list) and mlp_energy:
            layer_means.append(sum(float(v) for v in mlp_energy) / len(mlp_energy))
    if not layer_means:
        return None
    return sum(layer_means) / len(layer_means)


@dataclass
class BioBreakpoint:
    """A conditional breakpoint that halts model cognition when synaptic bio-state satisfies a rule."""

    name: str
    condition_fn: Callable[[int, int, Dict[str, Any]], bool]
    hit_count: int = 0
    enabled: bool = True


@dataclass
class CognitiveStackFrame:
    """Snapshot of active reasoning call-stack and synaptic state at a paused debugger step."""

    step: int
    token_id: int
    token_str: str
    reasoning_trace: str
    telemetry_snapshot: Dict[str, Any]
    hit_breakpoint: Optional[str] = None


class SynapticDebugger:
    """Interactive cognitive debugger attached to a living GPTSynaptic transformer."""

    def __init__(self, model: GPTSynaptic, tokenizer: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.scratchpad = WorkingMemoryScratchpad(model)
        self.stimulator = OptogeneticStimulator(model)
        self.trace_decoder = ReasoningTraceDecoder()
        self.breakpoints: List[BioBreakpoint] = []
        self.current_tokens: Optional[Tensor] = None
        self.call_stack: List[CognitiveStackFrame] = []
        self._energy_history: List[float] = []
        self.is_paused: bool = False
        # Live KV cache for incremental single-token stepping (engine.KVCache).
        self._kv_cache: Optional[Any] = None

    def add_breakpoint(self, bp: BioBreakpoint) -> None:
        """Register a biological state condition breakpoint."""
        self.breakpoints.append(bp)

    def set_camkii_threshold_breakpoint(self, threshold: float = 0.5, name: str = "CaMKII Latch Trigger") -> None:
        """Helper breakpoint firing when CaMKII phosphorylation exceeds threshold."""
        def cond(step: int, tok: int, telem: Dict[str, Any]) -> bool:
            for lyr in telem.get("layers", []):
                mlp = lyr.get("mlp", {})
                if "fc" in mlp and mlp["fc"].get("camkii", 0.0) >= threshold:
                    return True
            return False

        self.add_breakpoint(BioBreakpoint(name=name, condition_fn=cond))

    def _step_input(self) -> Tensor:
        """Input for one incremental decode step: only the NEW token when a KV
        cache is live (so the prefix is not re-forwarded — re-running it used to
        re-apply online plasticity to every prefix token each step, an O(k^2)
        contamination of both runtime and breakpoint semantics)."""
        assert self.current_tokens is not None
        if self._kv_cache is not None and self.current_tokens.shape[1] > 1:
            return self.current_tokens[:, -1:]
        return self.current_tokens

    def step_over(self, temperature: float = 1.0) -> Optional[CognitiveStackFrame]:
        """Execute a single token generation step, capturing telemetry and checking breakpoints."""
        if self.current_tokens is None:
            raise RuntimeError("No prompt or generation initialized. Call run_until_breakpoint first.")

        step = len(self.call_stack)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(self._step_input(), kv_cache=self._kv_cache)
            next_logits = logits[:, -1, :] / max(1e-3, temperature)
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            self.current_tokens = torch.cat([self.current_tokens, next_tok], dim=1)

            tok_val = int(next_tok.item())
            tok_str = self.tokenizer.decode([tok_val]) if self.tokenizer else str(tok_val)

            # Capture telemetry & dynamic energy trajectory from real bio channels.
            telem = self.model.bio_telemetry()
            energy_val = _mean_bio_energy(telem)
            if energy_val is not None:
                self._energy_history.append(energy_val)

            traj_slice = (
                self._energy_history[-3:]
                if len(self._energy_history) >= 2
                else ([energy_val, max(0.01, energy_val - 0.1)] if energy_val is not None else [])
            )
            trace_obj = self.trace_decoder.decode_energy_trajectory(traj_slice)
            reasoning_text = trace_obj.summary_narrative

            # Check active breakpoints
            hit_bp_name: Optional[str] = None
            for bp in self.breakpoints:
                if bp.enabled and bp.condition_fn(step, tok_val, telem):
                    bp.hit_count += 1
                    hit_bp_name = bp.name
                    break

            frame = CognitiveStackFrame(
                step=step,
                token_id=tok_val,
                token_str=tok_str,
                reasoning_trace=reasoning_text,
                telemetry_snapshot=telem,
                hit_breakpoint=hit_bp_name,
            )
            self.call_stack.append(frame)
            self.is_paused = (hit_bp_name is not None)
            return frame

    def run_until_breakpoint(
        self,
        prompt_tokens: Tensor,
        max_tokens: int = 16,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Optional[CognitiveStackFrame]]:
        """Autoregressively generate tokens until completion or until a breakpoint fires.

        Starts a FRESH debug session: transient per-sequence state (eligibility
        traces, CaMKII/PP1/BDNF latches, presyn state) is reset so one session's
        bio-state cannot leak into the next; trained weights are untouched. The
        prompt is prefilled into a KV cache and every subsequent step decodes
        incrementally from it.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        tokens = prompt_tokens.clone().to(device)
        # Transient-state reset only (vg9.4 contract); fast weights are trained
        # parameters and must survive across sessions.
        self.model.reset_sequence_state(reset_fast_weights=False)
        cfg = self.model.config
        cache = KVCache(
            batch_size=tokens.shape[0],
            num_heads=cfg.n_head,
            seq_len=cfg.sequence_len,
            head_dim=cfg.n_embd // cfg.n_head,
            num_layers=cfg.n_layer,
        )
        if tokens.shape[1] > 0:
            with torch.no_grad():
                self.model(tokens, kv_cache=cache)
            self._kv_cache: Optional[Any] = cache
        else:
            self._kv_cache = None
        self.current_tokens = tokens
        self.call_stack.clear()
        self._energy_history.clear()
        self.is_paused = False

        for _ in range(max_tokens):
            frame = self.step_over(temperature=temperature)
            if frame and frame.hit_breakpoint is not None:
                if self.current_tokens is None:
                    raise RuntimeError("Debugger state lost current tokens unexpectedly.")
                return self.current_tokens, frame

        if self.current_tokens is None:
            raise RuntimeError("Debugger state lost current tokens unexpectedly.")
        return self.current_tokens, None

    def edit_synaptic_state(
        self,
        var_name: str = "w_fast",
        value: float = 1.0,
        layer_idx: Optional[int] = None,
    ) -> None:
        """Live hot-patch of internal synaptic parameters while paused at a breakpoint."""
        clamp = SynapticClamp(
            layer_idx=layer_idx,
            variable_name=var_name,
            mode=ClampMode.PIN_VALUE,
            value=value,
        )
        self.stimulator.apply_clamps([clamp])

    def resume_generation(
        self,
        max_additional_tokens: int = 8,
        temperature: float = 1.0,
    ) -> Tensor:
        """Resume generation from the paused stack state."""
        if self.current_tokens is None:
            raise RuntimeError("No active generation session to resume.")

        self.is_paused = False

        for _ in range(max_additional_tokens):
            frame = self.step_over(temperature=temperature)
            if frame and frame.hit_breakpoint is not None:
                break

        if self.current_tokens is None:
            raise RuntimeError("Debugger state lost current tokens unexpectedly.")
        return self.current_tokens

    def log_debugger_frame(self, frame: CognitiveStackFrame, console: Optional[Console] = None) -> None:
        """Render Rich panel of paused debugger frame with reasoning call-stack."""
        c = console or Console()
        status_header = (
            f"[bold red]🛑 BREAKPOINT HIT: '{frame.hit_breakpoint}'[/bold red]"
            if frame.hit_breakpoint
            else "[bold green]Cognitive Step Completed[/bold green]"
        )
        c.rule(f"[bold cyan]Living Transformer Debugger — Step {frame.step}[/bold cyan]")
        c.print(status_header)
        c.print(f"Token: '[bold yellow]{frame.token_str}[/bold yellow]' (ID: {frame.token_id})")
        c.print(Panel(frame.reasoning_trace, title="Reasoning Trace Call-Stack", border_style="cyan"))
