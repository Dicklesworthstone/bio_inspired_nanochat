"""Schema'd in-silico patch-clamp recordings for synaptic generation.

The electrode consumes the model's public :meth:`GPTSynaptic.bio_telemetry`
contract instead of reaching into biological modules directly.  A fresh KV cache
keeps presynaptic state observable at every decode step while leaving the model's
training/evaluation mode as it was before recording.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear
from bio_inspired_nanochat.telemetry import BIO_TELEMETRY_SCHEMA


PATCH_CLAMP_SCHEMA = "patch-clamp/1"

_PRESYN_CHANNEL_NAMES = {
    "C": "calcium",
    "BUF": "buffer",
    "RRP": "rrp",
    "RES": "reserve_pool",
    "PR": "priming",
    "CL": "clamp",
    "E": "energy",
    "AMP": "amplitude",
}
_POSTSYN_CHANNELS = ("camkii", "pp1", "bdnf")


@dataclass
class ChannelRecording:
    """Time-series recording of a specific biological state channel at a synaptic site."""

    layer_idx: int
    site_type: str
    channel_name: str
    head_idx: int | None = None
    expert_idx: int | None = None
    site_name: str | None = None
    values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-safe channel payload."""
        return {
            "layer_idx": self.layer_idx,
            "site_type": self.site_type,
            "channel_name": self.channel_name,
            "head_idx": self.head_idx,
            "expert_idx": self.expert_idx,
            "site_name": self.site_name,
            "values": self.values,
        }


@dataclass
class ProbeTrace:
    """Complete multi-channel electrophysiological recording of a generation trajectory."""

    prompt_token_count: int
    token_ids: list[int]
    token_strings: list[str]
    recording_kind: str = "generation"
    sample_phase: str = "after_token_forward"
    sequence_state_policy: str = "attached"
    time_steps: list[int] = field(default_factory=list)
    channels: dict[str, ChannelRecording] = field(default_factory=dict)
    telemetry_history: list[dict[str, Any]] = field(default_factory=list)
    schema: str = field(default=PATCH_CLAMP_SCHEMA, init=False)

    @property
    def generated_token_ids(self) -> list[int]:
        """Tokens sampled while the electrode was attached."""
        return self.token_ids[self.prompt_token_count :]

    @property
    def recorded_token_ids(self) -> list[int]:
        """Token ids whose completed forwards align with telemetry rows."""
        if self.recording_kind == "generation":
            return self.generated_token_ids
        return self.token_ids

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe recording, including source telemetry."""
        return {
            "schema": self.schema,
            "source_schema": BIO_TELEMETRY_SCHEMA,
            "recording_kind": self.recording_kind,
            "sample_phase": self.sample_phase,
            "sequence_state_policy": self.sequence_state_policy,
            "prompt_token_count": self.prompt_token_count,
            "token_ids": self.token_ids,
            "generated_token_ids": self.generated_token_ids,
            "recorded_token_ids": self.recorded_token_ids,
            "token_strings": self.token_strings,
            "time_steps": self.time_steps,
            "channels": {key: recording.to_dict() for key, recording in self.channels.items()},
            "telemetry_history": self.telemetry_history,
        }


class PatchClampElectrode:
    """In-silico patch-clamp recording fixture attached to a GPTSynaptic model."""

    def __init__(self, model: GPTSynaptic):
        self.model = model

    def record_generation(
        self,
        prompt_tokens: Tensor,
        max_new_tokens: int = 8,
        temperature: float = 1.0,
        tokenizer: Any | None = None,
        *,
        update_memory: bool = False,
    ) -> ProbeTrace:
        """Generate one sequence while recording every public biological channel.

        The trace is single-sequence by design: channel keys identify layers,
        heads, experts, and sites without an ambiguous batch dimension. Each
        telemetry row is captured *after* the corresponding sampled token has
        completed its forward, including the final sampled token, so channel
        values align with ``generated_token_ids`` without a one-token phase lag.

        A fresh cache resets prompt-local presynaptic calcium/vesicle state. The
        electrode otherwise attaches to the model's current sequence-local and
        persistent postsynaptic state; call ``model.reset_sequence_state()``
        explicitly before attaching when a clean sequence boundary is desired.

        Args:
            prompt_tokens: non-empty integer tensor shaped ``(1, prompt_length)``.
            max_new_tokens: number of decode decisions to record.
            temperature: sampling temperature; ``0`` selects greedy decoding.
            tokenizer: optional object exposing ``decode([token_id])``.
            update_memory: opt in to normal online postsynaptic writes.  The
                default is observational and leaves those writes disabled. An
                observational probe rejects models with deferred training writes
                rather than silently flushing them.
        """
        self._validate_request(prompt_tokens, max_new_tokens, temperature)

        prompt = prompt_tokens.detach().clone()
        prompt_ids = [int(token_id) for token_id in prompt[0].tolist()]
        trace = ProbeTrace(
            prompt_token_count=len(prompt_ids),
            token_ids=prompt_ids,
            token_strings=[self._decode(tokenizer, token_id) for token_id in prompt_ids],
        )
        if max_new_tokens == 0:
            return trace

        self._assert_observationally_safe(update_memory)
        cache = self._new_cache()

        was_training = self.model.training
        self.model.eval()  # ubs:ignore — PyTorch mode switch, not built-in eval()
        try:
            with torch.no_grad():
                logits, _ = self.model(
                    prompt,
                    kv_cache=cache,
                    train_mode=update_memory,
                )
                for step in range(max_new_tokens):
                    next_token = self._sample(logits[:, -1, :], temperature)
                    token_id = int(next_token.item())
                    trace.token_ids.append(token_id)
                    trace.token_strings.append(self._decode(tokenizer, token_id))

                    # Forward the sampled token before recording. This makes each
                    # row a post-token state observation and records the final token.
                    logits, _ = self.model(
                        next_token,
                        kv_cache=cache,
                        train_mode=update_memory,
                    )
                    self._capture_step(trace, cache, step)
        finally:
            self.model.train(was_training)

        return trace

    def record_forward(
        self,
        input_tokens: Tensor,
        tokenizer: Any | None = None,
        *,
        update_memory: bool = False,
    ) -> ProbeTrace:
        """Record every token of a causal incremental forward.

        This is the general forward counterpart to :meth:`record_generation`.
        It advances a fresh KV cache one input token at a time so presynaptic and
        postsynaptic snapshots are available at every token boundary.
        """
        self._validate_tokens(input_tokens, name="input_tokens")
        if input_tokens.shape[1] > self.model.config.sequence_len:
            raise ValueError(
                "input tokens exceed the model context: "
                f"{input_tokens.shape[1]} > {self.model.config.sequence_len}"
            )
        self._assert_observationally_safe(update_memory)

        tokens = input_tokens.detach().clone()
        token_ids = [int(token_id) for token_id in tokens[0].tolist()]
        trace = ProbeTrace(
            prompt_token_count=len(token_ids),
            token_ids=token_ids,
            token_strings=[self._decode(tokenizer, token_id) for token_id in token_ids],
            recording_kind="forward",
        )
        cache = self._new_cache()
        was_training = self.model.training
        self.model.eval()  # ubs:ignore — PyTorch mode switch, not built-in eval()
        try:
            with torch.no_grad():
                for step in range(tokens.shape[1]):
                    self.model(
                        tokens[:, step : step + 1],
                        kv_cache=cache,
                        train_mode=update_memory,
                    )
                    self._capture_step(trace, cache, step)
        finally:
            self.model.train(was_training)
        return trace

    def _new_cache(self) -> KVCache:
        config = self.model.config
        return KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=config.sequence_len,
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )

    def _capture_step(self, trace: ProbeTrace, cache: KVCache, step: int) -> None:
        telemetry = self.model.bio_telemetry(
            presyn_state=cache.presyn_state,
            include_routing=True,
        )
        if telemetry.get("schema") != BIO_TELEMETRY_SCHEMA:
            raise RuntimeError(
                "bio_telemetry() returned an unsupported schema: "
                f"{telemetry.get('schema')!r}"
            )
        trace.time_steps.append(step)
        trace.telemetry_history.append(telemetry)
        self._record_telemetry_channels(trace, telemetry)

    def _assert_observationally_safe(self, update_memory: bool) -> None:
        if update_memory:
            return
        pending = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, SynapticLinear) and module._plasticity_pending
        ]
        if pending:
            preview = ", ".join(pending[:4])
            suffix = "" if len(pending) <= 4 else f" (+{len(pending) - 4} more)"
            raise RuntimeError(
                "observational patch-clamp recording cannot start while deferred "
                f"plasticity writes are pending at {preview}{suffix}; complete the "
                "training-step flush before probing, or opt in with update_memory=True"
            )

    def _validate_request(
        self,
        prompt_tokens: Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        self._validate_tokens(prompt_tokens, name="prompt_tokens")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0.0
        ):
            raise ValueError("temperature must be a finite, non-negative number")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer")
        total_tokens = int(prompt_tokens.shape[1]) + max_new_tokens
        if total_tokens > self.model.config.sequence_len:
            raise ValueError(
                "prompt plus generated tokens exceed the model context: "
                f"{total_tokens} > {self.model.config.sequence_len}"
            )

    @staticmethod
    def _validate_tokens(tokens: Tensor, *, name: str) -> None:
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(
                f"{name} must have shape (1, sequence_length); got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one token")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must use torch.int32 or torch.int64 token ids")

    @staticmethod
    def _decode(tokenizer: Any | None, token_id: int) -> str:
        return str(token_id) if tokenizer is None else str(tokenizer.decode([token_id]))

    @staticmethod
    def _sample(logits: Tensor, temperature: float) -> Tensor:
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    @staticmethod
    def _append_channel(
        trace: ProbeTrace,
        key: str,
        value: float,
        *,
        layer_idx: int,
        site_type: str,
        channel_name: str,
        head_idx: int | None = None,
        expert_idx: int | None = None,
        site_name: str | None = None,
    ) -> None:
        recording = trace.channels.get(key)
        if recording is None:
            recording = ChannelRecording(
                layer_idx=layer_idx,
                site_type=site_type,
                channel_name=channel_name,
                head_idx=head_idx,
                expert_idx=expert_idx,
                site_name=site_name,
            )
            trace.channels[key] = recording
        recording.values.append(float(value))

    def _record_telemetry_channels(
        self,
        trace: ProbeTrace,
        telemetry: dict[str, Any],
    ) -> None:
        """Flatten one ``bio-telemetry/1`` snapshot into addressable time series."""
        for layer in telemetry.get("layers", []):
            layer_idx = int(layer["index"])
            self._record_attention_channels(trace, layer_idx, layer.get("attention"))
            self._record_mlp_channels(trace, layer_idx, layer.get("mlp") or {})

    def _record_attention_channels(
        self,
        trace: ProbeTrace,
        layer_idx: int,
        attention: dict[str, Any] | None,
    ) -> None:
        if not attention:
            return
        for telemetry_name, channel_name in _PRESYN_CHANNEL_NAMES.items():
            batches = attention.get(telemetry_name)
            if not isinstance(batches, list) or not batches:
                continue
            head_values = batches[0]
            if not isinstance(head_values, list):
                continue
            for head_idx, value in enumerate(head_values):
                key = f"L{layer_idx}.attention.H{head_idx}.{channel_name}"
                self._append_channel(
                    trace,
                    key,
                    float(value),
                    layer_idx=layer_idx,
                    site_type="attention_presyn",
                    channel_name=channel_name,
                    head_idx=head_idx,
                )

    def _record_mlp_channels(
        self,
        trace: ProbeTrace,
        layer_idx: int,
        mlp: dict[str, Any],
    ) -> None:
        mlp_type = mlp.get("type")
        if mlp_type == "dense":
            for site_name in ("fc", "proj"):
                self._record_postsyn_site(
                    trace,
                    layer_idx,
                    mlp.get(site_name) or {},
                    key_prefix=f"L{layer_idx}.dense.{site_name}",
                    site_type="dense_synapse",
                    site_name=site_name,
                )
            return
        if mlp_type != "moe":
            return

        energy = mlp.get("energy") or []
        fatigue = mlp.get("fatigue") or []
        for expert_idx, value in enumerate(energy):
            self._append_channel(
                trace,
                f"L{layer_idx}.moe.E{expert_idx}.energy",
                float(value),
                layer_idx=layer_idx,
                site_type="moe_expert",
                channel_name="energy",
                expert_idx=expert_idx,
            )
        for expert_idx, value in enumerate(fatigue):
            self._append_channel(
                trace,
                f"L{layer_idx}.moe.E{expert_idx}.fatigue",
                float(value),
                layer_idx=layer_idx,
                site_type="moe_expert",
                channel_name="fatigue",
                expert_idx=expert_idx,
            )
        for expert_idx, expert in enumerate(mlp.get("experts") or []):
            for site_name in ("fc1", "fc2"):
                self._record_postsyn_site(
                    trace,
                    layer_idx,
                    expert.get(site_name) or {},
                    key_prefix=f"L{layer_idx}.moe.E{expert_idx}.{site_name}",
                    site_type="moe_synapse",
                    expert_idx=expert_idx,
                    site_name=site_name,
                )

    def _record_postsyn_site(
        self,
        trace: ProbeTrace,
        layer_idx: int,
        site: dict[str, Any],
        *,
        key_prefix: str,
        site_type: str,
        expert_idx: int | None = None,
        site_name: str,
    ) -> None:
        for channel_name in _POSTSYN_CHANNELS:
            if channel_name not in site:
                continue
            self._append_channel(
                trace,
                f"{key_prefix}.{channel_name}",
                float(site[channel_name]),
                layer_idx=layer_idx,
                site_type=site_type,
                channel_name=channel_name,
                expert_idx=expert_idx,
                site_name=site_name,
            )

    def plot_trace(
        self,
        trace: ProbeTrace,
        channel_keys: list[str] | None = None,
        *,
        max_channels: int = 12,
    ) -> Any:
        """Plot selected channel time series and return a Matplotlib figure.

        ``channel_keys=None`` selects up to ``max_channels`` channels in stable
        key order, which keeps notebook output readable for MoE models exposing
        hundreds of sites.
        """
        from matplotlib import pyplot as plt

        selected = sorted(trace.channels) if channel_keys is None else list(channel_keys)
        selected = selected[:max_channels]
        missing = [key for key in selected if key not in trace.channels]
        if missing:
            raise KeyError(f"unknown patch-clamp channels: {missing}")
        if not selected:
            raise ValueError("trace has no channels to plot")

        figure, axis = plt.subplots(figsize=(12, 6))
        for key in selected:
            values = trace.channels[key].values
            axis.plot(trace.time_steps[: len(values)], values, marker="o", label=key)
        axis.set_title("In-Silico Patch-Clamp Recording")
        axis.set_xlabel(
            "Generated token step"
            if trace.recording_kind == "generation"
            else "Input token step"
        )
        axis.set_ylabel("Channel value")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize="small")
        figure.tight_layout()
        return figure

    def log_trace_summary(self, trace: ProbeTrace, console: Console | None = None) -> None:
        """Render Rich table of patch-clamp recording electrophysiology."""
        c = console or Console()
        c.rule("[bold cyan]In-Silico Patch-Clamp Probing Electrode Summary[/bold cyan]")
        c.print(
            f"Recorded Trajectory: {len(trace.token_ids)} tokens | "
            f"Active Channels: [bold green]{len(trace.channels)}[/bold green]"
        )

        table = Table(title="Bio-State Dynamic Channels")
        table.add_column("Channel Key", style="bold")
        table.add_column("Site", justify="center")
        table.add_column("State Variable", justify="center")
        table.add_column("Initial Value", justify="right")
        table.add_column("Final Value", justify="right")
        table.add_column("Mean ± Std", justify="right")

        for key, rec in sorted(trace.channels.items()):
            if not rec.values:
                continue
            v_arr = np.asarray(rec.values, dtype=np.float64)
            table.add_row(
                key,
                rec.site_type,
                rec.channel_name,
                f"{rec.values[0]:.4f}",
                f"{rec.values[-1]:.4f}",
                f"{v_arr.mean():.4f} ± {v_arr.std():.4f}",
            )
        c.print(table)
