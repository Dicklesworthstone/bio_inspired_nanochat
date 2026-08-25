"""Persistent Lifelong Synaptic Memory System (bead `re4e.4`).

Enables cross-session personalized memory:
1. `UserMemoryPartition`: Logically namespaced storage for per-user fast weights and consolidated slow deltas.
2. `PersistentLifelongMemoryManager`: Manages session mounting, offline sleep consolidation,
   pseudonymous namespacing, and application-level partition removal.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear


@dataclass
class UserMemoryPartition:
    """Logically namespaced memory container for an individual user or agent."""

    user_id: str
    fast_weights: Dict[int, Tensor] = field(default_factory=dict)
    slow_deltas: Dict[int, Tensor] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class PersistentLifelongMemoryManager:
    """Coordinates lifecycle, cross-session persistence, and sleep consolidation of user memories."""

    def __init__(self, storage_dir: Path | str, max_delta_norm: float = 4.0):
        if not math.isfinite(max_delta_norm) or max_delta_norm < 0.0:
            raise ValueError("max_delta_norm must be finite and non-negative")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_delta_norm = max_delta_norm
        self.active_user: Optional[str] = None
        self._active_model: Optional[GPTSynaptic] = None
        self._active_slow_deltas: Dict[int, Tensor] = {}

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

    def _hash_user(self, user_id: str) -> str:
        """Derive a deterministic pseudonymous filename component from a user ID.

        The project prefix is public and fixed, so this is namespace hygiene rather than
        encryption or keyed protection against guessing low-entropy identifiers.
        """
        self._validate_user_id(user_id)
        return hashlib.sha256(f"nanochat_salt_{user_id}".encode("utf-8")).hexdigest()[:24]

    def _user_path(self, user_id: str) -> Path:
        return self.storage_dir / f"mem_{self._hash_user(user_id)}.pt"

    @staticmethod
    def _save_partition_data(path: Path, data: Dict[str, Any]) -> None:
        """Publish a partition atomically so interrupted writes preserve the prior copy."""
        temporary_path = path.with_name(f"{path.name}.tmp")
        torch.save(data, temporary_path)
        temporary_path.replace(path)

    @staticmethod
    def _load_partition_data(path: Path) -> Dict[str, Any]:
        data = torch.load(path, weights_only=True)
        if not isinstance(data, dict):
            raise ValueError(f"memory partition must contain a mapping: {path}")
        for field_name in ("fast_weights", "slow_deltas"):
            value = data.get(field_name, {})
            if not isinstance(value, dict):
                raise ValueError(f"memory partition field {field_name!r} must be a mapping")
            for idx, tensor in value.items():
                if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
                    raise ValueError(
                        f"memory partition contains an invalid layer index: {idx!r}"
                    )
                if (
                    not isinstance(tensor, Tensor)
                    or not tensor.is_floating_point()
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError(
                        f"memory partition {field_name} must contain finite floating tensors"
                    )
        return data

    @staticmethod
    def _validate_model_state(
        data: Dict[str, Any], syn_layers: list[SynapticLinear]
    ) -> None:
        for field_name, target_name in (
            ("slow_deltas", "w_slow"),
            ("fast_weights", "w_fast"),
        ):
            for idx, value in data.get(field_name, {}).items():
                if idx >= len(syn_layers):
                    raise ValueError(f"memory partition contains an invalid layer index: {idx!r}")
                target = getattr(syn_layers[idx], target_name)
                if target is None:
                    raise ValueError(f"memory partition contains invalid {field_name} state")
                if value.shape != target.shape:
                    raise ValueError(
                        f"memory partition {field_name} shape {tuple(value.shape)} does not "
                        f"match model shape {tuple(target.shape)} at layer {idx}"
                    )
                # Reject cross-precision partitions BEFORE any session teardown:
                # add_/sub_ on the slow path cannot cast, so a float32 partition
                # mounted into a bfloat16 model used to crash mid-mount with the
                # previous active session already torn down.
                if value.dtype != target.dtype:
                    raise ValueError(
                        f"memory partition {field_name} dtype {value.dtype} does not "
                        f"match model dtype {target.dtype} at layer {idx}"
                    )

    def load_partition(self, user_id: str) -> Optional[UserMemoryPartition]:
        """Load user memory partition from disk if it exists."""
        path = self._user_path(user_id)
        if not path.exists():
            return None
        data = self._load_partition_data(path)
        return UserMemoryPartition(
            user_id=user_id,
            fast_weights=data.get("fast_weights", {}),
            slow_deltas=data.get("slow_deltas", {}),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
        )

    def mount_user(self, model: GPTSynaptic, user_id: str) -> bool:
        """Mount user memory state into the living model for a session."""
        # Validate the destination before disturbing an already-mounted session.
        self._validate_user_id(user_id)
        if (
            self.active_user == user_id
            and self._active_model is not None
            and self._active_model is not model
        ):
            raise ValueError(
                "cannot implicitly remount the active user onto a different model; "
                "unmount the current session explicitly first"
            )
        path = self._user_path(user_id)
        syn_layers = [m for m in model.modules() if isinstance(m, SynapticLinear)]
        data = None
        # Validate a different user's durable state before unmounting the current session.
        # Remounting the same user must first save its live fast weights, then reload that
        # newly-written partition rather than a stale pre-unmount snapshot.
        if path.exists() and self.active_user != user_id:
            data = self._load_partition_data(path)
            self._validate_model_state(data, syn_layers)
        # Unmount existing user if needed to restore base model state
        if self.active_user is not None:
            if self._active_model is None:
                raise RuntimeError("active user has no associated model")
            self.unmount_user(self._active_model, self.active_user, consolidate=False)

        if not path.exists():
            # Brand new memory partition: start with clean zeroed fast scratchpad
            for mod in syn_layers:
                if mod.w_fast is not None:
                    mod.w_fast.data.zero_()
            self.active_user = user_id
            self._active_model = model
            self._active_slow_deltas.clear()
            return False

        if data is None:
            data = self._load_partition_data(path)
            self._validate_model_state(data, syn_layers)
        slow_deltas = data.get("slow_deltas", {})
        fast_weights = data.get("fast_weights", {})
        self._active_slow_deltas.clear()

        for idx, mod in enumerate(syn_layers):
            if idx in slow_deltas:
                delta = slow_deltas[idx].to(device=mod.w_slow.device, dtype=mod.w_slow.dtype)
                mod.w_slow.data.add_(delta)
                self._active_slow_deltas[idx] = delta.clone()
            if idx in fast_weights and mod.w_fast is not None:
                mod.w_fast.data.copy_(fast_weights[idx].to(mod.w_fast.device))
            elif mod.w_fast is not None:
                mod.w_fast.data.zero_()

        self.active_user = user_id
        self._active_model = model
        return True

    def unmount_user(
        self,
        model: GPTSynaptic,
        user_id: str,
        consolidate: bool = True,
        consolidation_lr: float = 0.1,
    ) -> None:
        """Consolidate fast weights into slow deltas, save partition to disk, and restore base model."""
        if not math.isfinite(consolidation_lr) or consolidation_lr < 0.0:
            raise ValueError("consolidation_lr must be finite and non-negative")
        if self.active_user != user_id:
            raise ValueError(
                f"cannot unmount user {user_id!r}: active user is {self.active_user!r}"
            )
        if self._active_model is not model:
            raise ValueError("cannot unmount user from a model other than the active model")

        syn_layers = [m for m in model.modules() if isinstance(m, SynapticLinear)]
        path = self._user_path(user_id)

        slow_deltas: Dict[int, Tensor] = {}
        fast_weights: Dict[int, Tensor] = {}
        created_at = time.time()

        if path.exists():
            old_data = self._load_partition_data(path)
            self._validate_model_state(old_data, syn_layers)
            slow_deltas = old_data.get("slow_deltas", {})
            created_at = old_data.get("created_at", created_at)

        for idx, mod in enumerate(syn_layers):
            if (
                consolidate
                and consolidation_lr > 0.0
                and mod.w_fast is not None
                and mod.w_fast.norm() > 1e-6
            ):
                # Fast->Slow consolidation
                transfer = consolidation_lr * mod.w_fast.detach().cpu()
                if idx in slow_deltas:
                    slow_deltas[idx].add_(transfer)
                else:
                    slow_deltas[idx] = transfer.clone()

                # Norm bounding
                curr_norm = float(slow_deltas[idx].norm().item())
                if curr_norm > self.max_delta_norm:
                    slow_deltas[idx].mul_(self.max_delta_norm / max(1e-6, curr_norm))

            elif mod.w_fast is not None:
                fast_weights[idx] = mod.w_fast.detach().cpu().clone()

        # Persist before mutating live state: a storage failure must leave the mounted
        # session intact so the caller can retry without losing personalized weights.
        self._save_partition_data(
            path,
            {
                "user_id_hash": self._hash_user(user_id),
                "slow_deltas": slow_deltas,
                "fast_weights": fast_weights,
                "created_at": created_at,
                "updated_at": time.time(),
            },
        )

        for idx, mod in enumerate(syn_layers):
            if mod.w_fast is not None:
                mod.w_fast.data.zero_()
            if idx in self._active_slow_deltas:
                delta = self._active_slow_deltas[idx]
                mod.w_slow.data.sub_(delta.to(device=mod.w_slow.device, dtype=mod.w_slow.dtype))

        self._active_slow_deltas.clear()
        self.active_user = None
        self._active_model = None

    def forget_user(self, user_id: str, model: Optional[GPTSynaptic] = None) -> bool:
        """Unlink the live partition and revert mounted state at the application layer."""
        if self.active_user == user_id and model is None:
            raise ValueError(
                "model is required to forget the active user without leaving live memory mounted"
            )
        if self.active_user == user_id and self._active_model is not model:
            raise ValueError("cannot forget an active user through a model other than the active model")

        if model is not None and self.active_user == user_id:
            # Purge live state before deleting the durable recovery copy. If model cleanup
            # fails, the partition remains available for a safe retry instead of being lost
            # while personalized state is still mounted.
            syn_layers = [m for m in model.modules() if isinstance(m, SynapticLinear)]
            for idx, mod in enumerate(syn_layers):
                if idx in self._active_slow_deltas:
                    delta = self._active_slow_deltas[idx]
                    mod.w_slow.data.sub_(delta.to(device=mod.w_slow.device, dtype=mod.w_slow.dtype))
                if mod.w_fast is not None:
                    mod.w_fast.data.zero_()
            self._active_slow_deltas.clear()
            self.active_user = None
            self._active_model = None

        path = self._user_path(user_id)
        erased = False
        if path.exists():
            path.unlink()
            erased = True

        return erased

    def log_store_status(self, console: Optional[Console] = None) -> None:
        """Render Rich summary table of persistent user memory store."""
        c = console or Console()
        c.rule("[bold cyan]Lifelong Persistent Synaptic Memory Registry[/bold cyan]")

        files = list(self.storage_dir.glob("mem_*.pt"))
        table = Table(title=f"User Partitions ({len(files)} registered)")
        table.add_column("Partition Hash", style="bold")
        table.add_column("Size (KB)", justify="right")
        table.add_column("Status", justify="center")

        for f in files:
            size_kb = f.stat().st_size / 1024.0
            table.add_row(
                f.stem,
                f"{size_kb:.1f} KB",
                "[bold green]Active[/bold green]"
                if self.active_user and self._hash_user(self.active_user) in f.stem
                else "Persisted",
            )
        c.print(table)
