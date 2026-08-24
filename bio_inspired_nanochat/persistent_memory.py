"""Persistent Lifelong Synaptic Memory System (bead `re4e.4`).

Enables cross-session personalized memory:
1. `UserMemoryPartition`: Isolated storage container for per-user fast weights and consolidated slow deltas.
2. `PersistentLifelongMemoryManager`: Manages session mounting, offline sleep consolidation,
   privacy namespacing, and hard right-to-be-forgotten erasure.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear
from bio_inspired_nanochat.working_memory_api import WorkingMemoryScratchpad


@dataclass
class UserMemoryPartition:
    """Isolated memory storage container for an individual user/agent namespace."""

    user_id: str
    fast_weights: Dict[int, Tensor] = field(default_factory=dict)
    slow_deltas: Dict[int, Tensor] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class PersistentLifelongMemoryManager:
    """Coordinates lifecycle, cross-session persistence, and sleep consolidation of user memories."""

    def __init__(self, storage_dir: Path | str, max_delta_norm: float = 4.0):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_delta_norm = max_delta_norm
        self.active_user: Optional[str] = None

    def _hash_user(self, user_id: str) -> str:
        """Compute salted cryptographic hash of user ID for filesystem isolation."""
        return hashlib.sha256(f"nanochat_salt_{user_id}".encode("utf-8")).hexdigest()[:24]

    def _user_path(self, user_id: str) -> Path:
        return self.storage_dir / f"mem_{self._hash_user(user_id)}.pt"

    def mount_user(self, model: GPTSynaptic, user_id: str) -> bool:
        """Mount user memory state into the living model for a session."""
        # Unmount existing user if needed
        if self.active_user is not None:
            self.unmount_user(model, self.active_user, consolidate=False)

        path = self._user_path(user_id)
        syn_layers = [m for m in model.modules() if isinstance(m, SynapticLinear)]

        if not path.exists():
            # Brand new memory partition: start with clean zeroed fast scratchpad
            for mod in syn_layers:
                if mod.w_fast is not None:
                    mod.w_fast.data.zero_()
            self.active_user = user_id
            return False

        data = torch.load(path, weights_only=True)

        for idx, mod in enumerate(syn_layers):
            if idx in data.get("slow_deltas", {}):
                mod.w_slow.data.add_(data["slow_deltas"][idx].to(mod.w_slow.device))
            if idx in data.get("fast_weights", {}) and mod.w_fast is not None:
                mod.w_fast.data.copy_(data["fast_weights"][idx].to(mod.w_fast.device))

        self.active_user = user_id
        return True

    def unmount_user(
        self,
        model: GPTSynaptic,
        user_id: str,
        consolidate: bool = True,
        consolidation_lr: float = 0.1,
    ) -> None:
        """Consolidate fast weights into slow deltas and save partition to disk."""
        syn_layers = [m for m in model.modules() if isinstance(m, SynapticLinear)]
        path = self._user_path(user_id)

        slow_deltas: Dict[int, Tensor] = {}
        fast_weights: Dict[int, Tensor] = {}

        if path.exists():
            old_data = torch.load(path, weights_only=True)
            slow_deltas = old_data.get("slow_deltas", {})

        for idx, mod in enumerate(syn_layers):
            if consolidate and mod.w_fast is not None and mod.w_fast.norm() > 1e-6:
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

                # Clear fast weights in living model
                mod.w_fast.data.zero_()

            elif mod.w_fast is not None:
                fast_weights[idx] = mod.w_fast.detach().cpu().clone()
                mod.w_fast.data.zero_()

        # Save partition to disk
        torch.save(
            {
                "user_id_hash": self._hash_user(user_id),
                "slow_deltas": slow_deltas,
                "fast_weights": fast_weights,
                "updated_at": time.time(),
            },
            path,
        )

        self.active_user = None

    def forget_user(self, user_id: str, model: Optional[GPTSynaptic] = None) -> bool:
        """Right-to-be-forgotten: Permanently delete user memory partition."""
        path = self._user_path(user_id)
        erased = False
        if path.exists():
            path.unlink()
            erased = True

        if model is not None and self.active_user == user_id:
            # Wipe model fast weights
            scratchpad = WorkingMemoryScratchpad(model)
            scratchpad.clear_scratchpad()
            self.active_user = None

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
