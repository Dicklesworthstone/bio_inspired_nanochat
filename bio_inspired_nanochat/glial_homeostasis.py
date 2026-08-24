"""Slow astrocyte-like homeostasis for sparse expert routing.

The controller models the regulatory side of a tripartite synapse: it pools
activity and metabolic state over small groups of experts, then emits a slow
additive routing signal (a computational analogue of gliotransmission).  The
signal is deliberately stateful, bounded, and zero-sum so it can counter a
winner-take-all router without changing the router's common logit offset.

This module owns only the control law.  ``SynapticMoE`` owns routing and calls
``observe`` after dispatch, which keeps the controller independent of expert
implementation details and of the Python/Rust/Triton metabolism backends.
"""

from __future__ import annotations

from bio_inspired_nanochat.torch_imports import Tensor, nn, torch


class GlialHomeostasis(nn.Module):
    """Integral homeostatic feedback over expert activity and pooled energy.

    ``activity_ema`` starts at the healthy uniform fixed point.  Balanced
    routing with equal energy therefore produces exactly zero correction, while
    persistent overuse integrates a negative bias and underuse a positive one.
    Both activity and energy are pooled by contiguous expert groups before the
    group-level part of the feedback is applied.
    """

    def __init__(
        self,
        num_experts: int,
        *,
        group_size: int,
        ema_rate: float,
        feedback_rate: float,
        energy_weight: float,
        bias_cap: float,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if not 0.0 < ema_rate <= 1.0:
            raise ValueError(f"ema_rate must be in (0, 1], got {ema_rate}")
        if not 0.0 < feedback_rate <= 1.0:
            raise ValueError(f"feedback_rate must be in (0, 1], got {feedback_rate}")
        if energy_weight < 0.0:
            raise ValueError(f"energy_weight must be >= 0, got {energy_weight}")
        if bias_cap <= 0.0:
            raise ValueError(f"bias_cap must be > 0, got {bias_cap}")

        self.num_experts = int(num_experts)
        self.group_size = min(int(group_size), self.num_experts)
        self.ema_rate = float(ema_rate)
        self.feedback_rate = float(feedback_rate)
        self.energy_weight = float(energy_weight)
        self.bias_cap = float(bias_cap)

        group_index = torch.arange(self.num_experts, dtype=torch.long) // self.group_size
        num_groups = int(group_index[-1].item()) + 1
        group_counts = torch.bincount(group_index, minlength=num_groups).to(torch.float32)

        self.register_buffer("group_index", group_index)
        self.register_buffer("group_counts", group_counts)
        self.register_buffer(
            "activity_ema",
            torch.full((self.num_experts,), 1.0 / float(self.num_experts)),
        )
        self.register_buffer("group_energy_ema", torch.ones(num_groups))
        self.register_buffer("gliotransmitter_bias", torch.zeros(self.num_experts))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    @property
    def routing_bias(self) -> Tensor:
        """Current bounded additive routing-logit correction."""
        return self.gliotransmitter_bias

    @torch.no_grad()
    def reset_(self) -> None:
        """Return the controller to its healthy, behavior-neutral fixed point."""
        self.activity_ema.fill_(1.0 / float(self.num_experts))
        self.group_energy_ema.fill_(1.0)
        self.gliotransmitter_bias.zero_()
        self.steps.zero_()

    @torch.no_grad()
    def observe(self, selection_counts: Tensor, expert_energy: Tensor) -> None:
        """Advance slow homeostasis from one dispatch observation.

        ``selection_counts`` is the number of top-k assignments per expert, not
        gate probability mass.  It is normalized internally, making the target
        independent of both token count and ``top_k``.
        """
        if selection_counts.numel() != self.num_experts:
            raise ValueError(
                "selection_counts must have one entry per expert, got "
                f"{selection_counts.numel()} for {self.num_experts} experts"
            )
        if expert_energy.numel() != self.num_experts:
            raise ValueError(
                "expert_energy must have one entry per expert, got "
                f"{expert_energy.numel()} for {self.num_experts} experts"
            )

        counts = selection_counts.detach().reshape(-1).to(self.activity_ema)
        energy = expert_energy.detach().reshape(-1).to(self.activity_ema)
        if not bool(torch.isfinite(counts).all()) or not bool(torch.isfinite(energy).all()):
            raise ValueError("glial observations must be finite")
        if bool((counts < 0).any()):
            raise ValueError("selection_counts must be non-negative")

        total = counts.sum()
        if float(total.item()) <= 0.0:
            return

        observed_share = counts / total
        self.activity_ema.lerp_(observed_share, self.ema_rate)

        pooled_energy = torch.zeros_like(self.group_energy_ema)
        pooled_energy.scatter_add_(0, self.group_index, energy)
        pooled_energy.div_(self.group_counts)
        self.group_energy_ema.lerp_(pooled_energy, self.ema_rate)

        expert_target = 1.0 / float(self.num_experts)
        expert_error = (expert_target - self.activity_ema) / expert_target

        group_activity = torch.zeros_like(self.group_energy_ema)
        group_activity.scatter_add_(0, self.group_index, self.activity_ema)
        group_target = self.group_counts / float(self.num_experts)
        group_error = (group_target - group_activity) / group_target

        pooled_energy_mean = (
            self.group_energy_ema * self.group_counts
        ).sum() / float(self.num_experts)
        energy_error = self.group_energy_ema - pooled_energy_mean

        drive = (
            expert_error
            + group_error.index_select(0, self.group_index)
            + self.energy_weight * energy_error.index_select(0, self.group_index)
        )
        drive.sub_(drive.mean())

        candidate = self.gliotransmitter_bias + self.feedback_rate * drive
        candidate.sub_(candidate.mean())
        scale = torch.clamp(
            self.bias_cap / candidate.abs().max().clamp_min(torch.finfo(candidate.dtype).eps),
            max=1.0,
        )
        self.gliotransmitter_bias.copy_(candidate * scale)
        self.steps.add_(1)

    def diagnostics(self) -> dict[str, Tensor]:
        """Detached controller state for structured telemetry and tests."""
        return {
            "activity_ema": self.activity_ema.detach().clone(),
            "group_energy_ema": self.group_energy_ema.detach().clone(),
            "routing_bias": self.gliotransmitter_bias.detach().clone(),
            "steps": self.steps.detach().clone(),
        }

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # A glia-enabled model may intentionally warm-start from a checkpoint
        # created before hy8.4.  Missing controller state means the neutral fixed
        # point, which is exactly the state constructed above.
        for name in (
            "group_index",
            "group_counts",
            "activity_ema",
            "group_energy_ema",
            "gliotransmitter_bias",
            "steps",
        ):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
