# nanochat/neuroscore.py
# -----------------------------------------------------------------------------
# NeuroScore: Evolutionary Credit Assignment for Synaptic Experts
# -----------------------------------------------------------------------------
# Measures:
#   1. Loss Contribution: first-order leave-one-expert-out advantage
#      ``-sum(<dL/d(expert_out), expert_out>)`` captured by backward hooks (uta.2),
#      falling back to the legacy sum-of-gates proxy when no fresh training gradients exist.
#   2. Specialization: How unique is this expert's input distribution? (Cosine distance from global mean)
#   3. Efficiency: Performance per unit of energy.
#   4. Resilience: Stability of contribution over time.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

from bio_inspired_nanochat.common import decouple_config
from bio_inspired_nanochat.torch_imports import F, Tensor, nn, torch

from .synaptic import SynapticMoE

FUSED_METRICS = decouple_config("BIO_FUSED_METRICS", default=False, cast=bool)


@dataclass
class NeuroScoreConfig:
    enabled: bool = True
    history_len: int = 1024
    update_every: int = 100  # Compute expensive metrics every N steps
    decay: float = 0.99  # EMA decay for resilience tracking

    # uta.2: estimator for per-expert loss contribution. "gradient" attributes the
    # first-order leave-one-expert-out advantage -sum(<dL/d(expert_out), expert_out>)
    # via backward hooks on each expert (signed: positive = genuinely useful).
    # Falls back to the routing proxy whenever no fresh training-time gradients
    # exist — eval/inference-only flows, or the first step after hooks are installed
    # (they attach lazily during the first step call). "proxy" keeps the legacy
    # sum-of-routing-gates heuristic unconditionally.
    credit_mode: str = "gradient"


class NeuroScore:
    """
    The 'Credit Assignment' engine.
    Tracks the true utility of experts to guide evolutionary decisions.
    """

    def __init__(self, cfg: NeuroScoreConfig, neuroviz=None):
        self.cfg = cfg
        self.neuroviz = neuroviz
        self.stats: Dict[str, Dict[str, Any]] = {}  # layer_name -> metrics
        self._last_loss = None

    def register_layer(self, name: str, num_experts: int, module: Optional[nn.Module] = None):
        if name not in self.stats:
            self.stats[name] = {
                "loss_contrib": torch.zeros(num_experts),  # Rolling sum
                "routing_freq": torch.zeros(num_experts),
                "specialization": torch.zeros(num_experts),
                "efficiency": torch.zeros(num_experts),
                "resilience": torch.zeros(num_experts),
                "prev_contrib": torch.zeros(num_experts),  # For resilience
                "updates": 0,
            }
        if module is not None:
            self._install_hooks(module)

    def _uses_gradient_credit(self) -> bool:
        return getattr(self.cfg, "credit_mode", "gradient") == "gradient"

    def _uses_fused_proxy_metrics(self, *, gates_on_cuda: bool) -> bool:
        """Whether the proxy-only fused kernel is semantically valid this step.

        ``update_metrics_fused`` accumulates routing-gate mass as contribution; it
        cannot consume the backward-hook stash used by gradient credit. Therefore a
        configured gradient estimator must always take the reference path, including
        on CUDA, instead of silently publishing proxy credit from the faster kernel.
        """
        return bool(
            FUSED_METRICS and gates_on_cuda and not self._uses_gradient_credit()
        )

    def _install_hooks(self, module: nn.Module) -> None:
        """Attach per-expert backward captures (uta.2), idempotently.

        A pre-forward hook on the MoE clears the stash at the start of every forward,
        so anything in ``module._ns_grad_stash`` afterwards provably belongs to THAT
        forward: each expert's hook registers a tensor hook on its output tensor, and
        when the training backward reaches it the per-selected-token first-order
        advantage ``<dL/dy_e, y_e>`` is stored under its expert index. Consequences:
        - an expert ABSENT from the stash was not invoked this forward (it received no
          tokens) — its marginal credit for the step is genuinely zero;
        - eval / inference forwards register no captures (grad mode off), so they wipe
          the snapshot without writing a new one — the next step falls back to the
          routing proxy instead of consuming stale numbers.
        Hooks are plain attributes + autograd hooks — nothing enters state_dict.
        """
        experts = getattr(module, "experts", None)
        if experts is None:
            return
        stash = cast(
            Optional[Dict[int, Any]], getattr(module, "_ns_grad_stash", None)
        )
        if stash is None:
            stash = {}
            object.__setattr__(module, "_ns_grad_stash", stash)
        if not getattr(module, "_ns_pre_attached", False):

            def _pre_clear(_mod: nn.Module, _args: tuple) -> None:
                stash.clear()

            module.register_forward_pre_hook(_pre_clear)
            object.__setattr__(module, "_ns_pre_attached", True)

        def _fwd_hook(_mod: nn.Module, _args: tuple, output: Any, _idx: int) -> None:
            out = output[0] if isinstance(output, tuple) else output
            # NOTE: ``Tensor`` from torch_imports is a typing shim (== Any), so the
            # runtime isinstance check must use the concrete torch class.
            if not (torch.is_grad_enabled() and isinstance(out, torch.Tensor) and out.requires_grad):
                return

            def _capture(grad: Tensor, _i: int = _idx, _out: Tensor = out) -> None:
                # First-order leave-one-expert-out advantage per selected token:
                # <dL/dy_e, y_e>. Zeroing expert e moves the loss by
                # -<dL/dy_e, y_e> + O(||y_e||^2), so the NEGATED sum is "how much
                # loss would RISE if this expert were removed" — signed, so a
                # consistently harmful expert lands below zero.
                stash[_i] = (grad.detach().float() * _out.detach().float()).sum(dim=-1).cpu()
            out.register_hook(_capture)

        for e_idx, expert in enumerate(experts):
            if getattr(expert, "_ns_hook_attached", False):
                continue  # survivor expert across a uta.4 resize: already armed
            expert.register_forward_hook(
                lambda m, a, o, _i=e_idx: _fwd_hook(m, a, o, _i)
            )
            object.__setattr__(expert, "_ns_hook_attached", True)

    def _collect_gradient_credit(
        self,
        module: nn.Module,
        st: Dict[str, Any],
        indices: Tensor,
        batch_size: int,
    ) -> Optional[Tensor]:
        """Turn the freshest backward stash into a normalized contribution vector.

        For every expert invoked this forward: negated sum of ``<dL/dy_e, y_e>``
        over its selected tokens, normalized by B*T — a cheap leave-one-expert-out
        counterfactual advantage (the uta.2 spec's second estimator). Because the
        expert output enters the mixture as ``gate * y_e``, autograd already hands
        back ``gate * dL/dout`` at the hook, so each token's dot product is
        gate-weighted with no extra bookkeeping. Positive = removing the expert
        would raise the loss (genuinely useful); negative = it currently hurts.
        Experts absent from the stash were not invoked (no tokens routed) and earn
        exactly zero. Returns None only on a genuine inconsistency (an entry whose
        size disagrees with the current routing — possible only under multi-backward
        accumulation); callers then fall back to the whole-vector proxy so the EMA
        never mixes epochs of bookkeeping.
        """
        stash = getattr(module, "_ns_grad_stash", None)
        if not stash:
            return None
        contrib = torch.zeros_like(st["loss_contrib"])
        for e in range(st["loss_contrib"].numel()):
            dot = stash.get(e)
            if dot is None:
                continue  # not invoked this forward -> true zero marginal credit
            mask = indices == e
            flat_pos = mask.any(dim=-1).view(-1).nonzero(as_tuple=False).squeeze(1)
            if dot.numel() != flat_pos.numel():
                return None  # snapshot predates the current routing pattern
            contrib[e] -= dot.sum()
        stash.clear()
        return contrib / float(batch_size)

    @torch.no_grad()
    def step(self, model: nn.Module, loss: Tensor, global_step: int):
        if not self.cfg.enabled:
            return

        # uta.2: contribution is estimated from REAL gradients when available. Backward
        # hooks on each expert's output capture <dL/d(expert_out), expert_out> during
        # the normal training backward (which has already fired by the time viz.step
        # runs), so the metric update itself stays under no_grad. The capture inherently
        # includes the routing gate (the output is mixed in as gate * y_e), making it a
        # cheap first-order leave-one-expert-out counterfactual. Without fresh grads
        # (eval-only flow, or the very first step after hook installation) we fall back
        # to the legacy forward-pass proxy: Contribution ~ sum(RoutingGates).

        for name, module in model.named_modules():
            if isinstance(module, SynapticMoE):
                if not hasattr(module, "last_ctx") or not module.last_ctx:
                    continue

                layer_name = name
                if self.neuroviz is not None:
                    mapped = self.neuroviz._name_of(module)
                    if mapped:
                        layer_name = mapped
                if layer_name not in self.stats:
                    self.register_layer(layer_name, module.num_experts, module=module)

                st = self.stats[layer_name]
                if st["loss_contrib"].numel() != int(module.num_experts):
                    # Layer was resized (uta.4 variable expert count): reset the
                    # bookkeeping; per-expert hook guards re-arm only new experts.
                    del self.stats[layer_name]
                    stash = getattr(module, "_ns_grad_stash", None)
                    if stash is not None:
                        stash.clear()
                    self.register_layer(layer_name, module.num_experts, module=module)
                    st = self.stats[layer_name]
                if self._uses_gradient_credit():
                    # uta.2: ensure the backward captures exist before we need them;
                    # idempotent and inert under no_grad.
                    self._install_hooks(module)
                ctx = module.last_ctx
                gates = ctx["gates"]  # (B,T,k)
                indices = ctx["indices"]  # (B,T,k)
                x_in = ctx["x"]  # (B,T,C)
                energy = module.energy # (E,)

                # Fused kernel path. This kernel implements the explicit routing
                # proxy only; gradient credit must consume the backward-hook stash
                # in the reference path below even when gates live on CUDA.
                if self._uses_fused_proxy_metrics(gates_on_cuda=gates.is_cuda):
                    try:
                        from bio_inspired_nanochat.kernels import update_metrics_fused

                        if update_metrics_fused(indices, gates, energy, st, self.cfg):
                            st["credit_source"] = "proxy"
                            st["updates"] += 1

                            if global_step % self.cfg.update_every == 0:
                                self._update_specialization(st, x_in, indices, module.num_experts)
                                if self.neuroviz:
                                    self._log_metrics(layer_name, st, global_step)
                            # Publish composite fitness for the lifecycle controller (de5l).
                            self._publish_score(module, st)
                            continue
                    except ImportError:
                        pass
                    except Exception as e:
                        print(f"Triton metrics kernel failed: {e}")

                # 1. Specialization (Diversity of inputs)
                # Calculate mean input vector per expert
                # This is expensive, so only do it occasionally
                if global_step % self.cfg.update_every == 0:
                    self._update_specialization(st, x_in, indices, module.num_experts)

                # 2. Loss Contribution (uta.2).
                # Preferred estimator: gradient-based marginal credit — for every token
                # routed to it, ||dL/d(expert output)|| captured by the per-expert
                # backward hooks installed above during the normal training backward.
                # This measures how much an expert's ACTUAL OUTPUT moves the loss, not
                # merely how often it gets routed. Fallbacks (eval/inference-only flows,
                # first step after hook installation, or cfg.credit_mode="proxy"): the
                # legacy sum-of-gates routing proxy.
                batch_size = gates.shape[0] * gates.shape[1]
                gates_flat = gates.view(-1)
                indices_flat = indices.view(-1)
                grad_credit = (
                    self._collect_gradient_credit(module, st, indices, batch_size)
                    if self._uses_gradient_credit()
                    else None
                )
                if grad_credit is not None:
                    contrib_update = grad_credit
                    st["credit_source"] = "gradient"
                else:
                    contrib_update = torch.zeros_like(st["loss_contrib"])
                    contrib_update.index_add_(0, indices_flat.cpu(), gates_flat.float().cpu())
                    contrib_update /= batch_size
                    st["credit_source"] = "proxy"

                # Routing frequency (counts per token, sums to top_k)
                freq_update = torch.zeros_like(st["routing_freq"])
                freq_update.index_add_(
                    0,
                    indices_flat.cpu(),
                    torch.ones_like(indices_flat.cpu(), dtype=torch.float32),
                )
                freq_update /= batch_size

                # EMA update
                st["loss_contrib"].mul_(self.cfg.decay).add_(contrib_update * (1 - self.cfg.decay))
                st["routing_freq"].mul_(self.cfg.decay).add_(freq_update * (1 - self.cfg.decay))

                # 3. Efficiency = Contribution / (Energy + epsilon)
                energy_cpu = module.energy.detach().float().cpu()
                st["efficiency"] = st["loss_contrib"] / (energy_cpu + 1e-6)

                # 4. Resilience = stability of contribution over time (1 / variation).
                # Gate by activity: a DEAD expert also has ~constant (zero) contribution, so an
                # ungated 1/|Δ| would score it as MAXIMALLY resilient (and min-max-normalized to 1.0
                # in composite_fitness), protecting it from reset — the lifecycle would never reclaim
                # it. This is the sibling of the dead-expert specialization inversion above. Only
                # experts routed this step earn resilience; unused experts decay toward 0 (low =
                # reclaimable). NOTE: keep parity with kernels.update_metrics_fused if that path gains
                # a resilience term.
                diff = (st["loss_contrib"] - st["prev_contrib"]).abs()
                stability = 1.0 / (diff + 1e-6)
                active = (freq_update > 0).float()
                st["resilience"].mul_(self.cfg.decay).add_(stability * active * (1 - self.cfg.decay))
                st["prev_contrib"].copy_(st["loss_contrib"])

                st["updates"] += 1

                # Publish the composite fitness for the split/merge controller (de5l).
                self._publish_score(module, st)

                # Log to NeuroViz/TensorBoard if connected
                if self.neuroviz and global_step % self.cfg.update_every == 0:
                    self._log_metrics(layer_name, st, global_step)

    @staticmethod
    def composite_fitness(
        efficiency: Tensor, specialization: Tensor, resilience: Tensor
    ) -> Tensor:
        """Blend the three NeuroScore metrics into a per-expert fitness in [0,1].

        Each metric is min-max normalized across experts first (so their heterogeneous
        scales — efficiency is contribution/energy, resilience is 1/variance — become
        comparable), then averaged. A degenerate all-equal metric maps to a neutral 0.5
        so it neither helps nor hurts. Higher = fitter (more split-worthy, less
        merge-worthy), matching the health convention the lifecycle controller uses.
        """
        def _norm(x: Tensor) -> Tensor:
            x = x.detach().float()
            lo = x.min()
            rng = x.max() - lo
            if float(rng) < 1e-8:
                return torch.full_like(x, 0.5)
            return (x - lo) / rng

        comp = (_norm(efficiency) + _norm(specialization) + _norm(resilience)) / 3.0
        # Never let a NaN/Inf metric leak into the lifecycle's health signal.
        return torch.nan_to_num(comp, nan=0.5).clamp(0.0, 1.0)

    def _publish_score(self, module: nn.Module, st: Dict[str, Any]) -> None:
        """Write the composite fitness onto the MoE module (``last_neuroscore``) so the
        SplitMergeController can blend it into health when ``use_neuroscore`` is on.
        Stored on the stats' device (CPU); the controller re-homes it to the layer."""
        comp = self.composite_fitness(
            st["efficiency"], st["specialization"], st["resilience"]
        )
        object.__setattr__(module, "last_neuroscore", comp)
        # uta.9: the raw per-expert credit relative to the mean (1.0 = an average expert), for
        # SplitMergeConfig.health_mode="credit". Unlike the min-max composite above it keeps the
        # absolute spread, so a uniform population reads 1.0 everywhere and nothing fires; only
        # a genuinely disproportionate or useless expert crosses a threshold.
        contrib = st["loss_contrib"].detach().float()
        scale = float(contrib.abs().mean()) + 1e-12
        object.__setattr__(module, "last_credit", contrib / scale)
        object.__setattr__(module, "last_credit_source", str(st.get("credit_source", "unknown")))

    def _update_specialization(self, st, x, indices, num_experts):
        """
        How 'unique' is the input subspace this expert sees?
        High specialization = Sees vectors very different from the global mean.
        """
        # x: (B,T,C)
        # indices: (B,T,k)
        B, T, C = x.shape
        
        # Compute global mean of inputs
        global_mean = x.mean(dim=(0, 1)).float()  # (C,)
        
        # We want mean input per expert.
        # Gather inputs for each expert? Too much memory.
        # Streaming approx:
        # Just sample a subset for speed
        mask_prob = 0.1
        mask = torch.rand(B, T, device=x.device) < mask_prob
        if not mask.any():
            return

        x_sub = x[mask].float() # (N, C)
        ind_sub = indices[mask] # (N, k)
        
        # For each expert, compute centroid of assigned inputs
        expert_sums = torch.zeros(num_experts, C, device=x.device, dtype=torch.float32)
        expert_counts = torch.zeros(num_experts, device=x.device, dtype=torch.float32)
        
        # Naive loop is slow, but x_sub is small. 
        # Vectorized scatter_add is better.
        # Expand x_sub for k assignments? 
        # (N, k, C)
        # This might OOM if k is large, but k=2 usually.
        
        for k_i in range(ind_sub.shape[1]):
            # idx: (N,)
            idx = ind_sub[:, k_i]
            expert_sums.index_add_(0, idx, x_sub)
            expert_counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            
        expert_means = expert_sums / (expert_counts.unsqueeze(1) + 1e-6)
        
        # Cosine distance from global mean
        # (E, C) vs (C,)
        sim = F.cosine_similarity(expert_means, global_mean.unsqueeze(0), dim=1)

        # Specialization = 1 - similarity (0 = generic, 1 = unique)
        spec = 1.0 - sim
        # An expert that received no tokens in this sample has an all-zero centroid, so its cosine
        # similarity is 0 and spec would be 1.0 — i.e. a dead expert would score as MAXIMALLY
        # specialized, inverting the lifecycle signal (protecting it from reset, attracting splits).
        # Keep the prior specialization for unused experts instead of the spurious max.
        used = expert_counts > 0
        prior = st["specialization"].to(spec.device)
        spec = torch.where(used, spec, prior)
        st["specialization"].copy_(spec.detach().cpu())

    def _gini(self, x: Tensor) -> float:
        x = x.float().sort()[0]
        n = x.shape[0]
        index = torch.arange(1, n + 1, device=x.device, dtype=torch.float32)
        return ((2 * index - n - 1) * x).sum() / (n * x.sum() + 1e-6)

    def _log_metrics(self, layer_name, st, step):
        # Push to TensorBoard via NeuroViz
        if not self.neuroviz or not getattr(self.neuroviz, "tb", None):
            return
        tb = self.neuroviz.tb
        
        # Gini Coefficient (Load Balancing)
        gini = self._gini(st["routing_freq"])
        tb.add_scalar(f"{layer_name}/score/gini_routing", gini, step)
        
        # Scalars (Means)
        tb.add_scalar(f"{layer_name}/score/mean_efficiency", st["efficiency"].mean(), step)
        tb.add_scalar(f"{layer_name}/score/mean_specialization", st["specialization"].mean(), step)
        tb.add_scalar(f"{layer_name}/score/mean_resilience", st["resilience"].mean(), step)
        
        # Histograms
        tb.add_histogram(f"{layer_name}/score/hist_efficiency", st["efficiency"], step)
        tb.add_histogram(f"{layer_name}/score/hist_specialization", st["specialization"], step)
        
        # Leaderboard (Top 5 Experts by Efficiency)
        top_k = 5
        vals, idxs = torch.topk(st["efficiency"], k=min(top_k, len(st["efficiency"])))
        
        # Create Markdown Table
        md = "| Rank | ID | Efficiency | Spec | Contrib |\n|---|---|---|---|---|\n"
        for rank, (val, idx) in enumerate(zip(vals, idxs)):
            i = idx.item()
            md += f"| {rank+1} | {i} | {val:.3f} | {st['specialization'][i]:.3f} | {st['loss_contrib'][i]:.3f} |\n"
            
        tb.add_text(f"{layer_name}/leaderboard", md, step)
