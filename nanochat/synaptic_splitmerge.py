# nanochat/synaptic_splitmerge.py
# Split/Merge controller for Synaptic MoE layers
#
# This controller performs:
#   • MERGE: pick expert pairs with high cosine similarity in the router-embedding
#            space AND low health; merge loser into winner (weighted average),
#            then CLONE the winner (with small noise) back into loser slot.
#   • SPLIT: clone strong experts into the weakest slots (optional).
#   • Keeps expert count constant; updates router columns, embeddings, synaptic state,
#     zeroes optimizer moments for changed parameters, and can broadcast in DDP.
#
# Works with:
#   - SynapticMoE, SynapticExpert, SynapticLinear, PostsynapticHebb from nanochat/synaptic.py
#   - GPTSynaptic from nanochat/gpt_synaptic.py
#
# Usage:
#   ctrl = SplitMergeController(model, SplitMergeConfig(...))
#   ctrl.step(global_step, optimizer=opt)    # call periodically (e.g. every 50k steps)

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict
import torch
import torch.nn as nn
import torch.distributed as dist

from .synaptic import SynapticMoE, SynapticExpert, SynapticLinear

Tensor = torch.Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SplitMergeConfig:
    enabled: bool = True
    # MERGE criteria
    merge_cosine_threshold: float = 0.85        # router-embedding cosine sim threshold
    merge_health_max: float = 0.25              # both experts must be below this health to be merge-candidates
    merges_per_call: int = 1                    # max merges per step
    # SPLIT criteria
    split_health_min: float = 0.80              # expert must be above this health to be split candidate
    splits_per_call: int = 1
    # Noise scales for cloned experts
    clone_noise_linear: float = 0.02            # noise scale for linear weights
    clone_noise_router: float = 0.01            # noise scale for router columns
    clone_noise_embed: float = 0.05             # noise scale (tangent) for router embedding
    # Scheduling
    min_step_interval: int = 10_000             # don't do anything more frequently than this
    warmup_steps: int = 20_000                   # no changes until after warmup
    # DDP
    ddp_broadcast: bool = True                  # broadcast parameters from rank 0 after changes
    # Expert weighting
    use_util_weighting: bool = True             # weight merge by winner/loser utilization (via fatigue proxy)
    # Logging
    verbose: bool = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def _world_size() -> int:
    return 1 if (not dist.is_available()) or (not dist.is_initialized()) else dist.get_world_size()

@torch.no_grad()
def _cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return a @ b.T

@torch.no_grad()
def _orthogonal_perturb_like(vec: Tensor, noise_scale: float) -> Tensor:
    """Return a unit-length vector: normalized(vec + noise in orthogonal subspace)."""
    d = vec.shape[-1]
    noise = torch.randn_like(vec)
    proj = (noise * vec).sum(dim=-1, keepdim=True) * vec
    tangent = noise - proj
    out = vec + noise_scale * tangent
    return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

@torch.no_grad()
def _add_noise_(t: Tensor, scale: float):
    if scale <= 0: return
    t.add_(torch.randn_like(t) * scale)

@torch.no_grad()
def _zero_optim_moments_for(optimizer: Optional[torch.optim.Optimizer], params: Iterable[nn.Parameter]):
    if optimizer is None: return
    pset = set(params)
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in pset:
                state = optimizer.state.get(p, None)
                if state:
                    for k in list(state.keys()):  # momentum, exp_avg, etc.
                        state[k].zero_() if torch.is_tensor(state[k]) else None

@torch.no_grad()
def _broadcast_module_params(module: nn.Module):
    if not dist.is_available() or not dist.is_initialized() or _world_size() == 1: return
    for t in module.state_dict().values():
        if torch.is_tensor(t):
            dist.broadcast(t, src=0)

# ---------------------------------------------------------------------------
# Parameter copy helpers (SynapticLinear & expert)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _copy_synaptic_linear_(dst: SynapticLinear, src: SynapticLinear):
    # weights & bias
    dst.w_slow.copy_(src.w_slow); dst.w_fast.copy_(src.w_fast)
    if (dst.bias is not None) and (src.bias is not None):
        dst.bias.copy_(src.bias)
    # postsyn state
    dst.post.U.copy_(src.post.U); dst.post.V.copy_(src.post.V)
    dst.post.H_fast.copy_(src.post.H_fast)
    dst.post.m_gate.copy_(src.post.m_gate)
    dst.post.camkii.copy_(src.post.camkii); dst.post.pp1.copy_(src.post.pp1)

@torch.no_grad()
def _merge_linear_into_(winner: SynapticLinear, loser: SynapticLinear, alpha: float):
    """winner = alpha * winner + (1-alpha) * loser"""
    winner.w_slow.mul_(alpha).add_((1.0 - alpha) * loser.w_slow)
    winner.w_fast.mul_(alpha).add_((1.0 - alpha) * loser.w_fast)
    if (winner.bias is not None) and (loser.bias is not None):
        winner.bias.mul_(alpha).add_((1.0 - alpha) * loser.bias)
    winner.post.U.mul_(alpha).add_((1.0 - alpha) * loser.post.U)
    winner.post.V.mul_(alpha).add_((1.0 - alpha) * loser.post.V)
    winner.post.H_fast.mul_(alpha).add_((1.0 - alpha) * loser.post.H_fast)
    # gate and enzymes: bias toward winner (more stable)
    winner.post.m_gate.mul_(0.9).add_(0.1 * loser.post.m_gate)
    winner.post.camkii.mul_(0.9).add_(0.1 * loser.post.camkii)
    winner.post.pp1.mul_(0.9).add_(0.1 * loser.post.pp1)

@torch.no_grad()
def _clone_linear_from_(dst: SynapticLinear, src: SynapticLinear, noise_scale: float):
    _copy_synaptic_linear_(dst, src)
    _add_noise_(dst.w_slow, noise_scale); _add_noise_(dst.w_fast, noise_scale)
    if dst.bias is not None: _add_noise_(dst.bias, noise_scale)
    # reset fast Hebbian traces for cloned expert
    dst.post.H_fast.zero_()
    dst.post.U.mul_(0.5); dst.post.V.mul_(0.5)  # keep some eligibility but dampen

@torch.no_grad()
def _merge_expert_into_and_clone_(layer: SynapticMoE, winner_idx: int, loser_idx: int,
                                  alpha: float, cfg: SplitMergeConfig):
    """Merge loser into winner (weighted), then clone winner (+noise) into loser slot."""
    winner: SynapticExpert = layer.experts[winner_idx]
    loser:  SynapticExpert = layer.experts[loser_idx]

    # 1) Merge parameters into winner
    _merge_linear_into_(winner.fc1, loser.fc1, alpha)
    _merge_linear_into_(winner.fc2, loser.fc2, alpha)

    # 2) Clone back into loser (to keep count constant)
    _clone_linear_from_(loser.fc1, winner.fc1, cfg.clone_noise_linear)
    _clone_linear_from_(loser.fc2, winner.fc2, cfg.clone_noise_linear)

    # 3) Router columns: average into winner, clone into loser (with noise)
    W = layer.router.weight  # shape: (n_embd, E) in PyTorch's (out_features, in_features) conv; here we defined Linear(n_embd->E): weight is (E, n_embd) if bias=False? Actually torch.nn.Linear(out,in) has weight (out,in)
    # In synaptic.py, router = nn.Linear(n_embd, num_experts, bias=False); so weight shape is (E, n_embd)
    # We'll operate on row vectors (expert rows):
    W_w = W[winner_idx]  # (n_embd,)
    W_l = W[loser_idx]
    W_w.mul_(alpha).add_((1.0 - alpha) * W_l)
    W_l.copy_(W_w)
    _add_noise_(W_l, cfg.clone_noise_router)

    # 4) Router embeddings: keep winner embedding; clone loser as orthogonalized perturbed winner
    emb = layer.router_embeddings  # (E, D)
    e_w = emb[winner_idx:winner_idx+1]  # (1,D)
    e_l = _orthogonal_perturb_like(e_w.clone(), cfg.clone_noise_embed)
    emb[loser_idx:loser_idx+1].copy_(e_l)

    # 5) Reset stats
    layer.fatigue[loser_idx] = 0.0
    layer.energy[loser_idx]  = 1.0

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class SplitMergeController:
    def __init__(self, model: nn.Module, cfg: SplitMergeConfig):
        self.model = model
        self.cfg = cfg
        self._last_step = -10**12    # ensure first call can run if warmup permits
        self._moe_layers: List[SynapticMoE] = self._find_moe_layers(model)

    def _find_moe_layers(self, module: nn.Module) -> List[SynapticMoE]:
        moes: List[SynapticMoE] = []
        for m in module.modules():
            if isinstance(m, SynapticMoE):
                moes.append(m)
        return moes

    @torch.no_grad()
    def _health(self, layer: SynapticMoE) -> Tensor:
        # Higher is better: combine (1 - fatigue) with energy in [0,1]
        fat = layer.fatigue.clamp(0, 1)          # EMA utilization proxy
        eng = layer.energy.clamp(0, 1)
        health = (1.0 - fat) * (0.5 + 0.5 * eng) # [0,1]
        return health

    @torch.no_grad()
    def _util_weight(self, layer: SynapticMoE, i: int, j: int) -> float:
        if not self.cfg.use_util_weighting:
            return 0.6  # mild bias toward first arg
        # invert fatigue → utilization proxy
        u_i = (1.0 - layer.fatigue[i]).clamp(0,1)
        u_j = (1.0 - layer.fatigue[j]).clamp(0,1)
        s = (u_i + u_j).clamp_min(1e-6)
        return float((u_i / s).item())

    @torch.no_grad()
    def _pick_merge_pairs(self, layer: SynapticMoE) -> List[Tuple[int,int]]:
        E = layer.num_experts
        emb = layer.router_embeddings  # (E, D)
        sim = _cosine(emb, emb)        # (E,E)
        health = self._health(layer)
        # candidate mask: high sim and both low health
        sim_mask = sim > self.cfg.merge_cosine_threshold
        low = (health <= self.cfg.merge_health_max)
        cand = sim_mask & low.unsqueeze(1) & low.unsqueeze(0)
        # remove diagonal
        idx = torch.arange(E, device=emb.device)
        cand[idx, idx] = False
        # score by similarity (higher first)
        scores = sim.masked_fill(~cand, -1.0)  # -1 for invalid
        pairs: List[Tuple[int,int]] = []
        used = set()
        for _ in range(self.cfg.merges_per_call):
            # find max entry
            val, linear_idx = scores.view(-1).max(dim=0)
            if val <= 0: break
            i = (linear_idx // E).item()
            j = (linear_idx %  E).item()
            if i in used or j in used:
                scores[i,:] = -1.0; scores[:,i] = -1.0
                scores[j,:] = -1.0; scores[:,j] = -1.0
                continue
            pairs.append((i,j))
            used.add(i); used.add(j)
            # invalidate rows/cols
            scores[i,:] = -1.0; scores[:,i] = -1.0
            scores[j,:] = -1.0; scores[:,j] = -1.0
        return pairs

    @torch.no_grad()
    def _pick_split_sources(self, layer: SynapticMoE) -> List[int]:
        health = self._health(layer)
        strong = (health >= self.cfg.split_health_min).nonzero(as_tuple=False).flatten().tolist()
        # take top k strongest
        strong_sorted = sorted(strong, key=lambda e: float(health[e].item()), reverse=True)
        return strong_sorted[:self.cfg.splits_per_call]

    @torch.no_grad()
    def _weakest_slots(self, layer: SynapticMoE, k: int) -> List[int]:
        health = self._health(layer)
        idx = torch.argsort(health)  # ascending
        return idx[:k].tolist()

    @torch.no_grad()
    def _split_into_slots(self, layer: SynapticMoE, sources: List[int], slots: List[int], optimizer: Optional[torch.optim.Optimizer]):
        for src, dst in zip(sources, slots):
            if src == dst: continue
            # Clone src → dst, with noise & embedding tweak
            _clone_linear_from_(layer.experts[dst].fc1, layer.experts[src].fc1, self.cfg.clone_noise_linear)
            _clone_linear_from_(layer.experts[dst].fc2, layer.experts[src].fc2, self.cfg.clone_noise_linear)
            # router weight row (expert row)
            W = layer.router.weight
            W[dst].copy_(W[src])
            _add_noise_(W[dst], self.cfg.clone_noise_router)
            # embedding
            layer.router_embeddings[dst:dst+1].copy_(_orthogonal_perturb_like(layer.router_embeddings[src:src+1].clone(), self.cfg.clone_noise_embed))
            # reset stats
            layer.fatigue[dst] = 0.0; layer.energy[dst] = 1.0
            # zero optimizer moments
            if optimizer is not None:
                changed = [layer.experts[dst].fc1.w_slow, layer.experts[dst].fc1.w_fast,
                           layer.experts[dst].fc2.w_slow, layer.experts[dst].fc2.w_fast, W]
                if layer.experts[dst].fc1.bias is not None: changed.append(layer.experts[dst].fc1.bias)
                if layer.experts[dst].fc2.bias is not None: changed.append(layer.experts[dst].fc2.bias)
                _zero_optim_moments_for(optimizer, changed)

    @torch.no_grad()
    def _do_merges(self, layer: SynapticMoE, optimizer: Optional[torch.optim.Optimizer]):
        pairs = self._pick_merge_pairs(layer)
        if self.cfg.verbose and len(pairs) > 0:
            print(f"[SplitMerge] Merging pairs: {pairs}")
        for i, j in pairs:
            # winner = healthier of the two
            health = self._health(layer)
            if health[i] >= health[j]:
                w,l = i,j
            else:
                w,l = j,i
            alpha = self._util_weight(layer, w, l)
            _merge_expert_into_and_clone_(layer, w, l, alpha, self.cfg)
            # zero optimizer moments for both experts + router rows
            if optimizer is not None:
                changed = [
                    layer.experts[w].fc1.w_slow, layer.experts[w].fc1.w_fast,
                    layer.experts[w].fc2.w_slow, layer.experts[w].fc2.w_fast,
                    layer.experts[l].fc1.w_slow, layer.experts[l].fc1.w_fast,
                    layer.experts[l].fc2.w_slow, layer.experts[l].fc2.w_fast,
                    layer.router.weight
                ]
                if layer.experts[w].fc1.bias is not None: changed.append(layer.experts[w].fc1.bias)
                if layer.experts[w].fc2.bias is not None: changed.append(layer.experts[w].fc2.bias)
                if layer.experts[l].fc1.bias is not None: changed.append(layer.experts[l].fc1.bias)
                if layer.experts[l].fc2.bias is not None: changed.append(layer.experts[l].fc2.bias)
                _zero_optim_moments_for(optimizer, changed)

    @torch.no_grad()
    def step(self, global_step: int, optimizer: Optional[torch.optim.Optimizer]=None):
        if not self.cfg.enabled: return
        if global_step < self.cfg.warmup_steps: return
        if global_step - self._last_step < self.cfg.min_step_interval: return

        if not _is_rank0():
            # Non-zero ranks just wait for broadcast after rank 0 modifies params
            if self.cfg.ddp_broadcast:  # ensure we hit the barrier roughly in sync
                if dist.is_available() and dist.is_initialized(): dist.barrier()
            return

        if self.cfg.verbose:
            print(f"[SplitMerge] step @ {global_step}")

        # Perform operations layer-by-layer on rank 0
        for layer in self._moe_layers:
            # 1) merges
            self._do_merges(layer, optimizer)
            # 2) splits
            sources = self._pick_split_sources(layer)
            if len(sources) > 0 and self.cfg.splits_per_call > 0:
                slots = self._weakest_slots(layer, min(len(sources), self.cfg.splits_per_call))
                if self.cfg.verbose:
                    print(f"[SplitMerge] Splitting {list(zip(sources,slots))}")
                self._split_into_slots(layer, sources, slots, optimizer)

        # Broadcast updated params to all ranks (DDP)
        if self.cfg.ddp_broadcast and dist.is_available() and dist.is_initialized():
            for layer in self._moe_layers:
                _broadcast_module_params(layer)
            dist.barrier()

        self._last_step = global_step
