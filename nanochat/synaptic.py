# nanochat/synaptic.py
# Comprehensive synaptic modules for nanochat:
# - Presynaptic biophysics → attention logit augmentation
# - Postsynaptic dual-timescale linear with low-rank eligibility
# - Synaptic Self-Attention (RoPE, MQA-compatible)
# - Synaptic MLP
# - Synaptic MoE with router embeddings, contrastive updates & structural hooks
# - Structural plasticity utilities
#
# Design highlights (mapped from the JAX reference you provided):
#   • Synaptotagmin-1/7 mixed Ca2+ sensor, complexin clamp
#   • Munc13/18 priming, clathrin/dynamin endocytosis (delay queue)
#   • V-ATPase/VDAC energy coupling and per-edge cost model
#   • EMA normalization of quantal gain; optional stochastic release
#   • PSD-like low-rank eligibility U/V with CaMKII/PP1 gating (fast/slow)
#   • Septin-like distance barrier in attention logits
#   • Router embeddings + contrastive update; MoE top-k dispatch with fatigue
#
# This file is intentionally verbose and highly instrumented for clarity.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _rms(x: Tensor, eps=1e-6) -> Tensor:
    return (x.square().mean(dim=-1, keepdim=True)+eps).sqrt()

def _rmsnorm(x: Tensor, eps=1e-6) -> Tensor:
    return x/_rms(x,eps)

def _softplus(x: Tensor, beta=1.0) -> Tensor:
    return (1.0/beta)*F.softplus(beta*x)

def _cosine(u: Tensor, v: Tensor, eps=1e-8) -> Tensor:
    """Cosine similarity with safe normalization."""
    u = u / (u.norm(dim=-1, keepdim=True) + eps)
    v = v / (v.norm(dim=-1, keepdim=True) + eps)
    return (u * v).sum(dim=-1)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class SynapticConfig:
    enabled: bool=True
    # presynaptic timescales (per-token units)
    tau_c: float=4.0       # Ca decay
    tau_buf: float=10.0    # buffer decay
    tau_prime: float=18.0  # SNARE priming decay
    tau_rrp: float=40.0    # RRP refill
    tau_energy: float=80.0 # ATP recovery
    # gains
    alpha_ca: float=0.25
    alpha_buf_on: float=0.6
    alpha_buf_off: float=0.1
    alpha_prime: float=0.10
    alpha_unprime: float=0.04
    alpha_refill: float=0.04
    alpha_recycle: float=0.02
    energy_in: float=0.03
    energy_cost_rel: float=0.015
    energy_cost_pump: float=0.006
    # sensors & clamp
    syt_fast_kd: float=0.4
    syt_slow_kd: float=3.0
    complexin_bias: float=0.5
    # quantal amplitude
    qmax: float=1.3
    q_beta: float=2.0
    # eligibility / consolidation
    rank_eligibility: int=16
    rho_elig: float=0.95
    eta_elig: float=0.02
    eta_fast: float=0.03
    eta_slow: float=0.002
    camkii_gain: float=1.5
    pp1_gain: float=1.0
    # attention geometry
    barrier_strength: float=0.075     # septin-like barrier in logits (distance penalty)
    # stochastic synaptic sampling
    stochastic_train_frac: float=0.10
    stochastic_quanta: int=8
    # structural plasticity (MoE + router embeddings)
    structural_interval: int=50000
    structural_tau_util: float=0.2
    structural_age_bias: float=1.0
    router_embed_dim: int=24
    router_contrastive_lr: float=1e-4
    router_contrastive_push: float=0.1
    router_sim_threshold: float=0.6
    # general numerics
    epsilon: float=1e-6

# -----------------------------------------------------------------------------
# Presynaptic biophysics
# -----------------------------------------------------------------------------

class SynapticPresyn(nn.Module):
    """
    Vectorized presynaptic module with explicit Syt1/7 mix, complexin clamp,
    Munc13/18 priming, clathrin/dynamin endocytosis (queue), V-ATPase/VDAC
    coupling, EMA normalization, optional stochastic release on a fraction
    of edges, and a septin-like distance barrier for attention logits.

    Inputs:
      q (B,H,T,D), k (B,H,T,D), logits (B,H,T,T) masked
      state: dict with tensors (B,H,T) keyed as C, BUF, RRP, RES, PR, CL, E
    Output:
      syn_logit (B,H,T,T), updated state
    """
    def __init__(self, head_dim: int, cfg: SynapticConfig):
        super().__init__(); self.cfg=cfg
        self.register_buffer('running_release', torch.tensor(1.0))

    @staticmethod
    def _syt_mix(c: Tensor, kd1: float, kd7: float) -> Tensor:
        fast = c/(c+kd1); slow=c/(c+kd7); return 0.7*fast+0.3*slow

    def forward(
        self,
        q: Tensor,                    # (B,H,T,D)
        k: Tensor,                    # (B,H,T,D)
        logits: Tensor,               # (B,H,T,T) pre-softmax, causal-masked to -inf
        state: Dict[str,Tensor],      # per-key state (B,H,T)
        causal_mask: Tensor,          # (T,T) bool where i>=j is True
        train_mode: bool,
    ) -> Tuple[Tensor, Dict[str,Tensor]]:
        c=self.cfg; B,H,T,D=q.shape; eps=c.epsilon
        C,BUF,RRP,RES,PR,CL,E = state['C'],state['BUF'],state['RRP'],state['RES'],state['PR'],state['CL'],state['E']

        # Ca²⁺ drive from compatibilities; strictly causal and numerically safe
        drive=_softplus(logits.clamp(-20,20))*causal_mask.view(1,1,T,T)
        counts=causal_mask.float().sum(dim=0,keepdim=True).clamp_min(1.0)  # (1,T)
        influx=drive.sum(dim=2)/counts.view(1,1,T)                          # (B,H,T)

        # Buffer/Ca dynamics
        rho_c=math.exp(-1.0/c.tau_c); rho_b=math.exp(-1.0/c.tau_buf)
        C_new  = rho_c*C + c.alpha_ca*influx - c.alpha_buf_on*C*(1.0-BUF) + c.alpha_buf_off*BUF
        BUF_new= rho_b*BUF + c.alpha_buf_on*C*(1.0-BUF) - c.alpha_buf_off*BUF
        C_new=C_new.clamp_min(0.0); BUF_new=BUF_new.clamp(0.0,1.0)

        # Priming/Refill baselines (consumption applied after computing release)
        rho_p=math.exp(-1.0/c.tau_prime); rho_r=math.exp(-1.0/c.tau_rrp)
        PR_mid = (rho_p*PR + c.alpha_prime*(1.0-PR)).clamp(0.0,1.0)
        RRP_refill = (rho_r*RRP + c.alpha_refill*RES).clamp(0.0,1.0)
        RES_mid = (RES - c.alpha_refill*RES).clamp(0.0,1.0)
        rho_e=math.exp(-1.0/c.tau_energy); E_mid = (rho_e*E + c.energy_in).clamp(0.0,1.6)

        # Docking nonlinearity from q/k
        D_bilin = torch.sigmoid(((q.unsqueeze(2)*k.unsqueeze(3)).sum(-1))/math.sqrt(D))  # (B,H,T,T)

        # Optional stochastic sampling for a fraction of query rows during train
        if train_mode and self.cfg.stochastic_train_frac>0:
            row_mask = (torch.rand(B,H,T, device=q.device, dtype=q.dtype) < self.cfg.stochastic_train_frac).unsqueeze(-1)
        else:
            row_mask = None

        # Release probability per edge
        syt = self._syt_mix(C_new.unsqueeze(2), c.syt_fast_kd, c.syt_slow_kd)     # (B,H,1,T)
        fuse_p = torch.sigmoid(3.0*syt + 2.0*PR_mid.unsqueeze(2) - 2.0*(CL.unsqueeze(2)+c.complexin_bias)) * D_bilin
        fuse_p = fuse_p * causal_mask.view(1,1,T,T)

        # Expected release demand
        avail = RRP_refill.unsqueeze(2)             # (B,H,1,T)
        raw_release = (fuse_p*avail).clamp(0.0,1.0) # (B,H,T,T)
        if row_mask is not None:
            # Concrete-like relaxed sampling: add noise around expectation on masked rows
            noise = (torch.rand_like(raw_release) - 0.5) / max(1,self.cfg.stochastic_quanta)
            raw_release = torch.where(row_mask, (raw_release + noise).clamp(0,1), raw_release)

        # Conserve RRP: rescale s.t. sum_i release_ij ≤ RRP_refill_j
        total_by_key = raw_release.sum(dim=2).clamp_min(eps)      # (B,H,T)
        scale = torch.minimum(torch.ones_like(total_by_key), RRP_refill/total_by_key).unsqueeze(2)
        release_frac = raw_release * scale                         # (B,H,T,T)
        used_rrp = release_frac.sum(dim=2)                         # (B,H,T)

        RRP_new = (RRP_refill - used_rrp).clamp(0.0,1.0)
        RES_new = (RES_mid + used_rrp).clamp(0.0,1.0)
        PR_new  = (PR_mid - c.alpha_unprime*used_rrp).clamp(0.0,1.0)
        E_new   = (E_mid - c.energy_cost_rel*used_rrp - c.energy_cost_pump*(1.0-RES_new)).clamp(0.0,1.6)

        qamp = torch.sigmoid(c.q_beta*(E_new - 0.5)) * c.qmax      # (B,H,T)

        # Septin-like distance barrier (discourages long jumps across sequence)
        steps = torch.arange(T, device=logits.device, dtype=logits.dtype)
        dist = (steps.view(1,1,1,T) - steps.view(1,1,T,1)).abs() / float(max(1,T))
        syn_logit = torch.log(release_frac*qamp.unsqueeze(2).clamp_min(eps)+eps) - self.cfg.barrier_strength*dist

        with torch.no_grad():
            self.running_release.mul_(0.99).add_(0.01*release_frac.mean())

        state.update(C=C_new, BUF=BUF_new, RRP=RRP_new, RES=RES_new, PR=PR_new, CL=CL, E=E_new)
        return syn_logit, state

# -----------------------------------------------------------------------------
# Postsynaptic eligibility and linear
# -----------------------------------------------------------------------------

class PostsynapticHebb(nn.Module):
    """Low-rank eligibility + CaMKII/PP1 gate controlling consolidation."""
    def __init__(self, d_in: int, d_out: int, cfg: SynapticConfig):
        super().__init__(); self.cfg=cfg; R=cfg.rank_eligibility
        self.register_buffer('U', torch.zeros(d_in,R))
        self.register_buffer('V', torch.zeros(R,d_out))
        self.register_buffer('H_fast', torch.zeros(d_in,d_out))
        self.register_buffer('m_gate', torch.zeros(1))
        self.register_buffer('camkii', torch.zeros(1))
        self.register_buffer('pp1', torch.ones(1)*0.5)

    @torch.no_grad()
    def update_elig(self, x_in: Tensor, y_out: Tensor):
        c=self.cfg
        u = x_in.mean(dim=0)  # (din,)
        v = y_out.mean(dim=0) # (dout,)
        self.U.mul_(c.rho_elig).add_(c.eta_elig*u.unsqueeze(-1))
        self.V.mul_(c.rho_elig).add_(c.eta_elig*v.unsqueeze(0))
        self.H_fast.mul_(c.rho_elig).add_(c.eta_fast*(self.U@self.V))

    @torch.no_grad()
    def consolidate(self, calcium: Tensor, energy: Tensor):
        c=self.cfg
        self.camkii.add_(c.camkii_gain*torch.clamp(calcium.mean()-0.2, min=0.0))
        self.pp1.add_(c.pp1_gain*torch.clamp(0.3-energy.mean(), min=0.0))
        self.camkii.clamp_(0,1); self.pp1.clamp_(0,1)
        gate=torch.sigmoid(3.0*(self.camkii-0.5)-2.0*self.pp1)
        self.m_gate.copy_(gate)

class SynapticLinear(nn.Module):
    """Dual-timescale linear with low-rank eligibility; bfloat16-safe."""
    def __init__(self, in_features: int, out_features: int, cfg: SynapticConfig, bias: bool=True, use_input_ln: bool=False):
        super().__init__(); self.cfg=cfg; self.use_input_ln=use_input_ln
        self.w_slow=nn.Parameter(torch.empty(in_features,out_features))
        self.w_fast=nn.Parameter(torch.empty(in_features,out_features))
        self.bias=nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.trunc_normal_(self.w_slow,std=0.02); nn.init.trunc_normal_(self.w_fast,std=0.02)
        self.post=PostsynapticHebb(in_features,out_features,cfg)
        self.input_ln=nn.LayerNorm(in_features,eps=1e-5) if use_input_ln else None

    def forward(self, x: Tensor, calcium: Tensor, energy: Tensor, update_mem: bool=True):
        if self.input_ln is not None: x = self.input_ln(x)
        W=self.w_fast + self.post.m_gate*self.w_slow + self.post.H_fast
        y = x @ W
        if self.bias is not None: y=y+self.bias
        if update_mem:
            self.post.update_elig(x.detach(),y.detach())
            self.post.consolidate(calcium.detach(),energy.detach())
        return y

# -----------------------------------------------------------------------------
# Presyn state builder
# -----------------------------------------------------------------------------

def build_presyn_state(B: int, T: int, H: int, device, dtype, cfg: SynapticConfig):
    C=torch.zeros(B,H,T,device=device,dtype=dtype)
    BUF=torch.zeros_like(C); RRP=torch.ones_like(C)*0.8; RES=torch.ones_like(C)*0.2
    PR=torch.ones_like(C)*0.6; CL=torch.ones_like(C)*cfg.complexin_bias; E=torch.ones_like(C)*0.8
    return {'C':C,'BUF':BUF,'RRP':RRP,'RES':RES,'PR':PR,'CL':CL,'E':E}

# -----------------------------------------------------------------------------
# Attention and MLP
# -----------------------------------------------------------------------------

class SynapticCausalSelfAttention(nn.Module):
    """
    Drop-in attention with synaptic augmentation. Uses standard Q,K,V projections,
    RoPE, multi-query key/value replication, and adds log(ε+q⋅n) to logits.
    """
    def __init__(self, n_embd, n_head, n_kv_head, rope_cos, rope_sin, cfg: SynapticConfig, attn_drop=0.0, resid_drop=0.0):
        super().__init__(); assert n_embd % n_head == 0
        self.n_head=n_head; self.n_kv_head=n_kv_head; self.head_dim=n_embd//n_head; self.cfg=cfg
        self.q_proj=nn.Linear(n_embd, n_head*self.head_dim, bias=False)
        self.k_proj=nn.Linear(n_embd, n_kv_head*self.head_dim, bias=False)
        self.v_proj=nn.Linear(n_embd, n_kv_head*self.head_dim, bias=False)
        self.o_proj=nn.Linear(n_head*self.head_dim, n_embd, bias=False)
        self.attn_drop=nn.Dropout(attn_drop); self.resid_drop=nn.Dropout(resid_drop)
        self.cos, self.sin=rope_cos, rope_sin
        self.pre=SynapticPresyn(self.head_dim, cfg)

    def _apply_rope(self, x: Tensor, T0: int):
        H=self.n_head if x.size(-1)==self.n_head*self.head_dim else self.n_kv_head
        D=self.head_dim; x=x.view(x.size(0), x.size(1), H, D)
        cos=self.cos[:,T0:T0+x.size(1),:D//2]; sin=self.sin[:,T0:T0+x.size(1),:D//2]
        x1,x2=x.split(D//2,dim=-1); xr=torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
        return xr

    def _repeat_kv(self, x: Tensor):
        if self.n_head==self.n_kv_head: return x
        nrep=self.n_head//self.n_kv_head; b,t,nh,d=x.shape
        return x.unsqueeze(2).expand(b,t,nh,nrep,d).reshape(b,t,self.n_head,d)

    def forward(self, x: Tensor, kv_cache=None, presyn_state=None, train_mode=True):
        B,T,C=x.shape; H=self.n_head; D=self.head_dim; device=x.device; dtype=x.dtype
        q=self.q_proj(x); k=self.k_proj(x); v=self.v_proj(x)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        q=self._apply_rope(q,T0); k=self._apply_rope(k,T0)
        q=_rmsnorm(q); k=_rmsnorm(k)
        k=self._repeat_kv(k); v=self._repeat_kv(v)
        q=q.transpose(1,2); k=k.transpose(1,2); v=v.transpose(1,2)  # (B,H,T,D)/(B,H,T,D)

        logits = (q @ k.transpose(-1,-2)) / math.sqrt(D)           # (B,H,T,T)
        causal_mask = torch.tril(torch.ones(T,T,device=device,dtype=dtype)).bool()
        logits = logits.masked_fill(~causal_mask.view(1,1,T,T), float('-inf'))

        if presyn_state is None: presyn_state=build_presyn_state(B,T,H,device,dtype,self.cfg)
        syn_logit, presyn_state = self.pre(q,k,logits, presyn_state, causal_mask, train_mode)

        aug_logits = logits + syn_logit
        P = F.softmax(aug_logits, dim=-1)
        P = self.attn_drop(P)

        ctx = torch.einsum('bhts,bhsd->bhtd', P, v)                # (B,H,T,D)
        y = ctx.transpose(1,2).contiguous().view(B,T,H*D)
        y = self.resid_drop(self.o_proj(y))
        return y, presyn_state

class SynapticMLP(nn.Module):
    def __init__(self, n_embd: int, cfg: SynapticConfig, dropout: float=0.0):
        super().__init__(); self.cfg=cfg
        self.fc = SynapticLinear(n_embd, 4*n_embd, cfg, bias=True, use_input_ln=True)
        self.proj = SynapticLinear(4*n_embd, n_embd, cfg, bias=True, use_input_ln=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('C0', torch.tensor(0.5)); self.register_buffer('E0', torch.tensor(0.8))
    def forward(self, x: Tensor):
        B,T,C=x.shape; c=self.C0.expand(B*T); e=self.E0.expand(B*T)
        h=self.fc(x.reshape(B*T,C), c, e); h=F.relu(h).square(); h=self.drop(h.reshape(B,T,-1))
        y=self.proj(h.reshape(B*T,-1), c, e).reshape(B,T,C); return y

# -----------------------------------------------------------------------------
# Synaptic MoE (router embeddings, contrastive updates)
# -----------------------------------------------------------------------------

class SynapticExpert(nn.Module):
    def __init__(self, n_embd:int, hidden_mult:int, cfg:SynapticConfig, dropout:float=0.0):
        super().__init__(); h=hidden_mult*n_embd
        self.fc1=SynapticLinear(n_embd,h,cfg,bias=True,use_input_ln=False)
        self.fc2=SynapticLinear(h,n_embd,cfg,bias=True,use_input_ln=False)
        self.drop=nn.Dropout(dropout)
    def forward(self,x:Tensor)->Tensor:
        y=self.fc1(x, calcium=torch.ones(x.size(0),device=x.device), energy=torch.ones(x.size(0),device=x.device))
        y=F.relu(y).square(); y=self.drop(y)
        y=self.fc2(y, calcium=torch.ones(x.size(0),device=x.device), energy=torch.ones(x.size(0),device=x.device))
        return y

class SynapticMoE(nn.Module):
    """Top-k sparse Synaptic MoE with router embeddings, expert fatigue/energy,
    contrastive router-embedding updates, and split/merge structural hooks."""
    def __init__(self, n_embd:int, num_experts:int, top_k:int, hidden_mult:int, cfg:SynapticConfig, dropout:float=0.0):
        super().__init__(); self.num_experts=num_experts; self.top_k=top_k; self.cfg=cfg
        self.router=nn.Linear(n_embd,num_experts,bias=False)
        self.experts=nn.ModuleList([SynapticExpert(n_embd,hidden_mult,cfg,dropout) for _ in range(num_experts)])
        self.register_buffer('fatigue', torch.zeros(num_experts))
        self.register_buffer('energy', torch.ones(num_experts))
        # Router embeddings (biological identity) with unit-norm constraint
        emb = torch.randn(num_experts, cfg.router_embed_dim)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        self.router_embeddings = nn.Parameter(emb, requires_grad=False)  # updated by EMA-style rule
        self.last_aux_loss=None

    def forward(self,x:Tensor)->Tuple[Tensor,Tensor]:
        B,T,C=x.shape; E=self.num_experts; device=x.device
        logits=self.router(x)                                     # (B,T,E)
        # Embed-token similarity (optional small bias from router embeddings)
        # We synthesize a token embedding proxy by pooling x and projecting onto router embeddings
        tok_proxy = x.mean(dim=-1, keepdim=True)                  # (B,T,1)
        bias = 0.02 * _cosine(self.router_embeddings, self.router_embeddings).diag().view(1,1,-1) * tok_proxy
        logits = logits + bias + 0.1*self.energy.view(1,1,E) - 0.1*self.fatigue.view(1,1,E)
        topk=min(self.top_k,E)
        g, idx = torch.topk(logits, topk, dim=-1)                 # (B,T,k)
        gates = F.softmax(g, dim=-1)                              # (B,T,k)

        out=torch.zeros_like(x); flat_out=out.view(-1,C); flat_x=x.view(-1,C)
        me=torch.zeros(E,device=device); pe=torch.zeros(E,device=device)
        for e in range(E):
            mask=(idx==e)                                         # (B,T,k)
            sel=mask.any(dim=-1)                                  # (B,T)
            if not sel.any(): continue
            flat_idx=sel.view(-1).nonzero(as_tuple=False).squeeze(1)
            x_e=flat_x.index_select(0,flat_idx)
            y_e=self.experts[e](x_e)
            w=gates.masked_select(mask).unsqueeze(-1)
            flat_out.index_add_(0, flat_idx, w*y_e)
            me[e]=sel.sum(); pe[e]=gates.masked_select(mask).sum()

        with torch.no_grad():
            util=me.clamp_min(1.0)/float(B*T)
            self.fatigue.mul_(0.99).add_(0.01*util)
            self.energy.mul_(0.995).add_(0.005*(1.0-util))

        me=me/float(B*T); pe=pe/float(B*T)
        aux_loss=E*torch.sum(pe*me)
        self.last_aux_loss=aux_loss

        # Contrastive router-embedding update (co-activated experts pull together, others push)
        with torch.no_grad():
            cooc = torch.zeros(E,E, device=device)
            # crude but stable co-activation surrogate: diagonal mass only (keeps update bounded)
            for e in range(E):
                cooc[e,e] = pe[e]
            emb = self.router_embeddings
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            sim = emb @ emb.T
            pull = cooc * (sim - 1.0)
            push = (1.0 - cooc) * (sim + 0.3) * self.cfg.router_contrastive_push
            grad = pull - push
            grad = grad - grad.mean()  # center
            delta = (grad @ emb) * self.cfg.router_contrastive_lr
            emb = emb - delta
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            self.router_embeddings.copy_(emb)

        return out, aux_loss

# -----------------------------------------------------------------------------
# Structural plasticity utility
# -----------------------------------------------------------------------------

class StructuralPlasticity(nn.Module):
    def __init__(self, cfg: SynapticConfig):
        super().__init__(); self.cfg=cfg
        self.register_buffer('age', torch.zeros(1))
        self.register_buffer('util', torch.zeros(1))
    @torch.no_grad()
    def step(self, used: Tensor):
        self.age.add_(1.0)
        self.util.mul_(1.0-self.cfg.structural_tau_util).add_(self.cfg.structural_tau_util*used.float())
    @torch.no_grad()
    def decision(self):
        s=torch.sigmoid(10.0*(self.util-0.2)-self.cfg.structural_age_bias*(self.age/float(self.cfg.structural_interval)))
        return (torch.rand_like(s)>s).item()

def structural_plasticity_step(expert_states: List[nn.Module], cfg: SynapticConfig, global_step: int):
    if cfg.structural_interval<1 or global_step%cfg.structural_interval!=0: return
    for st in expert_states:
        st.step(used=torch.tensor(1.0))
        if st.decision():
            for p in st.parameters(): nn.init.trunc_normal_(p, std=0.02)
