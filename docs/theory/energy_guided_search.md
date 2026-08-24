# Energy-Guided Tree Search & State-Space Rollouts — Design Note (beads `re4e.3`, `re4e.3.1`)

_Capability Frontier (`re4e`) · Physical Value Functions & System-2 Planning. Author: GoldenRiver · 2026-08-24._

## Purpose & Scope

Standard heuristic search (e.g. logprob beam search, best-of-$N$, Monte Carlo Tree Search with trained value heads) relies purely on token log-likelihoods or external discriminator models. However, log-likelihood is notoriously blind to epistemic coherence, global consistency, and internal tension.

In `bio_inspired_nanochat`, the synaptic transformer possesses an intrinsic physical scalar: the **Lyapunov Free Energy** $\mathcal{E}(h) = \frac{1}{2} \|h - \text{gate}(h)\|^2$ / $F = E - T \cdot S$.

**Energy-Guided Search (EGS)** uses internal state free energy as a principled physical value function $V(\mathbf{s}) = -F(\mathbf{s})$:
1. **Branching**: When per-token entropy or local energy exceeds threshold $\tau_{\text{branch}}$, expand top-$B$ token candidates.
2. **State-Space Rollouts**: Simulate fast-weight synaptic state updates $\mathbf{S}_{t+1} = \text{EMA}(\mathbf{S}_t, h_{t+1} h_{t+1}^T)$ across candidate paths without committing them.
3. **Energy-Pruning**: Prune high-energy branches whose Lyapunov barrier $\Delta \mathcal{E} > \tau_{\text{prune}}$.
4. **Value-Guided Terminal Scoring**: Rank complete rollouts by combined objective $J(\tau) = \log P(\tau) - \beta \cdot F_{\text{terminal}}(\tau)$.

---

## 1. Mathematical Formulation

Given a search node $u$ with prefix tokens $X_u$, hidden state $h_u$, and fast-weight synaptic matrix $\mathbf{S}_u$:
- **Candidate Expansion**: For next-token candidates $v \in \text{Top-K}(\pi(\cdot | X_u))$:
  $$h_v = f_{\theta}(h_u, v; \mathbf{S}_u)$$
  $$\mathcal{E}(h_v) = \frac{1}{2} \|h_v - \text{GELU}(h_v W_{\text{gate}})\|^2$$
- **Node Evaluation / Cumulative Cost**:
  $$G(v) = G(u) - \log P(v | X_u) + \lambda \cdot \mathcal{E}(h_v)$$
- **Pruning Criterion**: If $\mathcal{E}(h_v) > \mathcal{E}_{\text{cutoff}}$, prune node $v$ immediately.
- **Rollout Selection**: Best candidate chosen via $\arg\min_{v \in \text{Leaves}} G(v)$.

---

## 2. API & Deterministic Fallback

- `enabled: bool`: Flag to activate energy-guided search (default: False).
- `beam_width: int`: Number of parallel search trajectories $B \in [1, 16]$.
- `branching_factor: int`: Expansion candidates per step $K \in [1, 8]$.
- `energy_weight: float`: Penalty coefficient $\lambda \ge 0.0$.
- **Fallback**: When `enabled=False` or `energy_weight=0.0`, the search collapses deterministically to standard logprob beam search with zero overhead.
