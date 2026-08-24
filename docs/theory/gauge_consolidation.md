# Gauge-Theoretic Multi-Timescale Consolidation — Theory Note & Assumptions Ledger (bead `0642.7.1`)

> **Thrust D**: Gauge symmetry of the fast/slow decomposition, connection $\mathcal{A}$, holonomy bounds on catastrophic forgetting, and Fisher-geodesic schedules.

---

## 1. The Weight Bundle & Gauge Invariance

The total effective synaptic weight $W_{\text{total}} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ decomposes into:
$$W_{\text{total}} = W_{\text{slow}} + W_{\text{fast}} + U V$$
where $U \in \mathbb{R}^{D_{\text{in}} \times R}$ and $V \in \mathbb{R}^{R \times D_{\text{out}}}$.

### 1.1 Gauge Lie Group $G = GL(R) \times \mathbb{R}^D$

Under an internal transformation $(g, \Delta) \in G$:
$$U \mapsto U g, \quad V \mapsto g^{-1} V, \quad W_{\text{fast}} \mapsto W_{\text{fast}} - \Delta, \quad W_{\text{slow}} \mapsto W_{\text{slow}} + \Delta$$
The observable map $y = x W_{\text{total}}$ is strictly gauge-invariant:
$$W_{\text{total}} \mapsto (W_{\text{slow}} + \Delta) + (W_{\text{fast}} - \Delta) + (U g)(g^{-1} V) = W_{\text{total}}$$

---

## 2. Connection $\mathcal{A}$, Curvature $F$, and the Forgetting Bound

We define a gauge connection 1-form $\mathcal{A}$ transporting the fiber along training trajectories $\gamma(t)$:
$$\nabla^\mathcal{A} = d + \mathcal{A}$$
where $\mathcal{A}$ has non-trivial components determined by CaMKII/PP1 consolidation gating and BDNF scaling.

### 2.1 Curvature 2-Form

$$F = d\mathcal{A} + \mathcal{A} \wedge \mathcal{A}$$

### 2.2 Theorem (Holonomy Bound on Catastrophic Forgetting)

For a closed loop in task space $\gamma = \partial \Sigma$ (e.g. Task A $\to$ Task B $\to$ Task A):
$$\operatorname{Hol}_\gamma(\mathcal{A}) = \mathcal{P} \exp\left(-\oint_\gamma \mathcal{A}\right) = \mathcal{P} \exp\left(-\iint_\Sigma F\right)$$
The catastrophic forgetting delta on Task A satisfies:
$$\Delta \mathcal{L}_{\text{forgetting}} \le \frac{1}{2} \|\operatorname{Hol}_\gamma(\mathcal{A}) - I\|_{g_{\text{Fisher}}}^2 \le \frac{1}{2} \left(\iint_\Sigma \|F\|_{g_{\text{Fisher}}} \, d\sigma\right)^2$$
**Corollary**: If the connection $\mathcal{A}$ is flat ($F = 0$), holonomy is trivial and forgetting is mathematically zero ($\Delta \mathcal{L} = 0$).

---

## 3. Fisher-Geodesic Natural Gradient Consolidation

To minimize curvature along the transfer path:
$$\dot{W}(t) = - g_{\text{Fisher}}(W)^{-1} \nabla_W \mathcal{L}$$
where $g_{\text{Fisher}} = \mathbb{E}_{x \sim p_\theta}[\nabla \log p(x) \nabla \log p(x)^T]$.

### 3.1 Failure Modes & Fallbacks

1. **Ill-conditioned Fisher Matrix** ($\kappa(g_{\text{Fisher}}) > 10^6$):
   - Revert to diagonal empirical Fisher / Elastic Weight Consolidation (EWC) penalty:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \frac{\lambda_{\text{EWC}}}{2} \sum_i F_{ii} (W_i - W_i^*)^2$$
2. **Singular Rank-$R$ Modes**:
   - Tikhonov regularization $\mathcal{A}_{\text{reg}} = \mathcal{A} + \epsilon I_R$.

---

## 4. Proof Obligations & Assumptions Ledger

| ID | Claim / Guarantee | Formal Statement | Proof / Verification Method | Fallback on Breach |
|:---|:---|:---|:---|:---|
| **OBL-D1** | Behavioral Gauge Invariance | $W_{\text{total}}(g \cdot z) = W_{\text{total}}(z)$ | Exact matrix algebra | Fail-closed assertion |
| **OBL-D2** | Curvature Forgetting Bound | $\Delta \mathcal{L} \le \frac{1}{2} \iint \|F\|_{g} d\sigma$ | Stokes theorem on Riemann manifold | Diagonal EWC baseline |
| **OBL-D3** | Geodesic Consolidation | $\nabla_{\dot{\gamma}} \dot{\gamma} = 0$ in Fisher metric | Christoffel symbol solver | Standard momentum SGD |
