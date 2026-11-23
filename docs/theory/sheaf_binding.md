# Cellular Sheaf Binding & Operadic SNARE Routing — Theory Note & Assumptions Ledger (bead `0642.8.1`)

> **Thrust G Master Specification**: Cellular sheaf Laplacian $L = \delta^T \delta$, 1-cohomology obstruction $H^1$, operadic syntax tree routing, and provable compositional generalization.

---

## 1. Cellular Sheaf Model on the Token Graph

Let $\mathcal{G} = (V, E)$ be the relational dependency graph over tokens (or syntax tree nodes).
A **cellular sheaf** $(\mathcal{G}, \mathcal{F})$ associates:
1. Stalk vector spaces $\mathcal{F}(v) = \mathbb{R}^d$ for each vertex $v \in V$.
2. Edge stalk vector spaces $\mathcal{F}(e) = \mathbb{R}^d$ for each directed edge $e = (u, v) \in E$.
3. Linear restriction maps $\mathcal{F}_{u \trianglelefteq e}: \mathcal{F}(u) \to \mathcal{F}(e)$ and $\mathcal{F}_{v \trianglelefteq e}: \mathcal{F}(v) \to \mathcal{F}(e)$.

### 1.1 Cochain Spaces & Coboundary Operator

The 0-cochains $C^0(\mathcal{G}, \mathcal{F}) = \bigoplus_{v \in V} \mathcal{F}(v) \cong \mathbb{R}^{|V| \cdot d}$ assign local binding embeddings to each variable.  
The 1-cochains $C^1(\mathcal{G}, \mathcal{F}) = \bigoplus_{e \in E} \mathcal{F}(e) \cong \mathbb{R}^{|E| \cdot d}$ represent edge compatibility residuals.

The coboundary map $\delta: C^0(\mathcal{G}, \mathcal{F}) \to C^1(\mathcal{G}, \mathcal{F})$ is defined for $e = (u, v)$ by:
$$(\delta x)_e = \mathcal{F}_{v \trianglelefteq e}(x_v) - \mathcal{F}_{u \trianglelefteq e}(x_u)$$

---

## 2. Sheaf Laplacian & Cohomology Theorems

The **sheaf Laplacian** is the self-adjoint, positive semidefinite operator:
$$L = \delta^T \delta: C^0(\mathcal{G}, \mathcal{F}) \to C^0(\mathcal{G}, \mathcal{F})$$
Explicitly on block $(u, v)$:
$$L_{uu} = \sum_{e \ni u} \mathcal{F}_{u \trianglelefteq e}^T \mathcal{F}_{u \trianglelefteq e}, \quad L_{uv} = - \sum_{e = (u, v)} \mathcal{F}_{u \trianglelefteq e}^T \mathcal{F}_{v \trianglelefteq e}$$

### 2.1 Theorem 1 (0-Cohomology is the Space of Global Bindings)

$$H^0(\mathcal{G}, \mathcal{F}) = \ker(\delta) = \ker(L) = \{x \in C^0 : \delta x = 0\}$$
A 0-cochain $x$ satisfies the supplied restriction maps exactly when $Lx=0$, equivalently when its Dirichlet energy $x^T Lx=\|\delta x\|^2$ vanishes. Whether the binding yields the correct answer requires a separate target.

### 2.2 Theorem 2 (1-Cohomology as the Binding Obstruction)

$$H^1(\mathcal{G}, \mathcal{F}) = \operatorname{coker}(\delta) = C^1(\mathcal{G}, \mathcal{F}) / \operatorname{im}(\delta)$$
For a graph with no 2-cells, $\dim H^1 = \dim C^1-\operatorname{rank}(\delta)$.
A scalar identity sheaf on three vertices has a one-dimensional node kernel on
both a path and a triangle. The path has $\dim H^1=0$ and the triangle has
$\dim H^1=1$. An individual edge observation $b$ is compatible with some node
assignment exactly when $b\in\operatorname{im}\delta$. Determining its class
requires observing $b$. The implemented energy measures how well supplied node
features agree across edges. It can be positive even on the path.

### 2.3 Theorem 3 (Diffusion Convergence)

For step size $\eta < 2 / \lambda_{\max}(L)$, the discrete sheaf diffusion iteration:
$$x^{(t+1)} = (I - \eta L) x^{(t)}$$
converges exponentially to the orthogonal projection $\Pi_{\ker(L)}(x^{(0)})$ with rate:
$$\|x^{(t)} - \Pi_{\ker(L)}(x^{(0)})\| \le \max_{\lambda_i>0}|1-\eta\lambda_i|^t \|x^{(0)}\|.$$
For $0<\eta\le1/\lambda_{\max}$, the smallest positive eigenvalue determines
this factor. Larger stable steps can make the largest eigenvalue decisive.

---

## 3. Operadic Rab/SNARE Code-Based Routing

Vesicular SNARE receptors form an operadic colored tree algebra $\mathcal{O}_{\text{SNARE}}$:
$$\gamma: \mathcal{O}(k) \times \mathcal{O}(n_1) \times \dots \times \mathcal{O}(n_k) \to \mathcal{O}(n_1 + \dots + n_k)$$
`OperadicSNAREMatcher` scores the cosine affinity of two projected vectors. Its matched-code test establishes an ordering between that input pair and its sign reversal. Associative substitution and systematic language generalization remain separate empirical or mathematical obligations.

---

## 4. Proof Obligations, Assumptions & Failure Mode Ledger

| ID | Obligation / Theorem | Formal Statement | Proof / Verification Method | Fallback Behavior |
|:---|:---|:---|:---|:---|
| **OBL-G1** | Global Section Ker(L) | $L x = 0 \iff \delta x = 0$ | $x^T L x = \|\delta x\|^2 = 0$ | Eager linear projection |
| **OBL-G2** | Exponential diffusion | Positive-spectrum contraction factor from §2.3 | Both ends of the positive spectrum | Standard softmax attention |
| **OBL-G3** | Operadic Composition | Associative subtree substitution | Binary Hamming distance matching | Soft routing baseline |
| **OBL-G4** | AUROC Separation | $\text{AUROC} \ge 0.95$ on corrupted bindings | ROC curve over normalized Dirichlet energy | Monolithic transformer baseline |

### Failure Modes & Degradation Hierarchy
1. **Single-token / Disconnected Graph**: When $|E| = 0$, $L = 0$, $H^1 = 0$, and the monitor safely degrades to an identity passthrough.
2. **Spectral Collapse ($\lambda_2(L) \approx 0$)**: When the graph is disconnected into disjoint components, diffusion operates independently on each connected component's local sections.
3. **No-Op Toggle**: `enable_sheaf_binding=False` disables the monitor and diffusion layer completely with zero overhead.

---

## 5. Audit & Telemetry Specification

The node-monitor audit records the following quantities when its JSONL output is enabled.
- `step`: Global integer step
- `obstruction_score`: Per-sample Rayleigh quotient divided by $\lambda_{\max}(L)$, averaged across the batch.
- `spectral_gap`: Smallest non-zero eigenvalue $\lambda_2(L)$
- `dim_ker`: Dimension of $H^0$ global sections
- `is_certified`: The node operator has a numerical kernel and a positive spectral gap.
- `h1_dimension`: Edge-space dimension minus numerical rank, when supplied for a graph.
- `h1_obstruction`: Unknown for node-only observations. Output correctness requires independent labels.

## Held-out command composition

Eight paired seeds each trained two 21,310-parameter models on 228 synthetic commands.
Both models saw all twelve atomic commands. Seventy-two combinations of familiar atoms
were withheld. At the primary 400-update checkpoint the plain model solved 87/576
examples and three-step sheaf diffusion solved 52/576. The paired difference was
−6.08 percentage points (two-sided exact sign-flip p = 0.125).

The triangle's first cohomology had dimension four throughout. We measured the
specific class represented by the two predicted atomic action counts minus the predicted
composite count. All 52 correct diffusion outputs had zero residual, as did 32 wrong
outputs. Reversing a sequence preserves counts and can change its meaning. This
explains how a count-based score can rank errors well (mean per-seed AUROC 0.970)
while failing as a correctness certificate. The registered joint endpoint failed.
An independent recomputation checked 3,456 row-level predictions at all three budgets,
their 10,368 edge observations and all 256 paired sign assignments.
[Study](../../results/sheaf_compositional_2026-11-22.json) · [Raw predictions](../../results/sheaf_compositional_2026-11-22.jsonl.gz)
