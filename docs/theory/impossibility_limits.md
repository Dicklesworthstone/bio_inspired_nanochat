# Information-Theoretic & Thermodynamic Impossibility Results (bead `r00r.14`)

> **Scope**: Mathematical impossibility theorems and lower bounds establishing fundamental limits on biological neural network mechanisms.

---

## 1. Thermodynamic Uncertainty Relation (TUR) Precision Floor

### Theorem 1 (Synaptic TUR Lower Bound)

For any non-equilibrium continuous-time Markovian synaptic transmission process with net current $j$ (vesicle release flux) and steady-state entropy production rate $\dot{S}_{\text{ep}}$:
$$\frac{\operatorname{Var}(j)}{\langle j \rangle^2} \ge \frac{2 k_B}{\Delta S_{\text{ep}}}$$
where $\Delta S_{\text{ep}} = \dot{S}_{\text{ep}} \cdot \Delta t$ is the total thermodynamic dissipation over integration window $\Delta t$.

**Physical Implication**: Uncertainty estimation precision cannot be made arbitrarily small without burning proportional metabolic energy (ATP hydrolysis). Zero-uncertainty requires infinite entropy production.

---

## 2. Rate-Distortion & Memory Capacity Bound for Fast-Weights

### Theorem 2 (Exponential Memory Decay Under Finite Leakage)

Consider a linear fast-weight recurrence with decay factor $\lambda \in [0, 1)$ and input variance $\sigma_x^2$:
$$W_t = \lambda W_{t-1} + \eta x_t y_t^T + \xi_t, \quad \xi_t \sim \mathcal{N}(0, \sigma_\xi^2 I)$$
The mutual information $I(X_{1:t-H}; W_t)$ concerning tokens $H$ steps in the past satisfies:
$$I(X_{1:t-H}; W_t) \le \frac{D_{\text{in}} D_{\text{out}}}{2} \lambda^{2H} \log\left(1 + \frac{\eta^2 \sigma_x^2 \sigma_y^2}{(1 - \lambda^2) \sigma_\xi^2}\right)$$
For any fixed capacity threshold $C > 0$, the effective working memory retention horizon is strictly bounded by:
$$H \le \frac{1}{2 \ln(1/\lambda)} \ln\left(\frac{D_{\text{in}} D_{\text{out}}}{2 C} \log\left(1 + \frac{\eta^2 \sigma_x^2 \sigma_y^2}{(1 - \lambda^2) \sigma_\xi^2}\right)\right) = O\left(\frac{1}{1 - \lambda}\right)$$

**Algorithmic Implication**: Purely leaky online fast-weights cannot maintain working memory beyond $O(1/(1-\lambda))$ tokens without nonlinear bistable latches (CaMKII/PP1 cusp in Thrust F) or discrete consolidation into slow weights (Thrust D).
