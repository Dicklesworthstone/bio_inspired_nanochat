# Metacognition & Self-Model Layer — Design Note (beads `re4e.2`, `re4e.2.1`)

_Capability Frontier (`re4e`) · Introspective Alignment & Calibration. Author: GoldenRiver · 2026-08-24._

## Purpose & Scope

Standard autoregressive language models are famously uncalibrated and prone to overconfident hallucination: because they lack persistent internal state representations of epistemic uncertainty, their output softmax probabilities reflect token frequency and lexical patterns rather than ground-truth competence.

In `bio_inspired_nanochat`, the model maintains genuine physical state variables:
1. **Lyapunov Free Energy** $\mathcal{E}(h) = \frac{1}{2} \|h - \text{gate}(h)\|^2$ / $F = E - T \cdot S$ (Thrust A / `r00r.1`).
2. **Sheaf Inconsistency / Coboundary Obstruction** $\|\delta^0(s)\|_2$ across semantic spans (Thrust G / `r00r.5`).
3. **Presynaptic & Postsynaptic Plasticity Traces** (CaMKII, PP1, BDNF accumulation).

The **Metacognitive Self-Model Layer** fuses these multimodal internal state signals into a calibrated, tri-state epistemic competence report:
- **`KNOWN`** ($\mathcal{C} \ge \tau_{\text{known}}$): Low free energy, low sheaf obstruction, high sharpness.
- **`GUESSING`** ($\tau_{\text{unknown}} \le \mathcal{C} < \tau_{\text{known}}$): Intermediate energy / minor local sheaf obstruction (extrapolation).
- **`UNKNOWN`** ($\mathcal{C} < \tau_{\text{unknown}}$): High energy barrier, high coboundary obstruction, or collapse of synaptic activation.

---

## 1. Mathematical Formulation

For a generated span $X = (x_1, \dots, x_M)$ with hidden representations $H = (h_1, \dots, h_M)$, the competence score $\mathcal{C}(X) \in [0, 1]$ is computed as:

$$\mathcal{C}(X) = \sigma\left(w_0 + w_E \cdot \overline{\mathcal{E}}(H) + w_S \cdot \mathcal{O}_{\text{sheaf}}(H) + w_H \cdot \overline{\mathcal{H}}_{\text{entropy}}\right)$$

where:
- $\overline{\mathcal{E}}(H) = \frac{1}{M} \sum_{m=1}^M \mathcal{E}(h_m)$ is the average Lyapunov free-energy penalty.
- $\mathcal{O}_{\text{sheaf}}(H) = \|\delta^0(\mathbf{s})\|_2$ is the sheaf coboundary obstruction norm measuring pairwise inconsistency across span features.
- $\overline{\mathcal{H}}_{\text{entropy}} = \frac{1}{M \ln |\mathcal{V}|} \sum_{m=1}^M \mathcal{H}(p_m)$ is normalized token entropy.
- $w = (w_0, w_E, w_S, w_H)$ are calibrated logistic coefficients (optimized via Platt scaling on held-out calibration data).

---

## 2. Calibration & Evaluation Targets

1. **Expected Calibration Error (ECE)**:
   $$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right| \le 0.08$$
2. **AUROC vs Softmax Confidence Baseline**:
   $$\text{AUROC}(\text{Metacognitive Competence}) > \text{AUROC}(\text{Softmax Confidence}) + 0.05$$
   proving that internal biological state carries epistemic signal inaccessible to token logits alone.
3. **Three-Way Discrete Categorization**:
   - `KNOWN`: $\mathcal{C} \ge 0.75$
   - `GUESSING`: $0.35 \le \mathcal{C} < 0.75$
   - `UNKNOWN`: $\mathcal{C} < 0.35$

---

## 3. Downstream Composition

The metacognitive report feeds directly into:
- **`re4e.1` (Self-Correcting Loop)**: Spans labeled `GUESSING` or `UNKNOWN` trigger targeted deliberation and localized regeneration.
- **`re4e.10` (Conformal Certified Abstention)**: Guarantees finite-sample error rate $\le \alpha$ by abstaining when $\mathcal{C} < \tau_{\alpha}$.
