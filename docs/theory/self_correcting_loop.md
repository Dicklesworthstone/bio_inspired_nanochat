# Self-Correcting Generation Loop — Design Note & Termination Proof (beads `re4e.1`, `re4e.1.1`)

_Capability Frontier (`re4e`) · Emergent Compositions & Bounded Repair. Author: GoldenRiver · 2026-08-24._

## Purpose & Scope

The **Self-Correcting Generation Loop** composes:
1. **Sheaf Obstruction Detection** (`r00r.5`): Fast detection of representation discontinuities via a fixed-sheaf coboundary residual. This MVP detector is not an H¹ or semantic-correctness certificate.
2. **Span Localization**: Precise isolation of the offending token span $[t_{\text{start}}, t_{\text{end}}]$.
3. **Causal Deliberation & Localized Regeneration** (`r00r.1` / `r00r.15`): Free-energy gradient relaxation and constrained resampling over the corrupted span.
4. **Re-Checking & Bounded Abstention**: Post-regeneration obstruction check. If the residual remains above threshold after $N_{\text{max}}$ attempts, the loop terminates with an `ABSTAIN` token. This bounded fallback is separate from the conformal certificate implemented by `re4e.10`.

---

## 1. Closed Control Loop Algorithm

```text
Algorithm: Detect-Deliberate-Regenerate-Recheck (D2R2)
Input: Prompt tokens X_0, Model M, Sheaf Detector D_sheaf, Max Attempts N_max, Budget K_delib

1. Generate initial sequence X = (x_1, ..., x_T).
2. For attempt n in 1 .. N_max:
   a. Compute the fixed-sheaf obstruction measurement:
      C = D_sheaf.detect_inconsistencies(X, M.hidden_states(X))
   b. If not C.detected_obstruction:
      return X, Status.NO_OBSTRUCTION_DETECTED
   c. Locate maximal obstruction span [t_start, t_end] = C.corrupted_span.
   d. Rewind sequence to t_start - 1.
   e. Run full-state causal deliberation on prefix state h_{t_start - 1} with budget K_delib.
   f. Regenerate replacement span (x'_t_start, ..., x'_t_end) from the relaxed state.
   g. Reconstruct candidate sequence X' = X[1:t_start - 1] + X'_span + X[t_end + 1:T].
   h. Update X <- X'.
3. Return the configured abstention token, Status.ABSTAIN
```

---

## 2. Termination & Non-Oscillation Proof

### Theorem 1 (Strict Finite-Time Termination)
For any input prompt and finite generation length $T$, the self-correcting loop terminates in at most $N_{\text{max}}$ iterations, requiring at most $O(N_{\text{max}} \cdot (T + K_{\text{delib}}))$ total FLOPs.

**Proof**:
1. Each iteration $n \in [1, N_{\text{max}}]$ executes a finite forward pass of length $\le T$ and a bounded deliberation loop of at most $K_{\text{delib}}$ steps.
2. The outer loop counter $n$ strictly increments on each attempt.
3. If $n = N_{\text{max}}$ without achieving $\|\delta^0(s)\|_2 \le \tau$, the loop breaks unconditionally and returns the `ABSTAIN` status when abstention is enabled.
4. Thus, infinite loops and runtime divergence are strictly impossible. $\blacksquare$

### Deliberation Energy Check

The current relaxation takes fixed-size gradient steps on the learned quadratic energy proxy $\mathcal{E}(h)$. Tests check descent for the configured reference setting; the implementation does not claim that lower proxy energy certifies semantic coherence.

---

## 3. Configuration & API Interface

- `max_repair_attempts`: Integer $N_{\text{max}} \in [1, 5]$ (default: 3).
- `obstruction_threshold`: Sheaf threshold $\tau_{\text{obstruction}} \in (0, 1]$ (default: 0.40).
- `deliberation_budget`: Relaxation steps $K \in [0, 16]$ (default: 4).
- `abstain_on_exhaustion`: Boolean (default: True) returning `ABSTAIN` payload if unrepairable.
