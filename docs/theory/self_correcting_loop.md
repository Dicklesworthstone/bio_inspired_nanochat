# Self-Correcting Generation Loop — Design Note & Termination Proof (beads `re4e.1`, `re4e.1.1`)

_Capability Frontier (`re4e`) · Emergent Compositions & Bounded Repair. Author: GoldenRiver · 2026-08-24._

## Purpose & Scope

The generator measures a fixed-sheaf residual along adjacent tokens in a model-generated completion. A flagged result selects a bounded span around the largest edge residual. Deliberation relaxes the prefix representation, resamples the span and measures the new candidate. An exhausted repair budget can yield an abstention. A completion with fewer than two tokens has no edge and returns `UNCHECKED`. Correctness requires an independent answer label. [Generator](../../bio_inspired_nanochat/self_correcting_generator.py)

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

## 2. Repair-attempt and model-call bounds

For a valid prompt of length $P$ and $T\ge2$ generated tokens, let $R$ be the attempt cap, $L$ the maximum repair span and $K$ the per-token deliberation budget. The initial draft takes $T$ backbone calls. An attempt can make two full-candidate checks and regenerate at most $\min(L,T)$ tokens. The whole call therefore uses at most $T + R(2+\min(L,T))$ backbone evaluations and $RK\min(L,T)$ relaxation steps. Each backbone evaluation may process up to $P+T$ input positions. Operation counts depend on those lengths and the model architecture.

The outer counter advances on every attempt. Assuming each underlying call returns, the generator terminates by the cap even if successive repairs alternate between earlier answers. Finite iteration counts alone give no guarantee of numerical stability or answer correctness. [Generation and repair loop](../../bio_inspired_nanochat/self_correcting_generator.py)

### Deliberation Energy Check

The current relaxation takes fixed-size gradient steps on $\mathcal{E}(h)=\tfrac12\lVert g(h)-h\rVert^2$, where $g$ is a nonlinear gate. A freshly constructed controller initializes this gate separately from the language model. The reference tests measure descent of this proxy. [Controller and energy calculation](../../bio_inspired_nanochat/causal_deliberation.py)

---

## 3. Configuration & API Interface

- `max_repair_attempts` is a positive integer and defaults to 3.
- `obstruction_threshold` lies in $(0,1]$ and defaults to 0.40.
- `deliberation_budget` is a nonnegative integer number of relaxation steps and defaults to 4.
- `abstain_on_exhaustion` defaults to true and returns an abstention when all allowed attempts fail the detector check.

## Paired evaluation on trained binding answers

**Self-correction returned fewer correct answers and exceeded the registered compute budget.** Eight independently initialized two-layer GPTs trained on 2,048 variable-binding prompts apiece. Each prompt contained three variable/value bindings and four distractors. A correct completion supplied the queried value followed by a separator. Training supervised both positions, giving the detector one edge to inspect. Each seed contributed 32 new prompts. All 256 held-out prompts were distinct and excluded from every training stream. Each generation pair started from the same model, prompt and RNG seed, and its initial tokens matched exactly. The detector used threshold 0.40, at most two repair attempts and at most four relaxation steps per repaired token. The controller's relaxation gate was independently initialized. [Evaluation and frozen settings](../../scripts/e2e/self_correction_eval.py)

| Measured on the same 256 prompts | Single pass | Self-correction |
|---|---:|---:|
| Exact two-token answers | 29 | 14 |
| Incorrect or abstained | 227 | 242 |
| Answered | 256 | 141 |
| Backbone calls | 512 | 1,607 |
| Input tokens processed by backbone | 6,400 | 21,007 |
| Calls to the output head | 512 | 994 |
| Calls to the relaxation gate, including energy measurements | 1,024 | 5,844 |
| Instrumented CPU generation time | 7.46 s | 22.65 s |

Correction lost 15 initially correct answers and recovered zero of 227 initial errors. Error rose 5.86 percentage points, with a paired training-seed bootstrap 95% interval of +1.56 to +11.72 points. The paired t-test gave $p=0.090$ and the Wilcoxon test gave $p=0.0625$. Coverage was 141/256 and 127 of the 141 returned answers were wrong. Backbone calls rose 3.14× and wall time rose 3.04×, above the registered 3× limits. Timings exclude restoration of the matched model state and oracle scoring. [Seed-level uncertainty, joint outputs, costs and error curves](../../results/self_correction_binding_2026-11-22.json)

The loop attempted repairs on 125 prompts, executed 241 repairs and inspected 613 candidates or drafts. All ten final responses labelled `REPAIRED` failed the independent oracle. Twenty-two queries produced an exactly correct intermediate repair, yet 21 of these ended in abstention. Of the 15 initially correct answers lost, 13 ended in abstention and two ended in an incorrect accepted repair. These measurements concern this trained synthetic task and the fixed generation policy. Natural-language factuality and systematic compositional generalization require separate measurements. [Per-attempt tokens and obstruction scores](../../results/self_correction_binding_2026-11-22.json)
