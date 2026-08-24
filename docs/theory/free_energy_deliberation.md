# Free-Energy Deliberation + Energy-Based Decoding — Design Note (bead `r00r.1.1`)

_Capability Frontier (`r00r`) · metabolic System-2. Author: GoldenRiver · 2026-06-12._

## Purpose & scope

Turn the synaptic free energy `F = E − T·S` into **adaptive test-time compute**: on a hard token, run
*extra* free-energy-minimization steps ("ponder") so the synaptic state relaxes toward a lower-energy,
more self-consistent configuration before committing; on an easy token, stop after one step. Expose
`F` as a confidence/effort signal, halt early when it stops dropping, and add **energy-based
decoding** — sample `p ∝ exp(−F/kT)` — for controllable, constraint-aware generation. This is
o1-style "think longer," but **grounded in physics**: the model descends a genuine energy it is
guaranteed (Thrust A) to descend, so extra iterations are provably safe and bounded.

This note specifies (a) the deliberation loop, (b) the halting rule, (c) the Boltzmann decoder, (d)
the convergence argument, and (e) the API — the executable spec that `r00r.1.2` wires into the live
per-token decode path (toggle + fallback). The reference implementation is in
`bio_inspired_nanochat/metriplectic_integrator.py` (`deliberate`, `boltzmann_weights`), demonstrated
in `tests/test_free_energy_deliberation.py`.

**It rests entirely on already-delivered, tested results:** the metriplectic free energy `F` and its
Lyapunov property (`docs/theory/metriplectic.md`, `0642.1.1`) and the structure-preserving integrator
that makes `F` descend *monotonically and convergently at the discrete level* (`0642.1.2.1`). No new
dynamics are introduced — deliberation is just *running the existing descent longer*.

---

## 1. The deliberation loop

Per token, the model holds a synaptic state `z` (the metriplectic core `(C, B, h)`, and in the live
model the full calcium/buffer/pool/energy state). Standard decoding advances `z` one step. Deliberation
instead iterates the **structure-preserving step** under the guards until the state self-consistently
relaxes or a compute budget is hit:

```
deliberate(z, dt; eps, max_iters, T):
    F_prev = F(z)
    for k in 0 .. max_iters−1:
        z = guarded_step(z, dt)              # F monotonically ↓  (Thrust A Lyapunov)
        if |F_prev − F(z)| < eps:  return z, k+1, converged=True     # self-consistent
        F_prev = F(z)
    return z, max_iters, converged=False     # budget hit (hard token)
```

`guarded_step` is the discrete-gradient integrator with the conservation/degeneracy guards and the
clamped-Euler fallback (`0642.1.2.3`), so a deliberation step can **never** destabilize the state even
if a learned operator misbehaves.

---

## 2. The convergence argument (why extra iterations are safe)

This is the crux — "think longer" is only safe if the extra compute provably converges. It does, by
Thrust A:

1. **Monotone descent.** Each `guarded_step` satisfies, at the discrete level (any `dt`),
   `F(z_{k+1}) ≤ F(z_k)` — the discrete free-energy Lyapunov property proved and tested in
   `0642.1.2.1` (`F` non-increasing, machine-precision energy conservation, monotone entropy).
2. **Bounded below.** `F = E − T·S` with `E` conserved at `E₀` and `S ≤ E₀` on the compact energy
   shell (`metriplectic.md` §5), so `F ≥ (1−T)·E₀`. A monotone-decreasing sequence bounded below
   **converges**: `F(z_k) → F_∞`.
3. **Halting fires.** Convergence ⟹ `|F(z_{k+1}) − F(z_k)| → 0`, so for any `eps > 0` the halt
   condition is met after finitely many steps. By LaSalle on the compact shell the limit is the MaxEnt
   equilibrium `z* = (0,0,E₀)`; near it the descent is geometric, so the step count to reach `|ΔF| <
   eps` is `O(log(1/eps))` — **bounded and predictable**.

So deliberation is a contraction toward a unique attractor: more iterations always help (or are
no-ops) and never diverge. The compute is *spent*, not *risked*.

---

## 3. Halting rule, effort & confidence signals

- **Halt** when `|ΔF| < eps` (self-consistent — further pondering buys nothing) **or** when `k`
  reaches `max_iters` (budget). The two outcomes are distinguished (`halted_converged`).
- **Effort signal** = the number of iterations actually used. Easy tokens halt in ~1 step; hard
  tokens use many (verified: a far-from-equilibrium state takes ~100 steps vs 1 for a near-equilibrium
  one). This is the token-level difficulty estimate, available *for free*.
- **Confidence signal** = the final `F` (and the released `F_drop = F(z₀) − F_∞`). A low final `F` /
  small remaining drop ⟹ the state is self-consistent ⟹ confident; a budget-hit with `F` still
  falling ⟹ the model is "unsure" — a calibrated, physics-grounded `I'm-guessing` flag (feeds the
  metacognition layer `re4e.2`).

---

## 4. Energy-based (Boltzmann) decoding

Beyond pondering a single state, score candidate continuations by their relaxed free energy and sample

```
        p(candidate) ∝ exp(−F(candidate) / kT).
```

Lower free energy ⟹ more self-consistent ⟹ more probable. `kT` is the decoding temperature: `kT → 0`
recovers greedy `argmin F` (most self-consistent), large `kT` → uniform (more exploratory). **Hard
constraints enter as additive energy terms** `F + Σ_c λ_c·g_c(z)` (energy-based constrained /
controllable generation, `re4e.8`): forbidden configurations get high energy and vanishing
probability. The weights are numerically stabilized (subtract the max-logit). The Boltzmann decoder
composes with the per-token deliberation: relax each candidate, then sample by relaxed `F`.

### 4.1 Task-calibrated candidate energy (`r00r.1.7`)

The raw `F(mean C, mean B, 0)` readout is parameter-free but throws away almost all branch-local
structure. The optional `CandidateEnergyReadout` instead uses a frozen linear energy over the model
energy prior, the relaxation trajectory (`F_initial`, `F_final`, drop, effort, convergence), and
per-candidate moments of C/BUF/RRP/RES/PR/CL/E/AMP. It is trained by pairwise ranking: within each
model-top-k group, the gold continuation must have lower energy than every incorrect candidate.

The fit is deliberately leakage-safe:

1. the language model is trained first and its state is frozen;
2. each seed fits a readout only on its deterministic calibration sequences;
3. calibration and evaluation content rows are checked for zero overlap;
4. the immutable readout cannot update in the evaluation decode path;
5. ranking is compared on one fixed model-only candidate corpus, so policies cannot manufacture an
   easier candidate distribution;
6. an affine scale match preserves model-energy temperature, then a calibration-only cross-entropy
   grid chooses a conservative residual blend (ties choose model-only energy).

The five-seed copy-continuation falsification (`seeds=11,23,37,53,71`; 24 calibration and 8 held-out
sequences per seed) found:

- calibrated rank accuracy `0.8017` versus raw physical `F` `0.2412`; paired delta `+0.5605`, bootstrap
  95% interval `[+0.4845,+0.6283]`, paired-t `p=0.000182`;
- calibrated rank accuracy versus the already-strong raw total/model energy was a null: delta
  `+0.0067`, interval `[0.0000,+0.0200]`, `p=0.3739`;
- at `max_iters=32`, token accuracy **regressed** from `0.76875` to `0.70000` while effort rose to
  `279.82` relaxations/token; paired delta `-0.06875`, interval `[-0.10625,-0.03125]`, `p=0.0402`.

So `r00r.1.7` establishes a valid disjoint-split readout and decisively repairs ranking relative to
the raw aggregate, but does **not** establish a useful capability improvement. The result is a
predeclared regression, not a success relabel: `r00r.1` remains open. The default-off path is still
the recommended runtime behavior.

---

## 5. The compute-vs-quality control knob

Two scalars set the trade-off:

| knob | meaning | smaller / larger |
|---|---|---|
| `eps` | halting threshold on `\|ΔF\|` | smaller ⟹ deliberate longer (more compute; quality is empirical) |
| `max_iters` | per-token compute budget | larger ⟹ allow harder tokens more thinking |
| `kT` | decode temperature | smaller ⟹ greedier on self-consistency |

Because effort is *self-allocating* (easy tokens may halt early), the **average** compute is bounded
by `max_iters`; the budget bounds the worst case and the dynamics decide how much is spent. That is
the intended "compute scales with difficulty" mechanism. The `r00r.1.7` regression above shows that
self-allocation alone is not evidence of a favorable compute/quality trade-off.

---

## 6. The API (reference)

```python
# bio_inspired_nanochat/metriplectic_integrator.py
deliberate(z, dt, *, eps=1e-4, max_iters=64, T=TEMP, thresholds=None, **operators)
    -> DeliberationResult(z, iters, F_final, F_drop, halted_converged)

boltzmann_weights(free_energies, kT=1.0) -> np.ndarray   # p ∝ exp(−F/kT), stabilized
```

For `r00r.1.2` (the live wiring): the loop runs per token on the model's synaptic state; the toggle
gates it (default-off ⟹ single-step decode, the current behavior); the deliberation **state** is the
per-sequence presyn state already carried by the KV-cache path; and the fallback is the existing
single-step decode if the budget is exhausted or a guard trips. The `F`-trajectory and per-token
iteration count are logged (the user mandate for detailed per-step logging; schema `eqyk.2`).

---

## 7. Downstream this unblocks

`r00r.1.2` (implementation: per-token loop + F-halting + energy-guided sampler, toggle+fallback),
`r00r.1.3` (tests + F-trajectory logging), `r00r.1.4` (eval: does deliberation improve a
reasoning/consistency metric at controlled extra compute, vs single-step — `74f.3` stats). It is the
substrate for `re4e.1` (self-correcting loop), `re4e.2` (metacognition from the confidence signal),
`re4e.3` (energy-guided search), `re4e.8` (constrained generation via energy terms), and `re4e.9`
(decode the deliberation trajectory into a reasoning trace).

---

## References

- Internal: `docs/theory/metriplectic.md` (`0642.1.1`, the free energy + Lyapunov property),
  `bio_inspired_nanochat/metriplectic_integrator.py` (`0642.1.2.1`, the structure-preserving step),
  `tests/test_free_energy_deliberation.py` (this note's numerical demonstration).
- LaSalle, J.P. (1960). *Some extensions of Lyapunov's second method.* IRE Trans. Circuit Theory. —
  convergence to the invariant set (the halting guarantee).
- Hinton, G. & others on energy-based models; Boltzmann sampling `p ∝ exp(−E/kT)`.
- Adaptive-computation-time / "ponder" lineage (Graves 2016; o1-style test-time compute) — here
  grounded in a physical energy descent rather than a learned halting unit.
