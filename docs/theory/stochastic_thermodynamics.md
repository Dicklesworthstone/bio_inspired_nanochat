# Stochastic Thermodynamics of Vesicle Release — Theory Note (bead `0642.3.1`)

_Thrust E — fluctuation-theorem-calibrated UQ. Author: BeigeSquirrel · 2026-06-12._

## Purpose & scope

This note gives the mathematical foundation for treating **stochastic vesicle release as a
nonequilibrium thermodynamic process** and harvesting its fluctuation theorems as *calibration and
precision guarantees*. It establishes three results and the assumptions they rest on, so the
downstream implementation (`0642.3.2.1`, the TUR certificate + Crooks monitor) and falsification
(`0642.3.3.1`, the FT test + ECE/OOD-AUROC vs the softmax/MC baseline) build against a fixed contract:

1. **Vesicle release is a driven Markov jump process** with an explicit **entropy production** `Σ`,
   and its fluctuation theorems hold (`⟨e^{−Σ}⟩ = 1`, `P(Σ)/P(−Σ) = e^Σ`) (§1, subtask `0642.3.1.1`).
2. **The Thermodynamic Uncertainty Relation** `Var(J)/⟨J⟩² ≥ 2/⟨Σ⟩` — *precision costs entropy* — a
   provable lower bound on the relative uncertainty of any release current `J` (§2, `0642.3.1.2`).
3. **Crooks / Jarzynski ⟹ a calibration guarantee**: the empirical entropy-production histogram must
   obey an analytic detailed-balance relation, an *a-priori* calibration constraint rather than only a
   post-hoc ECE (§3, `0642.3.1.3`).

Everything is grounded in the *live* code: the physical engine is the already-tested stochastic
release `K ~ Binomial(N = RRP, p = release_prob)` (`synaptic._sample_binomial_counts`, gated by
`stochastic_train_frac`/ACh) with recovery rate `rec_rate`. The reference math is
`bio_inspired_nanochat/stochastic_thermo.py`; the qualitative and quantitative claims are corroborated
numerically in `tests/test_stochastic_thermo.py` (§5). The **baseline** this improves on is the
softmax-entropy + MC-dropout calibration of `u2t` — which has hysteresis-free UQ but no
fluctuation-theorem guarantee.

---

## 0. The dynamical system (as it actually is)

Per attention edge (head), a readily-releasable pool of `N` vesicles is drained each step: every
docked vesicle releases independently with probability `p` (the faithful Hill/SNARE release
probability), so the **released count** is `K ~ Binomial(N, p)` — the live stochastic-release path.
Released vesicles are recovered/recycled at rate `rec_rate` (the reverse jump; the
`vesicle_depletion_refill` conservation law `Δ(RRP+RES+Σdelay) = −released·(1−rec_rate)`).

Two competing jumps define a **two-state-per-vesicle Markov jump process** `Docked ⇌ Released`:

| jump | rate (per unit time) | bio | code |
|---|---|---|---|
| `D → R` (release) | `a` | calcium-driven vesicle fusion | `p · N` |
| `R → D` (recovery) | `b` | ATP-driven endocytosis/recycling | `rec_rate · N` |

The **release current** `J = N₊ − N₋` (releases minus recoveries) is the observable of interest. In
the Poisson (rare-release-per-step) limit, `N₊ ~ Poisson(a·t)` and `N₋ ~ Poisson(b·t)` are
independent, so `J` is a **Skellam** random variable. The metabolic drive sustains `a > b`, which is
exactly what breaks detailed balance: with `a = b` the cycle is at equilibrium (`Σ = 0`); with `a > b`
the synapse dissipates. `ACh`/temperature (`hy8`) sets the thermal scale `kT`.

---

## 1. The Markov jump process & entropy production  → subtask `0642.3.1.1`

**Entropy production.** A forward release jump `D → R` (probability `a`) against its time-reverse
`R → D` (probability `b`) contributes `ln(a/b)` to the medium entropy; a recovery jump contributes
`−ln(a/b)`. A trajectory with `N₊` releases and `N₋` recoveries therefore produces

```
        Σ[ω]  =  (N₊ − N₋)·ln(a/b)  =  J · A,        A := ln(a/b)   (the affinity).
```

`A > 0 ⟺ a > b` (driven), so `Σ = J·A` and `⟨Σ⟩ = ⟨J⟩·A = (a−b)t·ln(a/b) ≥ 0` — the **second law**,
with equality iff `a = b` (detailed balance). `A` is the entropy produced per net released vesicle.

**Fluctuation theorems (exact for the Skellam model).** Using `⟨z^{N}⟩ = e^{λ(z−1)}` for
`N ~ Poisson(λ)`:

```
   ⟨e^{−Σ}⟩  =  ⟨(b/a)^{N₊}⟩·⟨(a/b)^{N₋}⟩
             =  exp(a t (b/a − 1)) · exp(b t (a/b − 1))
             =  exp(t(b − a) + t(a − b))  =  e^0  =  1.                 (integral FT)
```

This is the integral fluctuation theorem (Jarzynski with `ΔF = 0`): the second law is then exactly
Jensen's inequality `⟨e^{−Σ}⟩ ≥ e^{−⟨Σ⟩}` on it. Likewise, the Skellam PMF symmetry
`P(J = +k) = (a/b)^k · P(J = −k)` gives the **detailed** fluctuation theorem

```
        P(Σ = +s) / P(Σ = −s)  =  e^{s}.                                (detailed FT)
```

*Numerical caveat (the reason the corroboration is split).* The MC estimator of `⟨e^{−Σ}⟩` is
dominated by exponentially-rare negative-`Σ` trajectories, so it only converges near equilibrium; the
identity itself is verified in **closed form** (`integral_ft_closed_form`) for any drive, and by
simulation only in the near-equilibrium regime where both signs of `J` are sampled.

---

## 2. The Thermodynamic Uncertainty Relation  → subtask `0642.3.1.2`

For any current `J` of a nonequilibrium steady state, the TUR bounds its **relative uncertainty** by
the entropy produced:

```
        Var(J) / ⟨J⟩²  ≥  2 / ⟨Σ⟩.                                     (TUR)
```

*Precision costs entropy.* For the release current, `Var(J)/⟨J⟩² = (a+b)/((a−b)²·t)` and
`2/⟨Σ⟩ = 2/((a−b)·t·ln(a/b))`, so the TUR reduces to the elementary inequality

```
        (a + b)·ln(a/b)  ≥  2·(a − b)          for all a, b > 0,
```

which holds with equality only as `a → b` (linear response): **the TUR is saturated near
equilibrium** and loosens as the drive grows. The per-head **TUR certificate**
(`tur_certificate`/`empirical_tur`) reports the measured precision `ε² = Var(J)/⟨J⟩²`, the bound
`2/⟨Σ⟩`, and the slack `ε² − 2/⟨Σ⟩ ≥ 0` — a provable precision/energy Pareto position for each head,
and the substrate for a fluctuation-theorem-optimal (Landauer) release temperature that maximizes
bits-per-joule.

---

## 3. Crooks / Jarzynski → the calibration guarantee  → subtask `0642.3.1.3`

Identifying the dissipated work `w = kT·Σ` (with `ΔF = 0` for a steady-state current), the detailed FT
**is** the Crooks relation `P_F(w)/P_R(−w) = e^{(w−ΔF)/kT}` and the integral FT is Jarzynski
`⟨e^{−w/kT}⟩ = e^{−ΔF/kT}`. This turns into a **calibration guarantee** with teeth: the empirical
entropy-production histogram produced by MC release sampling (`u2t`) must satisfy

```
        ln( P(+Σ) / P(−Σ) )  =  Σ      on every populated symmetric bin.
```

`crooks_calibration` checks exactly this (symmetric bins about 0; equal widths make the count ratio
the density ratio). Unlike a post-hoc ECE number, this is an **analytic relation the predictive
distribution must obey a priori** — and it is *falsifiable*: a distribution with no
fluctuation-theorem symmetry (e.g. a Gaussian) fails the check, which is the proof-ledger trigger to
drop the analytic-guarantee claim and report empirical ECE only (§4, R-fail). Hard constraints enter
as additive energy terms `Σ + Σ_c λ_c g_c` (energy-based constrained generation, `re4e.8`), and
`jarzynski_free_energy` recovers `ΔF` from the same nonequilibrium fluctuations (`≈ 0` for steady
state).

---

## 4. Proof-obligation & assumptions ledger  → consumed by `0642.3.2`, `0642.3.3`

| # | Assumption (discharged by) | Statement | Failure mode | Fallback |
|---|---|---|---|---|
| E1 | **Markov / Poisson limit**: release & recovery are independent Poisson jumps with rates `a ∝ p·N`, `b ∝ rec_rate·N` (the Skellam model; `simulate_currents`). | `Σ = J·ln(a/b)`; the integral & detailed FTs hold exactly. | Binomial saturation (`p` not small, `N` small) breaks the Poisson limit; correlated releases break independence. | Use the empirical FT test (`0642.3.3.1`) as the gate; if it fails, drop the analytic guarantee, report empirical ECE only, flag. |
| E2 | **Stationary drive**: `a, b` (hence `A`) ≈ constant over the measurement window. | `⟨Σ⟩ = ⟨J⟩·A`; the TUR `Var(J)/⟨J⟩² ≥ 2/⟨Σ⟩`. | A fast-ramping release-probability protocol makes `A` time-dependent; the steady-state TUR no longer applies verbatim. | Use the finite-time/transient TUR (generalized bound) or restrict to windows where `p` is quasi-stationary. |
| E3 | **Local detailed balance**: the reverse rate of a release is the recovery rate (`b`), i.e. no hidden third state. | The medium entropy `ln(a/b)` is the correct trajectory affinity; Crooks/Jarzynski close. | A hidden facilitation/priming state (`Doc2`/`SNARE`) adds cycles ⟹ the 1-cycle affinity is incomplete. | Extend to the multi-state network affinity (Schnakenberg); until then, treat the 2-state `Σ` as a lower bound on the true entropy production. |
| R | **Calibration claim**: the empirical `Σ` histogram obeys `ln(P(+Σ)/P(−Σ)) = Σ` within tolerance. | The predictive distribution is fluctuation-theorem-calibrated (an analytic guarantee). | The histogram fails the relation (E1/E3 broken). | Drop the analytic-guarantee claim; fall back to the post-hoc ECE of the `u2t` baseline; flag. |

**Composition note** (`0642.10`/`0642.11.1`): the FT calibration composes with the other thrusts only
while E1–E3 hold jointly with the presyn recurrence active (the release `p` and pool `N` are produced
by the calcium/RRP dynamics, so the stationarity E2 couples to the timescale-separation gauge).

---

## 5. Numerical corroboration

`tests/test_stochastic_thermo.py` checks the results against the reference Markov-jump model:

- **Entropy production & second law** — `Σ = J·ln(a/b)`, `⟨Σ⟩ ≥ 0`, `= 0` at `a = b`; affinity sign
  tracks the drive; `rates_from_release` is dissipative iff `p > rec_rate`.
- **Fluctuation theorems** — `integral_ft_closed_form ≡ 1` for every drive/duration; the simulator
  reproduces `⟨J⟩, Var(J), ⟨Σ⟩` and `⟨e^{−Σ}⟩ ≈ 1` near equilibrium; the detailed-FT ratio
  `P(+k)/P(−k) ≈ (a/b)^k` to ~5%.
- **TUR** — satisfied for every drive; relative slack `→ 0` as `a → b` (saturated near equilibrium);
  the empirical TUR holds on sampled currents.
- **Crooks/Jarzynski calibration** — `jarzynski_free_energy ≈ 0`; the `Σ` histogram obeys the detailed
  FT line; and the check **rejects** a misspecified (Gaussian) `Σ`, so the guarantee is falsifiable.

These corroborate the exact identities (closed form) and that the simulated release reproduces them —
which is what licenses the runtime certificate/monitor (`0642.3.2.1`) and the falsification
(`0642.3.3.1`).

---

## References

- Jarzynski, C. (1997). *Nonequilibrium equality for free energy differences.* PRL. — `⟨e^{−w/kT}⟩ = e^{−ΔF/kT}`.
- Crooks, G. (1999). *Entropy production fluctuation theorem and the nonequilibrium work relation.* PRE.
- Barato, A. & Seifert, U. (2015). *Thermodynamic uncertainty relation for biomolecular processes.* PRL. — the TUR.
- Seifert, U. (2012). *Stochastic thermodynamics, fluctuation theorems and molecular machines.* Rep. Prog. Phys.
- Skellam, J.G. (1946). *The frequency distribution of the difference between two Poisson variates.* — `J = N₊ − N₋`.
- Internal: `bio_inspired_nanochat/stochastic_thermo.py`, `docs/theory/singular_perturbation.md` (`0642.2.1`, the proof-ledger pattern), `synaptic._sample_binomial_counts` (`u2t`, the physical engine).
