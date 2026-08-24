# Stochastic Thermodynamics of Vesicle Release — Theory Note (bead `0642.3.1`)

_Thrust E — fluctuation-theorem-calibrated UQ. Author: BeigeSquirrel · 2026-06-12._

## Purpose & scope

This note gives the mathematical foundation for treating **stochastic vesicle release as a
nonequilibrium thermodynamic process** and testing its fluctuation-theorem consistency and precision
bounds. It establishes three results and the assumptions they rest on, so the
downstream implementation (`0642.3.2.1`, the TUR certificate + Crooks monitor) and falsification
(`0642.3.3.1`, the FT test + ECE/OOD-AUROC vs the softmax/MC baseline) build against a fixed contract:

1. **Vesicle release is a driven Markov jump process** with an explicit **entropy production** `Σ`,
   and its fluctuation theorems hold (`⟨e^{−Σ}⟩ = 1`, `P(Σ)/P(−Σ) = e^Σ`) (§1, subtask `0642.3.1.1`).
2. **The Thermodynamic Uncertainty Relation** `Var(J)/⟨J⟩² ≥ 2/⟨Σ⟩` — *precision costs entropy* — a
   provable lower bound on the relative uncertainty of any release current `J` (§2, `0642.3.1.2`).
3. **Crooks / Jarzynski ⟹ a release-trajectory consistency check**: the empirical
   entropy-production histogram must obey an analytic detailed-balance relation. This constrains the
   stochastic release process; it is not, by itself, a guarantee that token probabilities are
   calibrated (§3, `0642.3.1.3`).

Everything is grounded in the *live* code: the physical engine is the already-tested stochastic
release `K ~ Binomial(N = RRP, p = release_prob)` (`synaptic._sample_binomial_counts`, gated by
`stochastic_train_frac`/ACh) with recovery rate `rec_rate`. The reference math is
`bio_inspired_nanochat/stochastic_thermo.py`; the qualitative and quantitative claims are corroborated
numerically in `tests/test_stochastic_thermo.py` (§5). Softmax entropy and MC-dropout from `u2t` are
the empirical predictive-calibration baselines for the falsification experiment; no improvement is
assumed.

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

## 3. Crooks / Jarzynski → release-trajectory consistency  → subtask `0642.3.1.3`

Identifying the dissipated work `w = kT·Σ` (with `ΔF = 0` for a steady-state current), the detailed FT
**is** the Crooks relation `P_F(w)/P_R(−w) = e^{(w−ΔF)/kT}` and the integral FT is Jarzynski
`⟨e^{−w/kT}⟩ = e^{−ΔF/kT}`. This gives a **trajectory-consistency check** with teeth: the empirical
entropy-production histogram produced by MC release sampling (`u2t`) must satisfy

```
        ln( P(+Σ) / P(−Σ) )  =  Σ      on every populated symmetric bin.
```

`crooks_calibration` checks exactly this (symmetric bins about 0; equal widths make the count ratio
the density ratio). This is an **analytic relation the release-trajectory distribution must obey** —
and it is *falsifiable*: a distribution with no
fluctuation-theorem symmetry (e.g. a Gaussian) fails the check, which is the proof-ledger trigger to
drop the release-process claim and report empirical predictive calibration only (§4, R-fail). Passing
does **not** prove low ECE or calibrated token probabilities; those require the held-out baseline
comparison in §7. Hard constraints enter
as additive energy terms `Σ + Σ_c λ_c g_c` (energy-based constrained generation, `re4e.8`), and
`jarzynski_free_energy` recovers `ΔF` from the same nonequilibrium fluctuations (`≈ 0` for steady
state).

---

## 4. Energy-optimal (Landauer) release temperature  → subtask `0642.3.1.4`

The release temperature `kT` (set by ACh, `hy8.5`) is the exploration knob: hotter ⟹ more stochastic
release. How hot is *thermodynamically* optimal? Trade the information delivered against the metabolic
energy spent. A release that resolves a drive at signal-to-noise ratio `SNR` delivers
`½·log₂(1+SNR)` bits; by the TUR (§2) the minimum entropy to sustain that precision is `⟨Σ⟩ = 2·SNR`,
costing energy `E = kT·⟨Σ⟩`. At the **matched temperature** `kT = σ/√SNR` (thermal noise scaled to the
drive uncertainty `σ`), `E = 2σ·√SNR`, so the **bits-per-joule** is

```
        η(SNR)  ∝  log₂(1 + SNR) / √SNR .
```

`η` has a single interior maximum where `d/dSNR[ln(1+SNR)/√SNR] = 0`, i.e. at the universal
rate-distortion operating point

```
        2·SNR*/(1 + SNR*)  =  ln(1 + SNR*)        ⟹      SNR* ≈ 3.9215,
```

giving the **energy-optimal release temperature**

```
        kT*  =  σ / √SNR*  ≈  0.505 · σ.
```

*Interpretation.* The release resolves the signal **exactly to the level of its uncertainty**: any
finer (`kT < kT*`) wastes metabolic energy on spurious precision (the Landauer cost of bits you didn't
need), any coarser (`kT > kT*`) throws away signal. `SNR* ≈ 3.92` is a constant of the channel, not a
tunable — the thermodynamically-optimal attention always operates there.

**ACh coupling (state-dependent optimality).** Acetylcholine signals uncertainty/attention, scaling
the effective drive uncertainty `σ(ACh) = σ_base·(1 + g·ACh)`. So the optimal temperature
`kT*(ACh) = σ(ACh)/√SNR*` **rises with ACh**: more uncertainty ⟹ hotter ⟹ more exploration — exactly
the `hy8.5` direction (ACh → exploration), now with a *thermodynamic* justification rather than a
hand-set gain. (`landauer_optimal_temperature`, `ach_coupled_temperature`.)

---

## 5. Proof-obligation & assumptions ledger  → consumed by `0642.3.2`, `0642.3.3`

| # | Assumption (discharged by) | Statement | Failure mode | Fallback |
|---|---|---|---|---|
| E1 | **Markov / Poisson limit**: release & recovery are independent Poisson jumps with rates `a ∝ p·N`, `b ∝ rec_rate·N` (the Skellam model; `simulate_currents`). | `Σ = J·ln(a/b)`; the integral & detailed FTs hold exactly. | Binomial saturation (`p` not small, `N` small) breaks the Poisson limit; correlated releases break independence. | Use the empirical FT test (`0642.3.3.1`) as the gate; if it fails, drop the analytic guarantee, report empirical ECE only, flag. |
| E2 | **Stationary drive**: `a, b` (hence `A`) ≈ constant over the measurement window. | `⟨Σ⟩ = ⟨J⟩·A`; the TUR `Var(J)/⟨J⟩² ≥ 2/⟨Σ⟩`. | A fast-ramping release-probability protocol makes `A` time-dependent; the steady-state TUR no longer applies verbatim. | Use the finite-time/transient TUR (generalized bound) or restrict to windows where `p` is quasi-stationary. |
| E3 | **Local detailed balance**: the reverse rate of a release is the recovery rate (`b`), i.e. no hidden third state. | The medium entropy `ln(a/b)` is the correct trajectory affinity; Crooks/Jarzynski close. | A hidden facilitation/priming state (`Doc2`/`SNARE`) adds cycles ⟹ the 1-cycle affinity is incomplete. | Extend to the multi-state network affinity (Schnakenberg); until then, treat the 2-state `Σ` as a lower bound on the true entropy production. |
| R | **Trajectory-consistency claim**: the empirical `Σ` histogram obeys `ln(P(+Σ)/P(−Σ)) = Σ` within tolerance. | The sampled release process is fluctuation-theorem-consistent for the tested counter-protocol; predictive calibration remains empirical. | The histogram fails the relation (E1/E3 broken). | Drop the release-process claim; report predictive ECE/AUROC empirically; flag. |
| P | **Predictive-distribution claim** (`0642.3.4`): every named layer/head has fresh, finite, sufficiently covered paired-binomial evidence; its predeclared Crooks and TUR gates pass; and distinct matched seeds pass the held-out ECE/AUROC statistical rule. | The exact MC release draws that produced the token distribution are joined to their local thermodynamic evidence and empirical predictive-calibration result. | Empty, approximate-count, sparse, asymmetric, stale, duplicate-seed, FT/TUR-failed, or statistically null evidence breaks the join. | Set `predictive_distribution_claim=false`, remove the analytic label, and report plain empirical ECE. |

**Composition note** (`0642.10`/`0642.11.1`): the FT consistency check composes with the other thrusts only
while E1–E3 hold jointly with the presyn recurrence active (the release `p` and pool `N` are produced
by the calcium/RRP dynamics, so the stationarity E2 couples to the timescale-separation gauge).

---

## 6. Numerical corroboration

`tests/test_stochastic_thermo.py` checks the results against the reference Markov-jump model:

- **Entropy production & second law** — `Σ = J·ln(a/b)`, `⟨Σ⟩ ≥ 0`, `= 0` at `a = b`; affinity sign
  tracks the drive; `rates_from_release` is dissipative iff `p > rec_rate`.
- **Fluctuation theorems** — `integral_ft_closed_form ≡ 1` for every drive/duration; the simulator
  reproduces `⟨J⟩, Var(J), ⟨Σ⟩` and `⟨e^{−Σ}⟩ ≈ 1` near equilibrium; the detailed-FT ratio
  `P(+k)/P(−k) ≈ (a/b)^k` to ~5%.
- **TUR** — satisfied for every drive; relative slack `→ 0` as `a → b` (saturated near equilibrium);
  the empirical TUR holds on sampled currents.
- **Crooks/Jarzynski trajectory consistency** — `jarzynski_free_energy ≈ 0`; the `Σ` histogram obeys
  the detailed FT line; and the check **rejects** a misspecified (Gaussian) `Σ`, so the release-process
  claim is falsifiable.
- **Landauer temperature** — `optimal_exploration_snr ≈ 3.9215` solves the stationarity; bits-per-joule
  peaks there; `kT* = σ/√SNR* ≈ 0.505·σ` is linear in the drive uncertainty; the ACh coupling raises
  `kT*` with the signaled uncertainty.

These corroborate the exact identities (closed form) and that the simulated release reproduces them —
which is what licenses the runtime certificate/monitor (`0642.3.2.1`) and the falsification
(`0642.3.3.1`).

---

## 7. Matched-seed falsification verdict  → subtask `0642.3.3.2`

The runnable harness now has a matched-seed statistics path backed by the shared `74f.3` layer:
Student-t 95% intervals for each method, paired t and exact Wilcoxon tests, and a 10,000-resample
paired-bootstrap interval for every thermo-minus-baseline delta. The committed run used seeds
`11,23,37,53,71,89,107,131,149,167`, identical trained weights within each seed, and the default
CPU experiment configuration:

```bash
uv run python -m scripts.e2e.stochastic_thermo_uq \
  --seeds 11,23,37,53,71,89,107,131,149,167 \
  --bootstrap-samples 10000 \
  --output results/stochastic_thermo_uq_548ebe9f791d.json \
  --registry results/registry.jsonl
```

### 7.1 Calibration result: **null**, not a win

| method | ID ECE mean (Student-t 95% CI) | OOD AUROC mean (Student-t 95% CI) |
|---|---:|---:|
| softmax entropy | 0.071934 [0.061107, 0.082761] | 0.994748 [0.989499, 0.999998] |
| MC-dropout | 0.110240 [0.098622, 0.121858] | 0.997287 [0.993953, 1.000621] |
| thermo-UQ | 0.071938 [0.061379, 0.082498] | 0.994911 [0.989545, 1.000277] |

| paired thermo delta | mean | bootstrap 95% CI | paired-t p | Wilcoxon p | verdict |
|---|---:|---:|---:|---:|---|
| ECE vs softmax (lower is better) | +0.000004 | [-0.000316, +0.000224] | 0.9780 | 0.2324 | indistinguishable |
| AUROC vs softmax (higher is better) | +0.000163 | [-0.000217, +0.000553] | 0.4512 | 0.6250 | indistinguishable |
| ECE vs MC-dropout | -0.038302 | [-0.042648, -0.033841] | 5.81e-8 | 0.00195 | thermo better |
| AUROC vs MC-dropout | -0.002376 | [-0.004981, +0.000054] | 0.1178 | 0.1562 | no supported improvement |

The decision rule requires favorable bootstrap intervals **and** both paired tests for ECE and OOD
AUROC against **both** baselines. Thermo-UQ ties the stronger softmax baseline, and its ECE advantage
over MC-dropout does not extend to AUROC. The registry verdict is therefore **null**. This tiny cyclic
language-model benchmark also has near-saturated OOD AUROC, so it cannot establish a general
calibration advantage; a harder predictive-distribution benchmark would be fresh follow-up work, not
a reinterpretation of this result.

### 7.2 FT and TUR scope: one pass, one important limitation

The live one-step paired-binomial counter-protocol passed its detailed/integral fluctuation-theorem
thresholds on all 10 seeds. Mean maximum Crooks residual was `0.047982` (95% CI
`[0.026676, 0.069288]`), and mean integral-FT residual was `0.001773` (95% CI
`[0.000942, 0.002604]`). This remains an isolated local-detailed-balance check; the report explicitly
sets `predictive_distribution_claim=false` and does not turn it into a certificate for recurrent
hidden state or model predictions.

The classic continuous-time TUR bound is **non-vacuous** on the exact live protocol, but it does not
quite transfer to the finite one-step binomial process:

```text
exact relative variance       = 10.416664
classic entropy bound         = 10.445187
slack (variance - bound)      = -0.028523
bound ratio                   = 0.997269
```

Thus E1's Poisson/Skellam limit matters quantitatively: applying the classic bound directly to the
finite live binomial would assert a small false violation. The runtime continuous-time TUR remains
valid for its stated reference process, but a live-path certificate needs either a justified
rare-event limit or an appropriate finite/discrete-time bound. The harness records this as a theory
limitation instead of rounding it into a pass.

Evidence: `results/stochastic_thermo_uq_548ebe9f791d.json`; 30 schema-validated per-seed/method rows
in `results/registry.jsonl`, all provenance-stamped to implementation commit `55df5f5`.

### 7.3 Predictive-distribution integration gate (`0642.3.4`)

The predictive claim now has a concrete, fail-closed evidence path instead of inheriting the
one-step result. During MC inference, the orchestration layer assigns every `SynapticPresyn` its
stable module address. The canonical stochastic-release call supplies the exact probability, prior
RRP, validity mask, and sampled count that actually shaped each predictive draw. A private seeded
generator draws the matched recovery counter-protocol, so collecting evidence cannot advance or
otherwise perturb the model's PyTorch RNG stream. Evidence is then reduced separately for every
layer/head.

The local gate records the number of predictive samples, observed/tested/retained/degenerate events,
tested fraction, populated symmetric Crooks bins and residual, and the finite-binomial TUR bound
ratio. A deterministic random-priority reservoir bounds retained evidence at 100,000 events per head
without biasing capture toward early tokens or samples.
Approximate `normal_reparam` counts are explicitly degenerate; they are never rounded into exact
binomial evidence. Every stream is bound to the run ID, exact in-memory checkpoint fingerprint,
model-config hash, and predictive RNG seed. Reusing evidence after any binding changes marks it
stale.

The group claim is the conjunction of all of the following:

1. at least two distinct seed/run identities;
2. fresh, finite evidence for every observed layer/head;
3. the predeclared coverage, symmetric-support, Crooks-residual, and TUR-ratio gates for every head;
4. the matched-seed ECE/AUROC rule in §7.1, plus the live FT/TUR obligations in §7.2.

No partial pass is promoted. Structured reports use `null` for unavailable diagnostics rather than
JSON `NaN`/`Infinity`, retain the refusal reasons, and deterministically select
`empirical_ece_fallback` whenever the conjunction fails. This closes the software observability gap
while preserving the scientific limitation: a local FT pass is necessary evidence, not an automatic
token-calibration theorem.

The canonical 10-seed run exercised that refusal path on the exact predictive draws. It emitted 20
finite layer/head records (two heads at `h.0.attn.attn.pre` per seed), with 99,840 of 99,840 observed
events tested and retained, zero degenerate events, and 100% tested coverage. All 10 provenance
tuples had distinct run IDs, checkpoint fingerprints, config hashes, and predictive RNG seeds. The
finite-binomial TUR ratios passed the declared `0.95` floor (`2.530218`–`3.222809`), but all 20
Crooks residuals exceeded the predeclared `0.35` tolerance (`0.398946`–`1.097407`); eight heads also
had only one populated symmetric bin instead of the required two. Consequently, zero local seeds
passed. The matched-seed ECE/AUROC superiority rule remained null as reported in §7.1, so the group
claim stayed false with reasons `local_layer_head_gates_failed` and
`multi_seed_statistics_failed`, and calibration stayed in `empirical_ece_fallback` mode.

Evidence collection was observational: the training losses, method metrics, paired comparisons,
one-step FT aggregates, and live TUR result exactly matched the earlier no-collector artifact. The
strict-JSON report is `results/stochastic-thermo-uq-742e61d1bca7.json`; its 30 registry rows are
provenance-stamped to implementation commit `d2382cd`.

---

## References

- Jarzynski, C. (1997). *Nonequilibrium equality for free energy differences.* PRL. — `⟨e^{−w/kT}⟩ = e^{−ΔF/kT}`.
- Crooks, G. (1999). *Entropy production fluctuation theorem and the nonequilibrium work relation.* PRE.
- Barato, A. & Seifert, U. (2015). *Thermodynamic uncertainty relation for biomolecular processes.* PRL. — the TUR.
- Seifert, U. (2012). *Stochastic thermodynamics, fluctuation theorems and molecular machines.* Rep. Prog. Phys.
- Skellam, J.G. (1946). *The frequency distribution of the difference between two Poisson variates.* — `J = N₊ − N₋`.
- Internal: `bio_inspired_nanochat/stochastic_thermo.py`, `docs/theory/singular_perturbation.md` (`0642.2.1`, the proof-ledger pattern), `synaptic._sample_binomial_counts` (`u2t`, the physical engine).
