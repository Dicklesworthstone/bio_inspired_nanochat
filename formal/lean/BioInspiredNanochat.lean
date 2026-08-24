import Mathlib

/-!
# Machine-checked bio-inspired certificates

This file formalizes the reduced calcium/buffer/heat model implemented by
`bio_inspired_nanochat/metriplectic_integrator.py`.

The continuous theorems prove the algebraic chain for the explicit vector field:

* the Poisson operator annihilates the entropy gradient;
* the friction operator annihilates the energy gradient;
* energy rate is zero and entropy rate is nonnegative;
* positive-temperature free energy has nonpositive rate.

The discrete theorems deliberately use an abstract `DiscreteStep` contract. They prove that exact
energy conservation plus entropy production imply one-step and trajectory-level free-energy
monotonicity, and that nonnegative heat gives explicit component bounds on every reachable state.
The file also checks the algebraic core of the cusp-retention and stochastic-thermodynamic
certificates: the cusp fold half-width and its perturbation margin, the logarithmic inequality behind
the analytic TUR, and the Crooks histogram-calibration identity.  Every contract-to-runtime gap is
documented in `formal/lean/README.md`.
-/

namespace BioInspiredNanochat.Metriplectic

noncomputable section

/-- Reduced state `z = (C, B, h)`: free calcium, buffered calcium, and heat. -/
@[ext]
structure State where
  calcium : ℝ
  buffer : ℝ
  heat : ℝ

/-- Runtime parameters for reversible exchange, dissipation, and reference temperature. -/
structure Params where
  omega : ℝ
  gammaCalcium : ℝ
  gammaBuffer : ℝ
  temperature : ℝ

/-- `E(z) = 1/2 (C² + B²) + h`. -/
def energy (z : State) : ℝ :=
  (z.calcium ^ 2 + z.buffer ^ 2) / 2 + z.heat

/-- `S(z) = h`. -/
def entropy (z : State) : ℝ :=
  z.heat

/-- `F_T(z) = E(z) - T S(z)`. -/
def freeEnergy (temperature : ℝ) (z : State) : ℝ :=
  energy z - temperature * entropy z

/-- `∇E = (C, B, 1)`. -/
def gradEnergy (z : State) : State where
  calcium := z.calcium
  buffer := z.buffer
  heat := 1

/-- `∇S = (0, 0, 1)`. -/
def gradEntropy (_z : State) : State where
  calcium := 0
  buffer := 0
  heat := 1

def addState (left right : State) : State where
  calcium := left.calcium + right.calcium
  buffer := left.buffer + right.buffer
  heat := left.heat + right.heat

/-- Apply the constant skew Poisson operator `L` to a vector. -/
def poissonApply (p : Params) (v : State) : State where
  calcium := p.omega * v.buffer
  buffer := -p.omega * v.calcium
  heat := 0

/-- Apply `M = γ_C u uᵀ + γ_B v vᵀ` at state `z` to a vector. -/
def frictionApply (p : Params) (z v : State) : State where
  calcium := p.gammaCalcium * v.calcium - p.gammaCalcium * z.calcium * v.heat
  buffer := p.gammaBuffer * v.buffer - p.gammaBuffer * z.buffer * v.heat
  heat :=
    -p.gammaCalcium * z.calcium * v.calcium
      - p.gammaBuffer * z.buffer * v.buffer
      + (p.gammaCalcium * z.calcium ^ 2 + p.gammaBuffer * z.buffer ^ 2) * v.heat

/-- The explicit reduced vector field `L ∇E + M ∇S` used by the Python reference. -/
def field (p : Params) (z : State) : State :=
  addState (poissonApply p (gradEnergy z)) (frictionApply p z (gradEntropy z))

/-- `L ∇S`; its vanishing is the first GENERIC degeneracy condition. -/
def poissonOnEntropy (p : Params) (z : State) : State :=
  poissonApply p (gradEntropy z)

/-- `M ∇E`, expanded componentwise for the rank-two friction operator. -/
def frictionOnEnergy (p : Params) (z : State) : State :=
  frictionApply p z (gradEnergy z)

/-- The constant Poisson operator annihilates the heat-only entropy gradient. -/
theorem poisson_entropy_degeneracy (p : Params) (z : State) :
    poissonOnEntropy p z = { calcium := 0, buffer := 0, heat := 0 } := by
  ext <;> simp [poissonOnEntropy, poissonApply, gradEntropy]

/-- The constructed friction operator does no work on the energy gradient. -/
theorem friction_energy_degeneracy (p : Params) (z : State) :
    frictionOnEnergy p z = { calcium := 0, buffer := 0, heat := 0 } := by
  ext <;> simp [frictionOnEnergy, frictionApply, gradEnergy]
  ring

/-- Directional derivative of energy along `field p z`. -/
def energyRate (p : Params) (z : State) : ℝ :=
  z.calcium * (field p z).calcium
    + z.buffer * (field p z).buffer
    + (field p z).heat

/-- Directional derivative of entropy along `field p z`. -/
def entropyRate (p : Params) (z : State) : ℝ :=
  (field p z).heat

/-- Directional derivative of free energy along `field p z`. -/
def freeEnergyRate (p : Params) (z : State) : ℝ :=
  energyRate p z - p.temperature * entropyRate p z

/-- The explicit metriplectic flow conserves total energy exactly. -/
theorem energyRate_zero (p : Params) (z : State) : energyRate p z = 0 := by
  simp only [energyRate, field, addState, poissonApply, frictionApply, gradEnergy, gradEntropy]
  ring

/-- Nonnegative dissipation rates produce entropy. -/
theorem entropyRate_nonnegative (p : Params) (z : State)
    (hCalcium : 0 ≤ p.gammaCalcium) (hBuffer : 0 ≤ p.gammaBuffer) :
    0 ≤ entropyRate p z := by
  simpa [entropyRate, field, addState, poissonApply, frictionApply, gradEnergy, gradEntropy] using
      add_nonneg
        (mul_nonneg hCalcium (sq_nonneg z.calcium))
        (mul_nonneg hBuffer (sq_nonneg z.buffer))

/-- Positive-temperature free energy is a Lyapunov function for the continuous flow. -/
theorem freeEnergyRate_nonpositive (p : Params) (z : State)
    (hTemperature : 0 ≤ p.temperature)
    (hCalcium : 0 ≤ p.gammaCalcium) (hBuffer : 0 ≤ p.gammaBuffer) :
    freeEnergyRate p z ≤ 0 := by
  rw [freeEnergyRate, energyRate_zero]
  simpa only [zero_sub] using
    neg_nonpos.mpr (mul_nonneg hTemperature (entropyRate_nonnegative p z hCalcium hBuffer))

/-- Abstract contract discharged by an accepted discrete-gradient step. -/
structure DiscreteStep (before after : State) : Prop where
  energy_conserved : energy after = energy before
  entropy_produced : entropy before ≤ entropy after

/-- Exact discrete conservation plus entropy production imply the one-step Lyapunov property. -/
theorem freeEnergy_nonincreasing {before after : State} {temperature : ℝ}
    (hTemperature : 0 ≤ temperature) (step : DiscreteStep before after) :
    freeEnergy temperature after ≤ freeEnergy temperature before := by
  unfold freeEnergy
  rw [step.energy_conserved]
  have hScaled := mul_le_mul_of_nonneg_left step.entropy_produced hTemperature
  linarith

/-- Energy stays constant along an abstract trajectory of accepted discrete steps. -/
theorem trajectory_energy_constant (trajectory : ℕ → State)
    (steps : ∀ n, DiscreteStep (trajectory n) (trajectory (n + 1))) :
    ∀ n, energy (trajectory n) = energy (trajectory 0) := by
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
      calc
        energy (trajectory (n + 1)) = energy (trajectory n) := (steps n).energy_conserved
        _ = energy (trajectory 0) := ih

/-- Entropy at every reachable state is at least its initial value. -/
theorem trajectory_entropy_from_start (trajectory : ℕ → State)
    (steps : ∀ n, DiscreteStep (trajectory n) (trajectory (n + 1))) :
    ∀ n, entropy (trajectory 0) ≤ entropy (trajectory n) := by
  intro n
  induction n with
  | zero => exact le_rfl
  | succ n ih => exact ih.trans (steps n).entropy_produced

/-- Free energy never rises along a trajectory of accepted discrete steps. -/
theorem trajectory_freeEnergy_nonincreasing (trajectory : ℕ → State) {temperature : ℝ}
    (hTemperature : 0 ≤ temperature)
    (steps : ∀ n, DiscreteStep (trajectory n) (trajectory (n + 1))) :
    ∀ n, freeEnergy temperature (trajectory (n + 1)) ≤ freeEnergy temperature (trajectory n) := by
  intro n
  exact freeEnergy_nonincreasing hTemperature (steps n)

/-- Conserved coercive energy and nonnegative heat bound both mechanical coordinates. -/
theorem trajectory_component_bounds (trajectory : ℕ → State)
    (steps : ∀ n, DiscreteStep (trajectory n) (trajectory (n + 1)))
    (initial_heat_nonnegative : 0 ≤ (trajectory 0).heat) (n : ℕ) :
    (trajectory n).calcium ^ 2 + (trajectory n).buffer ^ 2
        ≤ 2 * energy (trajectory 0)
      ∧ 0 ≤ (trajectory n).heat
      ∧ (trajectory n).heat ≤ energy (trajectory 0) := by
  have hEnergy := trajectory_energy_constant trajectory steps n
  have hEntropy := trajectory_entropy_from_start trajectory steps n
  have hHeat : 0 ≤ (trajectory n).heat := by
    exact initial_heat_nonnegative.trans hEntropy
  have hCalciumSq := sq_nonneg (trajectory n).calcium
  have hBufferSq := sq_nonneg (trajectory n).buffer
  constructor
  · rw [← hEnergy]
    simp only [energy]
    nlinarith
  · constructor
    · exact hHeat
    · rw [← hEnergy]
      simp only [energy]
      nlinarith

#print axioms poisson_entropy_degeneracy
#print axioms friction_energy_degeneracy
#print axioms energyRate_zero
#print axioms entropyRate_nonnegative
#print axioms freeEnergyRate_nonpositive
#print axioms freeEnergy_nonincreasing
#print axioms trajectory_energy_constant
#print axioms trajectory_entropy_from_start
#print axioms trajectory_freeEnergy_nonincreasing
#print axioms trajectory_component_bounds

end
end BioInspiredNanochat.Metriplectic

namespace BioInspiredNanochat.CuspRetention

noncomputable section

/-!
The reduced equilibrium equation is the depressed cubic `u³ + a*u + b = 0`.  Theorems in this
namespace assume that the singular-perturbation reduction is valid and that `a < 0`, so the reduced
system is in the bistable cusp wedge.  They prove the exact fold algebra and the runtime monitor's
margin arithmetic.  They do not prove Fenichel persistence, floating-point coefficient extraction,
or the full clamped CaMKII/PP1 dynamics.
-/

/-- Cusp hysteresis half-width `δ*(a) = 2/(3√3) * (-a)^(3/2)`, written as
`(-a) * sqrt (-a)` to avoid an ambiguous fractional real power.  Monostable `a ≥ 0` has no
retention certificate. -/
def deltaStar (a : ℝ) : ℝ :=
  if a < 0 then 2 * (-a) * Real.sqrt (-a) / (3 * Real.sqrt 3) else 0

/-- The runtime certificate is fail-closed outside the bistable regime. -/
theorem deltaStar_of_nonnegative {a : ℝ} (ha : 0 ≤ a) : deltaStar a = 0 := by
  simp [deltaStar, not_lt.mpr ha]

/-- A bistable cusp has a strictly positive retention half-width. -/
theorem deltaStar_pos {a : ℝ} (ha : a < 0) : 0 < deltaStar a := by
  rw [deltaStar, if_pos ha]
  have hNegA : 0 < -a := neg_pos.mpr ha
  positivity

/-- `b = ±δ*(a)` lies exactly on the depressed-cubic fold discriminant
`4*a³ + 27*b² = 0`. -/
theorem fold_discriminant_zero {a : ℝ} (ha : a < 0) :
    4 * a ^ 3 + 27 * deltaStar a ^ 2 = 0 := by
  rw [deltaStar, if_pos ha]
  have hNegA : 0 ≤ -a := (neg_pos.mpr ha).le
  have hNumerator : (2 * (-a) * Real.sqrt (-a)) ^ 2 = -4 * a ^ 3 := by
    rw [mul_pow, mul_pow, Real.sq_sqrt hNegA]
    ring
  have hDenominator : (3 * Real.sqrt 3) ^ 2 = 27 := by
    rw [mul_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]
    norm_num
  rw [div_pow, hNumerator, hDenominator]
  ring

/-- Discriminant of the depressed cubic `u³ + a*u + b`. -/
def cuspDiscriminant (a b : ℝ) : ℝ :=
  -4 * a ^ 3 - 27 * b ^ 2

/-- The open retention window `|b| < δ*(a)` is exactly the positive-discriminant, three-real-root
region of the constructed depressed cubic.  This is a statement about the cubic chart, not a claim
that every root lies inside the clamped physical state interval. -/
theorem inside_retention_window_iff_discriminant_pos {a b : ℝ} (ha : a < 0) :
    |b| < deltaStar a ↔ 0 < cuspDiscriminant a b := by
  have hDelta : 0 < deltaStar a := deltaStar_pos ha
  have hFold := fold_discriminant_zero ha
  constructor
  · intro hInside
    have hSquare : b ^ 2 < deltaStar a ^ 2 := by
      rw [sq_lt_sq]
      simpa [abs_of_pos hDelta] using hInside
    unfold cuspDiscriminant
    nlinarith
  · intro hDiscriminant
    have hSquare : b ^ 2 < deltaStar a ^ 2 := by
      unfold cuspDiscriminant at hDiscriminant
      nlinarith
    have hAbs := sq_lt_sq.mp hSquare
    simpa [abs_of_pos hDelta] using hAbs

/-- Remaining distance from a resting bias to either cusp fold. -/
def retentionMargin (a b : ℝ) : ℝ :=
  deltaStar a - |b|

/-- A perturbation smaller than the certified remaining margin cannot cross either fold.  The
dynamical statement "the selected stable branch persists while no fold is crossed" is an explicit
cusp-model assumption, not hidden in this arithmetic theorem. -/
theorem perturbation_below_margin_stays_inside {a b noise : ℝ} (ha : a < 0)
    (hNoise : |noise| < retentionMargin a b) :
    |b + noise| < deltaStar a := by
  have _hPositiveCertificate : 0 < deltaStar a := deltaStar_pos ha
  have hTriangle : |b + noise| ≤ |b| + |noise| := abs_add_le b noise
  unfold retentionMargin at hNoise
  linarith

#print axioms deltaStar_of_nonnegative
#print axioms deltaStar_pos
#print axioms fold_discriminant_zero
#print axioms inside_retention_window_iff_discriminant_pos
#print axioms perturbation_below_margin_stays_inside

end
end BioInspiredNanochat.CuspRetention

namespace BioInspiredNanochat.StochasticThermodynamics

noncomputable section

/-!
These theorems model the analytic Skellam/Poisson reference used by
`bio_inspired_nanochat/stochastic_thermo.py`.  The TUR statement assumes positive stationary rates,
a driven release current `b < a`, positive observation time, and local detailed balance.  Empirical
finite-sample variance, binomial saturation, hidden states, and time-varying rates remain runtime
evidence obligations.
-/

/-- Gap whose nonnegativity for `x ≥ 1` is equivalent to
`2*(x-1) ≤ (x+1)*log x`. -/
def logMeanGap (x : ℝ) : ℝ :=
  (x + 1) * Real.log x - 2 * (x - 1)

private theorem hasDerivAt_logMeanGap {x : ℝ} (hx : 0 < x) :
    HasDerivAt logMeanGap (Real.log x + x⁻¹ - 1) x := by
  have hProduct := ((hasDerivAt_id x).const_add 1).mul (Real.hasDerivAt_log hx.ne')
  have hLinear := ((hasDerivAt_id x).sub_const 1).const_mul 2
  convert hProduct.sub hLinear using 1
  · funext y
    change (y + 1) * Real.log y - 2 * (y - 1) =
      (1 + y) * Real.log y - 2 * (y - 1)
    ring
  · simp only [id_eq]
    field_simp [hx.ne']
    ring

/-- The logarithmic-mean inequality underlying the release-current TUR. -/
theorem logMeanGap_nonnegative {x : ℝ} (hx : 1 ≤ x) : 0 ≤ logMeanGap x := by
  have hMonotone : MonotoneOn logMeanGap (Set.Ici 1) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 1)
    · intro y hy
      have hyPos : 0 < y := zero_lt_one.trans_le hy
      exact (hasDerivAt_logMeanGap hyPos).continuousAt.continuousWithinAt
    · intro y hy
      have hyOne : 1 < y := by simpa only [interior_Ici, Set.mem_Ioi] using hy
      exact (hasDerivAt_logMeanGap (zero_lt_one.trans hyOne)).differentiableAt.differentiableWithinAt
    · intro y hy
      have hyOne : 1 < y := by simpa only [interior_Ici, Set.mem_Ioi] using hy
      have hyPos : 0 < y := zero_lt_one.trans hyOne
      rw [(hasDerivAt_logMeanGap hyPos).deriv]
      linarith [Real.one_sub_inv_le_log_of_pos hyPos]
  have hAtOne := hMonotone (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx) hx
  simpa [logMeanGap] using hAtOne

/-- Cleared-denominator TUR inequality for positive forward/reverse rates in the driven regime. -/
theorem log_rate_tur_core {a b : ℝ} (hb : 0 < b) (hba : b ≤ a) :
    2 * (a - b) ≤ (a + b) * Real.log (a / b) := by
  have hRatio : 1 ≤ a / b := (le_div_iff₀ hb).2 (by simpa using hba)
  have hGap := logMeanGap_nonnegative hRatio
  have hRaw : 2 * (a / b - 1) ≤ (a / b + 1) * Real.log (a / b) := by
    unfold logMeanGap at hGap
    linarith
  have hScaled := mul_le_mul_of_nonneg_left hRaw hb.le
  calc
    2 * (a - b) = b * (2 * (a / b - 1)) := by field_simp [hb.ne']
    _ ≤ b * ((a / b + 1) * Real.log (a / b)) := hScaled
    _ = (a + b) * Real.log (a / b) := by field_simp [hb.ne']

/-- Analytic mean entropy production of the stationary release current. -/
def meanEntropy (a b observationTime : ℝ) : ℝ :=
  (a - b) * observationTime * Real.log (a / b)

/-- Analytic relative current variance `Var(J)/mean(J)^2`. -/
def relativeVariance (a b observationTime : ℝ) : ℝ :=
  (a + b) / ((a - b) ^ 2 * observationTime)

/-- Thermodynamic lower bound `2/meanEntropy`. -/
def entropyBound (a b observationTime : ℝ) : ℝ :=
  2 / meanEntropy a b observationTime

/-- The exact analytic TUR consumed by `tur_certificate`: relative variance cannot fall below the
entropy-production bound. -/
theorem tur_calibration_inequality {a b observationTime : ℝ} (hb : 0 < b) (hba : b < a)
    (hTime : 0 < observationTime) :
    entropyBound a b observationTime ≤ relativeVariance a b observationTime := by
  have hCore := log_rate_tur_core hb hba.le
  have hDiff : 0 < a - b := sub_pos.mpr hba
  have hRatio : 1 < a / b := (lt_div_iff₀ hb).2 (by simpa using hba)
  have hLog : 0 < Real.log (a / b) := Real.log_pos hRatio
  have hMeanEntropy : 0 < meanEntropy a b observationTime := by
    unfold meanEntropy
    positivity
  have hVarianceDenominator : 0 < (a - b) ^ 2 * observationTime := by positivity
  rw [entropyBound, relativeVariance, div_le_div_iff₀ hMeanEntropy hVarianceDenominator]
  have hScaled := mul_le_mul_of_nonneg_right hCore (mul_nonneg hDiff.le hTime.le)
  unfold meanEntropy
  convert hScaled using 1 <;> ring

/-- The detailed fluctuation-theorem ratio implies the histogram calibration equation checked by
`crooks_calibration`. -/
theorem crooks_log_calibration {positiveMass negativeMass sigma : ℝ}
    (hRatio : positiveMass / negativeMass = Real.exp sigma) :
    Real.log (positiveMass / negativeMass) = sigma := by
  rw [hRatio, Real.log_exp]

#print axioms logMeanGap_nonnegative
#print axioms log_rate_tur_core
#print axioms tur_calibration_inequality
#print axioms crooks_log_calibration

end
end BioInspiredNanochat.StochasticThermodynamics
