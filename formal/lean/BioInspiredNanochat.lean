import Mathlib

/-!
# Metriplectic Lyapunov certificate

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
The contract-to-floating-point gap is documented in `formal/lean/README.md`.
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
