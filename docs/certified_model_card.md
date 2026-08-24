# Live certificate model cards

`bio_inspired_nanochat.certificate_bundle` composes the project's runtime proof artifacts under one
declared model identity. It is a bounded certificate-policy deployment gate, not a general
model-safety claim. Well-formed missing, stale, fallback-covered, mismatched, or out-of-scope
evidence produces a visible refusal; malformed input is rejected before card artifacts are created.

## What the bundle certifies

| Gate | Required live evidence | Claim when it passes | Deterministic refusal/fallback |
| --- | --- | --- | --- |
| Provenance | One declared bundle identity with full target checkpoint SHA-256, config digest, and Git revision; predictive target/cohort run IDs retained separately | The observations consistently declare that identity and the live config recomputes its digest; checkpoint/revision association remains an explicit operator attestation | Keep source artifacts separate; refuse the aggregate label |
| Metriplectic stability | Non-empty torch-runtime `TorchStepRecord` stream, within the fixed deployment tolerances, with zero fallbacks | The observed guarded state updates conserve energy, produce entropy, and do not increase `F` | Use clamped Euler with no structure-preserving label |
| Cusp retention | Active `bistable_latch` + `cusp_latch`; live config passes `certify_retention` with `delta* > 0` | The active latch has the reported local retention half-width | Use the heuristic latch with `delta*=0` and no retention guarantee |
| Predictive calibration | Fresh target per-layer/head records and fixed local ECE/OOD-AUROC point-estimate gates; the exact ordered cohort seeds `11,23,37,41,53,67`; all four ECE/OOD-AUROC × softmax/MC-dropout comparisons; raw per-run Crooks/integral-FT histograms; and the scoped finite-binomial TUR diagnostic | The target passes the local point-estimate policy while the separate fixed cohort supplies population-level statistical support | Report empirical ECE/uncertainty only |
| Tropical robustness | Non-empty exact-affine, replayable live monitor records with valid scope/norm enums, non-vacuous radius semantics, complete source details, retained binding digests, and recomputed scope-specific readout/output consistency | The named selection/readout scope is stable in the stated norm ball | Retain soft/default routing and an uncertified fingerprint |
| Composition | Thrusts A/E/F are individually eligible and pairwise compatible under the configured timescale proxy | The four preceding claims may be presented together under that proxy assumption | Disable the higher-risk incompatible thrust named by the harness |

The tropical gate deliberately preserves scope. `moe_topk_membership` is a membership guarantee, not
an expert-output guarantee. Attention output stability appears only when the selected value was
explicitly frozen. The H certificate has its own exact-affine runtime gates; `composition.py` covers
the A/E/F timescale hierarchy.

## Runtime integration

Capture evidence from the objects that observed the deployed checkpoint. Do not reconstruct passing
booleans from a historical summary:

```python
import json
from dataclasses import asdict, replace
from pathlib import Path

from bio_inspired_nanochat.certificate_bundle import (
    ModelIdentity,
    PredictiveCalibrationObservation,
    RobustnessObservation,
    StabilityObservation,
    build_guarantee_bundle,
    make_evidence_manifest,
)
from bio_inspired_nanochat.checkpoint_manager import config_hash
from scripts.e2e.stochastic_thermo_uq import run_experiment, run_multi_seed

# Freeze the cohort before inference, then evaluate an independently trained target.
canonical_multi_seed_report = run_multi_seed(
    deployment_experiment_config,
    (11, 23, 37, 41, 53, 67),
)
deployed_target_report = run_experiment(
    replace(deployment_experiment_config, seed=10_000)
)
target_provenance = deployed_target_report.predictive_thermo_evidence.provenance
identity = ModelIdentity(
    run_id=run_id,
    checkpoint_id=target_provenance.checkpoint_id,
    config_hash=config_hash(asdict(synaptic_config)),
    predictive_config_hash=target_provenance.config_hash,
    git_sha=git_sha,
)
stability = StabilityObservation.from_torch_records(identity, live_metriplectic_records)
predictive = PredictiveCalibrationObservation.from_multi_seed_report(
    identity,
    canonical_multi_seed_report,
    target_report=deployed_target_report,
)
robustness = RobustnessObservation.from_monitor(identity, tropical_monitor)

bundle = build_guarantee_bundle(
    identity=identity,
    config=synaptic_config,
    stability=stability,
    predictive_calibration=predictive,
    robustness=robustness,
)
bundle.require_deployable()  # raises CertificationRefused unless every gate passed
bundle.write_artifacts("runs/certified_model_card")

manifest = make_evidence_manifest(
    identity=identity,
    config=synaptic_config,
    stability=stability,
    predictive_calibration=predictive,
    robustness=robustness,
)
Path("runs/certificate_evidence.json").write_text(
    json.dumps(manifest, allow_nan=False), encoding="utf-8"
)
```

`live_metriplectic_records` are the `TorchStepRecord` objects emitted by the torch-native guarded
recurrence. `from_monitor` is also available for a transparent NumPy-reference refusal card, but the
deployment gate rejects that source. These stability and tropical source objects do not themselves
carry model provenance: the operator must capture them in the declared deployment process without
mutating or relabeling the model. The declared target checkpoint digest is the lowercase
64-character SHA-256 of the parameter/buffer artifact, and `git_sha` is a full 40-character revision.
The `run_id` names the certificate assembly; it is not asserted to equal every predictive source run
ID. The predictive source retains both the normalized SynapticConfig digest and a seed-independent
full experiment/protocol SHA; both must match the identity. Target-checkpoint local evidence is
strictly separate from the research cohort: its run ID, RNG seed, and checkpoint digest may not
appear in that cohort. Every cohort checkpoint digest must also be unique, so repeated evaluations
of one checkpoint cannot masquerade as independent model units. Complete target and cohort head
records are retained rather than collapsed into pass booleans. The card emits the target artifact
digest, full target evidence, and target ECE/OOD-AUROC measurements in a block separate from the
ordered cohort digests. The target-calibration digest binds those measurements to the target's
predictive artifact, run, checkpoint, SynapticConfig, experiment protocol, and RNG seed. It also
records the exact evaluation-distribution label and prediction count for each of the equally sized
ID and OOD splits. The fixed target policy requires
ECE at most `0.10`, OOD AUROC at least `0.70`, and strict improvement over both target-local
baselines. Those measurements describe only the recorded synthetic held-out/OOD protocol; they do
not transfer to deployment traffic or another data distribution. The bundle also recomputes counts,
coverage, Crooks/TUR gates, claim state, and exact expected site coverage.

The cohort statistics artifact is content-digested and must contain exactly ECE and OOD-AUROC
against both softmax entropy and MC-dropout, with deltas in `[-1,1]` and fixed `alpha=0.05`. Its
digest binds the model identity, ordered cohort run/checkpoint/SynapticConfig/experiment/RNG
identities, and expected sites. Paired p-values and bootstrap intervals are recomputed under the
fixed 10,000-sample/seed-20260824 policy. Deployment authorization additionally requires the exact
ordered cohort `(11, 23, 37, 41, 53, 67)`, preventing after-the-fact favorable-subset selection at
this gate. Its run IDs, RNG seeds, and checkpoint digests are all unique. Report-only cohorts are
capped at 256 seeds, and every delta vector must have exactly one value per seed before bootstrap
allocation. Each live FT record retains its independent mechanism-protocol config digest,
experiment seed, distinct forward/reverse RNG seeds, and complete `-pool_size … +pool_size`
integer-current histogram. The bundle checks the trajectory total, derives the integral FT,
reconstructs every sufficiently supported Crooks point, and applies the fixed `0.25`/`0.04`
tolerances. FT is canonical-protocol evidence co-run with a cohort report, not evidence that the FT
harness observed that report's trained checkpoint. Its protocol must exactly match the scoped live
TUR protocol, whose moments, slack, ratio, non-vacuity, and exact nonnegative-slack verdict are also
recomputed. An empty supported Crooks curve remains valid failure evidence and produces a refusal
card. The matched-seed comparison normally needs at least six nonzero pairs before an exact
two-sided Wilcoxon p-value can be at most 0.05.

## Standalone audit-card generator

Generate an audit card from a serialized evidence manifest:

```bash
uv run python -m bio_inspired_nanochat.certificate_bundle \
  runs/certificate_evidence.json \
  --output-dir runs/certified_model_card
```

Generate a transparent refusal card without failing a report-only workflow only when explicitly
requested:

```bash
uv run python -m bio_inspired_nanochat.certificate_bundle \
  runs/certificate_evidence.json \
  --output-dir runs/certified_model_card \
  --allow-uncertified
```

Live runtime provenance is deliberately non-transferable: `make_evidence_manifest` omits private
in-process attestations, so replaying any JSON manifest cannot authorize deployment. The standalone
command therefore produces a refusal card and returns status `2` by default. Bounded authorization
is available only from the same-process `build_guarantee_bundle(...).require_deployable()` path that
still holds the original torch, predictive producer-report, and tropical monitor bindings. The
predictive adapter requires an explicit deployed target report plus its matched cohort report;
direct `from_evidences(...)` construction remains report-only. A future offline authorization path
would need a separately implemented trusted signature/verifier; a source label or content digest is
not one.

`--allow-uncertified` changes only the exit code for a well-formed refusal; it never changes the card
verdict. A well-formed refusal writes all three audit artifacts:

- `model_card.json` — strict JSON; untrusted manifests reject non-finite values, while failed
  in-process measurements are represented as `null`, never NaN/Infinity.
- `MODEL_CARD.md` — human-readable values, assumptions, failed gates, and fallbacks.
- `events.jsonl` — one `certificate_gate` event per gate plus the aggregate `certificate_bundle`
  event, with run provenance.

The JSON input rejects duplicate keys, NaN/Infinity, missing or unknown fields at every level,
truthy strings, negative counters/seeds, unknown enums, non-finite numbers, and malformed config
types. Input is capped at 16 MiB, and excessive nesting is reported as a clean refusal without
creating artifacts. Its identity hash is recomputed from the complete normalized config, and every
observation identity must match exactly.

## Required configuration and common refusals

A complete certificate normally needs:

- `enable_presyn=true` and `metriplectic_integrator=true`;
- `enable_hebbian=true`, `bistable_latch=true`, and `cusp_latch=true`;
- exact predictive sampling (`straight_through` or `gumbel_sigmoid_ste`) represented in the evidence;
- `tropical_skeleton=true`; and
- configured A/E/F timescales below the fixed maximum composition threshold `eps_max=0.5` (callers
  may select a stricter threshold but cannot widen it).

The stock config intentionally does not pass the full bundle. Its release-to-fast-weight boundary is
not separated, and the certificate toggles are default-off. This is a useful refusal, not an error to
paper over.

Other expected refusals include:

- a metriplectic fallback anywhere in the observed trajectory;
- empty stability or tropical evidence (vacuous truth is rejected);
- duplicate predictive run IDs, RNG seeds, or checkpoint digests; target/cohort identity overlap;
  missing/cherry-picked expected layer/head sites; a transplanted target-calibration artifact; or a
  source summary whose counts and verdicts do not recompute;
- stale or checkpoint/config-mismatched predictive streams;
- approximate `normal_reparam` evidence promoted to a thermodynamic claim;
- a pointwise/local-only tropical fingerprint presented as an affine-cell certificate; or
- a positive radius with unbound selection/Lipschitz/temperature artifacts, or inconsistent
  scope/readout/output/radius semantics.

## Scope and limitations

The card reports evidence for one exact declared model identity and the observed operating regime.
It does not prove arbitrary-distribution language-model safety, robustness of nonlinear
selected-expert outputs, future behavior after online weight/state mutation, or validity outside the
listed thresholds and threat norms. Individual manifest records are content-digested but unsigned.
They provide local content-consistency checks, not authenticity or deployment authorization:
relabeling a NumPy trace as `torch_runtime`, copying passing tropical booleans, or rewriting every
digest cannot recreate the private live-source binding. Recompute the checkpoint SHA-256 and re-run
evidence capture in the deployment process after any checkpoint, config, code, or relevant
runtime-state change.

The repository ships controlled composition tests and honest refusal tests; it does not ship a
passing production-model certificate card. A production authorization exists only after real
checkpoint evidence satisfies every gate.
