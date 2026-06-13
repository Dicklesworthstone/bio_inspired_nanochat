"""Bayesian MC ensembling via stochastic vesicle release at inference (bead `u2t.1`).

Covers `bio_inspired_nanochat/mc_ensemble.py` and the inference-time stochastic-release gate added to
`SynapticPresyn.release_canonical`:

  - `set_mc_sampling` / `mc_sampling` toggle inference-time release sampling on every presyn and
    restore the prior state (even on exception) — default-off;
  - `mc_predict` produces a normalized predictive distribution and the BALD uncertainty decomposition
    (predictive ≥ expected entropy; mutual information ≥ 0);
  - a bio model yields non-trivial, input-dependent predictive variance (the *stateful* noise), while a
    vanilla model is deterministic (no presyn ⟹ a graceful no-op);
  - the ensemble is reproducible under a fixed seed.

Note: epistemic uncertainty (the mutual information) is small on an *untrained* model because its
output is near-uniform, so the release noise barely moves it — the magnitude/calibration is a
trained-model property (`0642.3.3.1`); here we test the *mechanism* (variance is produced, structured,
and decomposes correctly). Run:  pytest tests/test_mc_ensemble.py -v
"""

from __future__ import annotations

import pytest

from _bio_testkit import make_tiny_synaptic, make_tiny_vanilla
from bio_inspired_nanochat import mc_ensemble as mc
from bio_inspired_nanochat.synaptic import SynapticPresyn
from bio_inspired_nanochat.torch_imports import torch

pytestmark = pytest.mark.unit


def _presyn(model):
    return [m for m in model.modules() if isinstance(m, SynapticPresyn)]


def _ids(b: int = 2, t: int = 12):
    return torch.randint(0, 90, (b, t))


def test_set_mc_sampling_toggles_and_counts_presyn():
    model = make_tiny_synaptic(seed=1234)
    n = mc.set_mc_sampling(model, True, frac=0.5)
    assert n == len(_presyn(model)) >= 1
    assert all(m._mc_sampling and m._mc_frac == 0.5 for m in _presyn(model))
    mc.set_mc_sampling(model, False)
    assert all(not m._mc_sampling for m in _presyn(model))


def test_mc_sampling_context_restores_state_even_on_error():
    model = make_tiny_synaptic(seed=1234)
    assert all(not getattr(m, "_mc_sampling", False) for m in _presyn(model))
    with pytest.raises(RuntimeError):
        with mc.mc_sampling(model):
            assert all(m._mc_sampling for m in _presyn(model))
            raise RuntimeError("boom")
    assert all(not m._mc_sampling for m in _presyn(model)), "must restore prior state on exception"


def test_mc_predict_produces_a_valid_predictive_distribution():
    model = make_tiny_synaptic(seed=1234)
    pred = mc.mc_predict(model, _ids(), n_samples=16)
    assert pred.mean_probs.shape[-1] == model.config.vocab_size
    sums = pred.mean_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "predictive distribution must normalize"
    assert pred.n_samples == 16


def test_uncertainty_decomposition_is_consistent():
    model = make_tiny_synaptic(seed=1234)
    pred = mc.mc_predict(model, _ids(), n_samples=24)
    # Independent recompute (NOT via the stored field): predictive_entropy must be the Shannon entropy
    # of the mean predictive distribution. Catches a predictive-entropy computed from the wrong tensor.
    indep_predictive = -(pred.mean_probs * (pred.mean_probs + 1e-12).log()).sum(dim=-1)
    assert torch.allclose(pred.predictive_entropy, indep_predictive, atol=1e-6)
    # Jensen / BALD — the substantive property: total (predictive) ≥ aleatoric (expected). This breaks
    # if predictive/expected are swapped or mis-aggregated (concavity of entropy forbids the reverse).
    assert (pred.predictive_entropy >= pred.expected_entropy - 1e-5).all()
    # ... and the epistemic gap is the (clamped) difference; entropies are in nats and finite.
    assert torch.allclose(pred.mutual_information,
                          (pred.predictive_entropy - pred.expected_entropy).clamp(min=0.0), atol=1e-6)
    assert torch.isfinite(pred.predictive_entropy).all() and (pred.expected_entropy >= 0).all()


def test_bio_model_has_nonzero_variance_vanilla_is_deterministic():
    bio = make_tiny_synaptic(seed=1234)
    van = make_tiny_vanilla(seed=1234)
    x = _ids()
    bio_pred = mc.mc_predict(bio, x, n_samples=16)
    van_pred = mc.mc_predict(van, x, n_samples=16)
    assert float(bio_pred.logit_variance.mean()) > 1e-5, "stochastic release must produce predictive variance"
    assert mc.set_mc_sampling(van, True) == 0, "a vanilla model has no presyn (graceful no-op)"
    assert float(van_pred.logit_variance.mean()) < 1e-5, "a vanilla model is deterministic in eval"


def test_predictive_variance_is_input_dependent_structured_noise():
    # The bio noise is stateful: different inputs drive different synaptic states (RRP depletion), so
    # the per-position predictive variance is NOT a fixed mask — the structural claim over MC-dropout.
    model = make_tiny_synaptic(seed=1234)
    torch.manual_seed(0)
    x1 = torch.randint(0, 90, (1, 16))
    x2 = torch.randint(0, 90, (1, 16))
    a = mc.mc_predict(model, x1, n_samples=20)
    b = mc.mc_predict(model, x2, n_samples=20)
    # Variance varies across positions (not constant), i.e. it is structured by the state.
    per_pos = a.logit_variance.mean(dim=-1).flatten()
    assert float(per_pos.std()) > 0.0, "predictive variance must vary across positions (structured)"
    # Two different inputs give different per-position uncertainty profiles.
    assert not torch.allclose(a.predictive_entropy, b.predictive_entropy, atol=1e-4)


def test_mc_predict_is_reproducible_under_a_fixed_seed():
    model = make_tiny_synaptic(seed=1234)
    x = _ids()
    torch.manual_seed(7)
    p1 = mc.mc_predict(model, x, n_samples=10)
    torch.manual_seed(7)
    p2 = mc.mc_predict(model, x, n_samples=10)
    assert torch.allclose(p1.mean_probs, p2.mean_probs, atol=0.0), "same seed ⟹ identical ensemble"


def test_mc_predict_validates_n_samples_and_supports_single_draw():
    model = make_tiny_synaptic(seed=1234)
    with pytest.raises(ValueError):
        mc.mc_predict(model, _ids(), n_samples=0)
    single = mc.mc_predict(model, _ids(), n_samples=1)
    assert single.n_samples == 1 and single.mean_probs.shape[0] == 2
    # n_samples=1: expected entropy equals predictive entropy (one sample), epistemic ≈ 0.
    assert torch.allclose(single.mutual_information, torch.zeros_like(single.mutual_information), atol=1e-5)


def test_summary_keys():
    model = make_tiny_synaptic(seed=1234)
    s = mc.mc_predict(model, _ids(), n_samples=8).summary()
    assert {"n_samples", "mean_predictive_entropy", "mean_aleatoric", "mean_epistemic", "mean_logit_std"} <= set(s)
