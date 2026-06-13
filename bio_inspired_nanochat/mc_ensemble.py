"""Bayesian MC ensembling via stochastic vesicle release at inference (bead `u2t.1`).

Exposes an inference mode that samples the stochastic vesicle release multiple times to produce a
**predictive distribution** (mean + uncertainty) — a structured, *stateful* alternative to MC-dropout.
The bio noise is not i.i.d.: a confident, high-RRP path can deplete its readily-releasable pool and so
release with low variance, while an uncertain, already-depleted path stays noisy — so the predictive
variance is shaped by the synaptic state, not a fixed dropout mask.

Mechanism: with `_mc_sampling` set on every `SynapticPresyn`, each forward pass draws an independent
stochastic release (the `Binomial(N=RRP, p)` sampler, `_sample_binomial_counts`), so `S` passes over
the same input give `S` samples of the next-token distribution. The standard epistemic/aleatoric
decomposition (BALD) then falls out:

    predictive_entropy  H[ E_s p_s ]          — total uncertainty
    expected_entropy    E_s H[ p_s ]          — aleatoric (data) uncertainty
    mutual_information   = predictive − expected — epistemic (model) uncertainty.

The predictive distribution this produces is exactly the nonequilibrium ensemble whose statistics obey
the fluctuation theorems of `stochastic_thermo.py` (Thrust E) — the substrate the TUR certificate /
Crooks calibration monitor (`0642.3.2.1`) is built to certify. Default-off: without `mc_predict`/
`mc_sampling` the model decodes deterministically as before.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

from bio_inspired_nanochat.synaptic import SynapticPresyn
from bio_inspired_nanochat.torch_imports import torch


def set_mc_sampling(model, enabled: bool, *, frac: float = 1.0) -> int:
    """Toggle inference-time stochastic release on every `SynapticPresyn` in `model`.

    Returns the number of presyn modules touched. `frac` is the fraction of query positions sampled
    stochastically (1.0 ⟹ a full predictive draw at every position). A no-op for a vanilla model.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, SynapticPresyn):
            module._mc_sampling = bool(enabled)
            module._mc_frac = float(frac)
            n += 1
    return n


@contextmanager
def mc_sampling(model, *, frac: float = 1.0):
    """Context manager: enable stochastic-release sampling on entry, restore the prior state on exit."""
    prior = [(m, getattr(m, "_mc_sampling", False), getattr(m, "_mc_frac", 1.0))
             for m in model.modules() if isinstance(m, SynapticPresyn)]
    set_mc_sampling(model, True, frac=frac)
    try:
        yield model
    finally:
        for m, was_on, was_frac in prior:
            m._mc_sampling = was_on
            m._mc_frac = was_frac


def _reset_sequence_state(model) -> None:
    """Reset the per-sequence synaptic scratchpad (fast/eligibility + consolidation), if supported.

    Called before each ensemble pass so the `S` draws are i.i.d. from the *same* clean baseline (a valid
    predictive expectation). The stateful/correlated noise the thesis exploits is preserved *within* a
    pass — the calcium/RRP evolve across the `T` positions of each forward — but does not leak across
    passes (which would make them a Markov chain rather than an ensemble).
    """
    fn = getattr(model, "reset_sequence_state", None)
    if callable(fn):
        fn(reset_fast_weights=True, reset_consolidation=True)


def _forward_logits(model, input_ids):
    """Call the model and return next-token logits `(B, T, V)`, handling GPT vs GPTSynaptic outputs."""
    kwargs = {}
    try:
        if "train_mode" in inspect.signature(model.forward).parameters:
            kwargs["train_mode"] = False  # eval-mode dynamics (no plasticity); MC flag drives the noise
    except (TypeError, ValueError):
        pass
    out = model(input_ids, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _entropy(probs, eps: float = 1e-12):
    """Shannon entropy `−Σ p·log p` along the last (vocab) axis (nats)."""
    return -(probs * (probs + eps).log()).sum(dim=-1)


@dataclass
class MCPrediction:
    """The MC-ensembled predictive distribution and its uncertainty decomposition (per position)."""

    mean_probs: torch.Tensor        # (B, T, V) — the predictive distribution E_s[p_s]
    predictive_entropy: torch.Tensor  # (B, T) — total uncertainty H[E_s p_s]
    expected_entropy: torch.Tensor    # (B, T) — aleatoric uncertainty E_s H[p_s]
    mutual_information: torch.Tensor  # (B, T) — epistemic uncertainty (BALD); ≥ 0
    logit_variance: torch.Tensor      # (B, T, V) — per-token logit variance across samples
    n_samples: int

    def summary(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "mean_predictive_entropy": float(self.predictive_entropy.mean()),
            "mean_aleatoric": float(self.expected_entropy.mean()),
            "mean_epistemic": float(self.mutual_information.mean()),
            "mean_logit_std": float(self.logit_variance.clamp(min=0).sqrt().mean()),
        }


@torch.no_grad()
def mc_predict(model, input_ids, *, n_samples: int = 16, temperature: float = 1.0) -> MCPrediction:
    """Run `n_samples` stochastic-release forward passes and aggregate the predictive distribution.

    Each pass is an independent draw (fresh per-call synaptic state ⟹ independent release noise), so the
    sample mean is the predictive distribution and the spread is the model's uncertainty. Restores the
    model's training/sampling state on exit. `n_samples == 1` reduces to a single stochastic forward.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    was_training = model.training
    model.eval()
    probs_sum = None
    entropy_sum = None
    logit_sum = None
    logit_sq_sum = None
    try:
        with mc_sampling(model):
            for _ in range(n_samples):
                _reset_sequence_state(model)  # i.i.d. draw from a clean per-sequence baseline
                logits = _forward_logits(model, input_ids).float()
                probs = torch.softmax(logits / temperature, dim=-1)
                ent = _entropy(probs)
                probs_sum = probs if probs_sum is None else probs_sum + probs
                entropy_sum = ent if entropy_sum is None else entropy_sum + ent
                logit_sum = logits if logit_sum is None else logit_sum + logits
                logit_sq_sum = logits * logits if logit_sq_sum is None else logit_sq_sum + logits * logits
    finally:
        # Restore train/eval mode even if a forward pass raises (the mc_sampling context manager
        # already restores the per-module _mc_sampling/_mc_frac flags in its own finally).
        if was_training:
            model.train()

    assert (probs_sum is not None and entropy_sum is not None
            and logit_sum is not None and logit_sq_sum is not None)  # n_samples >= 1 ⟹ the loop ran
    mean_probs = probs_sum / n_samples
    predictive_entropy = _entropy(mean_probs)
    expected_entropy = entropy_sum / n_samples
    mutual_information = (predictive_entropy - expected_entropy).clamp(min=0.0)  # BALD; ≥ 0 up to noise
    logit_mean = logit_sum / n_samples
    logit_variance = (logit_sq_sum / n_samples - logit_mean * logit_mean).clamp(min=0.0)
    return MCPrediction(
        mean_probs=mean_probs,
        predictive_entropy=predictive_entropy,
        expected_entropy=expected_entropy,
        mutual_information=mutual_information,
        logit_variance=logit_variance,
        n_samples=n_samples,
    )
