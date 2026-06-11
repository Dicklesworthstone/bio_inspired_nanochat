"""Single source of truth for bio mechanism toggles, ablation presets, and config
validation (bead bio_inspired_nanochat-hm4.7).

The whole research thesis depends on "modular, toggleable mechanisms for clean
ablation". This module makes that discipline explicit and enforceable:

* ``MECHANISMS`` — every bio mechanism, its ablation knob, default, off-value, and
  prerequisite mechanisms. The completeness of this list is guarded by a test
  (``test_ablation_registry.py``) so a new mechanism can't be added without a
  documented toggle.
* ``ABLATION_PRESETS`` — the canonical preset → field-override map. ``eval_matrix``
  applies presets through :func:`apply_preset`, so there is ONE definition.
* :func:`validate_config` — rejects configs whose mechanism is enabled without its
  prerequisite (e.g. ``bistable_latch`` without ``enable_hebbian``) or whose knobs are
  out of range, and warns on risky-but-legal combinations (e.g. prefill-only
  FlexAttention).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from bio_inspired_nanochat.synaptic import SynapticConfig


@dataclass(frozen=True)
class MechanismFlag:
    """One ablatable bio mechanism and the SynapticConfig field that gates it."""

    mechanism: str          # human label, e.g. "stochastic_release"
    field: str              # SynapticConfig field that toggles it
    default: Any            # value in a default SynapticConfig
    off_value: Any          # value that ABLATES the mechanism
    default_on: bool        # is the mechanism ON in a default config?
    requires: tuple[str, ...]  # fields that must themselves be "on" for this to do anything
    description: str


# Every bio mechanism with a documented ablation knob. `requires` references the
# `field` of the prerequisite mechanism(s).
MECHANISMS: tuple[MechanismFlag, ...] = (
    MechanismFlag(
        "presyn", "enable_presyn", True, False, True, (),
        "Presynaptic vesicle release augmenting the attention logits.",
    ),
    MechanismFlag(
        "hebbian", "enable_hebbian", True, False, True, (),
        "Postsynaptic Hebbian fast/slow plasticity.",
    ),
    MechanismFlag(
        "metabolism", "enable_metabolism", True, False, True, (),
        "MoE expert fatigue/energy metabolism bias.",
    ),
    MechanismFlag(
        "stochastic_release", "stochastic_train_frac", 0.12, 0.0, True, ("enable_presyn",),
        "Stochastic (Binomial STE) vesicle release during training.",
    ),
    MechanismFlag(
        "doc2", "doc2_gain", 0.08, 0.0, True, ("enable_presyn",),
        "Doc2 facilitation term in the release probability.",
    ),
    MechanismFlag(
        "septin_barrier", "barrier_strength", 0.1, 0.0, True, ("enable_presyn",),
        "Septin-like distance barrier in the attention logits.",
    ),
    MechanismFlag(
        "bdnf", "bdnf_scale", 1.0, 0.0, True, ("enable_hebbian",),
        "BDNF metaplasticity scaling of the slow-weight learning rate.",
    ),
    MechanismFlag(
        "bistable_latch", "bistable_latch", False, False, False, ("enable_hebbian",),
        "Bistable CaMKII/PP1 consolidation latch (sax.2).",
    ),
    MechanismFlag(
        "flex_attention", "use_flex_attention", False, False, False, ("enable_presyn",),
        "FlexAttention O(N) presyn path. PREFILL-ONLY: incompatible with KV-cache decode.",
    ),
    MechanismFlag(
        "native_genetics", "native_genetics", False, False, False, ("enable_metabolism",),
        "Fused Rust metabolism/genetics kernel (CUDA).",
    ),
)

_BY_FIELD: dict[str, MechanismFlag] = {m.field: m for m in MECHANISMS}

# Canonical ablation presets: preset id -> {SynapticConfig field: override value}.
# `bio_all` is the unmodified default; `vanilla` is handled at the model level
# (GPTSynapticConfig.synapses=False) and applies no synaptic overrides.
ABLATION_PRESETS: dict[str, dict[str, Any]] = {
    "vanilla": {},
    "bio_all": {},
    "bio_no_presyn": {"enable_presyn": False},
    "bio_no_hebbian": {"enable_hebbian": False},
    "bio_no_metabolism": {"enable_metabolism": False},
    "bio_no_stochastic_release": {"stochastic_train_frac": 0.0},
    "bio_no_doc2": {"doc2_gain": 0.0},
    "bio_no_bdnf": {"bdnf_scale": 0.0},
    "bio_no_septin_barrier": {"barrier_strength": 0.0},
}


def is_mechanism_on(cfg: SynapticConfig, field: str) -> bool:
    """Is the mechanism gated by ``field`` currently active in ``cfg``?"""
    flag = _BY_FIELD[field]
    return getattr(cfg, field) != flag.off_value


def apply_preset(preset: str, cfg: SynapticConfig) -> SynapticConfig:
    """Apply an ablation preset's field overrides to ``cfg`` in place and return it.

    This is the single application path; ``eval_matrix`` delegates here so the preset
    definitions cannot drift between modules.
    """
    if preset not in ABLATION_PRESETS:
        raise ValueError(
            f"Unknown ablation preset {preset!r}; known: {sorted(ABLATION_PRESETS)}"
        )
    for field, value in ABLATION_PRESETS[preset].items():
        setattr(cfg, field, value)
    return cfg


def validate_config(cfg: SynapticConfig) -> tuple[list[str], list[str]]:
    """Validate a SynapticConfig. Returns ``(errors, warnings)``.

    Errors are scientific/foot-gun bugs (a mechanism that silently does nothing, or an
    out-of-range knob); warnings are legal-but-risky combinations.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Prerequisite checks. An OPT-IN mechanism (default-off) that the user explicitly
    #    enabled without its prerequisite is a foot-gun -> error. A DEFAULT-ON dependent
    #    whose prerequisite was ablated (e.g. bio_no_presyn also silences doc2/stochastic,
    #    or bio_no_hebbian silences bdnf) is the EXPECTED ablation consequence, not an error.
    for m in MECHANISMS:
        if m.default_on or not is_mechanism_on(cfg, m.field):
            continue
        for prereq in m.requires:
            if not is_mechanism_on(cfg, prereq):
                errors.append(
                    f"{m.mechanism!r} is enabled ({m.field}={getattr(cfg, m.field)!r}) but "
                    f"its prerequisite {prereq!r} is off — it will silently do nothing. "
                    f"Enable {prereq} or disable {m.field}."
                )

    # 2. Range checks on knobs that can break dynamics if mis-set.
    if not 0.0 <= cfg.stochastic_train_frac <= 1.0:
        errors.append(
            f"stochastic_train_frac must be in [0,1], got {cfg.stochastic_train_frac}"
        )
    if cfg.bistable_latch:
        if cfg.latch_hill_n <= 0:
            errors.append(f"latch_hill_n must be > 0, got {cfg.latch_hill_n}")
        if cfg.latch_hill_k <= 0:
            errors.append(f"latch_hill_k must be > 0, got {cfg.latch_hill_k}")
        if not 0.0 <= cfg.latch_pp1_basal <= 1.0:
            errors.append(f"latch_pp1_basal must be in [0,1], got {cfg.latch_pp1_basal}")
        if cfg.latch_ltd_thr >= cfg.camkii_thr:
            errors.append(
                f"latch_ltd_thr ({cfg.latch_ltd_thr}) must be < camkii_thr "
                f"({cfg.camkii_thr}) so the BCM curve has a neutral zone where the latch holds"
            )

    # 3. Risky-but-legal combinations -> warnings.
    if cfg.use_flex_attention:
        warnings.append(
            "use_flex_attention=True is PREFILL-ONLY; it cannot serve KV-cache decoding. "
            "Use it for training/prefill benchmarks only."
        )

    return errors, warnings


def assert_valid_config(cfg: SynapticConfig) -> list[str]:
    """Raise ``ValueError`` on any validation error; return the warnings list."""
    errors, warnings = validate_config(cfg)
    if errors:
        raise ValueError(
            "Invalid SynapticConfig:\n  - " + "\n  - ".join(errors)
        )
    return warnings


def _registry_self_check() -> None:
    """Fail fast if the registry drifts from SynapticConfig (mirrors the census guard)."""
    cfg_fields = {f.name for f in fields(SynapticConfig)}
    for m in MECHANISMS:
        if m.field not in cfg_fields:
            raise RuntimeError(f"MECHANISMS references unknown field {m.field!r}")
        for prereq in m.requires:
            if prereq not in _BY_FIELD:
                raise RuntimeError(
                    f"{m.field!r} requires {prereq!r}, which is not a registered mechanism"
                )
    for preset, overrides in ABLATION_PRESETS.items():
        for field in overrides:
            if field not in cfg_fields:
                raise RuntimeError(
                    f"preset {preset!r} overrides unknown field {field!r}"
                )


_registry_self_check()
