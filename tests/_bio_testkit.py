"""
bio_testkit — shared test utilities for the bio_inspired_nanochat suite (bead eqyk.1).

This is the foundation the whole roadmap's test discipline builds on: every later
"unit tests + e2e + detailed logging" bead imports these helpers so tests are
*consistent, deterministic, and diagnostic*. Keep this module dependency-light
(torch + stdlib only) and side-effect free on import.

What lives here
---------------
- Determinism:        ``set_seed`` (python/numpy/torch + deterministic flags).
- Tensor diagnostics: ``tensor_stats`` / ``summarize`` / ``assert_finite`` — the
                      "detailed logging" primitive (mean/std/min/max/‖·‖/NaN/Inf).
- Tiny model factories: ``make_tiny_synaptic`` / ``make_tiny_vanilla`` —
                      small CPU models for fast unit tests, built from the exact
                      config fields proven in tests/test_engine.py.
- Golden artifacts:   ``assert_golden`` / ``save_golden`` — lock numerical
                      semantics under tests/golden/ (used by parity/property beads).
- Capability probes:  ``cuda_available`` / ``rustbpe_available``.

The companion ``conftest.py`` exposes most of these as pytest fixtures; this module
is the importable API for tests that want them directly.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
GOLDEN_DIR = TESTS_DIR / "golden"

# Tolerances used across the suite (kept here so every bead uses the same ones).
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4
# bf16/fp16 paths need looser tolerances.
LOOSE_ATOL = 1e-3
LOOSE_RTOL = 1e-2


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 0, *, deterministic: bool = True) -> torch.Generator:
    """Seed python ``random``, numpy, and torch (CPU+CUDA) for reproducible tests.

    Returns a seeded ``torch.Generator`` callers can thread into ops that take a
    ``generator=`` argument (preferred over relying on the global RNG).

    With ``deterministic=True`` we also request deterministic cuDNN/algorithms so
    same-seed runs match bit-for-bit where torch supports it. We DO NOT call
    ``torch.use_deterministic_algorithms(True)`` globally because several ops in
    the bio stack lack deterministic kernels and would raise; determinism here is
    "best effort + seeded", which is what unit tests need. Beads that require
    strict bitwise determinism (see hm4.4 seed/determinism policy) opt in locally.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


# --------------------------------------------------------------------------- #
# Tensor diagnostics — the "detailed logging" primitive
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TensorStats:
    """Cheap, log-friendly summary of a tensor. ``str(stats)`` is one tidy line."""

    shape: tuple[int, ...]
    dtype: str
    numel: int
    nan: int
    inf: int
    mean: float
    std: float
    min: float
    max: float
    absmax: float

    @property
    def finite(self) -> bool:
        return self.nan == 0 and self.inf == 0

    def __str__(self) -> str:  # pragma: no cover - formatting only
        flag = "" if self.finite else f"  ⚠ NaN={self.nan} Inf={self.inf}"
        return (
            f"{tuple(self.shape)} {self.dtype} "
            f"mean={self.mean:+.4g} std={self.std:.4g} "
            f"min={self.min:+.4g} max={self.max:+.4g} |max|={self.absmax:.4g}{flag}"
        )


def tensor_stats(t: torch.Tensor) -> TensorStats:
    """Compute a :class:`TensorStats` without ever raising on NaN/Inf/empty input."""
    t = t.detach()
    numel = t.numel()
    if numel == 0:
        return TensorStats(tuple(t.shape), str(t.dtype), 0, 0, 0,
                           math.nan, math.nan, math.nan, math.nan, math.nan)
    tf = t.float()
    nan = int(torch.isnan(tf).sum().item())
    inf = int(torch.isinf(tf).sum().item())
    finite_mask = torch.isfinite(tf)
    if finite_mask.any():
        f = tf[finite_mask]
        mean = float(f.mean().item())
        std = float(f.std(unbiased=False).item()) if f.numel() > 1 else 0.0
        mn = float(f.min().item())
        mx = float(f.max().item())
        absmax = float(f.abs().max().item())
    else:
        mean = std = mn = mx = absmax = math.nan
    return TensorStats(tuple(t.shape), str(t.dtype), numel, nan, inf,
                       mean, std, mn, mx, absmax)


def summarize(name: str, t: torch.Tensor) -> str:
    """One-line, log-ready summary, e.g. ``logits: (2,16,97) float32 mean=...``."""
    return f"{name}: {tensor_stats(t)}"


def assert_finite(t: torch.Tensor, name: str = "tensor") -> None:
    """Assert ``t`` has no NaN/Inf, with a diagnostic message on failure."""
    s = tensor_stats(t)
    assert s.finite, f"{name} is not finite -> {s}"


# --------------------------------------------------------------------------- #
# Capability probes
# --------------------------------------------------------------------------- #
def cuda_available() -> bool:
    return torch.cuda.is_available()


def rustbpe_available() -> bool:
    """The compiled Rust extension (tokenizer + CPU kernels) may be unbuilt."""
    try:
        import rustbpe  # noqa: F401  # ty: ignore[unresolved-import]
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Tiny model factories (CPU, fast). Field names mirror tests/test_engine.py.
# --------------------------------------------------------------------------- #
# Defaults sized for sub-second forward/backward on CPU.
TINY = dict(sequence_len=32, vocab_size=97, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)


def make_tiny_synaptic(seed: int = 0, *, train: bool = False, **overrides):
    """Build a tiny ``GPTSynaptic`` model on CPU. Returns the model (eval by default).

    ``overrides`` patch the :class:`GPTSynapticConfig` fields (e.g. ``n_layer=1``).
    """
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig

    set_seed(seed)
    # ty cannot statically match **dict unpacking to the dataclass fields here.
    cfg = GPTSynapticConfig(**{**TINY, **overrides})  # ty: ignore[invalid-argument-type]
    model = GPTSynaptic(cfg)
    model.train(train)
    return model


def make_tiny_vanilla(seed: int = 0, *, train: bool = False, **overrides):
    """Build a tiny vanilla ``GPT`` model on CPU (baseline for bio-vs-vanilla tests)."""
    from bio_inspired_nanochat.gpt import GPT, GPTConfig

    set_seed(seed)
    # ty cannot statically match **dict unpacking to the dataclass fields here.
    cfg = GPTConfig(**{**TINY, **overrides})  # ty: ignore[invalid-argument-type]
    model = GPT(cfg)
    model.train(train)
    return model


def random_tokens(batch: int = 2, seq: int = 16, vocab: int = 97, seed: int | None = None) -> torch.Tensor:
    """Deterministic random token ids ``(batch, seq)`` for feeding tiny models."""
    gen = torch.Generator()
    gen.manual_seed(0 if seed is None else seed)
    return torch.randint(0, vocab, (batch, seq), generator=gen, dtype=torch.long)


def count_params(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


# --------------------------------------------------------------------------- #
# Golden artifacts — lock numerical semantics (used by parity/property beads)
# --------------------------------------------------------------------------- #
# Set BIO_UPDATE_GOLDEN=1 to (re)write goldens after an *intended* semantics change.
UPDATE_GOLDEN_ENV = "BIO_UPDATE_GOLDEN"


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.npy"


def save_golden(name: str, tensor: torch.Tensor) -> Path:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    p = golden_path(name)
    np.save(p, tensor.detach().float().cpu().numpy())
    return p


def load_golden(name: str) -> np.ndarray:
    p = golden_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"golden '{name}' missing ({p}). Generate it with {UPDATE_GOLDEN_ENV}=1."
        )
    return np.load(p)


def assert_golden(
    name: str,
    tensor: torch.Tensor,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """Compare ``tensor`` against committed golden ``name``.

    If the golden is missing OR ``BIO_UPDATE_GOLDEN=1``, the golden is written and
    the check is skipped (first-run / intentional-update bootstrap). Otherwise the
    values must match within tolerance or the test fails with a stats diff.
    """
    arr = tensor.detach().float().cpu().numpy()
    updating = os.environ.get(UPDATE_GOLDEN_ENV) == "1"
    if updating or not golden_path(name).exists():
        save_golden(name, tensor)
        return
    ref = load_golden(name)
    assert ref.shape == arr.shape, f"golden '{name}' shape {ref.shape} != actual {arr.shape}"
    a = torch.from_numpy(arr)
    r = torch.from_numpy(ref)
    if not torch.allclose(a, r, atol=atol, rtol=rtol):
        diff = (a - r).abs()
        raise AssertionError(
            f"golden '{name}' mismatch: max|Δ|={diff.max().item():.3g} "
            f"(atol={atol}, rtol={rtol})\n  actual {tensor_stats(a)}\n  golden {tensor_stats(r)}"
        )
