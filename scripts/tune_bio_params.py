#!/usr/bin/env python3
"""
Bio-Hyperparameter Tuner using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

This script optimizes the 'Biological Hyperparameters' of the SynapticConfig
(time constants, gains, enzyme kinetics) to maximize performance on a
synthetic 'Associative Recall' task that stresses working memory.

Features:
- Uses cma (CMA-ES) for derivative-free optimization in high-dimensional space.
- Rich visualization of the optimization landscape and population stats.
- Synthetic task generation for fast, reproducible iteration.
- Robust logging.

Usage:
    # Run a single deterministic evaluation (baseline/default config):
    uv run python -m scripts.tune_bio_params eval --seed 42

    # Run a single evaluation for an explicit 10D vector (comma-separated):
    uv run python -m scripts.tune_bio_params eval --seed 42 --vector "0.9,0.6,0.06,0.05,0.08,8.0,3.0,0.55,0.0,1.0"

    # Validate a proxy objective vs full eval, using learning-curve extrapolation (LCE):
    uv run python -m scripts.tune_bio_params proxy --seed 123 --proxy-steps 30 --full-steps 100 --mode lce

    # Run CMA-ES over the default 10D search space:
    uv run python -m scripts.tune_bio_params optimize --seed 1337 --run-dir runs/cmaes/top10

    # Use proxy eval for early generations, then switch to full eval:
    uv run python -m scripts.tune_bio_params optimize --seed 1337 --run-dir runs/cmaes/top10 \\
        --steps 200 --proxy-steps 50 --proxy-generations 10 --proxy-mode lce

    # Distributed (multi-GPU) population eval via torchrun (rank0 controller):
    uv run torchrun --standalone --nproc_per_node=8 -m scripts.tune_bio_params \\
        optimize --distributed --seed 1337 --device cuda --run-dir runs/cmaes/top10
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import pickle  # nosec B403
import time
from collections.abc import Callable
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Protocol, Sequence, cast

import cma as cma_module
import numpy as np
import torch
import torch.distributed as torch_dist
from torch.utils.tensorboard import SummaryWriter

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich import box
from rich.syntax import Syntax

from bio_inspired_nanochat.checkpoint_manager import config_hash
from bio_inspired_nanochat.results_registry import DEFAULT_REGISTRY, append_record, make_record
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig


class _CmaApi(Protocol):
    CMAEvolutionStrategy: Callable[
        [Sequence[float] | np.ndarray, float, dict[str, Any]], Any
    ]


class _DistributedApi(Protocol):
    """Collectives used by distributed population evaluation."""

    def is_initialized(self) -> bool: ...

    def init_process_group(self, *, backend: str, init_method: str) -> None: ...

    def broadcast(self, tensor: torch.Tensor, *, src: int) -> None: ...

    def all_gather(
        self, tensor_list: list[torch.Tensor], tensor: torch.Tensor
    ) -> None: ...

    def destroy_process_group(self) -> None: ...


cma = cast(_CmaApi, cma_module)
dist = cast(_DistributedApi, torch_dist)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use a smaller model for fast tuning loop (biophysics should generalize)
MODEL_CONFIG = GPTSynapticConfig(
    sequence_len=256,
    vocab_size=1024,  # small vocab for synthetic task
    n_layer=2,        # shallow
    n_head=4,
    n_kv_head=4,
    n_embd=128,       # thin
    synapses=True,
    use_moe=False,    # disable MoE to focus on synaptic dynamics
)

# Optimization settings
POPULATION_SIZE = 8       # CMA-ES population size
MAX_GENERATIONS = 50      # How long to run
STEPS_PER_EVAL = 100      # Training steps per candidate evaluation
BATCH_SIZE = 16
PENALTY_LOSS = 100.0

# -----------------------------------------------------------------------------
# Parameter Definitions (Search Space)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    name: str
    init: float
    lower: float
    upper: float
    log_scale: bool


# NOTE: This script targets the Python reference synaptic attention path
# (`SynapticPresyn.release`), so these 10 parameters are chosen to actually
# affect model behavior in that path (vs Rust-only compat knobs).
TOP10_PARAM_SPECS: tuple[ParamSpec, ...] = (
    # The CANONICAL live presyn knobs (or4t). The legacy sigmoid release params alpha_c /
    # syt1_slope / syt7_slope / cpx_thresh were removed; tuning them had no effect on the model.
    # tau_c is an exp calcium-decay TIME CONSTANT (retention = exp(-1/tau_c); 8j9.2/x6z4).
    ParamSpec("tau_c", 6.0, 2.0, 20.0, True),
    ParamSpec("alpha_ca", 0.55, 0.05, 2.00, True),       # calcium influx gain
    ParamSpec("syt_fast_kd", 0.4, 0.10, 2.00, True),     # Syt1 (fast) Hill Kd
    ParamSpec("syt_slow_kd", 1.0, 0.20, 5.00, True),     # Syt7 (slow) Hill Kd
    ParamSpec("doc2_gain", 0.08, 0.00, 0.50, False),     # Doc2 facilitation gain
    ParamSpec("complexin_bias", 0.0, 0.00, 2.00, False),  # complexin inhibitory bias
    ParamSpec("prime_rate", 0.075, 0.005, 0.30, True),
    ParamSpec("unprime_per_release", 0.05, 0.001, 0.30, True),
    ParamSpec("nsf_recover", 0.08, 0.001, 0.30, True),
    ParamSpec("lambda_loge", 1.0, 0.00, 5.00, False),
)

POSTSYN_PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("post_fast_decay", 0.95, 0.80, 0.999, False),
    ParamSpec("post_fast_lr", 0.0015, 0.0001, 0.05, True),
    ParamSpec("post_slow_lr", 0.0005, 0.00005, 0.01, True),
    ParamSpec("post_trace_decay", 0.96, 0.80, 0.999, False),
    ParamSpec("fast_weight_eta", 0.5, 0.05, 2.0, True),
    ParamSpec("camkii_up", 0.05, 0.005, 0.5, True),
    ParamSpec("pp1_tau", 0.985, 0.80, 0.999, False),
    ParamSpec("bdnf_tau", 0.985, 0.80, 0.999, False),
    ParamSpec("bdnf_scale", 1.0, 0.1, 5.0, True),
)

STRUCTURAL_PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("energy_fill", 0.02, 0.001, 0.10, True),
    ParamSpec("energy_use", 0.02, 0.001, 0.10, True),
    ParamSpec("energy_cost_rel", 0.015, 0.001, 0.10, True),
    ParamSpec("structural_tau_util", 0.2, 0.05, 1.0, True),
)

FULL_LIVE_PARAM_SPECS: tuple[ParamSpec, ...] = (
    TOP10_PARAM_SPECS + POSTSYN_PARAM_SPECS + STRUCTURAL_PARAM_SPECS
)

PARAM_SPACES: dict[str, tuple[ParamSpec, ...]] = {
    "top10": TOP10_PARAM_SPECS,
    "presyn": TOP10_PARAM_SPECS,
    "postsyn": POSTSYN_PARAM_SPECS,
    "structural": STRUCTURAL_PARAM_SPECS,
    "full": FULL_LIVE_PARAM_SPECS,
}


def _validate_param_specs(specs: Sequence[ParamSpec]) -> None:
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("Duplicate ParamSpec.name in search space")
    for spec in specs:
        if spec.lower >= spec.upper:
            raise ValueError(f"Invalid bounds for {spec.name}: {spec.lower} >= {spec.upper}")
        if spec.log_scale and spec.lower <= 0:
            raise ValueError(f"log_scale params must be >0 lower bound: {spec.name}")


def _build_synaptic_config(overrides: Dict[str, float]) -> SynapticConfig:
    cfg = SynapticConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(
                f"SynapticConfig has no field {key!r}. "
                "Update TOP10_PARAM_SPECS to match the real config."
            )
        setattr(cfg, key, float(value))

    # For objective stability, disable stochastic release by default.
    cfg.stochastic_train_frac = 0.0
    return cfg


def encode_params(config: SynapticConfig, specs: Sequence[ParamSpec]) -> np.ndarray:
    """Extract optimization vector from config using the provided ParamSpec list."""
    vals: list[float] = []
    for spec in specs:
        v = float(getattr(config, spec.name))
        vals.append(math.log(v) if spec.log_scale else v)
    return np.array(vals, dtype=np.float64)


def decode_params(vector: np.ndarray, specs: Sequence[ParamSpec]) -> Dict[str, float]:
    """Convert optimization vector back to a bounded parameter dict."""
    if vector.shape != (len(specs),):
        raise ValueError(f"Expected vector shape ({len(specs)},), got {vector.shape}")
    out: dict[str, float] = {}
    for i, spec in enumerate(specs):
        val = float(vector[i])
        if spec.log_scale:
            val = math.exp(val)
        val = max(spec.lower, min(spec.upper, val))
        out[spec.name] = val
    return out


def _cma_bounds(specs: Sequence[ParamSpec]) -> tuple[list[float], list[float]]:
    lbs: list[float] = []
    ubs: list[float] = []
    for spec in specs:
        if spec.log_scale:
            lbs.append(math.log(spec.lower))
            ubs.append(math.log(spec.upper))
        else:
            lbs.append(spec.lower)
            ubs.append(spec.upper)
    return lbs, ubs


def _parse_vector(text: str, specs: Sequence[ParamSpec]) -> np.ndarray:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != len(specs):
        raise ValueError(
            f"--vector expected {len(specs)} comma-separated floats, got {len(parts)}"
        )
    vals: list[float] = []
    for part, spec in zip(parts, specs):
        v = float(part)
        if not (spec.lower <= v <= spec.upper):
            raise ValueError(
                f"--vector value for {spec.name}={v} out of bounds "
                f"[{spec.lower}, {spec.upper}]"
            )
        vals.append(math.log(v) if spec.log_scale else v)
    return np.array(vals, dtype=np.float64)

# -----------------------------------------------------------------------------
# Sanity / Toy Objectives
# -----------------------------------------------------------------------------


def _rosenbrock_2d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def run_rosenbrock_2d_cmaes(
    *,
    seed: int,
    iterations: int = 80,
    popsize: int = 8,
    sigma0: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    Deterministic toy convergence check for validating the CMA-ES code path.
    The Rosenbrock 2D minimum is at (1, 1).

    Note: `seed=0` is treated as falsy by some libraries; prefer non-zero seeds.
    """
    if seed == 0:
        raise ValueError("Use a non-zero seed for deterministic Rosenbrock sanity checks")

    es = cma.CMAEvolutionStrategy(
        [-1.2, 1.0],
        float(sigma0),
        {"seed": int(seed), "popsize": int(popsize), "verbose": -1},
    )
    for _ in range(int(iterations)):
        xs = es.ask()
        fs = [_rosenbrock_2d(np.asarray(x, dtype=np.float64)) for x in xs]
        es.tell(xs, fs)
        if es.stop():
            break

    xbest = np.asarray(es.result.xbest, dtype=np.float64)
    fbest = float(es.result.fbest)
    return xbest, fbest

# -----------------------------------------------------------------------------
# Synthetic Task: Associative Recall / Copy
# -----------------------------------------------------------------------------

def generate_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: str, *, seed: int | None = None
):
    """
    Generates a 'Needle in a Haystack' / Copy task.
    Sequence: [Key1] [Val1] ... [KeyK] [ValK] ... [Query:Key1] -> [Target:Val1]
    The model must use working memory (synapses) to store bindings.

    Pass ``seed`` for a REPRODUCIBLE batch (used to build a fixed held-out set that is
    identical across candidates, so the held-out objective is a fair comparison — 74f.1).
    """
    if seq_len % 2 != 0:
        raise ValueError("generate_batch copy task requires an even seq_len")

    # Simple repeated ngram pattern.
    # We generate a random sequence of (L/2) tokens, then repeat it:
    #   seq = [data, data]
    # Training uses *next-token* targets (like pretraining), so we only score
    # predictions that generate the second half from the first half.
    half = seq_len // 2
    if seed is None:
        data = torch.randint(0, vocab_size, (batch_size, half), device=device)
    else:
        gen = torch.Generator().manual_seed(int(seed))
        data = torch.randint(0, vocab_size, (batch_size, half), generator=gen).to(device)

    # Inputs: [Data] [Data]
    x = torch.cat([data, data], dim=1)  # (B, T)

    # Targets are aligned as next-token labels: targets[t] = x[t+1].
    # We ignore the first-half random tokens by setting targets to -1, and we
    # only score the transition that predicts the second half from the first.
    #
    # Example (T=8, half=4):
    #   x = [a b c d a b c d]
    #   y = [-1 -1 -1 a b c d -1]  (loss at positions 3..6)
    y = torch.full_like(x, -1)
    y[:, half - 1 : seq_len - 1] = x[:, half:seq_len]

    return x, y

# -----------------------------------------------------------------------------
# Evaluation Loop
# -----------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: str
    comm_device: torch.device


def _get_dist_env() -> tuple[int, int, int] | None:
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, local_rank, world_size
    return None


def _init_distributed(base_device: str) -> DistInfo:
    env = _get_dist_env()
    if env is None:
        return DistInfo(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            backend="",
            device=base_device,
            comm_device=torch.device("cpu"),
        )

    rank, local_rank, world_size = env
    use_cuda = base_device.startswith("cuda") and torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    if use_cuda:
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    device = f"cuda:{local_rank}" if use_cuda else "cpu"
    comm_device = torch.device(device) if backend == "nccl" else torch.device("cpu")
    return DistInfo(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        device=device,
        comm_device=comm_device,
    )


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    progress_jsonl: Path
    best_params_json: Path
    es_latest_pkl: Path
    es_manifest_json: Path
    tb_dir: Path
    tb_writer: SummaryWriter | None


def _prepare_run_artifacts(args: argparse.Namespace) -> RunArtifacts | None:
    if args.run_dir is None:
        if args.resume:
            raise ValueError("--resume requires --run-dir")
        return None

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tb"
    run_id = f"tune-{config_hash({'run_dir': str(run_dir.resolve()), 'seed': int(args.seed)})}"

    tb_writer = None
    if not args.no_tensorboard:
        tb_writer = SummaryWriter(log_dir=str(tb_dir))

    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        progress_jsonl=run_dir / "progress.jsonl",
        best_params_json=run_dir / "best_params.json",
        es_latest_pkl=run_dir / "es_latest.pkl",
        es_manifest_json=run_dir / "es_manifest.json",
        tb_dir=tb_dir,
        tb_writer=tb_writer,
    )


def _load_best_params(best_params_json: Path) -> tuple[float, dict[str, float]]:
    if not best_params_json.exists():
        return float("inf"), {}
    try:
        data = json.loads(best_params_json.read_text(encoding="utf-8"))
        best_loss = float(data["best_loss"])
        best_params = {str(k): float(v) for k, v in data["best_params"].items()}
        return best_loss, best_params
    except Exception:
        return float("inf"), {}


def _save_best_params(
    best_params_json: Path,
    *,
    run_id: str,
    best_loss: float,
    best_params: dict[str, float],
    gen: int,
) -> None:
    payload = {
        "run_id": run_id,
        "best_loss": float(best_loss),
        "best_params": {k: float(v) for k, v in best_params.items()},
        "generation": int(gen),
        "saved_at_unix": time.time(),
    }
    best_params_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_verified_cma_checkpoint(
    artifacts: RunArtifacts,
    *,
    specs: Sequence[ParamSpec],
    allow_unverified: bool = False,
    console: Console | None = None,
) -> Any:
    """Safely resume CMAEvolutionStrategy state with cryptographic integrity verification.

    Refuses to deserialize unverified pickle bytes from untrusted or tampered run directories.
    Fails closed with clear Rich error diagnostics if the SHA-256 hash does not match the manifest
    or if the checkpoint dimensions do not match the target search space.
    """
    _console = console or Console()
    if not artifacts.es_latest_pkl.exists():
        _console.print(f"[bold red]Checkpoint file missing: {artifacts.es_latest_pkl}[/bold red]")
        raise FileNotFoundError(f"Missing checkpoint: {artifacts.es_latest_pkl}")

    raw_bytes = artifacts.es_latest_pkl.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    if artifacts.es_manifest_json.exists():
        try:
            manifest_doc = json.loads(artifacts.es_manifest_json.read_text(encoding="utf-8"))
        except Exception as exc:
            _console.print(
                f"[bold red]Checkpoint manifest at {artifacts.es_manifest_json} is corrupted: {exc}[/bold red]"
            )
            raise ValueError(
                f"Corrupt checkpoint manifest at {artifacts.es_manifest_json}: {exc}"
            ) from exc

        if not isinstance(manifest_doc, dict) or manifest_doc.get("format") != "cmaes-checkpoint-manifest-v1":
            _console.print(
                f"[bold red]Unsupported checkpoint manifest format in {artifacts.es_manifest_json}[/bold red]"
            )
            raise ValueError(
                f"Unsupported checkpoint manifest format in {artifacts.es_manifest_json}: "
                f"{manifest_doc.get('format') if isinstance(manifest_doc, dict) else type(manifest_doc)}"
            )

        expected_sha256 = manifest_doc.get("sha256")
        if not isinstance(expected_sha256, str) or not hmac.compare_digest(actual_sha256, expected_sha256):
            _console.print(
                f"[bold red]Security Error: Checkpoint integrity verification failed for {artifacts.es_latest_pkl}! "
                f"Computed SHA-256 ({actual_sha256[:16]}...) does not match manifest SHA-256 ({str(expected_sha256)[:16]}...). "
                f"Refusing to deserialize potentially tampered checkpoint.[/bold red]"
            )
            raise ValueError(
                f"Checkpoint integrity verification failed for {artifacts.es_latest_pkl}: "
                f"SHA-256 digest mismatch against {artifacts.es_manifest_json}"
            )

        expected_dim = manifest_doc.get("dim")
        if expected_dim is not None and int(expected_dim) != len(specs):
            _console.print(
                f"[bold red]Checkpoint dimension mismatch: manifest recorded dim={expected_dim}, "
                f"current search space has dim={len(specs)}.[/bold red]"
            )
            raise ValueError(
                f"Checkpoint search space mismatch: manifest has dim={expected_dim}, expected {len(specs)}"
            )
    else:
        if not allow_unverified:
            _console.print(
                f"[bold red]Security Error: Checkpoint manifest {artifacts.es_manifest_json} not found. "
                f"Refusing to deserialize unverified legacy pickle bytes from {artifacts.es_latest_pkl}. "
                f"Pass --allow-unverified-checkpoint to explicitly authorize loading legacy unverified checkpoints.[/bold red]"
            )
            raise ValueError(
                f"Unverified checkpoint: {artifacts.es_manifest_json} is missing. "
                f"Refusing to execute unverified pickle bytes from {artifacts.es_latest_pkl}. "
                f"Pass --allow-unverified-checkpoint to authorize."
            )
        _console.print(
            f"[bold yellow]Warning: Loading unverified legacy checkpoint {artifacts.es_latest_pkl} "
            f"without cryptographic manifest verification (--allow-unverified-checkpoint enabled).[/bold yellow]"
        )

    try:
        es = pickle.loads(raw_bytes)  # nosec B301
    except Exception as exc:
        _console.print(f"[bold red]Failed to deserialize verified checkpoint: {exc}[/bold red]")
        raise ValueError(f"Failed to deserialize checkpoint {artifacts.es_latest_pkl}: {exc}") from exc

    es_dim = getattr(es, "N", None)
    if es_dim is not None and int(es_dim) != len(specs):
        _console.print(
            f"[bold red]Restored CMA-ES dimension mismatch: object has N={es_dim}, expected {len(specs)}.[/bold red]"
        )
        raise ValueError(
            f"Restored CMA-ES dimension mismatch: object has N={es_dim}, expected {len(specs)}"
        )

    return es


def _append_progress(progress_jsonl: Path, payload: dict[str, object]) -> None:
    with progress_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _log_tensorboard(
    tb_writer: SummaryWriter,
    *,
    gen: int,
    fitnesses: list[float],
    best_loss: float,
    es: object,
) -> None:
    writer = tb_writer
    if writer is None:
        return

    gen_min = float(min(fitnesses))
    gen_mean = float(np.mean(fitnesses))
    writer.add_scalar("fitness/min", gen_min, gen)
    writer.add_scalar("fitness/mean", gen_mean, gen)
    writer.add_scalar("fitness/best_so_far", float(best_loss), gen)

    fit_t = torch.tensor(fitnesses, dtype=torch.float32)
    writer.add_histogram("fitness/population", fit_t, gen)

    cov = getattr(es, "C", None)
    if cov is not None:
        cov = np.asarray(cov, dtype=np.float32)
        cov_abs = np.abs(cov)
        denom = float(cov_abs.max()) if cov_abs.size else 1.0
        denom = denom if denom > 0 else 1.0
        img = (cov_abs / denom).clip(0.0, 1.0)
        img3 = np.stack([img, img, img], axis=0)  # (3, D, D)
        writer.add_image("covariance/abs_norm", img3, gen, dataformats="CHW")

    sigma = getattr(es, "sigma", None)
    if sigma is not None:
        writer.add_scalar("cma/sigma", float(sigma), gen)

    writer.flush()


def _stagnation_improvement_frac(
    *,
    best_loss_history: Sequence[float],
    window_gens: int,
) -> float | None:
    """Return fractional best-loss improvement over the last `window_gens` generations.

    Example: if best loss went from 1.0 to 0.99, returns 0.01.
    Returns None if there isn't enough history or values are non-finite.
    """
    window = int(window_gens)
    if window <= 0:
        return None
    if len(best_loss_history) < window + 1:
        return None
    prev = float(best_loss_history[-(window + 1)])
    cur = float(best_loss_history[-1])
    if not (math.isfinite(prev) and math.isfinite(cur) and prev > 0.0):
        return None
    return (prev - cur) / prev


@dataclass(frozen=True)
class CandidateEvalResult:
    mean_last_loss: float
    steps_run: int
    lce_pred_loss: float | None = None
    losses: list[float] | None = None
    held_out_loss: float | None = None  # loss on a FIXED held-out set (74f.1)

    @property
    def objective(self) -> float:
        """The value to minimize: the held-out loss when available (74f.1 — it is far
        less noisy than the last-k training loss and measures generalization), else the
        mean last-k training loss."""
        if self.held_out_loss is not None and math.isfinite(self.held_out_loss):
            return self.held_out_loss
        return self.mean_last_loss


def _lce_predict_from_points(
    points: Sequence[tuple[int, float]],
    *,
    target_step: int,
    exponent: float,
) -> float | None:
    """Predict loss at `target_step` via fixed-exponent power-law tail fit.

    Model: loss(step) = a + b * step^{-exponent}, exponent fixed.
    We fit (a, b) via least squares over the provided points, and only accept
    b>=0 (monotone-improving curve). Returns None on insufficient/invalid data.
    """
    if target_step <= 0:
        return None
    if not (math.isfinite(exponent) and exponent > 0.0):
        return None

    steps: list[float] = []
    losses: list[float] = []
    for step, loss in points:
        step_i = int(step)
        loss_f = float(loss)
        if step_i <= 0 or step_i > int(target_step) or not math.isfinite(loss_f):
            continue
        steps.append(float(step_i))
        losses.append(loss_f)

    if len(losses) < 4:
        return None

    z = np.power(np.asarray(steps, dtype=np.float64), -float(exponent))
    X = np.stack([np.ones_like(z), z], axis=1)
    y = np.asarray(losses, dtype=np.float64)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    if not (math.isfinite(a) and math.isfinite(b)) or b < 0.0:
        return None

    pred = a + b * (float(target_step) ** (-float(exponent)))
    return float(pred) if math.isfinite(pred) else None


def evaluate_candidate_detailed(
    solution_vector: np.ndarray,
    *,
    specs: Sequence[ParamSpec],
    seed: int,
    steps: int,
    batch_size: int,
    device: str,
    lr: float,
    weight_decay: float,
    timeout_seconds: float | None,
    max_retries: int,
    raise_on_error: bool,
    mean_last: int = 10,
    lce_target_steps: int | None = None,
    lce_exponent: float = 0.5,
    lce_tail_points: int = 50,
    lce_stride: int = 1,
    record_losses: bool = False,
    model_config: GPTSynapticConfig = MODEL_CONFIG,
    reset_state: bool = True,
    held_out_batches: int = 8,
    held_out_seed: int = 12345,
) -> CandidateEvalResult:
    """
    Instantiates a model with specific bio-parameters and runs a short training loop.

    Returns a structured result that can optionally include an LCE prediction and/or
    the full per-step loss curve.
    """
    steps_i = max(0, int(steps))
    mean_last_i = max(1, min(int(mean_last), steps_i if steps_i > 0 else 1))
    lce_target = int(lce_target_steps) if lce_target_steps is not None else None
    lce_stride_i = max(1, int(lce_stride))
    lce_tail_i = max(4, int(lce_tail_points))

    use_lce = lce_target is not None and lce_target > steps_i

    attempts = max(0, int(max_retries)) + 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            # 1) Decode parameters
            param_dict = decode_params(solution_vector, specs)

            # 2) Build config
            syn_cfg = _build_synaptic_config(param_dict)
            model_cfg = replace(model_config, syn_cfg=syn_cfg)

            # 3) Build model
            _seed_everything(seed)
            model = GPTSynaptic(model_cfg).to(device)
            model.train()
            can_reset = reset_state and hasattr(model, "reset_sequence_state")

            # 4) Optimizer
            optim = torch.optim.AdamW(
                model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
            )

            # 5) Train loop
            mean_window: deque[float] = deque(maxlen=mean_last_i)
            lce_points: deque[tuple[int, float]] | None = (
                deque(maxlen=lce_tail_i) if use_lce else None
            )
            all_losses: list[float] | None = [] if record_losses else None

            deadline = (
                (time.monotonic() + float(timeout_seconds))
                if timeout_seconds is not None
                else None
            )
            for step_idx in range(steps_i):
                if deadline is not None and (step_idx % 10) == 0 and time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Candidate evaluation exceeded timeout_seconds={timeout_seconds}"
                    )

                # Per-sequence reset of plasticity state (74f.1): each synthetic batch is
                # an independent sequence. WITHOUT this the eligibility/fast-weight state
                # accumulates across batches and the synaptic candidate diverges to NaN ->
                # PENALTY_LOSS, which is a major reason the objective was uninformative.
                if can_reset:
                    model.reset_sequence_state(reset_fast_weights=True)
                x, y = generate_batch(
                    batch_size, model_cfg.sequence_len, model_cfg.vocab_size, device
                )
                _logits, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss encountered during evaluation")

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

                loss_val = float(loss.item())
                mean_window.append(loss_val)
                if all_losses is not None:
                    all_losses.append(loss_val)
                if lce_points is not None and ((step_idx + 1) % lce_stride_i) == 0:
                    lce_points.append((step_idx + 1, loss_val))

            mean_last_loss = float(np.mean(list(mean_window))) if mean_window else PENALTY_LOSS
            if not math.isfinite(mean_last_loss):
                raise FloatingPointError("Non-finite mean_last_loss encountered during evaluation")

            # Held-out evaluation (74f.1): score generalization on a FIXED held-out set
            # (identical across all candidates), which is far less noisy than the last-k
            # training loss and is the objective the optimizer should actually minimize.
            held_out_loss: float | None = None
            if held_out_batches > 0:
                model.eval()
                ho_losses: list[float] = []
                with torch.no_grad():
                    for j in range(int(held_out_batches)):
                        if can_reset:
                            model.reset_sequence_state(reset_fast_weights=True)
                        xho, yho = generate_batch(
                            batch_size,
                            model_cfg.sequence_len,
                            model_cfg.vocab_size,
                            device,
                            seed=int(held_out_seed) + j,
                        )
                        _, lho = model(xho, yho)
                        ho_losses.append(float(lho.item()))
                ho_mean = float(np.mean(ho_losses)) if ho_losses else PENALTY_LOSS
                held_out_loss = ho_mean if math.isfinite(ho_mean) else PENALTY_LOSS

            lce_pred = None
            if use_lce and lce_target is not None and lce_points is not None:
                lce_pred = _lce_predict_from_points(
                    list(lce_points),
                    target_step=int(lce_target),
                    exponent=float(lce_exponent),
                )
                if lce_pred is not None and not math.isfinite(lce_pred):
                    lce_pred = None

            return CandidateEvalResult(
                mean_last_loss=mean_last_loss,
                steps_run=steps_i,
                lce_pred_loss=lce_pred,
                losses=all_losses,
                held_out_loss=held_out_loss,
            )

        except Exception as exc:
            if raise_on_error:
                raise
            last_exc = exc
            if attempt + 1 >= attempts:
                return CandidateEvalResult(mean_last_loss=PENALTY_LOSS, steps_run=steps_i)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    if raise_on_error and last_exc is not None:
        raise last_exc
    return CandidateEvalResult(mean_last_loss=PENALTY_LOSS, steps_run=steps_i)


# -----------------------------------------------------------------------------
# Multi-seed objective + proxy readiness gate (bead 74f.1)
# -----------------------------------------------------------------------------
def multi_seed_objective(
    solution_vector: np.ndarray,
    *,
    specs: Sequence[ParamSpec],
    seeds: Sequence[int],
    steps: int,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
    lr: float = 3e-3,
    weight_decay: float = 0.0,
    model_config: GPTSynapticConfig = MODEL_CONFIG,
    held_out_batches: int = 8,
) -> dict[str, Any]:
    """Held-out objective averaged over several TRAINING seeds (74f.1).

    Averaging over >=3 seeds shrinks the objective's seed-noise by ~sqrt(n), so a signal
    that was below single-seed noise becomes measurable. Returns the per-seed objectives
    plus their mean and (sample) std.
    """
    per_seed: dict[int, float] = {}
    for s in seeds:
        res = evaluate_candidate_detailed(
            solution_vector,
            specs=specs,
            seed=int(s),
            steps=steps,
            batch_size=batch_size,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            timeout_seconds=None,
            max_retries=0,
            raise_on_error=False,
            model_config=model_config,
            held_out_batches=held_out_batches,
        )
        per_seed[int(s)] = res.objective
    vals = np.array(list(per_seed.values()), dtype=np.float64)
    return {
        "per_seed": per_seed,
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "n": int(vals.size),
    }


def readiness_from_objectives(
    good_per_seed: dict[int, float],
    bad_per_seed: dict[int, float],
    *,
    sigma_gate: float = 3.0,
    rel_gate: float = 0.005,
) -> dict[str, Any]:
    """Compute the proxy readiness gate from two configs' per-seed objectives.

    Separated from the (expensive) training so it can be unit-tested deterministically.
    The proxy is READY when the good/bad configs separate by > ``sigma_gate`` units of
    seed-noise AND by >= ``rel_gate`` relative improvement, and the paired test (matched
    seeds, via eval_stats) is significant — i.e. the objective carries signal over noise.
    """
    from bio_inspired_nanochat.eval_stats import paired_comparison

    g = np.array(list(good_per_seed.values()), dtype=np.float64)
    b = np.array(list(bad_per_seed.values()), dtype=np.float64)
    g_std = float(g.std(ddof=1)) if g.size > 1 else 0.0
    b_std = float(b.std(ddof=1)) if b.size > 1 else 0.0
    seed_sigma = math.sqrt((g_std**2 + b_std**2) / 2.0)
    signal = abs(float(g.mean()) - float(b.mean()))
    sigma_sep = signal / seed_sigma if seed_sigma > 1e-12 else float("inf")
    rel = signal / abs(float(b.mean())) if abs(float(b.mean())) > 1e-12 else float("inf")
    paired = paired_comparison(good_per_seed, bad_per_seed, lower_is_better=True)
    p_t = paired.t_p_value if paired else float("nan")
    p_w = paired.wilcoxon_p_value if paired else float("nan")
    return {
        "good_mean": float(g.mean()),
        "bad_mean": float(b.mean()),
        "signal": signal,
        "seed_sigma": seed_sigma,
        "sigma_separation": sigma_sep,
        "relative_improvement": rel,
        "paired_t_p": p_t,
        "paired_wilcoxon_p": p_w,
        "ready": bool(sigma_sep >= sigma_gate and rel >= rel_gate),
        "sigma_gate": sigma_gate,
        "rel_gate": rel_gate,
    }


def proxy_signal_check(
    good_vector: np.ndarray,
    bad_vector: np.ndarray,
    *,
    specs: Sequence[ParamSpec],
    seeds: Sequence[int],
    steps: int,
    sigma_gate: float = 3.0,
    rel_gate: float = 0.005,
    **eval_kw: Any,
) -> dict[str, Any]:
    """End-to-end proxy readiness gate (74f.1 acceptance): run the multi-seed held-out
    objective for a known-good and known-bad config on the SAME seeds and report whether
    they separate by > ``sigma_gate`` of seed-noise."""
    good = multi_seed_objective(good_vector, specs=specs, seeds=seeds, steps=steps, **eval_kw)
    bad = multi_seed_objective(bad_vector, specs=specs, seeds=seeds, steps=steps, **eval_kw)
    gate = readiness_from_objectives(
        good["per_seed"], bad["per_seed"], sigma_gate=sigma_gate, rel_gate=rel_gate
    )
    return {"good": good, "bad": bad, **gate}


def evaluate_candidate(
    solution_vector: np.ndarray,
    *,
    specs: Sequence[ParamSpec],
    seed: int,
    steps: int,
    batch_size: int,
    device: str,
    lr: float,
    weight_decay: float,
    timeout_seconds: float | None,
    max_retries: int,
    raise_on_error: bool,
) -> float:
    """
    Instantiates a model with specific bio-parameters and runs a short training loop.
    Returns: Final Validation Loss (lower is better).
    """
    res = evaluate_candidate_detailed(
        solution_vector,
        specs=specs,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        raise_on_error=raise_on_error,
        mean_last=10,
        lce_target_steps=None,
        record_losses=False,
    )
    return float(res.mean_last_loss)


def _merge_allgathered_fitness(gathered: list[torch.Tensor]) -> list[float]:
    pop = int(gathered[0].numel()) if gathered else 0
    out: list[float] = []
    for i in range(pop):
        v = float("nan")
        for t in gathered:
            val = float(t[i].item())
            if math.isfinite(val):
                v = val
                break
        out.append(v if math.isfinite(v) else PENALTY_LOSS)
    return out


def _distributed_worker_loop(
    args: argparse.Namespace,
    *,
    specs: Sequence[ParamSpec],
    dist_info: DistInfo,
) -> None:
    dim = len(specs)
    comm_device = dist_info.comm_device

    ctrl_len = 6
    cmd_stop = 0
    cmd_eval = 1
    mode_lce = 1

    while True:
        ctrl = torch.empty((ctrl_len,), dtype=torch.int64, device=comm_device)
        dist.broadcast(ctrl, src=0)
        cmd = int(ctrl[0].item())
        pop = int(ctrl[1].item())
        dim_in = int(ctrl[2].item())
        steps_to_run = int(ctrl[3].item())
        mode = int(ctrl[4].item())
        target_steps = int(ctrl[5].item())
        if cmd == cmd_stop:
            break
        if cmd != cmd_eval:
            raise ValueError(f"Unknown distributed command={cmd}")
        if dim_in != dim:
            raise ValueError(f"Worker expected dim={dim}, got {dim_in}")

        sols = torch.empty((pop, dim), dtype=torch.float64, device=comm_device)
        dist.broadcast(sols, src=0)

        fitness = torch.full((pop,), float("nan"), dtype=torch.float64, device=comm_device)
        use_lce = mode == mode_lce and target_steps > steps_to_run
        for sol_idx in range(dist_info.rank, pop, dist_info.world_size):
            res = evaluate_candidate_detailed(
                sols[sol_idx].detach().cpu().numpy(),
                specs=specs,
                seed=int(args.seed),
                steps=int(steps_to_run),
                batch_size=int(args.batch_size),
                device=dist_info.device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                timeout_seconds=args.timeout_seconds,
                max_retries=int(args.retries),
                raise_on_error=False,
                mean_last=10,
                lce_target_steps=int(target_steps) if use_lce else None,
                lce_exponent=float(args.lce_exponent),
                lce_tail_points=int(args.lce_tail_points),
                lce_stride=int(args.lce_stride),
                record_losses=False,
            )
            if use_lce and res.lce_pred_loss is not None:
                fitness[sol_idx] = float(res.lce_pred_loss)
            else:
                fitness[sol_idx] = float(res.mean_last_loss)

        gathered = [torch.empty_like(fitness) for _ in range(dist_info.world_size)]
        dist.all_gather(gathered, fitness)

# -----------------------------------------------------------------------------
# Main Optimization Script
# -----------------------------------------------------------------------------

def _cmd_eval(args: argparse.Namespace, *, console: Console, specs: Sequence[ParamSpec]) -> int:
    defaults = SynapticConfig()
    x0 = encode_params(defaults, specs)
    if args.vector is not None:
        x = _parse_vector(args.vector, specs)
    else:
        x = x0

    loss = evaluate_candidate(
        x,
        specs=specs,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        timeout_seconds=None,
        max_retries=0,
        raise_on_error=True,
    )
    decoded = decode_params(x, specs)

    table = Table(title="Single Candidate Evaluation", box=box.SIMPLE)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Seed", str(args.seed))
    table.add_row("Device", args.device)
    table.add_row("Steps", str(args.steps))
    table.add_row("Batch", str(args.batch_size))
    table.add_row("Final Loss (mean last 10)", f"{loss:.6f}")
    console.print(table)
    console.print(
        Panel(
            Syntax(
                "syn_cfg = SynapticConfig(\n"
                + "\n".join([f"    {k}={v:.6f}," for k, v in decoded.items()])
                + "\n)",
                "python",
                theme="monokai",
            ),
            title="Decoded SynapticConfig Overrides",
            border_style="green",
        )
    )
    return 0


def _cmd_sanity(args: argparse.Namespace, *, console: Console, specs: Sequence[ParamSpec]) -> int:
    console.print(
        Panel.fit(
            "[bold cyan]CMA-ES Sanity Checks[/bold cyan]\n"
            "Toy convergence + real objective smoke test (cheap gate before Phase 1).",
            border_style="cyan",
        )
    )

    xbest, fbest = run_rosenbrock_2d_cmaes(
        seed=int(args.seed),
        iterations=int(args.rosen_iterations),
        popsize=int(args.rosen_popsize),
        sigma0=float(args.rosen_sigma0),
    )
    rosen_err = float(np.linalg.norm(xbest - np.array([1.0, 1.0], dtype=np.float64)))
    rosen_ok = (rosen_err <= float(args.rosen_tol)) and (fbest <= float(args.rosen_f_tol))

    defaults = SynapticConfig()
    x0 = encode_params(defaults, specs)
    smoke_loss = evaluate_candidate(
        x0,
        specs=specs,
        seed=int(args.seed),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        device=args.device,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        timeout_seconds=None,
        max_retries=0,
        raise_on_error=True,
    )
    smoke_ok = bool(np.isfinite(smoke_loss) and smoke_loss < float(args.max_loss))

    table = Table(title="Sanity Results", box=box.SIMPLE)
    table.add_column("Check", style="cyan")
    table.add_column("Details", style="white")
    table.add_column("Status", style="white", justify="right")
    table.add_row(
        "Rosenbrock 2D",
        f"xbest=[{xbest[0]:.6f}, {xbest[1]:.6f}], "
        f"fbest={fbest:.3e}, |x-[1,1]|={rosen_err:.3e}",
        "[bold green]PASS[/bold green]" if rosen_ok else "[bold red]FAIL[/bold red]",
    )
    table.add_row(
        "Objective smoke",
        f"loss={smoke_loss:.6f} (threshold < {float(args.max_loss):.3g})",
        "[bold green]PASS[/bold green]" if smoke_ok else "[bold red]FAIL[/bold red]",
    )
    console.print(table)

    if not rosen_ok or not smoke_ok:
        console.print(
            Panel.fit(
                "[bold red]Sanity checks failed[/bold red]\n"
                "Fix failures before running Phase 1 CMA-ES.",
                border_style="red",
            )
        )
        return 1

    console.print(
        Panel.fit(
            "[bold green]All sanity checks passed[/bold green]\n"
            "Safe to proceed to Phase 1 runs.",
            border_style="green",
        )
    )
    return 0


def _cmd_proxy(args: argparse.Namespace, *, console: Console, specs: Sequence[ParamSpec]) -> int:
    rng = np.random.default_rng(int(args.seed))
    full_steps = int(args.full_steps)
    proxy_steps = int(args.proxy_steps)
    if full_steps <= 0 or proxy_steps <= 0:
        raise ValueError("--full-steps and --proxy-steps must be > 0")
    if proxy_steps >= full_steps:
        raise ValueError("--proxy-steps must be < --full-steps for a meaningful proxy check")

    mode = str(args.mode)
    target_steps = int(args.target_steps) if args.target_steps is not None else full_steps
    if target_steps < full_steps:
        raise ValueError("--target-steps must be >= --full-steps (or omit it)")

    console.print(
        Panel.fit(
            "[bold cyan]Proxy Objective Check[/bold cyan]\n"
            f"mode={mode}, proxy_steps={proxy_steps}, full_steps={full_steps}, target_steps={target_steps}",
            border_style="cyan",
        )
    )

    table = Table(title="Proxy vs Full Evaluation", box=box.SIMPLE)
    table.add_column("Idx", justify="right", style="cyan", no_wrap=True)
    table.add_column("Proxy Loss", justify="right", style="yellow")
    table.add_column("Pred (LCE)", justify="right", style="green")
    table.add_column("Full Loss", justify="right", style="white")
    table.add_column("|Err|", justify="right", style="magenta")

    preds: list[float] = []
    fulls: list[float] = []

    for idx in range(int(args.candidates)):
        vec_vals: list[float] = []
        for spec in specs:
            v = float(rng.uniform(spec.lower, spec.upper))
            vec_vals.append(math.log(v) if spec.log_scale else v)
        x = np.asarray(vec_vals, dtype=np.float64)

        proxy_res = evaluate_candidate_detailed(
            x,
            specs=specs,
            seed=int(args.seed),
            steps=proxy_steps,
            batch_size=int(args.batch_size),
            device=str(args.device),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            timeout_seconds=args.timeout_seconds,
            max_retries=int(args.retries),
            raise_on_error=True,
            mean_last=10,
            lce_target_steps=target_steps if mode == "lce" else None,
            lce_exponent=float(args.lce_exponent),
            lce_tail_points=int(args.lce_tail_points),
            lce_stride=int(args.lce_stride),
            record_losses=False,
        )

        full_res = evaluate_candidate_detailed(
            x,
            specs=specs,
            seed=int(args.seed),
            steps=full_steps,
            batch_size=int(args.batch_size),
            device=str(args.device),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            timeout_seconds=args.timeout_seconds,
            max_retries=int(args.retries),
            raise_on_error=True,
            mean_last=10,
            lce_target_steps=None,
            record_losses=False,
        )

        proxy_loss = float(proxy_res.mean_last_loss)
        pred = (
            float(proxy_res.lce_pred_loss)
            if mode == "lce" and proxy_res.lce_pred_loss is not None
            else proxy_loss
        )
        full_loss = float(full_res.mean_last_loss)
        err = abs(pred - full_loss)

        preds.append(pred)
        fulls.append(full_loss)
        table.add_row(
            f"{idx}",
            f"{proxy_loss:.6f}",
            f"{pred:.6f}",
            f"{full_loss:.6f}",
            f"{err:.6f}",
        )

    pearson = float("nan")
    if len(preds) >= 2 and np.std(preds) > 0 and np.std(fulls) > 0:
        pearson = float(np.corrcoef(np.asarray(preds), np.asarray(fulls))[0, 1])
    mae = float(np.mean(np.abs(np.asarray(preds) - np.asarray(fulls))))
    rmse = float(np.sqrt(np.mean((np.asarray(preds) - np.asarray(fulls)) ** 2)))

    console.print(
        Panel.fit(
            f"[bold cyan]Summary[/bold cyan]\npearson_r={pearson:.3f}  MAE={mae:.4f}  RMSE={rmse:.4f}",
            border_style="cyan",
        )
    )
    console.print(table)
    return 0


def _cmd_optimize(args: argparse.Namespace, *, console: Console, specs: Sequence[ParamSpec]) -> int:
    dist_info = DistInfo(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        backend="",
        device=args.device,
        comm_device=torch.device("cpu"),
    )
    if args.distributed:
        dist_info = _init_distributed(args.device)
        if not dist_info.enabled:
            raise ValueError(
                "--distributed requires torchrun (RANK/LOCAL_RANK/WORLD_SIZE env vars)"
            )
        if dist_info.rank != 0:
            _distributed_worker_loop(args, specs=specs, dist_info=dist_info)
            if dist.is_initialized():
                dist.destroy_process_group()
            return 0

    actual_device = dist_info.device if args.distributed else args.device
    gpu_count = 0
    if str(actual_device).startswith("cuda") and torch.cuda.is_available():
        gpu_count = dist_info.world_size if args.distributed else 1
    cost_per_gpu_hour = float(args.gpu_cost_per_hour)
    if cost_per_gpu_hour < 0:
        raise ValueError("--gpu-cost-per-hour must be >= 0")
    total_gpu_hours = 0.0
    total_cost_usd = 0.0
    console.print(
        Panel.fit(
            "[bold green]Bio-Inspired Hyperparameter Tuning[/bold green]\n"
            f"Optimizing {len(specs)} biological parameters using CMA-ES\n"
            f"Device: {actual_device} | GPUs: {gpu_count} | $/GPU-hr: {cost_per_gpu_hour:.4f}\n"
            "Checkpoint template: PLAN_TO_USE_CMAES_FOR_HYPERPARAMETER_EXPLORATION_AND_OPTIMIZATION_ACROSS_ALL_BIO_INSPIRED_FEATURES.md",
            border_style="green",
        )
    )

    artifacts = _prepare_run_artifacts(args)
    run_id = (
        artifacts.run_id
        if artifacts is not None
        else f"tune-s{int(args.seed)}-{time.time_ns()}"
    )
    if artifacts is not None:
        console.print(f"[dim]Run {run_id} dir: {artifacts.run_dir}[/dim]")

    run_config = {
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"registry_path", "run_dir", "save_best"}
        },
        "model": asdict(MODEL_CONFIG),
        "search_space": [asdict(spec) for spec in specs],
    }

    defaults = SynapticConfig()
    x0 = encode_params(defaults, specs)
    sigma0 = float(args.sigma0)

    if artifacts is not None and args.resume:
        es = _load_verified_cma_checkpoint(
            artifacts,
            specs=specs,
            allow_unverified=bool(getattr(args, "allow_unverified_checkpoint", False)),
            console=console,
        )
    else:
        lbs, ubs = _cma_bounds(specs)
        es = cma.CMAEvolutionStrategy(
            x0,
            sigma0,
            {
                "popsize": int(args.popsize),
                "bounds": [lbs, ubs],
                "verbose": -1,
                "seed": int(args.seed),
            },
        )

    best_loss, best_params = (
        _load_best_params(artifacts.best_params_json)
        if artifacts is not None
        else (float("inf"), {})
    )
    best_loss_history: list[float] = [float(best_loss)]
    restart_events = 0

    table = Table(title="Optimization History", box=box.SIMPLE)
    table.add_column("Gen", justify="right", style="cyan", no_wrap=True)
    table.add_column("Min Loss", justify="right", style="green")
    table.add_column("Mean Loss", justify="right", style="yellow")
    table.add_column("Best Param Change", justify="left", style="white")

    start_gen = int(getattr(es, "countiter", 0))
    if start_gen and artifacts is not None:
        console.print(f"[dim]Resuming from generation {start_gen}[/dim]")

    dim = len(specs)
    comm_device = dist_info.comm_device
    cmd_stop = 0
    cmd_eval = 1
    mode_mean_last = 0
    mode_lce = 1

    def _broadcast_stop() -> None:
        if not args.distributed:
            return
        ctrl = torch.tensor(
            [cmd_stop, 0, dim, 0, mode_mean_last, 0],
            dtype=torch.int64,
            device=comm_device,
        )
        dist.broadcast(ctrl, src=0)

    try:
        with Live(table, refresh_per_second=4, console=console):
            for _ in range(start_gen, int(args.generations)):
                if es.stop():
                    break

                gen_start = time.monotonic()
                next_gen = int(getattr(es, "countiter", 0)) + 1
                full_steps = int(args.steps)
                proxy_steps = int(args.proxy_steps) if args.proxy_steps is not None else 0
                proxy_gens = int(args.proxy_generations)
                proxy_enabled = (
                    proxy_steps > 0
                    and proxy_gens > 0
                    and proxy_steps < full_steps
                    and next_gen <= proxy_gens
                )
                eval_steps = proxy_steps if proxy_enabled else full_steps
                proxy_mode = str(args.proxy_mode)
                proxy_target_steps = (
                    int(args.proxy_target_steps)
                    if args.proxy_target_steps is not None
                    else full_steps
                )
                use_lce = (
                    proxy_enabled
                    and proxy_mode == "lce"
                    and proxy_target_steps > eval_steps
                )
                mode_int = mode_lce if use_lce else mode_mean_last
                lce_target_steps = proxy_target_steps if use_lce else 0

                solutions = es.ask()
                pop = int(len(solutions))

                if args.distributed:
                    ctrl = torch.tensor(
                        [
                            cmd_eval,
                            pop,
                            dim,
                            int(eval_steps),
                            int(mode_int),
                            int(lce_target_steps),
                        ],
                        dtype=torch.int64,
                        device=comm_device,
                    )
                    dist.broadcast(ctrl, src=0)
                    sols_np = np.asarray(solutions, dtype=np.float64)
                    sols_t = torch.as_tensor(sols_np, dtype=torch.float64, device=comm_device)
                    dist.broadcast(sols_t, src=0)

                    fitness = torch.full(
                        (pop,), float("nan"), dtype=torch.float64, device=comm_device
                    )
                    for sol_idx in range(dist_info.rank, pop, dist_info.world_size):
                        res = evaluate_candidate_detailed(
                            sols_t[sol_idx].detach().cpu().numpy(),
                            specs=specs,
                            seed=int(args.seed),
                            steps=int(eval_steps),
                            batch_size=int(args.batch_size),
                            device=dist_info.device,
                            lr=float(args.lr),
                            weight_decay=float(args.weight_decay),
                            timeout_seconds=args.timeout_seconds,
                            max_retries=int(args.retries),
                            raise_on_error=False,
                            mean_last=10,
                            lce_target_steps=int(lce_target_steps) if use_lce else None,
                            lce_exponent=float(args.lce_exponent),
                            lce_tail_points=int(args.lce_tail_points),
                            lce_stride=int(args.lce_stride),
                            record_losses=False,
                        )
                        if use_lce and res.lce_pred_loss is not None:
                            fitness[sol_idx] = float(res.lce_pred_loss)
                        else:
                            fitness[sol_idx] = float(res.mean_last_loss)

                    gathered = [
                        torch.empty_like(fitness) for _ in range(dist_info.world_size)
                    ]
                    dist.all_gather(gathered, fitness)
                    fitnesses = _merge_allgathered_fitness(gathered)
                else:
                    fitnesses = []
                    for sol in solutions:
                        res = evaluate_candidate_detailed(
                            np.asarray(sol, dtype=np.float64),
                            specs=specs,
                            seed=int(args.seed),
                            steps=int(eval_steps),
                            batch_size=int(args.batch_size),
                            device=args.device,
                            lr=float(args.lr),
                            weight_decay=float(args.weight_decay),
                            timeout_seconds=args.timeout_seconds,
                            max_retries=int(args.retries),
                            raise_on_error=False,
                            mean_last=10,
                            lce_target_steps=int(lce_target_steps) if use_lce else None,
                            lce_exponent=float(args.lce_exponent),
                            lce_tail_points=int(args.lce_tail_points),
                            lce_stride=int(args.lce_stride),
                            record_losses=False,
                        )
                        if use_lce and res.lce_pred_loss is not None:
                            fitnesses.append(float(res.lce_pred_loss))
                        else:
                            fitnesses.append(float(res.mean_last_loss))

                es.tell(solutions, fitnesses)

                gen = int(getattr(es, "countiter", 0))
                gen_min = float(min(fitnesses))
                gen_mean = float(np.mean(fitnesses))
                elapsed_s = float(time.monotonic() - gen_start)
                gpu_hours = (elapsed_s / 3600.0) * float(gpu_count)
                cost_usd = gpu_hours * cost_per_gpu_hour
                total_gpu_hours += gpu_hours
                total_cost_usd += cost_usd

                improved = gen_min < best_loss
                if improved:
                    best_loss = gen_min
                    best_idx = int(np.argmin(fitnesses))
                    best_params = decode_params(
                        np.array(solutions[best_idx], dtype=np.float64), specs
                    )

                    diffs: list[str] = []
                    for key, value in best_params.items():
                        def_val = float(getattr(defaults, key))
                        if abs(value - def_val) / (abs(def_val) + 1e-9) > 0.2:
                            diffs.append(f"{key}: {def_val:.3g}->{value:.3g}")
                    diff_str = ", ".join(diffs[:3]) + ("..." if len(diffs) > 3 else "")
                else:
                    diff_str = "-"

                best_loss_history.append(float(best_loss))
                stagnation_improve = _stagnation_improvement_frac(
                    best_loss_history=best_loss_history,
                    window_gens=int(args.stagnation_gens),
                )
                stagnation_triggered = (
                    stagnation_improve is not None
                    and stagnation_improve < float(args.stagnation_min_improve_frac)
                    and args.stagnation_action != "none"
                )
                stagnation_action_taken: str | None = None
                sigma_before: float | None = None
                sigma_after: float | None = None
                if stagnation_triggered:
                    stagnation_action_taken = str(args.stagnation_action)
                    if args.stagnation_action == "stop":
                        pass
                    elif args.stagnation_action == "sigma_reset":
                        sigma_before = float(getattr(es, "sigma", float("nan")))
                        try:
                            es.sigma = float(sigma0)
                        except Exception:
                            # Fall back to a best-effort mutating update.
                            es.sigma *= float(sigma0) / max(sigma_before, 1e-12)
                        sigma_after = float(getattr(es, "sigma", float("nan")))
                        restart_events += 1
                    else:
                        raise ValueError(
                            f"Unknown stagnation_action={args.stagnation_action!r}"
                        )

                table.add_row(f"{gen}", f"{gen_min:.4f}", f"{gen_mean:.4f}", diff_str)

                if artifacts is not None:
                    _append_progress(
                        artifacts.progress_jsonl,
                        {
                            "run_id": run_id,
                            "generation": gen,
                            "min_loss": gen_min,
                            "mean_loss": gen_mean,
                            "best_loss": best_loss,
                            "popsize": pop,
                            "elapsed_s": elapsed_s,
                            "gpu_count": int(gpu_count),
                            "gpu_hours": float(gpu_hours),
                            "gpu_hours_total": float(total_gpu_hours),
                            "cost_usd": float(cost_usd),
                            "cost_usd_total": float(total_cost_usd),
                            "cost_per_gpu_hour": float(cost_per_gpu_hour),
                            "eval_steps": int(eval_steps),
                            "proxy_enabled": bool(proxy_enabled),
                            "proxy_mode": proxy_mode if proxy_enabled else None,
                            "proxy_target_steps": int(proxy_target_steps)
                            if proxy_enabled
                            else None,
                            "lce_exponent": float(args.lce_exponent) if use_lce else None,
                            "lce_tail_points": int(args.lce_tail_points) if use_lce else None,
                            "lce_stride": int(args.lce_stride) if use_lce else None,
                            "stagnation_window_gens": int(args.stagnation_gens),
                            "stagnation_min_improve_frac": float(
                                args.stagnation_min_improve_frac
                            ),
                            "stagnation_improve_frac": float(stagnation_improve)
                            if stagnation_improve is not None
                            else None,
                            "stagnation_triggered": bool(stagnation_triggered),
                            "stagnation_action": stagnation_action_taken,
                            "sigma_before": sigma_before,
                            "sigma_after": sigma_after,
                            "restart_events": int(restart_events),
                            "saved_at_unix": time.time(),
                        },
                    )

                    if improved:
                        _save_best_params(
                            artifacts.best_params_json,
                            run_id=run_id,
                            best_loss=best_loss,
                            best_params=best_params,
                            gen=gen,
                        )

                    if not args.no_checkpoints:
                        es_bytes = es.pickle_dumps()
                        es_sha256 = hashlib.sha256(es_bytes).hexdigest()
                        artifacts.es_latest_pkl.write_bytes(es_bytes)
                        (artifacts.run_dir / f"es_gen_{gen:04d}.pkl").write_bytes(es_bytes)
                        manifest = {
                            "format": "cmaes-checkpoint-manifest-v1",
                            "run_id": run_id,
                            "generation": int(gen),
                            "countiter": int(getattr(es, "countiter", gen)),
                            "best_loss": float(best_loss),
                            "dim": len(specs),
                            "sha256": es_sha256,
                            "saved_at_unix": time.time(),
                        }
                        manifest_raw = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                        artifacts.es_manifest_json.write_text(manifest_raw, encoding="utf-8")
                        (artifacts.run_dir / f"es_manifest_gen_{gen:04d}.json").write_text(
                            manifest_raw, encoding="utf-8"
                        )

                    if artifacts.tb_writer is not None:
                        _log_tensorboard(
                            artifacts.tb_writer,
                            gen=gen,
                            fitnesses=fitnesses,
                            best_loss=best_loss,
                            es=es,
                        )
                        if stagnation_improve is not None:
                            artifacts.tb_writer.add_scalar(
                                "stagnation/improve_frac", float(stagnation_improve), gen
                            )
                        if stagnation_triggered:
                            artifacts.tb_writer.add_scalar("stagnation/triggered", 1.0, gen)
                        else:
                            artifacts.tb_writer.add_scalar("stagnation/triggered", 0.0, gen)
                        if sigma_before is not None:
                            artifacts.tb_writer.add_scalar(
                                "stagnation/sigma_before", float(sigma_before), gen
                            )
                        if sigma_after is not None:
                            artifacts.tb_writer.add_scalar(
                                "stagnation/sigma_after", float(sigma_after), gen
                            )
                        artifacts.tb_writer.add_scalar(
                            "budget/gpu_hours", float(gpu_hours), gen
                        )
                        artifacts.tb_writer.add_scalar(
                            "budget/gpu_hours_total", float(total_gpu_hours), gen
                        )
                        artifacts.tb_writer.add_scalar(
                            "budget/cost_usd", float(cost_usd), gen
                        )
                        artifacts.tb_writer.add_scalar(
                            "budget/cost_usd_total", float(total_cost_usd), gen
                        )
                        artifacts.tb_writer.add_scalar(
                            "budget/cost_per_gpu_hour", float(cost_per_gpu_hour), gen
                        )
                        artifacts.tb_writer.flush()

                if stagnation_triggered and args.stagnation_action == "stop":
                    console.print(
                        f"[yellow]Stopping early due to stagnation: improve_frac={stagnation_improve} "
                        f"< {float(args.stagnation_min_improve_frac)} over "
                        f"{int(args.stagnation_gens)} gens.[/yellow]"
                    )
                    break
    finally:
        _broadcast_stop()
        if artifacts is not None and artifacts.tb_writer is not None:
            artifacts.tb_writer.close()
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()

    console.print("\n[bold green]Optimization Complete![/bold green]")
    console.print(f"Best Loss: {best_loss:.6f}")
    console.print(
        Panel(
            Syntax(
                "best_config = SynapticConfig(\n"
                + "\n".join([f"    {k}={v:.6f}," for k, v in best_params.items()])
                + "\n)",
                "python",
                theme="monokai",
            ),
            title="Optimized Configuration",
            border_style="green",
        )
    )

    if args.save_best is not None:
        path = args.save_best
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated by scripts/tune_bio_params.py\n")
            f.write("from bio_inspired_nanochat.synaptic import SynapticConfig\n\n")
            f.write("OPTIMIZED_SYNAPTIC_CONFIG = SynapticConfig(\n")
            for k, v in best_params.items():
                f.write(f"    {k}={v:.6f},\n")
            f.write(")\n")
        console.print(f"[dim]Saved best config to {path}[/dim]")

    final_generation = int(getattr(es, "countiter", 0))
    if not math.isfinite(best_loss):
        raise RuntimeError("CMA-ES completed without a finite objective")
    artifact_note = str(artifacts.run_dir) if artifacts is not None else "none"
    append_record(
        make_record(
            "tune",
            {
                "tune_objective": float(best_loss),
                "tune_generation": float(final_generation),
            },
            run_id=run_id,
            config=run_config,
            seed=int(args.seed),
            dataset_shards=["synthetic:associative_recall"],
            timestamp=time.time(),
            notes=f"artifact_dir={artifact_note}; best_params={len(best_params)}",
        ),
        args.registry_path,
    )
    console.print(f"[dim]Appended {run_id} to {args.registry_path}[/dim]")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _validate_param_specs(TOP10_PARAM_SPECS)
    console = Console()

    parser = argparse.ArgumentParser(description="Tune synaptic bio parameters with CMA-ES")
    parser.add_argument(
        "--search-space",
        type=str,
        choices=list(PARAM_SPACES.keys()),
        default="top10",
        help="Subsystem search space: top10 (presyn), postsyn, structural, full",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    eval_p = sub.add_parser("eval", help="Run a single deterministic candidate evaluation")
    eval_p.add_argument("--vector", type=str, default=None, help="Comma-separated 10D vector")
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--device", type=str, default=DEVICE)
    eval_p.add_argument("--steps", type=int, default=STEPS_PER_EVAL)
    eval_p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    eval_p.add_argument("--lr", type=float, default=1e-3)
    eval_p.add_argument("--weight-decay", type=float, default=1e-2)

    sanity_p = sub.add_parser(
        "sanity",
        help="Run CMA-ES sanity checks (toy convergence + objective smoke)",
    )
    sanity_p.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Non-zero seed for deterministic toy convergence checks",
    )
    sanity_p.add_argument("--device", type=str, default=DEVICE)
    sanity_p.add_argument("--steps", type=int, default=30)
    sanity_p.add_argument("--batch-size", type=int, default=8)
    sanity_p.add_argument("--lr", type=float, default=1e-3)
    sanity_p.add_argument("--weight-decay", type=float, default=1e-2)
    sanity_p.add_argument(
        "--max-loss",
        type=float,
        default=20.0,
        help="Fail the smoke test if loss is >= this threshold",
    )
    sanity_p.add_argument("--rosen-iterations", type=int, default=80)
    sanity_p.add_argument("--rosen-popsize", type=int, default=8)
    sanity_p.add_argument("--rosen-sigma0", type=float, default=0.5)
    sanity_p.add_argument("--rosen-tol", type=float, default=0.05)
    sanity_p.add_argument("--rosen-f-tol", type=float, default=1e-3)

    proxy_p = sub.add_parser(
        "proxy",
        help="Validate proxy objective (proxy steps) vs full eval, optionally using LCE",
    )
    proxy_p.add_argument("--seed", type=int, default=123)
    proxy_p.add_argument("--device", type=str, default=DEVICE)
    proxy_p.add_argument("--proxy-steps", type=int, default=30)
    proxy_p.add_argument("--full-steps", type=int, default=100)
    proxy_p.add_argument("--target-steps", type=int, default=None)
    proxy_p.add_argument("--candidates", type=int, default=3)
    proxy_p.add_argument("--mode", type=str, choices=["mean_last", "lce"], default="lce")
    proxy_p.add_argument("--batch-size", type=int, default=8)
    proxy_p.add_argument("--lr", type=float, default=1e-3)
    proxy_p.add_argument("--weight-decay", type=float, default=1e-2)
    proxy_p.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Per-candidate evaluation timeout (seconds); raises on timeout",
    )
    proxy_p.add_argument("--retries", type=int, default=0)
    proxy_p.add_argument("--lce-exponent", type=float, default=0.5)
    proxy_p.add_argument("--lce-tail-points", type=int, default=50)
    proxy_p.add_argument("--lce-stride", type=int, default=1)

    opt_p = sub.add_parser("optimize", help="Run CMA-ES over the 10D search space")
    opt_p.add_argument("--seed", type=int, default=1337)
    opt_p.add_argument("--device", type=str, default=DEVICE)
    opt_p.add_argument(
        "--distributed",
        action="store_true",
        help="Use torchrun/DDP workers for population eval (rank0 controller)",
    )
    opt_p.add_argument("--generations", type=int, default=MAX_GENERATIONS)
    opt_p.add_argument("--popsize", type=int, default=POPULATION_SIZE)
    opt_p.add_argument("--sigma0", type=float, default=0.2)
    opt_p.add_argument("--steps", type=int, default=STEPS_PER_EVAL)
    opt_p.add_argument(
        "--proxy-steps",
        type=int,
        default=None,
        help="If set, use fewer steps per candidate in early generations (proxy eval)",
    )
    opt_p.add_argument(
        "--proxy-generations",
        type=int,
        default=0,
        help="How many early generations use --proxy-steps (0 disables proxy)",
    )
    opt_p.add_argument(
        "--proxy-mode",
        type=str,
        choices=["mean_last", "lce"],
        default="lce",
        help="Proxy fitness mode: raw proxy loss or LCE-predicted loss",
    )
    opt_p.add_argument(
        "--proxy-target-steps",
        type=int,
        default=None,
        help="If proxy-mode=lce, predict loss at this step count (default: --steps)",
    )
    opt_p.add_argument("--lce-exponent", type=float, default=0.5)
    opt_p.add_argument("--lce-tail-points", type=int, default=50)
    opt_p.add_argument("--lce-stride", type=int, default=1)
    opt_p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    opt_p.add_argument("--lr", type=float, default=1e-3)
    opt_p.add_argument("--weight-decay", type=float, default=1e-2)
    opt_p.add_argument(
        "--gpu-cost-per-hour",
        type=float,
        default=0.0,
        help="USD cost per GPU-hour for budget tracking (0 disables cost estimates)",
    )
    opt_p.add_argument(
        "--stagnation-gens",
        type=int,
        default=20,
        help="Stagnation window (gens) for patience rule; <=0 disables",
    )
    opt_p.add_argument(
        "--stagnation-min-improve-frac",
        type=float,
        default=0.01,
        help="Trigger if best loss improves by less than this fraction over the stagnation window",
    )
    opt_p.add_argument(
        "--stagnation-action",
        type=str,
        choices=["none", "stop", "sigma_reset"],
        default="stop",
        help="Action on stagnation: stop or reset CMA-ES sigma to sigma0",
    )
    opt_p.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Per-candidate evaluation timeout (seconds); returns penalty on timeout",
    )
    opt_p.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Retries per candidate on exception/timeout before penalty",
    )
    opt_p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional run directory for progress/checkpoints/TensorBoard",
    )
    opt_p.add_argument(
        "--registry-path",
        type=str,
        default=DEFAULT_REGISTRY,
        help="Committed JSONL results registry path",
    )
    opt_p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --run-dir/es_latest.pkl",
    )
    opt_p.add_argument(
        "--allow-unverified-checkpoint",
        action="store_true",
        help="Allow resuming from legacy unverified checkpoints lacking a cryptographic manifest (warning: security risk)",
    )
    opt_p.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable writing CMA-ES pickle checkpoints even if --run-dir is set",
    )
    opt_p.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging even if --run-dir is set",
    )
    opt_p.add_argument("--save-best", type=str, default=None, help="Optional path to save best config python")

    args = parser.parse_args(argv)
    specs = PARAM_SPACES.get(getattr(args, "search_space", "top10"), TOP10_PARAM_SPECS)
    _validate_param_specs(specs)

    if args.cmd == "eval":
        return _cmd_eval(args, console=console, specs=specs)
    if args.cmd == "sanity":
        return _cmd_sanity(args, console=console, specs=specs)
    if args.cmd == "proxy":
        return _cmd_proxy(args, console=console, specs=specs)
    if args.cmd == "optimize":
        return _cmd_optimize(args, console=console, specs=specs)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
