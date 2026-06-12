"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import glob
import hashlib
import json
import logging
import os
import random
import re
import subprocess
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any, Optional, cast

from bio_inspired_nanochat.torch_imports import torch

if TYPE_CHECKING:
    from bio_inspired_nanochat.synaptic import SynapticConfig

from bio_inspired_nanochat.common import get_base_dir
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.tokenizer import get_tokenizer
from bio_inspired_nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)


def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)


# --------------------------------------------------------------------------- #
# SynapticConfig checkpoint round-trip (vg9.6)
#
# build_model used to rebuild synaptic models with SynapticConfig() DEFAULTS, so a model
# trained/tuned with custom bio kinetics silently reloaded as a DIFFERENT model (only the
# learned buffers survived). These helpers persist the full SynapticConfig into meta_data and
# rebuild from it, with provenance (git SHA + a stable config hash) for reproducibility.
# --------------------------------------------------------------------------- #
def synaptic_config_to_meta(syn_cfg) -> dict:
    """Serialize a SynapticConfig to a JSON-able dict for checkpoint meta_data."""
    return asdict(syn_cfg)


def synaptic_config_from_meta(meta_data) -> "SynapticConfig":
    """Rebuild a SynapticConfig from checkpoint meta_data.

    Unknown saved fields are ignored and new schema fields take their defaults (forward/back
    compat). Falls back to SynapticConfig() defaults for pre-vg9.6 checkpoints that did not
    persist the config (logged loudly so the reproducibility risk is visible).
    """
    from bio_inspired_nanochat.synaptic import SynapticConfig

    saved = (meta_data or {}).get("synaptic_config")
    if not saved:
        log0(
            "[checkpoint] no 'synaptic_config' in meta_data; rebuilding with SynapticConfig() "
            "DEFAULTS (pre-vg9.6 checkpoint — bio kinetics may NOT match the trained model)."
        )
        return SynapticConfig()
    known = {f.name for f in fields(SynapticConfig)}
    unknown = sorted(set(saved) - known)
    if unknown:
        log0(f"[checkpoint] ignoring {len(unknown)} unknown synaptic_config field(s): {unknown}")
    return SynapticConfig(**{k: v for k, v in saved.items() if k in known})


def config_hash(cfg_dict: dict) -> str:
    """Stable short hash of a config dict (order-independent)."""
    blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def config_provenance(syn_cfg) -> dict:
    """Provenance stamp for a synaptic checkpoint: git SHA + a stable bio-config hash."""
    return {"git_sha": _git_sha(), "synaptic_config_hash": config_hash(asdict(syn_cfg))}

# --------------------------------------------------------------------------- #
# Atomic write + RNG capture (hwxb.2.6 — crash-safe, resumable long runs)
# --------------------------------------------------------------------------- #
# A multi-hour 2×4090 run that crashes mid-checkpoint must never leave a corrupt
# half-written file that a resume then loads. We always write to ``<path>.tmp`` and
# ``os.replace`` it into place (atomic on POSIX), so any reader sees either the old
# complete file or the new complete file — never a partial one. Stray ``*.tmp`` files
# from a crash are ignored by the loaders (which open the exact final names).
# {step:06d} pads to >=6 digits, so allow 6 OR MORE (a run past 1e6 steps must still
# be seen by rotation, else the disk silently fills — the very thing rotation prevents).
_CKPT_RE = re.compile(r"^(model|meta|optim|train)_(\d{6,})(?:_rank\d+)?\.(pt|json)$")


def _atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def capture_rng_state() -> dict:
    """Snapshot every RNG that affects training so a resume is bit-comparable.

    The synaptic forward is *stochastic* during training (stochastic vesicle release
    draws from the global torch RNG), so without restoring RNG a resumed run diverges
    from the uninterrupted one — verified in tests/test_scaleup_checkpoint.py. RNG state
    is per-rank (each rank draws independently), so it is saved in the per-rank blob.
    """
    state: dict = {"torch": torch.get_rng_state(), "python": random.getstate()}
    try:
        import numpy as np

        # legacy=True returns the MT19937 tuple; cast because numpy's overload also
        # types a dict form that ty otherwise infers.
        nstate = cast("tuple[Any, ...]", np.random.get_state(legacy=True))  # (type, uint32[624], pos, has_gauss, cached)
        # Tensor-encode the key array so the on-disk blob loads under the safe
        # weights_only=True default (a raw numpy array would require arbitrary unpickling).
        state["numpy"] = {
            "type": str(nstate[0]),
            "keys": torch.from_numpy(nstate[1].astype("int64")),
            "pos": int(nstate[2]),
            "has_gauss": int(nstate[3]),
            "cached": float(nstate[4]),
        }
    except Exception:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    """Restore RNGs saved by :func:`capture_rng_state` (no-op on None / missing keys)."""
    if not state:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("python") is not None:
        # torch.save/load round-trips the python state tuple as nested lists; setstate
        # requires tuples (version, internal-state-tuple, gauss).
        py = state["python"]
        random.setstate((int(py[0]), tuple(int(x) for x in py[1]), py[2]))
    if state.get("numpy") is not None:
        try:
            import numpy as np

            n = state["numpy"]
            np.random.set_state(
                (n["type"], n["keys"].numpy().astype("uint32"), int(n["pos"]),
                 int(n["has_gauss"]), float(n["cached"]))
            )
        except Exception:
            pass
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0, train_state=None):
    """Atomically persist a checkpoint.

    ``train_state`` (per-rank, optional) carries everything needed for a *bit-comparable*
    training resume beyond model+optimizer: RNG state (``capture_rng_state()``), the loop
    step, and any stateful-controller snapshots (split/merge ``_last_step`` + router-logit
    bias, neuromod EMAs, divergence-guard last-good). See docs/scale_up_checkpointing.md
    for the full persistence contract (and what is safely *rebuilt* rather than saved).
    """
    # Every rank ensures the directory exists BEFORE any rank writes: non-zero ranks
    # write their own optim/train files below, and there is no barrier guaranteeing rank 0
    # has created the dir first. makedirs(exist_ok=True) is idempotent and race-safe.
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        # Save the model state parameters (atomic).
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        _atomic_torch_save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Ensure meta_data exists and mark synaptic models
        meta_data = meta_data or {}
        # Check if model_data contains synaptic-specific keys (heuristic detection)
        # This is a fallback; ideally the caller should set synapses=True in meta_data
        if "synapses" not in meta_data:
            # Check for synaptic-specific buffer names in state dict
            synaptic_keys = [k for k in model_data.keys() if any(x in k for x in ["pre.", "post.", "H_fast", "U_buf", "V_buf", "gate_m"])]
            if synaptic_keys:
                meta_data["synapses"] = True
        # Save the metadata dict as json (atomic).
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        _atomic_write_json(meta_data, meta_path)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        _atomic_torch_save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
    # Per-rank training state (RNG + controller snapshots) for bit-comparable resume.
    if train_state is not None:
        train_path = os.path.join(checkpoint_dir, f"train_{step:06d}_rank{rank:d}.pt")
        _atomic_torch_save(train_state, train_path)
        logger.info(f"Saved train state to: {train_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0, load_train_state=False):
    # Load the model state. weights_only=True (the safe default) is sufficient: our
    # checkpoints are tensor-only state dicts (and RNG is tensor-encoded), so no
    # arbitrary-pickle deserialization is ever required to resume.
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device, weights_only=True)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device, weights_only=True)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    if load_train_state:
        train_path = os.path.join(checkpoint_dir, f"train_{step:06d}_rank{rank:d}.pt")
        # ALWAYS load RNG state onto CPU, regardless of the compute device: torch's RNG
        # ByteTensors are CPU tensors and torch.set_rng_state rejects a moved/typed copy
        # (it would crash a GPU resume). restore_rng_state() routes the CUDA RNG sub-state
        # to the GPU itself via torch.cuda.set_rng_state_all. Tensor-encoded by
        # capture_rng_state() so the safe weights_only=True default loads it.
        train_state = (
            torch.load(train_path, map_location="cpu", weights_only=True)
            if os.path.exists(train_path) else None
        )
        return model_data, optimizer_data, meta_data, train_state
    return model_data, optimizer_data, meta_data


def list_checkpoint_steps(checkpoint_dir: str) -> list[int]:
    """Sorted ascending list of steps that have a saved ``model_*.pt``."""
    steps = []
    for f in glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
        m = re.match(r"model_(\d{6,})\.pt$", os.path.basename(f))
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def prune_checkpoints(checkpoint_dir: str, keep_last: int, *, best_step: Optional[int] = None) -> list[int]:
    """Rotate checkpoints: keep the ``keep_last`` most recent steps + ``best_step``.

    Disk on a long run is finite; without rotation a multi-day run fills the volume and
    crashes. For each *superseded* step this deletes the **complete** checkpoint — the
    rank-0 model/meta AND every rank's optim/train shard (globbed by ``*_rank*``) — so it
    never leaves an inconsistent partial checkpoint behind, and it can be called once
    (e.g. on rank 0) to clean a whole DDP run. Only files matching the strict checkpoint
    name pattern in ``checkpoint_dir`` are touched. Opt-in: the caller passes an explicit
    ``keep_last``. Returns the list of pruned steps. Every deletion is logged.
    """
    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")
    steps = list_checkpoint_steps(checkpoint_dir)
    keep = set(steps[-keep_last:])
    if best_step is not None:
        keep.add(int(best_step))
    pruned = [s for s in steps if s not in keep]
    for s in pruned:
        paths = [
            os.path.join(checkpoint_dir, f"model_{s:06d}.pt"),
            os.path.join(checkpoint_dir, f"meta_{s:06d}.json"),
        ]
        # optim/train are per-rank; remove every rank's shard for this superseded step.
        paths += glob.glob(os.path.join(checkpoint_dir, f"optim_{s:06d}_rank*.pt"))
        paths += glob.glob(os.path.join(checkpoint_dir, f"train_{s:06d}_rank*.pt"))
        for path in paths:
            # Defensive: only ever remove files matching the checkpoint pattern.
            if os.path.exists(path) and _CKPT_RE.match(os.path.basename(path)):
                os.remove(path)
                logger.info(f"[checkpoint] pruned superseded checkpoint file: {path}")
    if pruned:
        logger.info(f"[checkpoint] rotation kept steps {sorted(keep)}, pruned {pruned}")
    return pruned


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    
    # Check if this is a synaptic model
    if meta_data.get("synapses", False):
        try:
            from bio_inspired_nanochat.gpt_synaptic import (
                GPTSynaptic,
                GPTSynapticConfig,
            )
        except Exception as e:
            raise ImportError(
                "Synaptic checkpoint requires synaptic modules, but they failed to import."
            ) from e
        # vg9.6: rebuild the bio kinetics from the checkpoint instead of silently using defaults.
        syn_cfg = synaptic_config_from_meta(meta_data)
        model_config = GPTSynapticConfig(
            sequence_len=model_config_kwargs["sequence_len"],
            vocab_size=model_config_kwargs["vocab_size"],
            n_layer=model_config_kwargs["n_layer"],
            n_head=model_config_kwargs["n_head"],
            n_kv_head=model_config_kwargs.get("n_kv_head", model_config_kwargs["n_head"]),
            n_embd=model_config_kwargs["n_embd"],
            synapses=True,
            syn_cfg=syn_cfg,
            dropout=model_config_kwargs.get("dropout", 0.0),
            use_moe=model_config_kwargs.get("use_moe", False),
            num_experts=model_config_kwargs.get("num_experts", 8),
            moe_top_k=model_config_kwargs.get("moe_top_k", 2),
            moe_hidden_mult=model_config_kwargs.get("moe_hidden_mult", 4),
            moe_balance_loss=model_config_kwargs.get("moe_balance_loss", 0.01),
            structural_every=model_config_kwargs.get("structural_every", 0),
            init_type=model_config_kwargs.get("init_type", "baseline"),
            init_seed=int(model_config_kwargs.get("init_seed", 42)),
            tie_embeddings=bool(model_config_kwargs.get("tie_embeddings", False)),  # hwxb.2.9
        )
        with torch.device("meta"):
            model = GPTSynaptic(model_config)
    else:
        model_config = GPTConfig(**model_config_kwargs)
        with torch.device("meta"):
            model = GPT(model_config)
    
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # hwxb.2.9: re-establish the wte/lm_head tie that assign=True breaks (no-op when untied).
    model.tie_weights()
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
