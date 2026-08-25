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


def checkpoint_model_config(model, base_config: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe architecture metadata, including live per-layer MoE counts.

    Structural birth/death can make expert counts heterogeneous after construction;
    persisting only the initial uniform count reconstructs the wrong module graph and
    makes strict state loading fail.
    """
    out = dict(base_config)
    config = getattr(model, "config", None)
    for name in (
        "dropout",
        "use_moe",
        "num_experts",
        "moe_top_k",
        "moe_hidden_mult",
        "moe_balance_loss",
        "structural_every",
        "init_type",
        "init_seed",
        "tie_embeddings",
        # Attention-architecture surface: without these, a checkpoint saved
        # through this metadata path rebuilt as attention_type="standard" and
        # the strict load failed on the unexpected ultrametric projection keys.
        "attention_type",
        "ultrametric_k",
        "ultrametric_p",
        "ultrametric_alpha",
        "ultrametric_lcp_beta",
        "ultrametric_query_chunk_size",
    ):
        if config is not None and hasattr(config, name):
            out[name] = getattr(config, name)
    if out.get("use_moe"):
        from bio_inspired_nanochat.synaptic import SynapticMoE

        counts: list[int] = []
        for block in getattr(model, "h", ()):
            mlp = getattr(block, "mlp", None)
            if not isinstance(mlp, SynapticMoE):
                raise ValueError("use_moe checkpoint contains a non-MoE model layer")
            counts.append(int(mlp.num_experts))
        if not counts:
            raise ValueError("use_moe checkpoint contains no MoE layers")
        out["moe_experts_per_layer"] = counts
    return out


def load_checkpoint_metadata(checkpoint_dir: str, step: int) -> dict[str, Any]:
    """Load only JSON metadata, so training can rebuild topology before tensor load."""
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed checkpoint metadata: {meta_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint metadata must be a JSON object: {meta_path}")
    return metadata


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
            timeout=5.0,
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


def _fsync_parent_dir(path: str) -> None:
    """fsync the containing directory so the rename itself is durable.

    Without this, a power loss right after ``os.replace`` can persist the new
    directory entry while the file's data blocks were still in page cache —
    leaving a zero-length/truncated file under the FINAL checkpoint name, which
    is exactly the corruption the tmp+rename scheme exists to prevent.
    """
    fd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_parent_dir(path)


def _atomic_write_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_parent_dir(path)


def capture_rng_state() -> dict:
    """Snapshot every RNG that affects training so a resume is bit-comparable.

    The synaptic forward is *stochastic* during training (stochastic vesicle release
    draws from the global torch RNG), so without restoring RNG a resumed run diverges
    from the uninterrupted one — verified in tests/test_scaleup_checkpoint.py. RNG state
    is per-rank and each rank's exact stream is preserved, so a resumed rank
    reproduces its pre-save draw sequence bit-for-bit. NOTE (jgkf): ranks are
    SEEDED identically at run start (compute_init), so their streams are
    correlated, not independent — the per-rank blobs exist to preserve whatever
    each rank's stream actually was across a save/resume boundary.
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
    torch_state = state.get("torch")
    python_state = None
    numpy_state = None
    cuda_state = state.get("cuda")

    # Validate all CPU RNG payloads against isolated generators before changing any
    # process-global stream. A corrupt later payload must not leave an earlier RNG restored.
    if torch_state is not None:
        try:
            torch.Generator(device="cpu").set_state(torch_state)
        except Exception as exc:
            raise RuntimeError("Failed to restore the saved PyTorch RNG state") from exc
    if state.get("python") is not None:
        try:
            # torch.save/load round-trips the Python state tuple as nested lists; setstate
            # requires tuples (version, internal-state-tuple, gauss).
            py = state["python"]
            python_state = (int(py[0]), tuple(int(x) for x in py[1]), py[2])
            random.Random().setstate(python_state)
        except Exception as exc:
            raise RuntimeError("Failed to restore the saved Python RNG state") from exc
    if state.get("numpy") is not None:
        try:
            import numpy as np

            n = state["numpy"]
            numpy_state = (
                n["type"],
                n["keys"].numpy().astype("uint32"),
                int(n["pos"]),
                int(n["has_gauss"]),
                float(n["cached"]),
            )
            np.random.RandomState().set_state(numpy_state)
        except Exception as exc:
            raise RuntimeError("Failed to restore the saved NumPy RNG state") from exc
    if cuda_state is not None and torch.cuda.is_available():
        try:
            if not isinstance(cuda_state, (list, tuple)):
                raise TypeError("CUDA RNG state must be a sequence")
            device_count = torch.cuda.device_count()
            if len(cuda_state) != device_count:
                raise ValueError(
                    f"saved CUDA RNG state count {len(cuda_state)} does not match "
                    f"available device count {device_count}"
                )
            for device_index, generator_state in enumerate(cuda_state):
                torch.Generator(device=f"cuda:{device_index}").set_state(generator_state)
        except Exception as exc:
            raise RuntimeError("Failed to restore the saved CUDA RNG state") from exc

    if torch_state is not None:
        torch.set_rng_state(torch_state)
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


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
        # Work on a copy: adding inferred metadata must not mutate caller-owned state.
        meta_to_save = dict(meta_data) if meta_data is not None else {}
        # Check if model_data contains synaptic-specific keys (heuristic detection)
        # This is a fallback; ideally the caller should set synapses=True in meta_data
        if "synapses" not in meta_to_save:
            # Check for synaptic-specific buffer names in state dict
            # Real registered names (synaptic.py): presyn lives under ".pre."
            # (incl. ema_e/_presyn_train_* buffers), postsyn under ".post.",
            # eligibility traces are "u_buf"/"v_buf", fast weights are "w_fast".
            # The previous markers ("H_fast", "U_buf", "V_buf", "gate_m") matched
            # NOTHING, so a caller that omitted meta["synapses"] got a vanilla-GPT
            # rebuild that failed the strict load with a confusing key dump.
            synaptic_keys = [
                k
                for k in model_data.keys()
                if any(x in k for x in ("pre.", "post.", "u_buf", "v_buf", "w_fast"))
            ]
            if synaptic_keys:
                meta_to_save["synapses"] = True
        # Save the metadata dict as json (atomic).
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        _atomic_write_json(meta_to_save, meta_path)
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
    # uta-review/0qvh: declare the step complete ONLY after every rank's shards
    # are durably on disk. The barrier makes all ranks wait for the slowest
    # writer; rank 0 then writes the commit marker strictly last, so discovery
    # (list/find) can never auto-select a half-written checkpoint set.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        _atomic_write_json(
            {"step": int(step), "world_size": _world_size_or_none()},
            os.path.join(checkpoint_dir, f"commit_{step:06d}.json"),
        )


def _world_size_or_none() -> Optional[int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return None


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
    meta_data = load_checkpoint_metadata(checkpoint_dir, step)
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


def _uses_commit_markers(checkpoint_dir: str) -> bool:
    """True once ANY step in this directory carries a ``commit_*.json`` marker.

    Regime detection keeps pre-marker checkpoints resumable (legacy directories
    keep the old model-file-only discovery) while every save that CAN write a
    marker is required to have one before the step counts as resumable.
    """
    return bool(glob.glob(os.path.join(checkpoint_dir, "commit_*.json")))


def _checkpoint_is_complete(checkpoint_dir: str, step: int, *, markers_in_use: bool) -> bool:
    """A step is resumable iff its rank-0 model AND metadata exist and — in
    marker-regime directories — the rank-0 commit marker declares every rank's
    shards landed. A crash mid-save used to leave exactly such a partial set,
    which ``find_last_step`` then auto-selected and resume died on."""
    if not (
        os.path.exists(os.path.join(checkpoint_dir, f"model_{step:06d}.pt"))
        and os.path.exists(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"))
    ):
        return False
    if not markers_in_use:
        return True  # legacy directory written before commit markers existed
    return os.path.exists(os.path.join(checkpoint_dir, f"commit_{step:06d}.json"))


def list_checkpoint_steps(checkpoint_dir: str) -> list[int]:
    """Sorted ascending list of COMPLETE, resumable steps.

    In directories that use commit markers, a step with a ``model_*.pt`` but no
    ``commit_{step}.json`` is debris from a crashed save and is excluded — this
    is what stops auto-resume from selecting a set whose optimizer shards never
    finished writing.
    """
    markers = _uses_commit_markers(checkpoint_dir)
    steps = []
    for f in glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
        m = re.match(r"model_(\d{6,})\.pt$", os.path.basename(f))
        if m:
            step = int(m.group(1))
            if _checkpoint_is_complete(checkpoint_dir, step, markers_in_use=markers):
                steps.append(step)
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
            # 0qvh: the completion marker is part of the checkpoint set; leaving
            # it behind would keep the pruned step "complete" if its files were
            # ever recreated by a retried save.
            os.path.join(checkpoint_dir, f"commit_{s:06d}.json"),
        ]
        # optim/train are per-rank; remove every rank's shard for this superseded step.
        paths += glob.glob(os.path.join(checkpoint_dir, f"optim_{s:06d}_rank*.pt"))
        paths += glob.glob(os.path.join(checkpoint_dir, f"train_{s:06d}_rank*.pt"))
        for path in paths:
            # Defensive: only ever remove files matching the checkpoint pattern.
            basename = os.path.basename(path)
            if os.path.exists(path) and (
                _CKPT_RE.match(basename) or re.match(r"^commit_\d{6,}\.json$", basename)
            ):
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
    if phase not in {"train", "eval"}:
        raise ValueError(f"phase must be 'train' or 'eval', got {phase!r}")
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
            moe_experts_per_layer=(
                tuple(model_config_kwargs["moe_experts_per_layer"])
                if model_config_kwargs.get("moe_experts_per_layer") is not None
                else None
            ),
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
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    model_vocab_size = model_config_kwargs["vocab_size"]
    if tokenizer_vocab_size != model_vocab_size:
        raise ValueError(
            "checkpoint tokenizer vocabulary mismatch: "
            f"tokenizer={tokenizer_vocab_size}, model={model_vocab_size}"
        )
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
    # A checkpoint becomes discoverable only after both atomic rank-0 files exist. A crash
    # between publishing model and metadata can leave a valid-looking model-only step that
    # must not eclipse the previous complete checkpoint during automatic resume.
    steps = [
        step
        for step in list_checkpoint_steps(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"))
    ]
    if not steps:
        raise FileNotFoundError(f"No complete checkpoints found in {checkpoint_dir}")
    return steps[-1]

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
