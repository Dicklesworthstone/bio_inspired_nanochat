"""
Common utilities for bio_inspired_nanochat.
"""

import os
import re
import logging
import hashlib
import math
import shutil
import time
import urllib.request
import urllib.error
from bio_inspired_nanochat.torch_imports import torch
import torch.distributed as dist
from filelock import FileLock
from decouple import Config as DecoupleConfig, RepositoryEnv, RepositoryEmpty

# Initialize decouple config (project-local .env if exists; fallback to empty config)
_env_path = ".env"
if os.path.exists(_env_path):
    decouple_config = DecoupleConfig(RepositoryEnv(_env_path))
else:
    # Fallback to empty repository when .env doesn't exist (e.g., in CI)
    decouple_config = DecoupleConfig(RepositoryEmpty())

DEFAULT_DOWNLOAD_TIMEOUT_SEC = float(decouple_config("NANOCHAT_DOWNLOAD_TIMEOUT_SEC", default="30.0"))
DEFAULT_DOWNLOAD_CHUNK_SIZE = int(decouple_config("NANOCHAT_DOWNLOAD_CHUNK_SIZE_BYTES", default=str(1024 * 1024)))
DEFAULT_DOWNLOAD_MAX_ATTEMPTS = int(decouple_config("NANOCHAT_DOWNLOAD_MAX_ATTEMPTS", default="3"))

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    default_dir = os.path.join(os.path.expanduser("~"), ".cache", "bio_inspired_nanochat")
    nanochat_dir = decouple_config("NANOCHAT_BASE_DIR", default=default_dir)
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    temp_path = file_path + ".tmp"
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        print(f"Downloading {url}...")
        timeout_sec = DEFAULT_DOWNLOAD_TIMEOUT_SEC
        chunk_size = DEFAULT_DOWNLOAD_CHUNK_SIZE

        for attempt in range(1, DEFAULT_DOWNLOAD_MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(url, timeout=timeout_sec) as response:
                    with open(temp_path, "wb") as f:
                        shutil.copyfileobj(response, f, length=chunk_size)
                os.replace(temp_path, file_path)
                print(f"Downloaded to {file_path}")
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                # HTTP errors like 403/404 are almost never transient; don't retry.
                if isinstance(e, urllib.error.HTTPError) and (400 <= e.code < 500):
                    raise
                if attempt >= DEFAULT_DOWNLOAD_MAX_ATTEMPTS:
                    raise
                print(f"Download attempt {attempt}/{DEFAULT_DOWNLOAD_MAX_ATTEMPTS} failed: {e}")
                time.sleep(2**attempt)
            finally:
                # Best-effort cleanup of partial temp file
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp() and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


def _stable_seed_u64(seed: int, *, salt: str) -> int:
    """Return a deterministic 64-bit seed derived from (seed, salt).

    Note: Python's built-in hash() is intentionally randomized across processes, so we use a
    stable hash (blake2b) to ensure reproducibility across runs and machines.
    """
    seed_u64 = int(seed) & 0xFFFFFFFFFFFFFFFF
    digest = hashlib.blake2b(salt.encode("utf-8"), digest_size=8).digest()
    salt_u64 = int.from_bytes(digest, byteorder="little", signed=False)
    return (seed_u64 ^ salt_u64) & 0xFFFFFFFFFFFFFFFF


def _eca_rule_table(rule: int) -> "torch.Tensor":
    """Return uint8 lookup table of shape (8,) for an elementary CA rule."""
    if rule < 0 or rule > 255:
        raise ValueError(f"ECA rule must be in [0, 255], got {rule}")
    return torch.tensor([(rule >> i) & 1 for i in range(8)], dtype=torch.uint8, device="cpu")


def _eca_bits_grid(
    *,
    steps: int,
    width: int,
    rule: int,
    seed: int,
    salt: str,
) -> "torch.Tensor":
    """Generate a (steps, width) uint8 grid from a 1D elementary cellular automaton.

    The initial state is seeded deterministically from (seed, salt) and uses periodic
    (wrap-around) boundary conditions.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")

    rule_table = _eca_rule_table(rule)
    g = torch.Generator(device="cpu")
    g.manual_seed(_stable_seed_u64(seed, salt=salt) % (2**63))

    state = (torch.rand(width, generator=g, device="cpu") > 0.5).to(torch.uint8)
    if int(state.sum().item()) == 0:
        state[width // 2] = 1

    grid = torch.empty((steps, width), dtype=torch.uint8, device="cpu")
    for t in range(steps):
        grid[t].copy_(state)
        left = torch.roll(state, shifts=1, dims=0)
        right = torch.roll(state, shifts=-1, dims=0)
        pattern = (left << 2) | (state << 1) | right
        state = rule_table[pattern.to(torch.int64)]
    return grid


@torch.no_grad()
def ca_init_weight_(
    weight: "torch.Tensor",
    *,
    rule: int,
    seed: int,
    salt: str,
    layout: str = "out_in",
    eps: float = 1e-6,
) -> None:
    """Initialize `weight` using a variance-corrected CA pattern.

    - Supports elementary CA rules 0..255 (we use 30 and 116 by convention).
    - Mean-centers and scales to fan_avg variance: Var = 1 / ((fan_in + fan_out)/2).
    - Generates the CA on CPU for determinism and to avoid per-step GPU kernel launch overhead.

    Args:
        weight: Tensor to initialize (ndim >= 2).
        rule: Wolfram ECA rule (e.g., 30 or 116).
        seed: Global init seed.
        salt: Stable per-tensor salt (e.g., parameter-qualified name).
        layout:
            - "out_in": interpret weight as (fan_out, fan_in) after flattening dims 1..N.
            - "in_out": interpret 2D weight as (fan_in, fan_out) (used by SynapticLinear).
        eps: Numerical epsilon for variance scaling.
    """
    if weight.ndim < 2:
        raise ValueError(f"CA init requires weight.ndim >= 2, got {weight.ndim}")
    if layout not in ("out_in", "in_out"):
        raise ValueError(f"layout must be 'out_in' or 'in_out', got {layout!r}")
    if layout == "in_out" and weight.ndim != 2:
        raise ValueError(f"layout='in_out' requires a 2D tensor, got weight.ndim={weight.ndim}")

    if layout == "out_in":
        fan_out = int(weight.shape[0])
        fan_in = int(weight.numel() // fan_out)
        transpose = False
    else:
        fan_in = int(weight.shape[0])
        fan_out = int(weight.shape[1])
        transpose = True

    bits = _eca_bits_grid(steps=fan_out, width=fan_in, rule=rule, seed=seed, salt=salt)
    vals = bits.to(torch.float32).mul_(2.0).sub_(1.0)  # {0,1} -> {-1,+1}
    vals.sub_(vals.mean())
    var = vals.var(unbiased=False)
    if not torch.isfinite(var) or float(var.item()) <= 0.0:
        raise ValueError(f"CA init produced degenerate variance for {salt!r}: var={var}")

    fan_avg = 0.5 * (fan_in + fan_out)
    target_var = 1.0 / float(fan_avg)
    vals.mul_(math.sqrt(target_var) / (torch.sqrt(var + float(eps)) + float(eps)))

    if transpose:
        vals = vals.transpose(0, 1).contiguous()
    vals = vals.reshape(weight.shape).to(device=weight.device, dtype=weight.dtype)
    weight.copy_(vals)
