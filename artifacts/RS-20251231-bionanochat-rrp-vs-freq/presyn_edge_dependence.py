import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bio_inspired_nanochat.synaptic import (  # noqa: E402
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
)


def run_once(q: torch.Tensor, k: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    cfg = SynapticConfig()
    cfg.enable_presyn = True

    B, H, T, D = q.shape
    pre = SynapticPresyn(d_head=D, cfg=cfg)
    state = build_presyn_state(B=B, T=T, H=H, device=q.device, dtype=q.dtype, cfg=cfg)

    syn_logit, _state = pre.forward(q=q, k=k, logits=logits, state=state, mask=None, train_mode=False)
    return syn_logit


def main() -> None:
    torch.manual_seed(0)

    B, H, T, D = 1, 1, 8, 4

    # Synthetic causal logits: baseline -2 for all causal keys, strong preference for key 0.
    logits = torch.full((B, H, T, T), -20.0, dtype=torch.float32)
    for t in range(T):
        logits[0, 0, t, : t + 1] = -2.0
        logits[0, 0, t, 0] = 2.0

    q_uniform = torch.ones((B, H, T, D), dtype=torch.float32)
    k_uniform = torch.ones((B, H, T, D), dtype=torch.float32)

    y_uniform = run_once(q_uniform, k_uniform, logits)

    # Random q/k structure (same logits).
    torch.manual_seed(0)
    q_rand = torch.randn((B, H, T, D), dtype=torch.float32)
    k_rand = torch.randn((B, H, T, D), dtype=torch.float32)

    y_rand = run_once(q_rand, k_rand, logits)

    # Potency sanity: repeat the random case with same seed.
    torch.manual_seed(0)
    q_rand2 = torch.randn((B, H, T, D), dtype=torch.float32)
    k_rand2 = torch.randn((B, H, T, D), dtype=torch.float32)

    y_rand2 = run_once(q_rand2, k_rand2, logits)

    out = {
        "setup": {
            "B": B,
            "H": H,
            "T": T,
            "D": D,
            "logits": {
                "causal_baseline": -2.0,
                "key0_boost": 2.0,
                "future_fill": -20.0,
            },
        },
        "metrics": {
            "mean_abs_diff_uniform_vs_random": float((y_uniform - y_rand).abs().mean()),
            "mean_abs_diff_random_repeat": float((y_rand - y_rand2).abs().mean()),
            "y_uniform_mean": float(y_uniform.mean()),
            "y_random_mean": float(y_rand.mean()),
        },
    }

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
