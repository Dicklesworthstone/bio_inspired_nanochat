"""
Deciding experiment for the "infinite local context" claim (bead hwxb.9, bridge plan G2).

PRE-REGISTERED DESIGN — frozen before the first run; a change is a new protocol_id.

Question. Do online Hebbian fast weights improve within-sequence recall when the model is trained in
a regime where the writes can act within the sequence (GPTSynaptic.chunked_train_step) and read in
the same regime (synthetic_tasks.retrieval_accuracy(chunk_len=1))? Every earlier null was measured in
a regime that could not see the writes (docs/online_learning_status.md).

Arms (2 x 2 per seed): Hebbian ON / OFF (SynapticConfig.enable_hebbian) x training regime
'full' (one forward per batch) / 'chunked' (chunk_len tokens per forward, truncated BPTT).
Model: GPTSynaptic 2 layers x 64 dims, 4 heads, vocab 97, default SynapticConfig otherwise.
Task: associative recall (synthetic_tasks.associative_recall), pairs drawn uniformly from
2..max_pairs per batch of 16; AdamW lr 3e-3; a FIXED number of steps declared up front (SL-4: no
peeking-based extension).
Reads: recall_accuracy_by_pairs at (2, 4, 8, 16) ∩ [2, max_pairs], batch 64, fixed eval seed, read
both 'full' (one forward) and 'chunked' (chunk_len=1).

Seeds. Discovery {0, 1}: may be looked at, used to tune nothing but to sanity-check the apparatus.
Confirmation {2, 3, 4}: the decision rule is evaluated on these only (SL-1). Both are reported.

Effect. acc(ON, chunked-train, chunked-read) - acc(OFF, chunked-train, chunked-read) at the largest
evaluated pair count, mean over confirmation seeds. Null spread sigma = std over confirmation seeds of
the OFF arm under the same read (NC-7). Minimum detectable effect = 3 sigma / sqrt(n_conf).

Controls, run every time (NC-1 / NC-2 — an apparatus that has never reported "nothing" and never
detected a planted witness has shown nothing):
- negative: OFF models read chunked == read full, bit-identical (nothing to see -> nothing reported);
- planted witness: for ON models, max |logits_chunked - logits_full| > 0 on an eval batch (the writes
  are visible to the chunked read); recorded per model, together with the smallest value seen;
- attention baseline: OFF/full-train accuracy above chance (1/97) at 2 pairs shows the probe measures
  recall at all.
Countermetric: seconds per training step, chunked / full; final training loss for both regimes.

Decision rule (confirmation seeds only, preregistered budget only): if the effect at >= 8 pairs is
>= 3 sigma / sqrt(n_conf) AND the controls pass, the claim is "demonstrated at toy scale" and
--hebb_chunk_len becomes a training-recipe flag in the D1 matrix; otherwise README demotes "infinite
local context" to a hypothesis and enable_hebbian moves to the add-one-in set. A run under any other
budget is labelled 'pilot' and triggers no decision.

Evidence class of any outcome: EMPIRICAL_PATTERN at toy scale on CPU. Not evidence about LM scale.

Run (preregistered):  uv run --no-sync python -m scripts.e2e.hebbian_chunked_regime --budget preregistered
Run (pilot, cheaper): uv run --no-sync python -m scripts.e2e.hebbian_chunked_regime --budget pilot --seeds 0 1 --steps 300 --max-pairs 8
Writes results/hebbian_chunked_regime_<date>_<budget>.json (stamped with measurement_regime).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.results_registry import measurement_regime
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.synthetic_tasks import _logits_for, associative_recall, recall_accuracy_by_pairs

logger = logging.getLogger("bio_inspired_nanochat.hebbian_chunked_regime")
console = Console()

PROTOCOL_ID = "hebbian-chunked-regime-v1-2026-09-02"
VOCAB = 97
SEQ = 64
TRAIN_BATCH = 16
EVAL_BATCH = 64
EVAL_SEED = 777
DISCOVERY_SEEDS = (0, 1)
CONFIRMATION_SEEDS = (2, 3, 4)
PREREGISTERED = {"steps": 2000, "chunk_len": 8, "max_pairs": 16, "lr": 3e-3}
CHANCE = 1.0 / VOCAB


def _model(seed: int, *, hebbian: bool) -> GPTSynaptic:
    torch.manual_seed(seed)
    cfg = GPTSynapticConfig(
        sequence_len=SEQ, vocab_size=VOCAB, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
        synapses=True, syn_cfg=SynapticConfig(enable_hebbian=hebbian),
    )
    return GPTSynaptic(cfg)


def train(model: GPTSynaptic, *, seed: int, regime: str, steps: int, chunk_len: int, max_pairs: int, lr: float) -> dict[str, Any]:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(500_000 + seed)
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        pairs = int(torch.randint(2, max_pairs + 1, (1,), generator=g))
        b = associative_recall(batch=TRAIN_BATCH, num_pairs=pairs, vocab_size=VOCAB, seed=seed * 1_000_000 + step)
        model.reset_sequence_state(reset_fast_weights=True)
        if regime == "chunked":
            loss = model.chunked_train_step(b.inputs, b.targets, chunk_len=chunk_len)
        else:
            _, loss = model(b.inputs, b.targets, train_mode=True)
            loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(f"seed={seed} regime={regime}: non-finite loss at step {step}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
    wall = time.perf_counter() - t0
    return {"final_loss": statistics.fmean(losses[-20:]), "first_loss": statistics.fmean(losses[:20]), "sec_per_step": wall / max(steps, 1)}


@torch.no_grad()
def evaluate(model: GPTSynaptic, *, eval_pairs: tuple[int, ...]) -> dict[str, Any]:
    model.eval()
    out: dict[str, Any] = {}
    for read, chunk in (("full", None), ("chunked", 1)):
        res = recall_accuracy_by_pairs(model, vocab_size=VOCAB, num_pairs=eval_pairs, batch=EVAL_BATCH, seed=EVAL_SEED, chunk_len=chunk)
        out[read] = {str(k): float(v) for k, v in res["by_pairs"].items()}
    # apparatus check: are the writes visible to the chunked read at all?
    probe = associative_recall(batch=8, num_pairs=eval_pairs[-1], vocab_size=VOCAB, seed=EVAL_SEED + 1)
    model.reset_sequence_state(reset_fast_weights=True)
    full = _logits_for(model, probe.inputs, chunk_len=None).clone()
    model.reset_sequence_state(reset_fast_weights=True)
    chunked = _logits_for(model, probe.inputs, chunk_len=1).clone()
    out["max_abs_logit_diff_chunked_vs_full"] = float((chunked - full).abs().max())
    return out


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _std(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", choices=["preregistered", "pilot", "smoke"], default="preregistered")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DISCOVERY_SEEDS + CONFIRMATION_SEEDS))
    parser.add_argument("--steps", type=int, default=PREREGISTERED["steps"])
    parser.add_argument("--chunk-len", type=int, default=PREREGISTERED["chunk_len"])
    parser.add_argument("--max-pairs", type=int, default=PREREGISTERED["max_pairs"])
    parser.add_argument("--lr", type=float, default=PREREGISTERED["lr"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.budget == "preregistered":
        declared = {"steps": args.steps, "chunk_len": args.chunk_len, "max_pairs": args.max_pairs, "lr": args.lr}
        if declared != PREREGISTERED or set(args.seeds) != set(DISCOVERY_SEEDS + CONFIRMATION_SEEDS):
            raise SystemExit("--budget preregistered requires the preregistered steps/chunk_len/max_pairs/lr and all five seeds; use --budget pilot")
    eval_pairs = tuple(p for p in (2, 4, 8, 16) if p <= args.max_pairs)
    out_path = Path(args.out or f"results/hebbian_chunked_regime_{_dt.date.today().isoformat()}_{args.budget}.json")
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        git_sha = ""

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        for hebbian in (True, False):
            for regime in ("chunked", "full"):
                model = _model(seed, hebbian=hebbian)
                tr = train(model, seed=seed, regime=regime, steps=args.steps, chunk_len=args.chunk_len, max_pairs=args.max_pairs, lr=args.lr)
                ev = evaluate(model, eval_pairs=eval_pairs)
                row = {"seed": seed, "hebbian": hebbian, "train_regime": regime, "set": "discovery" if seed in DISCOVERY_SEEDS else "confirmation", **tr, "eval": ev}
                runs.append(row)
                logger.info("[regime] seed=%d hebbian=%s train=%s loss=%.3f s/step=%.2f read_full=%s read_chunked=%s diff=%.2e",
                            seed, hebbian, regime, tr["final_loss"], tr["sec_per_step"], ev["full"], ev["chunked"], ev["max_abs_logit_diff_chunked_vs_full"])

    top = str(eval_pairs[-1])

    def acc(rows, hebbian, regime, read, pairs=top):
        return [r["eval"][read][pairs] for r in rows if r["hebbian"] is hebbian and r["train_regime"] == regime]

    def summarize(rows):
        on = acc(rows, True, "chunked", "chunked")
        off = acc(rows, False, "chunked", "chunked")
        sigma = _std(off)
        n = len(on)
        m_on, m_off = _mean(on), _mean(off)
        effect = (m_on - m_off) if (m_on is not None and m_off is not None) else None
        mde = (3 * sigma / (n ** 0.5)) if (sigma is not None and n) else None
        return {
            "n_seeds": n,
            "acc_on_chunked_train_chunked_read": _mean(on),
            "acc_off_chunked_train_chunked_read": _mean(off),
            "effect_at_top_pairs": effect,
            "null_sigma_off_arm": sigma,
            "minimum_detectable_effect_3sigma": mde,
            "acc_on_chunked_train_full_read": _mean(acc(rows, True, "chunked", "full")),
            "acc_on_full_train_chunked_read": _mean(acc(rows, True, "full", "chunked")),
            "acc_off_full_train_full_read": _mean(acc(rows, False, "full", "full")),
            "acc_off_full_train_2pairs_full_read": _mean(acc(rows, False, "full", "full", pairs=str(eval_pairs[0]))),
        }

    disc = [r for r in runs if r["set"] == "discovery"]
    conf = [r for r in runs if r["set"] == "confirmation"]
    controls = {
        "negative_off_reads_identical": all(
            r["eval"]["max_abs_logit_diff_chunked_vs_full"] == 0.0 for r in runs if not r["hebbian"]),
        "planted_witness_on_writes_visible": all(
            r["eval"]["max_abs_logit_diff_chunked_vs_full"] > 0.0 for r in runs if r["hebbian"]),
        "min_visible_diff_on": min((r["eval"]["max_abs_logit_diff_chunked_vs_full"] for r in runs if r["hebbian"]), default=None),
        "attention_baseline_above_chance": (lambda v: v is not None and v > 2 * CHANCE)(
            _mean(acc(runs, False, "full", "full", pairs=str(eval_pairs[0])))),
    }
    chunk_rows = [r for r in runs if r["train_regime"] == "chunked"]
    full_rows = [r for r in runs if r["train_regime"] == "full"]
    countermetric = {
        "sec_per_step_chunked": _mean([r["sec_per_step"] for r in chunk_rows]),
        "sec_per_step_full": _mean([r["sec_per_step"] for r in full_rows]),
        "final_loss_chunked": _mean([r["final_loss"] for r in chunk_rows]),
        "final_loss_full": _mean([r["final_loss"] for r in full_rows]),
    }
    conf_summary = summarize(conf)
    decision = "pilot: no decision (budget != preregistered)"
    if args.budget == "preregistered" and conf_summary["n_seeds"] == len(CONFIRMATION_SEEDS):
        passes = all(v for k, v in controls.items() if k != "min_visible_diff_on")
        eff, mde = conf_summary["effect_at_top_pairs"], conf_summary["minimum_detectable_effect_3sigma"]
        if not passes:
            decision = "controls failed: the apparatus is not trusted; no decision"
        elif eff is not None and mde is not None and eff >= mde:
            decision = "demonstrated at toy scale: ON > OFF under chunked training and reading at the top pair count"
        else:
            decision = "not demonstrated: README demotes 'infinite local context' to a hypothesis; enable_hebbian moves to add-one-in"

    payload = {
        "protocol_id": PROTOCOL_ID,
        "budget": {"label": args.budget, "steps": args.steps, "chunk_len": args.chunk_len, "max_pairs": args.max_pairs, "lr": args.lr,
                   "seeds": args.seeds, "eval_pairs": list(eval_pairs), "preregistered": PREREGISTERED,
                   "discovery_seeds": list(DISCOVERY_SEEDS), "confirmation_seeds": list(CONFIRMATION_SEEDS)},
        "git_sha": git_sha,
        "measurement_regime": measurement_regime(read="both"),
        "evidence_class": "EMPIRICAL_PATTERN (toy scale, CPU)",
        "controls": controls,
        "countermetric": countermetric,
        "summary": {"discovery": summarize(disc), "confirmation": conf_summary},
        "decision": decision,
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table = Table(title=f"hebbian chunked regime [{args.budget}] top pairs = {top}")
    for col in ("seed", "set", "hebb", "train", "read full", "read chunked", "loss", "s/step", "|Δlogit|"):
        table.add_column(col, justify="right")
    for r in runs:
        table.add_row(str(r["seed"]), r["set"][:4], "on" if r["hebbian"] else "off", r["train_regime"],
                      f"{r['eval']['full'][top]:.3f}", f"{r['eval']['chunked'][top]:.3f}", f"{r['final_loss']:.3f}",
                      f"{r['sec_per_step']:.2f}", f"{r['eval']['max_abs_logit_diff_chunked_vs_full']:.1e}")
    console.print(table)
    console.print(f"controls: {controls}")
    console.print(f"confirmation summary: {conf_summary}")
    console.print(f"[bold]decision:[/bold] {decision}")
    console.print(f"[bold]written[/bold] {out_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
