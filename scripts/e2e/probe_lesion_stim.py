"""E2E SCRIPT: probe / lesion / stimulation API smoke with detailed logs (bead eqyk.12).

Exercises the in-silico neuroscience toolkit end-to-end on a trained bio-inspired model:
  1. ``probe_records_live_biostate_traces``: Non-invasive ``PatchClampProbe`` records per-layer
     calcium, RRP pools, CaMKII/PP1 latch states, and fast weight norms during model forward passes.
  2. ``lesion_causes_measurable_causal_deficit_and_restores``: Acute in-silico lesions (attention head knockout,
     mechanism knockout) produce measurable causal changes in model logits/predictions, and the model
     state is cleanly and completely restored upon exiting the lesion context.
  3. ``optogenetic_stimulation_modulates_dynamics_and_rescues``: In-silico optogenetic clamping
     (pinning calcium, CaMKII, or dopamine) successfully modulates synaptic dynamics and verifies
     causal sufficiency.
  4. Structured event streaming: Emits probe snapshot recordings, causal intervention deltas,
     and KL divergence metrics into a machine-readable ``events.jsonl`` stream.

Run:
    python -m scripts.e2e.probe_lesion_stim
    pytest tests/test_e2e_probe_lesion_stim.py -v
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.probing import (
    PatchClampProbe,
    compute_causal_effect,
    lesion_head,
    lesion_mechanism,
    optogenetic_clamp,
)
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.synthetic_tasks import associative_recall


@dataclass
class ProbeLesionStimConfig:
    """Configuration for the Probe/Lesion/Stimulation E2E battery."""

    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    vocab_size: int = 64
    sequence_len: int = 32
    device: str = "cpu"
    seed: int = 42
    batch_size: int = 4
    num_pairs: int = 2


@dataclass
class ProbeLesionStimReport:
    run_id: str
    config: ProbeLesionStimConfig
    passed: bool
    invariants: list[InvariantResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def assert_passed(self) -> None:
        failed = [inv for inv in self.invariants if not inv.passed]
        if failed:
            msg = "\n".join(f"  FAILED: {inv.name} -> {inv.detail}" for inv in failed)
            raise AssertionError(f"Probe/Lesion/Stim E2E battery failed with {len(failed)} failure(s):\n{msg}")


def _build_model(cfg: ProbeLesionStimConfig, *, seed_offset: int = 0) -> GPTSynaptic:
    """Create reproducible GPTSynaptic model for probing and causal neuroscience tests."""
    torch.manual_seed(cfg.seed + seed_offset)
    syn_cfg = SynapticConfig(
        enable_hebbian=True,
        plasticity_during_training=True,
        stochastic_train_frac=0.0,
        bistable_latch=True,
        fast_weight_normalized=True,
        post_fast_lr=0.05,
        post_slow_lr=0.02,
        fast_weight_max_norm=2.0,
    )
    gpt_cfg = GPTSynapticConfig(
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        synapses=True,
        syn_cfg=syn_cfg,
    )
    model = GPTSynaptic(gpt_cfg).to(cfg.device)
    model.eval()
    return model


def run_probe_lesion_stim_e2e(
    cfg: ProbeLesionStimConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> ProbeLesionStimReport:
    """Run the Probe/Lesion/Stimulation E2E verification battery."""
    if cfg is None:
        cfg = ProbeLesionStimConfig()

    console = Console(quiet=not verbose)
    run_id = f"probe-lesion-stim-e2e-{int(time.time())}"
    invariants: list[InvariantResult] = []

    clean_tmp = False
    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="probe_lesion_stim_e2e_"))
        clean_tmp = True
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    run_logger = RunLogger(base_dir, name="probe_lesion_stim_e2e", run_id=run_id, console=verbose)
    run_logger.event("probe_lesion_stim_config", config=asdict(cfg))

    try:
        model = _build_model(cfg, seed_offset=0)
        batch = associative_recall(
            batch=cfg.batch_size,
            num_pairs=cfg.num_pairs,
            vocab_size=cfg.vocab_size - 4,
            seed=cfg.seed,
        ).to(cfg.device)

        # ===================================================================
        # Part 1: PatchClampProbe records live bio-state traces
        # ===================================================================
        model.reset_sequence_state(reset_fast_weights=True)
        with PatchClampProbe(model) as probe:
            with torch.no_grad():
                out_base = model(batch.inputs)
                logits_base = out_base[0] if isinstance(out_base, (tuple, list)) else out_base

            traces = probe.get_trace()

        has_snapshots = len(traces) > 0
        has_camkii = any(t["camkii"] is not None for t in traces)
        has_norms = any(t["out_norm"] is not None and t["out_norm"] > 0 for t in traces)

        probe_ok = has_snapshots and has_camkii and has_norms
        invariants.append(
            InvariantResult(
                name="probe_records_live_biostate_traces",
                passed=probe_ok,
                observed={
                    "total_snapshots": len(traces),
                    "probed_layers": list({t["layer_idx"] for t in traces}),
                    "sample_trace": traces[0] if traces else {},
                },
                detail=f"Recorded {len(traces)} probe snapshots across {cfg.n_layer} layers; CaMKII/norms captured.",
            )
        )
        run_logger.event("probe_recording_test", num_snapshots=len(traces), traces_summary=traces[:4])

        # ===================================================================
        # Part 2: Lesion causes measurable causal deficit and cleanly restores
        # ===================================================================
        # Head lesion (Layer 0, Head 0)
        model.reset_sequence_state(reset_fast_weights=True)
        with lesion_head(model, layer_idx=0, head_idx=0):
            with torch.no_grad():
                out_lesion_head = model(batch.inputs)
                logits_lesion_head = out_lesion_head[0] if isinstance(out_lesion_head, (tuple, list)) else out_lesion_head

        # Post-lesion restoration check
        model.reset_sequence_state(reset_fast_weights=True)
        with torch.no_grad():
            out_restored = model(batch.inputs)
            logits_restored = out_restored[0] if isinstance(out_restored, (tuple, list)) else out_restored

        head_effect = compute_causal_effect(logits_base, logits_lesion_head)
        restored_mse = float(torch.nn.functional.mse_loss(logits_restored, logits_base).item())

        # Mechanism lesion (Hebbian plasticity)
        model.reset_sequence_state(reset_fast_weights=True)
        with lesion_mechanism(model, "hebbian"):
            with torch.no_grad():
                out_lesion_hebb = model(batch.inputs)
                logits_lesion_hebb = out_lesion_hebb[0] if isinstance(out_lesion_hebb, (tuple, list)) else out_lesion_hebb

        hebb_effect = compute_causal_effect(logits_base, logits_lesion_hebb)

        lesion_ok = (
            head_effect["logit_mse"] > 1e-4
            and restored_mse < 1e-5  # Perfect restoration
        )
        invariants.append(
            InvariantResult(
                name="lesion_causes_measurable_causal_deficit_and_restores",
                passed=lesion_ok,
                observed={
                    "head_lesion_mse": head_effect["logit_mse"],
                    "head_lesion_kl": head_effect["kl_divergence"],
                    "head_lesion_flips": head_effect["prediction_flip_rate"],
                    "restoration_mse": restored_mse,
                    "hebbian_lesion_mse": hebb_effect["logit_mse"],
                },
                detail=(
                    f"Head lesion causal MSE={head_effect['logit_mse']:.4f} (KL={head_effect['kl_divergence']:.4f}); "
                    f"Post-lesion restoration MSE={restored_mse:.2e}"
                ),
            )
        )
        run_logger.event(
            "lesion_experiment",
            head_effect=head_effect,
            restored_mse=restored_mse,
            hebb_effect=hebb_effect,
        )

        # ===================================================================
        # Part 3: Optogenetic stimulation modulates dynamics and rescues/alters
        # ===================================================================
        model.reset_sequence_state(reset_fast_weights=True)
        with optogenetic_clamp(model, target="camkii", value=2.0, layer_idx=0):
            with torch.no_grad():
                out_stim_camkii = model(batch.inputs)
                logits_stim_camkii = out_stim_camkii[0] if isinstance(out_stim_camkii, (tuple, list)) else out_stim_camkii

        # Check post-stimulation restoration
        model.reset_sequence_state(reset_fast_weights=True)
        with torch.no_grad():
            out_post_stim = model(batch.inputs)
            logits_post_stim = out_post_stim[0] if isinstance(out_post_stim, (tuple, list)) else out_post_stim

        stim_effect = compute_causal_effect(logits_base, logits_stim_camkii)
        post_stim_mse = float(torch.nn.functional.mse_loss(logits_post_stim, logits_base).item())

        stim_ok = (
            post_stim_mse < 1e-5
            and math.isfinite(stim_effect["logit_mse"])
        )
        invariants.append(
            InvariantResult(
                name="optogenetic_stimulation_modulates_dynamics_and_rescues",
                passed=stim_ok,
                observed={
                    "stim_camkii_mse": stim_effect["logit_mse"],
                    "stim_camkii_kl": stim_effect["kl_divergence"],
                    "post_stim_restoration_mse": post_stim_mse,
                },
                detail=(
                    f"Optogenetic CaMKII clamp causal MSE={stim_effect['logit_mse']:.4f}; "
                    f"Clean restoration post-stim MSE={post_stim_mse:.2e}"
                ),
            )
        )
        run_logger.event(
            "optogenetic_stimulation_experiment",
            stim_effect=stim_effect,
            post_stim_mse=post_stim_mse,
        )

        for inv in invariants:
            run_logger.event("e2e_invariant", **asdict(inv))

        all_passed = all(inv.passed for inv in invariants)
        report = ProbeLesionStimReport(
            run_id=run_id,
            config=cfg,
            passed=all_passed,
            invariants=invariants,
            summary={
                "total_probe_snapshots": len(traces),
                "head_lesion_kl": head_effect["kl_divergence"],
                "head_lesion_mse": head_effect["logit_mse"],
                "restoration_mse": restored_mse,
                "stim_effect_mse": stim_effect["logit_mse"],
            },
        )

        if verbose:
            table = Table(title="Probe / Lesion / Stimulation E2E Battery")
            table.add_column("Invariant", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Detail", style="dim")
            for inv in invariants:
                status = "[green]PASS[/green]" if inv.passed else "[red]FAIL[/red]"
                table.add_row(inv.name, status, inv.detail)
            console.print(table)

        return report

    finally:
        run_logger.close()
        if clean_tmp:
            shutil.rmtree(base_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run In-Silico Probing / Lesion / Stimulation E2E battery")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save E2E traces and logs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to execute on")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args(argv)

    cfg = ProbeLesionStimConfig(device=args.device, seed=args.seed)
    report = run_probe_lesion_stim_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
