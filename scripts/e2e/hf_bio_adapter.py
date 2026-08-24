"""External-checkpoint E2E for the cross-architecture Hugging Face bio adapter.

The default run downloads a pinned, public GPT-2 checkpoint with the repository-mandated user
agent, then switches to local-only Transformers loading.  It checks function preservation,
performs a short adapter-only fine-tune with live biological dynamics, packages the adapter, and
strictly reloads it into a fresh base checkpoint.

Run with:

    uv run python -m scripts.e2e.hf_bio_adapter
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

from bio_inspired_nanochat.hf_bio_adapter import (
    bio_adapter_metrics,
    bio_adapter_parameters,
    inject_bio_adapters,
    load_bio_adapter,
    save_bio_adapter,
    set_bio_adaptation,
)
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.torch_imports import torch

USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"
DEFAULT_REPOSITORY = "sshleifer/tiny-gpt2"
DEFAULT_REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"


@dataclass(frozen=True)
class ExternalAdapterReport:
    """Strict, machine-readable evidence for ``bio_inspired_nanochat-r00r.8``."""

    schema_version: int
    bead: str
    repository: str
    revision: str
    model_type: str
    adapter_names: list[str]
    adapted_parameters: int
    initial_max_abs_logit_delta: float
    losses: list[float]
    adapter_metrics: dict[str, dict[str, float | int | str | bool]]
    dynamics_active: bool
    output_finite: bool
    roundtrip_max_abs_logit_delta: float
    bundle_manifest: dict[str, Any]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_passed(self) -> None:
        if not self.passed:
            raise AssertionError(
                "external HF bio-adapter E2E failed: "
                f"initial_delta={self.initial_max_abs_logit_delta:.6g}, "
                f"dynamics_active={self.dynamics_active}, output_finite={self.output_finite}, "
                f"roundtrip_delta={self.roundtrip_max_abs_logit_delta:.6g}"
            )


def _download_checkpoint(
    repository: str,
    revision: str,
    *,
    cache_dir: str | Path | None,
) -> Path:
    """Fetch only model/tokenizer artifacts, always with the mandated user-agent string."""
    downloaded = snapshot_download(
        repository,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "merges.txt",
            "pytorch_model.bin",
            "*.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
        user_agent=USER_AGENT,
        headers={"user-agent": USER_AGENT},
    )
    return Path(downloaded)


def _token_batch(tokenizer: Any) -> tuple[Any, Any]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        [
            "Biological synapses adapt across several interacting timescales.",
            "A tiny transformer can still expose real plasticity telemetry.",
        ],
        return_tensors="pt",
        padding=True,
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return encoded, labels


def run_external_adapter_e2e(
    *,
    repository: str = DEFAULT_REPOSITORY,
    revision: str = DEFAULT_REVISION,
    local_model_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    steps: int = 4,
    seed: int = 20260824,
) -> ExternalAdapterReport:
    """Run the external checkpoint -> adapt -> package -> reload evidence pipeline."""
    if steps < 2:
        raise ValueError("steps must be at least 2 so a deferred Hebbian write can land")
    torch.manual_seed(seed)
    source_dir = (
        Path(local_model_dir)
        if local_model_dir is not None
        else _download_checkpoint(repository, revision, cache_dir=cache_dir)
    )
    output_dir = Path(run_dir) if run_dir is not None else Path("runs/e2e/hf_bio_adapter")

    with RunLogger(
        output_dir,
        name="hf_bio_adapter",
        console=False,
        provenance={
            "bead": "bio_inspired_nanochat-r00r.8",
            "repository": repository,
            "revision": revision,
            "seed": seed,
            "steps": steps,
        },
    ) as logger:
        # Bandit B615 is inapplicable: this is an already-downloaded local path, obtained from
        # the pinned revision above, and network access is explicitly disabled for Transformers.
        tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            source_dir, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            source_dir, local_files_only=True
        ).eval()
        encoded, labels = _token_batch(tokenizer)
        with torch.no_grad():
            reference_logits = model(**encoded).logits.detach()

        config = SynapticConfig(
            enable_presyn=True,
            enable_hebbian=True,
            enable_metabolism=True,
            stochastic_train_frac=0.0,
            fast_weight_eta=0.02,
            fast_weight_max_norm=0.1,
            post_slow_lr=1e-5,
        )
        injection = inject_bio_adapters(model, config)
        set_bio_adaptation(model, False)
        with torch.no_grad():
            injected_logits = model(**encoded).logits.detach()
        initial_delta = float((injected_logits - reference_logits).abs().max().item())
        logger.event(
            "hf_bio_adapter_injected",
            injection=injection.to_dict(),
            initial_max_abs_logit_delta=initial_delta,
        )

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        parameters = bio_adapter_parameters(model)
        for parameter in parameters:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(parameters, lr=2e-4)

        losses: list[float] = []
        model.train()
        set_bio_adaptation(model, True)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**encoded, labels=labels).loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise AssertionError(f"non-finite adapter loss at step {step}: {loss}")
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().item())
            losses.append(loss_value)
            logger.log_metrics(step=step, loss=loss_value)

        metrics = bio_adapter_metrics(model)
        dynamics_active = all(
            int(values["adaptation_steps"]) >= steps
            and float(values["calcium"]) > 0.0
            and float(values["eligibility_norm"]) > 0.0
            for values in metrics.values()
        ) and any(float(values["fast_weight_norm"]) > 0.0 for values in metrics.values())
        logger.event(
            "hf_bio_adapter_dynamics",
            dynamics_active=dynamics_active,
            adapter_metrics=metrics,
        )

        # Flush the final deferred online write without advancing the runtime state.
        model.eval()
        set_bio_adaptation(model, False)
        with torch.no_grad():
            model(**encoded)

        resolved_bundle_dir = (
            Path(bundle_dir)
            if bundle_dir is not None
            else output_dir / "bundles" / uuid.uuid4().hex[:12]
        )
        manifest = save_bio_adapter(model, resolved_bundle_dir)
        with torch.no_grad():
            expected_logits = model(**encoded).logits.detach()

        restored = AutoModelForCausalLM.from_pretrained(  # nosec B615
            source_dir,
            local_files_only=True,
        ).eval()
        load_bio_adapter(
            restored,
            resolved_bundle_dir,
            adaptation_enabled=False,
        )
        with torch.no_grad():
            restored_logits = restored(**encoded).logits.detach()
        output_finite = bool(torch.isfinite(restored_logits).all().item())
        roundtrip_delta = float((restored_logits - expected_logits).abs().max().item())

        finite_losses = all(math.isfinite(value) for value in losses)
        passed = (
            initial_delta <= 1e-5
            and finite_losses
            and dynamics_active
            and output_finite
            and roundtrip_delta <= 1e-7
        )
        report = ExternalAdapterReport(
            schema_version=1,
            bead="bio_inspired_nanochat-r00r.8",
            repository=repository,
            revision=revision,
            model_type=str(getattr(model.config, "model_type", "unknown")),
            adapter_names=list(injection.adapter_names),
            adapted_parameters=injection.adapted_parameters,
            initial_max_abs_logit_delta=initial_delta,
            losses=losses,
            adapter_metrics=metrics,
            dynamics_active=dynamics_active,
            output_finite=output_finite,
            roundtrip_max_abs_logit_delta=roundtrip_delta,
            bundle_manifest=manifest,
            passed=passed,
        )
        logger.event("hf_bio_adapter_result", **report.to_dict())
        report.assert_passed()
        return report


def _render(report: ExternalAdapterReport, console: Console) -> None:
    table = Table(title="External Hugging Face bio-adapter E2E")
    table.add_column("Check")
    table.add_column("Evidence", justify="right")
    table.add_row("checkpoint", f"{report.repository}@{report.revision[:12]}")
    table.add_row("adapters", str(len(report.adapter_names)))
    table.add_row("initial logit max |Δ|", f"{report.initial_max_abs_logit_delta:.3e}")
    table.add_row("loss", f"{report.losses[0]:.6f} → {report.losses[-1]:.6f}")
    table.add_row("dynamics active", str(report.dynamics_active))
    table.add_row("bundle reload max |Δ|", f"{report.roundtrip_max_abs_logit_delta:.3e}")
    table.add_row("verdict", "PASS" if report.passed else "FAIL")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-model-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=Path("results/hf_bio_adapter.json"))
    args = parser.parse_args()

    report = run_external_adapter_e2e(
        repository=args.repository,
        revision=args.revision,
        local_model_dir=args.local_model_dir,
        cache_dir=args.cache_dir,
        bundle_dir=args.bundle_dir,
        run_dir=args.run_dir,
        steps=args.steps,
        seed=args.seed,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result artifact: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _render(report, Console())


if __name__ == "__main__":
    main()
