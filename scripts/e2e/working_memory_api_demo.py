"""Deterministic end-to-end demo for the synaptic working-memory API.

The demo writes a key/value association into the final MLP projection of a
tiny synaptic GPT.  The value is chosen analytically to make the runner-up
token overtake the original next-token prediction.  It then verifies that the
predicted and observed logit margins agree and writes the full operation trace
to ``events.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticLinear
from bio_inspired_nanochat.working_memory_api import (
    WorkingMemoryPolicy,
    WorkingMemoryScratchpad,
)


@dataclass(frozen=True, slots=True)
class WorkingMemoryDemoResult:
    """Machine-readable evidence that a neural-memory write changed generation."""

    baseline_token: int
    injected_token: int
    target_token: int
    predicted_margin: float
    observed_margin: float
    margin_error: float
    module: str
    events_path: str

    @property
    def passed(self) -> bool:
        return (
            self.baseline_token != self.target_token
            and self.injected_token == self.target_token
            and self.predicted_margin > 0.0
            and self.margin_error < 1e-4
        )

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "passed": self.passed,
            "baseline_token": self.baseline_token,
            "injected_token": self.injected_token,
            "target_token": self.target_token,
            "predicted_margin": self.predicted_margin,
            "observed_margin": self.observed_margin,
            "margin_error": self.margin_error,
            "module": self.module,
            "events_path": self.events_path,
        }


def _tiny_model(seed: int) -> GPTSynaptic:
    torch.manual_seed(seed)
    return GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
            logit_softcap=0.0,
            synapses=True,
            use_moe=False,
        )
    )


def run_demo(run_dir: str | Path, *, seed: int = 19) -> WorkingMemoryDemoResult:
    """Run the deterministic next-token intervention and return its evidence."""
    model = _tiny_model(seed)
    model.eval()
    module_name = "h.0.mlp.mlp.proj"
    projection = dict(model.named_modules()).get(module_name)
    if not isinstance(projection, SynapticLinear) or projection.post is None:
        raise RuntimeError(f"expected a writable SynapticLinear at {module_name}")

    run_path = Path(run_dir)
    with RunLogger(
        run_path,
        name="working_memory_api_demo",
        console=False,
        provenance={"seed": seed, "bead": "bio_inspired_nanochat-r00r.9"},
    ) as logger:
        scratchpad = WorkingMemoryScratchpad(
            model,
            policy=WorkingMemoryPolicy(
                max_vector_norm=16.0,
                max_abs_scale=1.0,
                max_delta_norm=16.0,
                max_norm_growth=16.0,
            ),
            logger=logger,
        )
        scratchpad.clear_scratchpad()
        snapshot = scratchpad.read_scratchpad()
        site = next(item for item in snapshot["sites"] if item["module"] == module_name)

        prompt = torch.tensor([[1, 7, 3, 11, 5, 2]], dtype=torch.long)
        captured: list[torch.Tensor] = []

        def capture_input(_module: torch.nn.Module, args: tuple[object, ...]) -> None:
            value = args[0]
            if torch.is_tensor(value):
                captured.append(value.detach().clone())

        handle = projection.register_forward_pre_hook(capture_input)
        try:
            with torch.no_grad():
                baseline_logits, _ = model(prompt, train_mode=False)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("final projection input was not captured")

        baseline = baseline_logits[0, -1].detach().float()
        ranking = torch.argsort(baseline, descending=True)
        baseline_token = int(ranking[0].item())
        target_token = int(ranking[1].item())
        baseline_margin = float((baseline[baseline_token] - baseline[target_token]).item())

        projection_input = captured[-1][-1].to(projection.w_fast)
        input_norm = projection_input.norm()
        input_norm_value = float(input_norm.detach().item())
        if input_norm_value <= 1e-8:
            raise RuntimeError("demo projection input is degenerate")
        key = projection_input / input_norm

        post = projection.post
        transform = torch.diag(1.0 + post.fast + post.slow) + post.U @ post.V
        head_delta = (
            model.lm_head.weight[target_token] - model.lm_head.weight[baseline_token]
        ).to(transform)
        value_gradient = transform @ head_delta
        gradient_norm = value_gradient.norm()
        gradient_norm_value = float(gradient_norm.detach().item())
        if gradient_norm_value <= 1e-8:
            raise RuntimeError("demo target has a degenerate readout direction")

        fast_gate = 0.5 * 0.8
        desired_margin = 1e-3
        value_norm = (baseline_margin + desired_margin) / (
            fast_gate * input_norm_value * gradient_norm_value
        )
        value = (value_gradient / gradient_norm) * value_norm

        receipt = scratchpad.write_association(
            int(site["site_index"]),
            key,
            value,
            expected_module=module_name,
        )
        with torch.no_grad():
            injected_logits, _ = model(prompt, train_mode=False)
        injected = injected_logits[0, -1].detach().float()
        injected_token = int(injected.argmax().item())

        predicted_gain = (
            fast_gate
            * input_norm_value
            * float(torch.dot(value, value_gradient).item())
            * float(receipt["effective_scale"])
        )
        predicted_margin = float(baseline[target_token] - baseline[baseline_token]) + predicted_gain
        observed_margin = float((injected[target_token] - injected[baseline_token]).item())
        margin_error = abs(predicted_margin - observed_margin)

        result = WorkingMemoryDemoResult(
            baseline_token=baseline_token,
            injected_token=injected_token,
            target_token=target_token,
            predicted_margin=predicted_margin,
            observed_margin=observed_margin,
            margin_error=margin_error,
            module=module_name,
            events_path=str(logger.events_path),
        )
        logger.event(
            "working_memory_demo_result",
            passed=result.passed,
            baseline_token=result.baseline_token,
            injected_token=result.injected_token,
            target_token=result.target_token,
            predicted_margin=result.predicted_margin,
            observed_margin=result.observed_margin,
            margin_error=result.margin_error,
            module=result.module,
            events_path=result.events_path,
        )
        if not result.passed:
            raise AssertionError(f"working-memory generation intervention failed: {result.to_dict()}")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="artifact directory (default: runs/e2e/working_memory_api/<timestamp>)",
    )
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runs/e2e/working_memory_api") / str(time.time_ns())
    result = run_demo(run_dir, seed=args.seed)
    Console().print_json(data=json.dumps(result.to_dict()))


if __name__ == "__main__":
    main()
