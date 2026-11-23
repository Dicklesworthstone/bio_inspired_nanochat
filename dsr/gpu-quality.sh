#!/usr/bin/env bash
# Hardware acceptance on the trusted dual-4090 worker, in a private checkout.
set -euo pipefail
out="$PWD/runs/dsr/${DSR_RUN_ID:?}/gpu"
mkdir -p "$out"
git rev-parse HEAD
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv
nvidia-smi topo -m
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv sync --frozen --python 3.14 --extra gpu --dev
uv run --no-sync maturin develop --release
uv run --no-sync python - <<'PY'
import torch
import triton
import rustbpe
from rich import print

cards = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(cards) < 2 or any("RTX 4090" not in card for card in cards[:2]):
    raise SystemExit(f"dual-4090 qualification requires two visible RTX 4090s, got {cards}")
if getattr(rustbpe, "PRESYN_CPU_ABI", None) != 1:
    raise SystemExit("the installed CPU reference-comparison extension requires ABI 1")
print({"cards": cards, "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__})
PY
uv run --no-sync python -m pytest -m gpu -ra --junitxml="$out/gpu-tests.xml"
uv run --no-sync python - "$out/gpu-tests.xml" <<'PY'
import sys
from xml.etree import ElementTree

cases = ElementTree.parse(sys.argv[1]).findall(".//testcase")
skipped = [case.get("name") for case in cases if case.find("skipped") is not None]
if not cases or skipped:
    raise SystemExit(f"GPU qualification ran {len(cases) - len(skipped)}/{len(cases)} cases; skipped={skipped}")
PY
uv run --no-sync python -m pytest -ra tests/test_presyn_canonical.py tests/test_presyn_golden.py \
  tests/test_presyn_rust_dispatch.py tests/test_presyn_rust_parity.py tests/test_rust_kernels.py \
  --junitxml="$out/reference-tests.xml"
perf_status=0
uv run --no-sync python -m scripts.benchmark_presyn_live --output "$out/presyn-timings.json" || perf_status=1
uv run --no-sync python -m scripts.perf_regression_gate --device cuda --mode check --tolerance 0.45 \
  --baselines results/perf_baselines_rtx4090.json --check-budget \
  --observations-json "$out/throughput-observations.json" --output-json "$out/throughput.json" || perf_status=1
exit "$perf_status"
