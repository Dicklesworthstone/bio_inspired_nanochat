#!/usr/bin/env bash
#
# Convenience launcher for a full bio-inspired NanoChat experiment on a dual 4090 rig.
# The script will:
#   1. Make sure we have enough FineWeb shards.
#   2. Run the fast "verify_evolution" smoke test (produces NeuroViz dashboards immediately).
#   3. Start TensorBoard (if not already running) so visualizations appear as soon as training begins.
#   4. Launch a torchrun job that trains a Synaptic GPT for a couple of hours with split/merge enabled.
#
# Usage:
#   scripts/run_bio_dual4090.sh            # uses defaults below
#   DATA_SHARDS=128 RUN_NAME=mybio scripts/run_bio_dual4090.sh
#

set -Eeuo pipefail

export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
export WANDB_MODE=offline

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_SHARDS="${DATA_SHARDS:-96}"          # ~12 GB download, tweak as desired
DATA_WORKERS="${DATA_WORKERS:-8}"
RUN_NAME="${RUN_NAME:-bio_dual4090}"
MODEL_TAG="${MODEL_TAG:-bio_d20_dual4090}"
TB_PORT="${TB_PORT:-6006}"

log() {
  printf '\n\033[1;36m[run_bio]\033[0m %s\n' "$*"
}

have_shards() {
  shopt -s nullglob
  local files=("$ROOT_DIR/base_data"/shard_*.parquet)
  (( ${#files[@]} >= DATA_SHARDS ))
}

log "Ensuring at least ${DATA_SHARDS} FineWeb shards are present..."
if ! have_shards; then
  log "Downloading shards via bio_inspired_nanochat.dataset (this may take a while)..."
  uv run python -m bio_inspired_nanochat.dataset --num-files "${DATA_SHARDS}" --num-workers "${DATA_WORKERS}"
else
  log "Found >= ${DATA_SHARDS} shards in base_data/ (skipping download)."
fi

log "Running the fast NeuroViz/NeuroScore smoke test (scripts/verify_evolution.py)..."
mkdir -p runs
PYTHONPATH=. uv run scripts/verify_evolution.py | tee runs/verify_evolution_console.log

if ! pgrep -f "tensorboard --logdir" >/dev/null 2>&1; then
  log "Starting TensorBoard on port ${TB_PORT} (background)..."
  nohup env PYTHONPATH=. uv run tensorboard --logdir runs --port "${TB_PORT}" \
        --load_fast=true --samples_per_plugin scalars=500,histograms=200 >/tmp/tensorboard.log 2>&1 &
  log "TensorBoard is streaming to http://localhost:${TB_PORT}  (logs -> /tmp/tensorboard.log)"
else
  log "TensorBoard already running; leaving it alone."
fi

# Start Streamlit Dashboard
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
mkdir -p runs/neuroviz  # Ensure directory exists so dashboard doesn't crash immediately
if ! pgrep -f "streamlit run scripts/dashboard.py" >/dev/null 2>&1; then
  log "Starting Bio-Nanochat Dashboard on port ${DASHBOARD_PORT} (background)..."
  nohup env PYTHONPATH=. uv run streamlit run scripts/dashboard.py --server.port "${DASHBOARD_PORT}" --server.headless true >/tmp/dashboard.log 2>&1 &
  log "Dashboard is streaming to http://localhost:${DASHBOARD_PORT} (logs -> /tmp/dashboard.log)"
  
  # Try to open browser
  if command -v xdg-open >/dev/null; then
      xdg-open "http://localhost:${DASHBOARD_PORT}" || true
  elif command -v open >/dev/null; then
      open "http://localhost:${DASHBOARD_PORT}" || true
  fi
else
  log "Dashboard already running; leaving it alone."
fi

log "Launching torchrun training job (${RUN_NAME})..."
set -x
uv run torchrun --nproc_per_node=2 --master_port=29500 \
  -m scripts.base_train \
    --synapses=1 \
    --run="${RUN_NAME}" \
    --model_tag="${MODEL_TAG}" \
    --depth=20 \
    --max_seq_len=2048 \
    --device_batch_size=12 \
    --splitmerge_every=2000 \
    --merges_per_call=1 \
    --splits_per_call=2 \
    --sm_verbose=1 \
    --eval_every=500 \
    --core_metric_every=2000 \
    --sample_every=2000 \
    --save_every=2000

