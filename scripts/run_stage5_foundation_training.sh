#!/usr/bin/env bash
set -euo pipefail

# Optional environment variables:
#   TRAIN_DEVICE=cuda             # device string for trainer (default: cuda)
#   DDP_PROCS=2                   # number of GPUs for torchrun; set 1 for single GPU
#   ENABLE_MAMBAVISION=1          # include MambaVision configs once dependencies installed
#   TRAIN_LOG_DIR=runs/logs/stage5_foundation  # directory for log files (default: runs/logs/stage5_foundation)
#   TRAIN_LOG_FILE=/tmp/stage5.log            # absolute path for log file; overrides TRAIN_LOG_DIR
TRAIN_DEVICE=${TRAIN_DEVICE:-cuda}
DDP_PROCS=${DDP_PROCS:-2}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DEFAULT_LOG_DIR=${TRAIN_LOG_DIR:-runs/logs/stage5_foundation}
if [[ -n "${TRAIN_LOG_FILE:-}" ]]; then
  LOG_FILE=$TRAIN_LOG_FILE
  mkdir -p "$(dirname "$LOG_FILE")"
else
  mkdir -p "$DEFAULT_LOG_DIR"
  LOG_FILE="$DEFAULT_LOG_DIR/stage5_foundation_training_${TIMESTAMP}.log"
fi

exec > >(tee -a "$LOG_FILE")
exec 2>&1

printf 'Stage 5 foundation training logs streaming to %s\n' "$LOG_FILE"

CACHE_DIR=${HF_HOME:-$(uv run python - <<'PY'
import os
print(os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface'))
PY
)}

SINGLE_SUMMARY=experiments/results/stage5_single_source_summary.json
MULTI_SUMMARY=experiments/results/stage5_multi_source_summary.json

SINGLE_CONFIGS=(
  "experiments/configs/foundation/general/siglip_set_a.yaml"
  "experiments/configs/foundation/general/siglip_set_b.yaml"
  "experiments/configs/foundation/general/siglip2_base_set_a.yaml"
  "experiments/configs/foundation/general/siglip2_base_set_b.yaml"
  "experiments/configs/foundation/general/siglip2_so400m_set_a.yaml"
  "experiments/configs/foundation/general/siglip2_so400m_set_b.yaml"
  "experiments/configs/foundation/general/siglip2_giant_opt_set_a.yaml"
  "experiments/configs/foundation/general/siglip2_giant_opt_set_b.yaml"
  "experiments/configs/foundation/general/dinov2_set_a.yaml"
  "experiments/configs/foundation/general/dinov2_set_b.yaml"
  "experiments/configs/foundation/general/mambavision_set_a.yaml"
  "experiments/configs/foundation/general/mambavision_set_b.yaml"
  "experiments/configs/foundation/medical/biomedclip_set_a.yaml"
  "experiments/configs/foundation/medical/biomedclip_set_b.yaml"
  "experiments/configs/foundation/medical/biovil_t_set_a.yaml"
  "experiments/configs/foundation/medical/biovil_t_set_b.yaml"
  "experiments/configs/foundation/medical/radimagenet_resnet50_set_a.yaml"
  "experiments/configs/foundation/medical/radimagenet_resnet50_set_b.yaml"
)

SINGLE_NAFLEX_CONFIGS=(
  "experiments/configs/foundation/general/siglip2_so400m_naflex_set_a.yaml"
  "experiments/configs/foundation/general/siglip2_so400m_naflex_set_b.yaml"
)

should_skip() {
  local cfg=$1

  if [[ "${FORCE_RETRAIN:-0}" != "0" ]]; then
    return 1
  fi

  local output_dir
  output_dir=$(uv run python - "$cfg" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open('r', encoding='utf-8') as fh:
    data = yaml.safe_load(fh)

exp = data.get('experiment', {}) if isinstance(data, dict) else {}
out = exp.get('output_dir')
if out:
    print(out, end='')
PY
) || output_dir=""

  if [[ -z "$output_dir" ]]; then
    return 1
  fi

  if [[ -f "$output_dir/metadata.json" ]]; then
    printf '↷ Skipping %s (metadata present in %s). Set FORCE_RETRAIN=1 to rerun.\n' "$cfg" "$output_dir"
    return 0
  fi

  local has_stale_weights=0
  if compgen -G "$output_dir/best_model_epoch_*.pth" > /dev/null; then
    has_stale_weights=1
  fi
  if [[ -f "$output_dir/last_model.pth" ]]; then
    has_stale_weights=1
  fi

  if [[ "$has_stale_weights" -eq 1 ]]; then
    printf '↻ Found checkpoints for %s in %s without metadata.json; removing stale weights before retraining.\n' "$cfg" "$output_dir"
    rm -f "$output_dir"/best_model_epoch_*.pth || true
    rm -f "$output_dir/last_model.pth" || true
  fi

  return 1
}

MULTI_CONFIGS=(
  "experiments/configs/foundation/general/siglip_set_ab.yaml"
  "experiments/configs/foundation/general/siglip2_base_set_ab.yaml"
  "experiments/configs/foundation/general/siglip2_so400m_set_ab.yaml"
  "experiments/configs/foundation/general/siglip2_giant_opt_set_ab.yaml"
  "experiments/configs/foundation/general/dinov2_set_ab.yaml"
  "experiments/configs/foundation/medical/biomedclip_set_ab.yaml"
  "experiments/configs/foundation/medical/biovil_t_set_ab.yaml"
  "experiments/configs/foundation/medical/radimagenet_resnet50_set_ab.yaml"
)

MULTI_NAFLEX_CONFIGS=(
  "experiments/configs/foundation/general/siglip2_so400m_naflex_set_ab.yaml"
)

run_training() {
  local cfg=$1
  local summary=$2
  printf '\n>>> Training %s with %s (DDP processes: %s)\n' "$cfg" "$TRAIN_DEVICE" "$DDP_PROCS"
  if [[ "$DDP_PROCS" -gt 1 ]]; then
    uv run torchrun --standalone --nproc-per-node="$DDP_PROCS" \
      --module src.training.cli train \
      --config "$cfg" \
      --device "$TRAIN_DEVICE" \
      --summary-output "$summary"
  else
    uv run knee-oa-train train \
      --config "$cfg" \
      --device "$TRAIN_DEVICE" \
      --summary-output "$summary"
  fi
}

aggregate_results() {
  uv run python scripts/aggregate_single_source_baselines.py \
    --roots experiments/models/foundation \
    --output-dir experiments/reports \
    --overwrite
}

printf 'Hugging Face cache directory: %s\n' "$CACHE_DIR"

process_configs() {
  local -n configs=$1
  local summary=$2

  for cfg in "${configs[@]}"; do
    if [[ "$cfg" == *"mambavision"* && "${ENABLE_MAMBAVISION:-0}" != "1" ]]; then
      printf '↷ Skipping %s (set ENABLE_MAMBAVISION=1 to include once dependencies are installed)\n' "$cfg"
      continue
    fi
    if should_skip "$cfg"; then
      continue
    fi

    run_training "$cfg" "$summary"
    aggregate_results
    sleep 2
    sync || true
    free -h || true
    nvidia-smi || true
    printf '\n'
  done
}

process_configs SINGLE_CONFIGS "$SINGLE_SUMMARY"
process_configs SINGLE_NAFLEX_CONFIGS "$SINGLE_SUMMARY"

process_configs MULTI_CONFIGS "$MULTI_SUMMARY"
process_configs MULTI_NAFLEX_CONFIGS "$MULTI_SUMMARY"

printf '✓ All scheduled Stage 5 foundation trainings have finished.\n'
