#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR=${HF_HOME:-$(uv run python - <<'PY'
import os
print(os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface'))
PY
)}

MODELS=(
  "foundation_general:siglip_base_patch16_384"
  "foundation_general:siglip2_base_patch16_384"
  "foundation_general:siglip2_so400m_patch14_384"
  "foundation_general:siglip2_so400m_patch16_naflex"
  "foundation_general:siglip2_giant_opt_patch16_384"
  "foundation_general:dinov2_vit_l14"
  "foundation_medical:biomedclip_vit_b16"
  "foundation_medical:biovil_t"
  "foundation_medical:radimagenet_resnet50"
)

printf 'Hugging Face cache directory: %s\n\n' "$CACHE_DIR"

for spec in "${MODELS[@]}"; do
  if [[ "$spec" == "foundation_general:mambavision_t2_1k" ]]; then
    printf '↷ Skipping %s (mamba_ssm build pending)\n\n' "$spec"
    continue
  fi
  printf '>>> Prefetching weights for %s\n' "$spec"
  uv run python scripts/smoke_test_foundation_models.py \
    --models "$spec" \
    --pretrained \
    --batch-size 1 \
    --num-classes 5 \
    --device cpu || {
      printf '!! Prefetch failed for %s\n' "$spec"
      exit 1
    }
  printf '\n'
  sleep 1
done

printf '✓ All requested Hugging Face weights are cached under %s\n' "$CACHE_DIR"
