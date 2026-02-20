#!/bin/bash
# Run Softmax Baseline experiments (standard classification)
# Purpose: Provide comparison against ordinal regression methods
#
# Usage:
#   bash scripts/run_softmax_baselines.sh           # Run all experiments (1-6)
#   START_FROM=3 bash scripts/run_softmax_baselines.sh  # Resume from experiment 3

set -e

echo "=========================================="
echo "Stage 6 Softmax Baseline Experiments"
echo "=========================================="
echo ""
echo "Purpose: Evaluate standard classification (Softmax + CE)"
echo "Comparison: vs CORAL ordinal regression"
echo ""

DEVICE="${DEVICE:-cuda:0}"
START_FROM="${START_FROM:-1}"  # Default: start from experiment 1

echo "Using device: $DEVICE"
echo "Starting from experiment: $START_FROM"
echo ""

# Total: 6 experiments (2 models × 3 datasets)
# Estimated time: ~12 hours (2h per experiment)

# ==========================================
# SigLIP Experiments (3 datasets)
# ==========================================

if [ "$START_FROM" -le 1 ]; then
echo "📊 Experiment 1/6: SigLIP + Set A + Softmax"
echo "   Backbone: google/siglip-base-patch16-384"
echo "   Loss: Cross Entropy"
echo ""

uv run knee-oa-train train \
  --config experiments/configs/stage6/ablation/stage6_siglip_set_a_softmax.yaml \
  --device "$DEVICE"

echo ""
echo "✅ Experiment 1/6 completed"
echo ""
else
echo "⏭️  Skipping Experiment 1/6 (already completed)"
fi

# ---

if [ "$START_FROM" -le 2 ]; then
echo "📊 Experiment 2/6: SigLIP + Set B + Softmax"
echo ""

uv run knee-oa-train train \
  --config experiments/configs/stage6/ablation/stage6_siglip_set_b_softmax.yaml \
  --device "$DEVICE"

echo ""
echo "✅ Experiment 2/6 completed"
echo ""
else
echo "⏭️  Skipping Experiment 2/6 (already completed)"
fi

# ---

if [ "$START_FROM" -le 3 ]; then
echo "📊 Experiment 3/6: SigLIP + Set AB + Softmax"
echo ""

uv run knee-oa-train train \
  --config experiments/configs/stage6/ablation/stage6_siglip_set_ab_softmax.yaml \
  --device "$DEVICE"

echo ""
echo "✅ Experiment 3/6 completed"
echo ""
else
echo "⏭️  Skipping Experiment 3/6 (already completed)"
fi

# ==========================================
# SigLIP2 Experiments (3 datasets)
# ==========================================

if [ "$START_FROM" -le 4 ]; then
echo "📊 Experiment 4/6: SigLIP2 + Set A + Softmax"
echo "   Backbone: google/siglip2-so400m-patch14-384"
echo "   Loss: Cross Entropy"
echo ""

uv run knee-oa-train train \
  --config experiments/configs/stage6/ablation/stage6_siglip2_set_a_softmax.yaml \
  --device "$DEVICE"

echo ""
echo "✅ Experiment 4/6 completed"
echo ""
else
echo "⏭️  Skipping Experiment 4/6 (already completed)"
fi

# ---

if [ "$START_FROM" -le 5 ]; then
echo "📊 Experiment 5/6: SigLIP2 + Set B + Softmax"
echo ""

uv run knee-oa-train train \
  --config experiments/configs/stage6/ablation/stage6_siglip2_set_b_softmax.yaml \
  --device "$DEVICE"

echo ""
echo "✅ Experiment 5/6 completed"
echo ""
else
echo "⏭️  Skipping Experiment 5/6 (already completed)"
fi

# ---

if [ "$START_FROM" -le 6 ]; then
echo "📊 Experiment 6/6: SigLIP2 + Set AB + Softmax"
echo ""

uv run knee-oa-train train \
  --config experiments/configs/stage6/ablation/stage6_siglip2_set_ab_softmax.yaml \
  --device "$DEVICE"

echo ""
echo "✅ Experiment 6/6 completed"
echo ""
else
echo "⏭️  Skipping Experiment 6/6 (already completed)"
fi

# ==========================================
# Summary
# ==========================================

echo "=========================================="
echo "All Softmax baseline experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - experiments/models/stage6/siglip_softmax/"
echo "  - experiments/models/stage6/siglip2_softmax/"
echo ""
echo "Metadata files:"
echo "  - experiments/models/stage6/siglip_softmax/set_a_softmax/metadata.json"
echo "  - experiments/models/stage6/siglip_softmax/set_b_softmax/metadata.json"
echo "  - experiments/models/stage6/siglip_softmax/set_ab_softmax/metadata.json"
echo "  - experiments/models/stage6/siglip2_softmax/set_a_softmax/metadata.json"
echo "  - experiments/models/stage6/siglip2_softmax/set_b_softmax/metadata.json"
echo "  - experiments/models/stage6/siglip2_softmax/set_ab_softmax/metadata.json"
echo ""
echo "Next steps:"
echo "  1. Update ablation summary with Softmax baseline results"
echo "  2. Generate updated comparison tables"
echo "  3. Calculate improvement: CORAL vs Softmax"
echo ""
