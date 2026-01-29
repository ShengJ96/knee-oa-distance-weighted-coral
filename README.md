
# Distance-Weighted Ordinal Regression for Kellgren-Lawrence Grading

This repository contains code to reproduce the experiments in:
"Distance-Weighted Ordinal Regression for Kellgren-Lawrence Grading: A Dual Public Cohort Study."

## Overview
We propose a distance-weighted CORAL model for KL grading and evaluate:
- within-dataset performance on two public datasets
- cross-dataset transportability under domain shift
- threshold-based clinical metrics (KL>=2 / KL>=3)
- a pilot reader study (if applicable)

## Data
This work uses only publicly available, de-identified datasets:
- **Set A (OAI-derived ROI dataset)**: Mendeley Data, DOI:10.17632/56rmx5bjcr.1 (CC BY 4.0)
- **Set B (Multi-center Indian radiographs)**: Mendeley Data, DOI:10.17632/t9ndx37v5h.1 (CC BY 4.0)

**Note:** The data are not redistributed in this repository. Please download from the official sources.

## Environment
Python 3.10+ is required.

We recommend using `uv`:
```bash
uv venv
uv pip install -r requirements.txt

Minimal install:

uv pip install torch torchvision scikit-learn pillow timm matplotlib

## Project Structure

dataset/
  set_a/{train,val,test}/{0-4}/
  set_b/{train,val,test}/{0-4}/
scripts/
src/
experiments/

## Reproduce Main Experiments

### Baseline (classical ML)

uv run python scripts/run_baseline_experiment.py --data dataset/set_a --out experiments/results/baseline

### Deep Learning (example)

uv run knee-oa-train train --config experiments/configs/advanced_cnn/convnext_tiny.yaml --device cuda

### Evaluate a checkpoint

uv run knee-oa-train evaluate --config <config.yaml> --checkpoint <path>

### Recompute metrics from existing predictions (no training)

uv run python scripts/recompute_set_b_metrics_from_npz.py

## Results

Key metrics and summaries are saved under:

experiments/reports/
experiments/results/

## License

MIT License (see LICENSE).

## Citation

If you use this code, please cite:

[Add citation after acceptance]

## Contact

Corresponding author: Yunhai Li
Email: 10199253@sina.com
