# Scripts Directory

This folder is intentionally minimal and keeps only core scripts for:
- data processing
- model training
- model inference/evaluation

## Data Curation
- `dataset_statistics.py`

## Training
- `run_baseline_experiment.py`
- `run_ablation.py`
- `run_softmax_baselines.sh`
- `run_stage5_foundation_training.sh`
- `smoke_test_foundation_models.py`
- `prefetch_foundation_weights.sh`

## Inference / Evaluation
- `save_predictions.py`
- `aggregate_single_source_baselines.py`

Core model training/evaluation entry points are also available via the
`knee-oa-train` CLI defined in `pyproject.toml`.
