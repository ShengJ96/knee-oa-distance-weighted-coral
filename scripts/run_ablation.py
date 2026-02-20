#!/usr/bin/env python
"""
消融实验运行脚本
Run ablation experiments for knee OA classification models
"""

import argparse
import json
import os
from pathlib import Path
import yaml
import copy
from typing import Any, Dict, List
import subprocess
import time

def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load base configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _ensure_model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = config.setdefault('model', {})
    return model_cfg.setdefault('params', {})


def _disable_ordinal_head(params: Dict[str, Any]) -> None:
    head_cfg = params.get('head')
    if isinstance(head_cfg, dict) and head_cfg.get('type') == 'ordinal':
        params.pop('head')


def _enable_ordinal_head(params: Dict[str, Any], mode: str) -> None:
    head_cfg = dict(params.get('head') or {})
    head_cfg['type'] = 'ordinal'
    head_cfg['mode'] = mode
    if 'dropout' not in head_cfg:
        default_dropout = params.get('classifier_dropout', 0.0)
        head_cfg['dropout'] = default_dropout
    params['head'] = head_cfg

def create_ablation_config(
    base_config: Dict[str, Any],
    ablation_type: str,
    ablation_value: str,
    output_dir: str
) -> Dict[str, Any]:
    """Create modified config for ablation experiment"""
    config = copy.deepcopy(base_config)

    # Update experiment name
    base_name = config['experiment']['name']
    config['experiment']['name'] = f"{base_name}_ablation_{ablation_type}_{ablation_value}"
    config['experiment']['output_dir'] = output_dir
    config['experiment']['figures_dir'] = f"{output_dir}/figures"
    config['experiment']['reports_dir'] = f"{output_dir}/reports"

    # Apply ablation modifications
    if ablation_type == 'loss':
        config = modify_loss_config(config, ablation_value)
    elif ablation_type == 'augmentation':
        config = modify_augmentation_config(config, ablation_value)
    elif ablation_type == 'preprocessing':
        config = modify_preprocessing_config(config, ablation_value)
    elif ablation_type == 'resolution':
        config = modify_resolution_config(config, ablation_value)
    elif ablation_type == 'pretrained':
        config = modify_pretrained_config(config, ablation_value)

    return config

def modify_loss_config(config: Dict[str, Any], loss_type: str) -> Dict[str, Any]:
    """Modify configuration for loss function ablation"""
    optimization = config.setdefault('optimization', {})
    params = _ensure_model_params(config)

    def reset_optional_keys():
        optimization.pop('class_weights', None)
        optimization.pop('focal_gamma', None)
        optimization.pop('focal_alpha', None)

    if loss_type == 'ce':
        reset_optional_keys()
        _disable_ordinal_head(params)
        optimization['criterion'] = 'cross_entropy'
    elif loss_type == 'weighted_ce':
        reset_optional_keys()
        _disable_ordinal_head(params)
        optimization['criterion'] = 'cross_entropy'
        optimization['class_weights'] = [1.0, 1.2, 1.5, 1.5, 2.0]
    elif loss_type == 'coral':
        reset_optional_keys()
        _enable_ordinal_head(params, mode='coral')
        optimization['criterion'] = 'coral'
    elif loss_type == 'corn':
        reset_optional_keys()
        _enable_ordinal_head(params, mode='corn')
        optimization['criterion'] = 'corn'
    elif loss_type == 'focal':
        reset_optional_keys()
        _disable_ordinal_head(params)
        optimization['criterion'] = 'focal'
        optimization['focal_gamma'] = 2.0
        optimization['focal_alpha'] = 0.25

    return config

def modify_augmentation_config(config: Dict[str, Any], aug_type: str) -> Dict[str, Any]:
    """Modify configuration for augmentation ablation"""
    aug_configs = {
        'none': {
            'library': 'torchvision',
            'variant': 'none',
            'augment_train': False
        },
        'basic': {
            'library': 'torchvision',
            'variant': 'light',
            'augment_train': True
        },
        'medical': {
            'library': 'monai',
            'variant': 'medium',
            'augment_train': True
        },
        'strong': {
            'library': 'torchvision',
            'variant': 'heavy',
            'augment_train': True,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0
        },
        'tta': {
            'library': 'torchvision',
            'variant': 'medium',
            'augment_train': True,
            'test_time_augmentation': True,
            'tta_transforms': 5
        }
    }

    if aug_type in aug_configs:
        config['data']['augmentation'] = aug_configs[aug_type]

    return config

def modify_preprocessing_config(config: Dict[str, Any], prep_type: str) -> Dict[str, Any]:
    """Modify configuration for preprocessing ablation"""
    prep_configs = {
        'none': {
            'medical_variant': 'none',
            'normalize': True
        },
        'normalize': {
            'medical_variant': 'none',
            'normalize': True
        },
        'clahe': {
            'medical_variant': 'clahe',
            'normalize': True
        },
        'histeq': {
            'medical_variant': 'histogram_equalization',
            'normalize': True
        },
        'unsharp': {
            'medical_variant': 'unsharp_mask',
            'normalize': True
        }
    }

    if prep_type in prep_configs:
        config['data'].update(prep_configs[prep_type])

    return config

def modify_resolution_config(config: Dict[str, Any], resolution: str) -> Dict[str, Any]:
    """Modify configuration for input resolution ablation"""
    resolutions = {
        '224': [224, 224],
        '288': [288, 288],
        '384': [384, 384],
        '512': [512, 512]
    }

    if resolution in resolutions:
        config['data']['target_size'] = resolutions[resolution]
        # Adjust batch size based on resolution (memory constraints)
        base_batch = config['data'].get('batch_size', 32)
        scale_factors = {'224': 1.0, '288': 0.75, '384': 0.5, '512': 0.35}
        config['data']['batch_size'] = int(base_batch * scale_factors[resolution])

    return config

def modify_pretrained_config(config: Dict[str, Any], pretrained_type: str) -> Dict[str, Any]:
    """Modify configuration for pretrained model ablation"""
    if pretrained_type == 'none':
        config['model']['params']['pretrained'] = False
    elif pretrained_type == 'imagenet':
        config['model']['params']['pretrained'] = True
        config['model']['params']['pretrained_source'] = 'imagenet'
    elif pretrained_type == 'medical':
        config['model']['params']['pretrained'] = True
        config['model']['params']['pretrained_source'] = 'radimagenet'
    elif pretrained_type == 'clip':
        config['model']['params']['pretrained'] = True
        config['model']['params']['pretrained_source'] = 'openai_clip'

    return config

def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def run_experiment(config_path: str, gpu_device: str = "cuda:0") -> Dict[str, Any]:
    """Run a single ablation experiment"""
    # Use the project CLI entrypoint (knee-oa-train) instead of the legacy
    # module path, and stream logs to both console and a file for debugging.
    # 默认输出 epoch 级进度（保留 tqdm），若需完全静默可设置环境变量 ABLA_QUIET=1。
    cmd = [
        "uv",
        "run",
        "knee-oa-train",
        "train",
        "--config",
        config_path,
        "--device",
        gpu_device,
    ]
    # 默认显示进度条；若想静默，可设置 ABLA_QUIET=1
    if os.environ.get("ABLA_QUIET") in {"1", "true", "True"}:
        cmd.append("--quiet")

    log_dir = os.path.join(os.path.dirname(config_path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    print(f"Running command: {' '.join(cmd)}")
    print(f"Streaming logs to: {log_path}")

    start_time = time.time()

    # Stream stdout/stderr to both console and file
    with open(log_path, "a") as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            log_f.write(line)
        process.wait()

    if process.returncode != 0:
        elapsed_time = time.time() - start_time
        err_msg = f"Command exited with {process.returncode}"
        print(f"Error running experiment: {err_msg}")
        return {
            "success": False,
            "error": err_msg,
            "config": config_path,
            "log_path": log_path,
            "elapsed_time": elapsed_time,
        }

    try:
        elapsed_time = time.time() - start_time

        results = {
            "success": True,
            "elapsed_time": elapsed_time,
            "config": config_path,
            "log_path": log_path,
        }

        # Try to load metrics from results file
        config_dir = os.path.dirname(config_path)
        results_file = os.path.join(config_dir, "../reports/final_metrics.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                metrics = json.load(f)
                results.update(metrics)

        return results

    except Exception as e:  # catch unexpected issues
        elapsed_time = time.time() - start_time
        print(f"Error running experiment: {e}")
        return {
            "success": False,
            "error": str(e),
            "config": config_path,
            "log_path": log_path,
            "elapsed_time": elapsed_time,
        }

def run_ablation_batch(
    base_config_path: str,
    ablation_type: str,
    ablation_values: List[str],
    output_base_dir: str,
    gpu_device: str = "cuda:0",
    dry_run: bool = False
) -> Dict[str, Any]:
    """Run a batch of ablation experiments"""
    base_config = load_base_config(base_config_path)
    results = {}

    for value in ablation_values:
        print(f"\n{'='*60}")
        print(f"Running {ablation_type} ablation with value: {value}")
        print('='*60)

        # Create output directory
        output_dir = os.path.join(output_base_dir, f"{ablation_type}_{value}")
        os.makedirs(output_dir, exist_ok=True)

        # Create modified config
        ablation_config = create_ablation_config(
            base_config, ablation_type, value, output_dir
        )

        # Save config
        config_path = os.path.join(output_dir, "config.yaml")
        save_config(ablation_config, config_path)

        if dry_run:
            print(f"[DRY RUN] Would run experiment with config: {config_path}")
            results[value] = {'dry_run': True, 'config': config_path}
        else:
            # Run experiment
            result = run_experiment(config_path, gpu_device)
            results[value] = result

            # Save intermediate results
            results_path = os.path.join(output_base_dir, f"{ablation_type}_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

    return results

def summarize_results(results_dir: str, output_path: str):
    """Summarize all ablation results into a markdown report"""
    all_results = {}

    # Load all result files
    for results_file in Path(results_dir).glob("*_results.json"):
        ablation_type = results_file.stem.replace('_results', '')
        with open(results_file, 'r') as f:
            all_results[ablation_type] = json.load(f)

    def _inject_metrics_if_missing(entry: Dict[str, Any]) -> Dict[str, Any]:
        """If metrics are missing, try to pull from sibling metadata.json"""
        if entry.get('success') and 'test_accuracy' not in entry:
            cfg_path = entry.get('config')
            if cfg_path:
                meta_path = Path(cfg_path).with_name("metadata.json")
                if meta_path.exists():
                    try:
                        meta = json.load(meta_path.open())
                        tm = ((meta.get("results") or {}).get("test_metrics")) or {}
                        entry['test_accuracy'] = tm.get('acc')
                        entry['test_f1_macro'] = tm.get('f1_macro')
                        entry['test_cohen_kappa'] = tm.get('kappa')
                    except Exception:
                        pass
        return entry

    # Create markdown report
    report = ["# Ablation Study Results\n\n"]
    report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for ablation_type, results in all_results.items():
        report.append(f"## {ablation_type.title()} Ablation\n\n")
        report.append("| Setting | Accuracy | F1-Macro | Cohen's κ | Time (min) |\n")
        report.append("|---------|----------|----------|-----------|------------|\n")

        for setting, metrics in results.items():
            metrics = _inject_metrics_if_missing(metrics)
            if metrics.get('success', False):
                acc = metrics.get('test_accuracy', 0) * 100
                f1 = metrics.get('test_f1_macro', 0)
                kappa = metrics.get('test_cohen_kappa', 0)
                time_min = metrics.get('elapsed_time', 0) / 60
                report.append(f"| {setting} | {acc:.1f}% | {f1:.3f} | {kappa:.3f} | {time_min:.1f} |\n")
            else:
                report.append(f"| {setting} | Failed | - | - | - |\n")

        report.append("\n")

    # Save report
    with open(output_path, 'w') as f:
        f.write(''.join(report))

    print(f"Summary report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument('--base-config', type=str, required=True,
                        help='Path to base configuration file')
    parser.add_argument('--ablation-type', type=str, required=True,
                        choices=['loss', 'augmentation', 'preprocessing', 'resolution', 'pretrained'],
                        help='Type of ablation to perform')
    parser.add_argument('--ablation-values', type=str, nargs='+',
                        help='Values to test for the ablation')
    parser.add_argument('--output-dir', type=str, default='experiments/ablations',
                        help='Output directory for ablation results')
    parser.add_argument('--gpu-device', type=str, default='cuda:0',
                        help='GPU device to use')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate configs without running experiments')
    parser.add_argument('--summarize-only', action='store_true',
                        help='Only generate summary from existing results')

    args = parser.parse_args()

    if args.summarize_only:
        summarize_results(args.output_dir,
                         os.path.join(args.output_dir, 'ablation_summary.md'))
    else:
        # Define default ablation values if not provided
        default_values = {
            'loss': ['ce', 'weighted_ce', 'coral', 'corn', 'focal'],
            'augmentation': ['none', 'basic', 'medical', 'strong', 'tta'],
            'preprocessing': ['none', 'normalize', 'clahe', 'histeq'],
            'resolution': ['224', '288', '384', '512'],
            'pretrained': ['none', 'imagenet', 'medical', 'clip']
        }

        ablation_values = args.ablation_values or default_values.get(args.ablation_type, [])

        results = run_ablation_batch(
            args.base_config,
            args.ablation_type,
            ablation_values,
            args.output_dir,
            args.gpu_device,
            args.dry_run
        )

        # Generate summary
        if not args.dry_run:
            summarize_results(args.output_dir,
                            os.path.join(args.output_dir, 'ablation_summary.md'))

if __name__ == '__main__':
    main()
