from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_risky_split_scripts_removed() -> None:
    removed_scripts = [
        "scripts/clean_set_b_keep_ratio.py",
        "scripts/clean_set_b_target_totals.py",
        "scripts/analyze_set_b_dedup.py",
        "scripts/analyze_set_b_experts.py",
        "scripts/cross_domain_evaluation.py",
    ]
    for rel_path in removed_scripts:
        assert not (REPO_ROOT / rel_path).exists(), rel_path


def test_cli_entrypoints_exist() -> None:
    entrypoint_modules = [
        "src/training/cli.py",
        "src/evaluation/cli.py",
        "src/experiments/cli.py",
    ]
    for rel_path in entrypoint_modules:
        assert (REPO_ROOT / rel_path).exists(), rel_path
