"""
Training utilities and scripts

Contains:
- Training loops for different model types
- Hyperparameter tuning utilities
- Curriculum learning strategies
"""

# Import available modules only
try:
    from .dl_trainer import DeepLearningTrainer
except ImportError:
    pass

try:
    from .distillation import (
        DistillationConfig,
        build_teacher_from_config,
        build_curriculum_hook,
    )
except ImportError:
    pass

# TODO: Import other modules when implemented
# from .baseline_trainer import BaselineTrainer
# from .hyperparameter_tuning import HyperparameterTuner
