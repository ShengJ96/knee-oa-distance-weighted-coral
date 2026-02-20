"""
Data processing and management module

Utilities for:
- Loading and preprocessing knee X-ray images
- Feature extraction for traditional ML models
- Data validation and quality control
- Dataset creation and management
"""

# NOTE:
# Keep imports lazy to avoid ImportError during early stages
# where some modules are not implemented yet.

def get_feature_extractor():
    """Lazy import for feature extractor (Stage 2)."""
    try:
        from .feature_extraction import MedicalImageFeatureExtractor  # type: ignore
        return MedicalImageFeatureExtractor
    except Exception as e:  # ImportError or not implemented
        raise ImportError(
            "特征提取模块尚未实现。请在Stage 2创建 src/data/feature_extraction.py"
        ) from e


def get_data_loader():
    """Lazy import for basic data loader (no torch required)."""
    try:
        from .data_loader import KneeOADataLoader  # type: ignore
        return KneeOADataLoader
    except Exception as e:
        raise ImportError(
            "数据加载器不可用。请确保文件 src/data/data_loader.py 存在。"
        ) from e


def get_data_validator():
    """Lazy import for optional data validator."""
    try:
        from .validation import DataValidator  # type: ignore
        return DataValidator
    except Exception:
        return None


def check_deep_learning_available() -> bool:
    """Check whether torch is available (Stage 3+)."""
    try:
        import torch  # type: ignore
        _ = torch.__version__
        return True
    except Exception:
        return False
