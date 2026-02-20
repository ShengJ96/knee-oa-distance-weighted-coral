"""
Baseline traditional machine learning models

Contains implementations of:
- Classical ML algorithms (Logistic Regression, SVM, etc.)
- Ensemble methods (Random Forest, XGBoost, etc.)
- Feature engineering utilities
"""

# 导入已实现的模块
from .classical import BaselineClassifiers

# 延迟导入未实现的模块
def get_ensemble_models():
    """延迟导入集成模型（待实现）"""
    try:
        from .ensemble import EnsembleModels
        return EnsembleModels
    except ImportError:
        raise ImportError("集成模型模块尚未实现。请创建 src/models/baseline/ensemble.py")

def get_feature_engineering():
    """延迟导入特征工程（待实现）"""
    try:
        from .features import FeatureEngineering
        return FeatureEngineering
    except ImportError:
        raise ImportError("特征工程模块尚未实现。请创建 src/models/baseline/features.py")
