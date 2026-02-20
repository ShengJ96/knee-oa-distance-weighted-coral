"""
Machine learning models module

This module contains:
- Baseline traditional ML models
- Deep learning architectures
- Custom model implementations
"""

# 导入当前 Stage可用的模块
from .baseline import BaselineClassifiers

# 延迟导入功能
def get_ensemble_models():
    """获取集成模型类"""
    return __import__('src.models.baseline', fromlist=['get_ensemble_models']).get_ensemble_models()

def get_deep_learning_models():
    """延迟导入深度学习模型"""
    try:
        from .deep_learning import BasicCNN, TransferLearningModels
        return {'BasicCNN': BasicCNN, 'TransferLearningModels': TransferLearningModels}
    except ImportError:
        raise ImportError("深度学习模块尚未实现或需要安装PyTorch")

def get_custom_models():
    """延迟导入自定义模型"""
    try:
        from .custom import CustomModels
        return CustomModels
    except ImportError:
        raise ImportError("自定义模型模块尚未实现")
