"""
Knee Osteoarthritis Image Classification Project

A comprehensive machine learning project for classifying knee osteoarthritis severity
using the Kellgren-Lawrence (KL) grading system.

This package contains modules for:
- Data processing and feature extraction
- Traditional machine learning models
- Deep learning models
- Model training and evaluation
- Experiment management and comparison
"""

__version__ = "0.1.0"
__author__ = "Medical AI Research Team"
__description__ = "Knee Osteoarthritis Image Classification using ML and Deep Learning"

# Keep package import side-effect free during early stages.
# Modules should be imported explicitly by consumers, e.g.:
#   from src.data.data_loader import KneeOADataLoader
# to avoid ImportError from partially implemented subpackages.

models = None  # populated on demand by callers via explicit imports
evaluation = None

# 延迟导入深度学习相关模块
def get_training_module():
    """延迟导入训练模块（需要torch）"""
    try:
        from . import training
        return training
    except ImportError:
        return None

# Stage 功能检测
def get_available_stages():
    """获取当前可用的功能 Stage"""
    stages = {
        'traditional_ml': True,  # 当前 Stage
        'deep_learning': False,
        'advanced_dl': False
    }
    
    # 检查深度学习
    try:
        import torch
        stages['deep_learning'] = True
    except ImportError:
        pass
    
    # 检查高级深度学习
    try:
        import timm
        stages['advanced_dl'] = True
    except ImportError:
        pass
    
    return stages
