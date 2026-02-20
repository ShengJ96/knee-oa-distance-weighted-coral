"""
Deep learning models module

Contains implementations of:
- Basic CNN architectures
- Transfer learning models (ResNet, EfficientNet, etc.)
- Advanced architectures (Vision Transformer, ConvNeXt, etc.)
"""

# Import available modules only. Optional dependencies are handled inside each submodule.
try:
    from .basic_cnn import SimpleCNN, MediumCNN, CNNRegistry
except ImportError:  # pragma: no cover - optional import guard
    SimpleCNN = MediumCNN = None  # type: ignore
    CNNRegistry = None  # type: ignore

try:
    from .transfer_learning import (
        TransferLearningModel,
        MedicalImageTransferModel,
        TransferLearningRegistry,
    )
except ImportError:  # pragma: no cover - optional import guard
    TransferLearningModel = MedicalImageTransferModel = None  # type: ignore
    TransferLearningRegistry = None  # type: ignore

try:
    from .advanced_cnn import AdvancedCNNModel, AdvancedCNNRegistry, AdvancedCNNSpec
except ImportError:  # pragma: no cover - optional import guard
    AdvancedCNNModel = AdvancedCNNRegistry = AdvancedCNNSpec = None  # type: ignore

try:
    from .vision_transformer import (
        VisionTransformerModel,
        VisionTransformerRegistry,
        VisionTransformerSpec,
    )
except ImportError:  # pragma: no cover - optional dependency (transformers)
    VisionTransformerModel = VisionTransformerRegistry = VisionTransformerSpec = None  # type: ignore


try:
    from .foundation_general import (
        FoundationGeneralModel,
        FoundationGeneralRegistry,
        FoundationGeneralSpec,
    )
except ImportError:  # pragma: no cover - optional dependency (transformers)
    FoundationGeneralModel = FoundationGeneralRegistry = FoundationGeneralSpec = None  # type: ignore

try:
    from .foundation_medical import (
        FoundationMedicalModel,
        FoundationMedicalRegistry,
        FoundationMedicalSpec,
    )
except ImportError:  # pragma: no cover - optional dependency (transformers/timm)
    FoundationMedicalModel = FoundationMedicalRegistry = FoundationMedicalSpec = None  # type: ignore

__all__ = [
    name
    for name in (
        "SimpleCNN",
        "MediumCNN",
        "CNNRegistry",
        "TransferLearningModel",
        "MedicalImageTransferModel",
        "TransferLearningRegistry",
        "AdvancedCNNModel",
        "AdvancedCNNRegistry",
        "AdvancedCNNSpec",
        "VisionTransformerModel",
        "VisionTransformerRegistry",
        "VisionTransformerSpec",
        "FoundationGeneralModel",
        "FoundationGeneralRegistry",
        "FoundationGeneralSpec",
        "FoundationMedicalModel",
        "FoundationMedicalRegistry",
        "FoundationMedicalSpec",
    )
    if globals().get(name) is not None
]
