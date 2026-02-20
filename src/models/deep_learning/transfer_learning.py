"""Transfer learning models for knee osteoarthritis classification.

This module provides pre-trained CNN models adapted for medical image
classification. Uses ImageNet pre-trained models and fine-tunes them
for the 5-class KL grading task.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
    VGG16_Weights, VGG19_Weights,
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights
)

# MONAI DenseNet backbones are no longer supported in this project.


class TransferLearningModel(nn.Module):
    """Base class for transfer learning models."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_features: bool = False,
        dropout_rate: float = 0.5
    ):
        """Initialize transfer learning model.
        
        Args:
            model_name: Name of the pre-trained model
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_features: Whether to freeze feature extraction layers
            dropout_rate: Dropout rate for final classifier
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_features = freeze_features
        
        # Create the base model
        self.backbone = self._create_backbone(model_name, pretrained)
        
        # Freeze feature extraction layers if specified
        if freeze_features:
            self._freeze_features()
        
        # Get the number of features from the backbone
        num_features = self._get_num_features()
        
        # Create custom classifier
        self.classifier = self._create_classifier(num_features, dropout_rate)
        
        # Replace the original classifier
        self._replace_classifier()
    
    def _create_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """Create the backbone model."""
        weights = "IMAGENET1K_V1" if pretrained else None
        
        if model_name == "resnet18":
            return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "resnet34":
            return models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "resnet50":
            return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "resnet101":
            return models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "vgg16":
            return models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "vgg19":
            return models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "mobilenet_v3_large":
            return models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "mobilenet_v3_small":
            return models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "efficientnet_b0":
            return models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "efficientnet_b1":
            return models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _freeze_features(self):
        """Freeze feature extraction layers."""
        if hasattr(self.backbone, 'features'):  # VGG
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        elif hasattr(self.backbone, 'conv1'):  # ResNet
            # Freeze all layers except the last residual block
            for name, param in self.backbone.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
        else:  # Other models
            # Freeze all parameters except classifier
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _get_num_features(self) -> int:
        """Get number of features from the backbone."""
        # Common torchvision models
        if hasattr(self.backbone, 'fc'):  # ResNet, EfficientNet
            return self.backbone.fc.in_features
        elif hasattr(self.backbone, 'classifier'):  # VGG, MobileNet
            if isinstance(self.backbone.classifier, nn.Sequential):
                # Find the last Linear layer
                for layer in reversed(self.backbone.classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            else:
                return self.backbone.classifier.in_features

        raise ValueError(f"Cannot determine feature size for {self.model_name}")
    
    def _create_classifier(self, num_features: int, dropout_rate: float) -> nn.Module:
        """Create custom classifier."""
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.num_classes)
        )
    
    def _replace_classifier(self):
        """Replace the original classifier with our custom one."""
        if hasattr(self.backbone, 'fc'):  # ResNet, EfficientNet
            self.backbone.fc = self.classifier
        elif hasattr(self.backbone, 'classifier'):  # VGG, MobileNet
            self.backbone.classifier = self.classifier
        else:
            raise ValueError(f"Cannot replace classifier for {self.model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def unfreeze_all(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers: int = 1):
        """Unfreeze top N layers for gradual fine-tuning."""
        # This is model-specific and would need implementation
        # For now, just unfreeze all
        self.unfreeze_all()


class MedicalImageTransferModel(TransferLearningModel):
    """Specialized transfer learning model for medical images."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_features: bool = False,
        dropout_rate: float = 0.5,
        use_medical_preprocessing: bool = True
    ):
        """Initialize medical image transfer model.
        
        Args:
            model_name: Name of the pre-trained model
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_features: Whether to freeze feature extraction layers
            dropout_rate: Dropout rate for final classifier
            use_medical_preprocessing: Apply medical image specific preprocessing
        """
        super().__init__(model_name, num_classes, pretrained, freeze_features, dropout_rate)
        
        self.use_medical_preprocessing = use_medical_preprocessing
        
        # Add medical-specific preprocessing if needed
        if use_medical_preprocessing:
            self._add_medical_preprocessing()
    
    def _add_medical_preprocessing(self):
        """Add medical image specific preprocessing layers."""
        # For now, we'll use standard preprocessing
        # Could add histogram equalization, contrast enhancement, etc.
        pass
    
    def _create_classifier(self, num_features: int, dropout_rate: float) -> nn.Module:
        """Create medical-specific classifier with more regularization."""
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.num_classes)
        )


class TransferLearningRegistry:
    """Registry for transfer learning models."""
    
    _models = {
        # ResNet family
        "resnet18": lambda **kwargs: TransferLearningModel("resnet18", **kwargs),
        "resnet34": lambda **kwargs: TransferLearningModel("resnet34", **kwargs),
        "resnet50": lambda **kwargs: TransferLearningModel("resnet50", **kwargs),
        "resnet101": lambda **kwargs: TransferLearningModel("resnet101", **kwargs),
        
        # VGG family
        "vgg16": lambda **kwargs: TransferLearningModel("vgg16", **kwargs),
        "vgg19": lambda **kwargs: TransferLearningModel("vgg19", **kwargs),
        
        # MobileNet family
        "mobilenet_v3_large": lambda **kwargs: TransferLearningModel("mobilenet_v3_large", **kwargs),
        "mobilenet_v3_small": lambda **kwargs: TransferLearningModel("mobilenet_v3_small", **kwargs),
        
        # EfficientNet family
        "efficientnet_b0": lambda **kwargs: TransferLearningModel("efficientnet_b0", **kwargs),
        "efficientnet_b1": lambda **kwargs: TransferLearningModel("efficientnet_b1", **kwargs),
        
        # Medical specialized models
        "medical_resnet50": lambda **kwargs: MedicalImageTransferModel("resnet50", **kwargs),
        "medical_efficientnet_b0": lambda **kwargs: MedicalImageTransferModel("efficientnet_b0", **kwargs),
        
        # (MONAI DenseNet removed)
    }
    
    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_classes: int = 5,
        **kwargs
    ) -> nn.Module:
        """Get a transfer learning model by name.
        
        Args:
            model_name: Name of the model
            num_classes: Number of output classes
            **kwargs: Additional model arguments
            
        Returns:
            Instantiated model
        """
        if model_name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        
        model_factory = cls._models[model_name]
        return model_factory(num_classes=num_classes, **kwargs)
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List available model names."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, any]:
        """Get information about a model."""
        model = cls.get_model(model_name, num_classes=5)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_name": model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "pretrained": True  # All our models use pretrained weights
        }


def compare_models(model_names: list[str], input_shape: tuple = (1, 3, 224, 224)) -> None:
    """Compare different transfer learning models.
    
    Args:
        model_names: List of model names to compare
        input_shape: Input tensor shape for testing
    """
    print("🔍 Transfer Learning Models Comparison")
    print("=" * 70)
    print(f"{'Model':<25} {'Total Params':<15} {'Trainable':<15} {'Size (MB)':<10}")
    print("-" * 70)
    
    for model_name in model_names:
        try:
            info = TransferLearningRegistry.get_model_info(model_name)
            print(f"{model_name:<25} {info['total_parameters']:<15,} "
                  f"{info['trainable_parameters']:<15,} {info['model_size_mb']:<10.1f}")
        except Exception as e:
            print(f"{model_name:<25} Error: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test transfer learning models
    print("Testing Transfer Learning Models...")
    print("=" * 50)
    
    # List available models
    available_models = TransferLearningRegistry.list_models()
    print(f"Available models: {len(available_models)}")
    
    # Compare key models
    key_models = ["resnet18", "resnet50", "efficientnet_b0", "medical_resnet50"]
    compare_models(key_models)
    
    # Test a model
    print(f"\n🧪 Testing ResNet50...")
    model = TransferLearningRegistry.get_model("resnet50", num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Transfer learning models working correctly!")

