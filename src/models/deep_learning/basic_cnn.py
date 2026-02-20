"""Basic CNN models for knee osteoarthritis classification.

This module provides simple CNN architectures suitable for medical image
classification. Models are designed to be lightweight yet effective for
the 5-class KL grading task.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Simple CNN for knee OA classification.
    
    A lightweight CNN with 3 convolutional blocks followed by
    fully connected layers. Suitable for quick experimentation.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 3,
        dropout_rate: float = 0.5
    ):
        """Initialize SimpleCNN.
        
        Args:
            num_classes: Number of output classes (KL grades)
            input_channels: Number of input channels (3 for RGB)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class MediumCNN(nn.Module):
    """Medium-sized CNN for knee OA classification.
    
    A more sophisticated CNN with 4 convolutional blocks,
    residual connections, and deeper fully connected layers.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 3,
        dropout_rate: float = 0.5
    ):
        """Initialize MediumCNN.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # First block
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Convolutional blocks with residual-like connections
        self.block1 = self._make_block(64, 64, 2)
        self.block2 = self._make_block(64, 128, 2, stride=2)
        self.block3 = self._make_block(128, 256, 2, stride=2)
        self.block4 = self._make_block(256, 512, 2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_block(self, in_channels: int, out_channels: int, num_layers: int, stride: int = 1):
        """Create a convolutional block."""
        layers = []
        
        # First layer (may have stride > 1)
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial conv and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class CNNRegistry:
    """Registry for CNN models."""
    
    _models = {
        "simple_cnn": SimpleCNN,
        "medium_cnn": MediumCNN,
    }
    
    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_classes: int = 5,
        **kwargs
    ) -> nn.Module:
        """Get a model by name.
        
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
        
        model_class = cls._models[model_name]
        return model_class(num_classes=num_classes, **kwargs)
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List available model names."""
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a new model class."""
        cls._models[name] = model_class


def get_model_info(model: nn.Module) -> Dict[str, any]:
    """Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }


def test_model(model: nn.Module, input_shape: tuple = (1, 3, 224, 224)) -> None:
    """Test a model with dummy input.
    
    Args:
        model: PyTorch model to test
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    model.eval()
    
    # Create dummy input
    x = torch.randn(input_shape)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Print results
    info = get_model_info(model)
    print(f"Model: {info['model_class']}")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {info['trainable_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")


if __name__ == "__main__":
    # Test models
    print("Testing CNN models...")
    print("=" * 50)
    
    # Test SimpleCNN
    print("\n1. SimpleCNN:")
    simple_model = CNNRegistry.get_model("simple_cnn")
    test_model(simple_model)
    
    # Test MediumCNN
    print("\n2. MediumCNN:")
    medium_model = CNNRegistry.get_model("medium_cnn")
    test_model(medium_model)
    
    # List available models
    print(f"\nAvailable models: {CNNRegistry.list_models()}")







