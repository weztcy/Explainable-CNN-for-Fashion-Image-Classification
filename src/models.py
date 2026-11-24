"""Model architectures for Fashion-MNIST classification.

Includes:
1. BaselineMLP: Multi-layer perceptron baseline.
2. StandardCNN: Classic 2-stage convolutional network.
3. RegularizedCNN: Deeper CNN with BatchNorm, Dropout, and residual-friendly conv layers.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) baseline for Fashion-MNIST.
    
    Accepts (B, 1, 28, 28) or (B, 784), outputs (B, 10).
    """

    def __init__(self, input_dim: int = 784, hidden_dims: Tuple[int, ...] = (256, 128), num_classes: int = 10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=False))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.network(x)


class StandardCNN(nn.Module):
    """Standard 2-stage CNN for Fashion-MNIST.
    
    Accepts (B, 1, 28, 28), outputs (B, 10).
    Has named conv layers for easy hook attachment in Grad-CAM.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv block 1: 1 -> 32 -> Pool -> 14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2: 32 -> 64 -> Pool -> 7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Classifier: 64 * 7 * 7 = 3136 -> 128 -> 10
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(128, num_classes)

    def get_target_layer(self) -> nn.Module:
        """Return target conv layer for Grad-CAM."""
        return self.conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


class RegularizedCNN(nn.Module):
    """Regularized CNN with Batch Normalization, Dropout, and dual-conv blocks.
    
    Accepts (B, 1, 28, 28), outputs (B, 10).
    Achieves high generalization and clear localization heatmaps.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super().__init__()
        # Block 1: 28x28 -> 14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(dropout_rate)

        # Block 2: 14x14 -> 7x7
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(dropout_rate)

        # Classifier: 64 * 7 * 7 = 3136 -> 256 -> 10
        self.fc1 = nn.Linear(64 * 7 * 7, 256, bias=False)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def get_target_layer(self) -> nn.Module:
        """Return target conv layer for Grad-CAM."""
        return self.conv4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Classifier
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        return self.fc2(x)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function for instantiating models."""
    name_lower = model_name.lower().replace("-", "_").replace(" ", "_")
    if name_lower in ["mlp", "baseline_mlp", "mlp_baseline"]:
        return BaselineMLP(**kwargs)
    elif name_lower in ["standard_cnn", "base_cnn", "cnn_standard"]:
        return StandardCNN(**kwargs)
    elif name_lower in ["regularized_cnn", "modern_cnn", "cnn_regularized", "augmented_cnn"]:
        return RegularizedCNN(**kwargs)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from: 'mlp', 'standard_cnn', 'regularized_cnn'."
        )


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
