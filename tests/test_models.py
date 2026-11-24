"""Unit tests for model architectures and gradient backprop."""

import pytest
import torch
import torch.nn as nn

from src.models import BaselineMLP, RegularizedCNN, StandardCNN, count_parameters, get_model


@pytest.mark.parametrize("model_class", [BaselineMLP, StandardCNN, RegularizedCNN])
def test_model_forward_shape(model_class):
    model = model_class(num_classes=10)
    model.eval()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_name", ["mlp", "standard_cnn", "regularized_cnn"])
def test_model_backpropagation(model_name):
    model = get_model(model_name)
    model.train()
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    # Check all parameter gradients are finite and non-empty
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_target_layers_for_grad_cam():
    cnn = StandardCNN()
    assert hasattr(cnn, "get_target_layer")
    assert isinstance(cnn.get_target_layer(), nn.Conv2d)

    reg_cnn = RegularizedCNN()
    assert hasattr(reg_cnn, "get_target_layer")
    assert isinstance(reg_cnn.get_target_layer(), nn.Conv2d)
