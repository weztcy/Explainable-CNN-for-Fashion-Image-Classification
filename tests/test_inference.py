"""Unit tests for checkpoint loading, inference, and explainability."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.explain import GradCAM, create_heatmap_overlay
from src.models import RegularizedCNN, StandardCNN


def test_grad_cam_execution():
    model = StandardCNN(num_classes=10)
    model.eval()
    target_layer = model.get_target_layer()

    grad_cam = GradCAM(model, target_layer)
    x = torch.randn(1, 1, 28, 28, requires_grad=True)

    heatmap, pred_class, conf = grad_cam.generate_heatmap(x, target_class=3)
    grad_cam.remove_hooks()

    assert heatmap.shape == (28, 28)
    assert 0 <= pred_class < 10
    assert 0.0 <= conf <= 1.0
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

    # Test overlay creation
    raw_img = (np.random.rand(28, 28) * 255).astype(np.uint8)
    overlay = create_heatmap_overlay(raw_img, heatmap)
    assert overlay.shape == (28, 28, 3)
    assert np.isfinite(overlay).all()
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0


def test_checkpoint_roundtrip(tmp_path):
    model = RegularizedCNN(num_classes=10)
    ckpt_path = tmp_path / "test_model.pt"

    torch.save(
        {
            "model_name": "regularized_cnn",
            "state_dict": model.state_dict(),
            "epoch": 5,
            "val_loss": 0.25,
            "val_acc": 91.5,
            "train_mean": 0.286,
            "train_std": 0.353,
        },
        ckpt_path,
    )

    # Load back
    loaded = torch.load(ckpt_path, map_location="cpu")
    new_model = RegularizedCNN(num_classes=10)
    new_model.load_state_dict(loaded["state_dict"])
    new_model.eval()

    x = torch.randn(2, 1, 28, 28)
    out = new_model(x)
    assert out.shape == (2, 10)
