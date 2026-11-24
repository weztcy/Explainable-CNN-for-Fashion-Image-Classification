"""Explainability module using Grad-CAM (Gradient-weighted Class Activation Mapping)."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.data import CLASS_NAMES, find_dataset_files, read_idx_images, read_idx_labels
from src.evaluate import load_model_from_checkpoint
from src.visualize import plot_explainability_heatmaps


class GradCAM:
    """Gradient-weighted Class Activation Mapping for PyTorch CNNs."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach().clone()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach().clone()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate_heatmap(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM heatmap for a single input tensor (1, 1, 28, 28).
        
        Returns:
            heatmap: (28, 28) numpy float array in [0, 1].
            predicted_class: integer index.
            confidence: softmax probability for the target/predicted class.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        pred_class = int(torch.argmax(logits, dim=1).item())
        if target_class is None:
            target_class = pred_class

        confidence = float(probs[0, target_class].item())

        # Backward pass on target class score
        score = logits[0, target_class]
        score.backward()

        # Compute weights via Global Average Pooling of gradients
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Discard negative contributions

        # Resize CAM to input dimensions (28, 28)
        cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence


def create_heatmap_overlay(
    original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap: str = "jet"
) -> np.ndarray:
    """Blend 28x28 grayscale image with Grad-CAM heatmap."""
    # Normalize original image to [0, 1] float
    if original_image.dtype == np.uint8:
        img_norm = original_image.astype(np.float32) / 255.0
    else:
        img_norm = (original_image - original_image.min()) / (
            original_image.max() - original_image.min() + 1e-8
        )

    # Convert grayscale to 3-channel RGB
    img_rgb = np.stack([img_norm] * 3, axis=-1)

    # Apply colormap to heatmap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Discard alpha channel (28, 28, 3)

    # Alpha blending
    overlay = (1.0 - alpha) * img_rgb + alpha * heatmap_colored
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def run_explainability_study(
    checkpoint_path: str = "outputs/models/best_cnn.pt",
    data_dir: str = "dataset",
    output_dir: str = "outputs",
    num_correct_samples: int = 4,
    num_confused_samples: int = 4,
) -> None:
    """Generate Grad-CAM heatmaps for correct and confused test samples, saving Figure 10."""
    device = torch.device("cpu")
    model, ckpt = load_model_from_checkpoint(checkpoint_path, device)
    train_mean = ckpt["train_mean"]
    train_std = ckpt["train_std"]

    # Target conv layer
    if hasattr(model, "get_target_layer"):
        target_layer = model.get_target_layer()
    else:
        # Fallback to last conv2d
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        target_layer = conv_layers[-1]

    grad_cam = GradCAM(model, target_layer)

    # Load raw test images and labels
    file_map = find_dataset_files(data_dir)
    test_images = read_idx_images(file_map["test_images"])
    test_labels = read_idx_labels(file_map["test_labels"])

    # Evaluate predictions across test samples
    samples_data = []

    # First pass: find candidates
    correct_candidates = []
    confused_candidates = []

    for i in range(len(test_images)):
        img_raw = test_images[i]
        label = int(test_labels[i])

        # Preprocess exactly as in training
        img_norm = (img_raw.astype(np.float32) / 255.0 - train_mean) / train_std
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            pred = int(torch.argmax(logits, dim=1).item())

        if pred == label:
            correct_candidates.append(i)
        else:
            confused_candidates.append(i)

        if len(correct_candidates) > 100 and len(confused_candidates) > 100:
            break

    # Pick diverse correct samples (different classes)
    chosen_correct_indices = []
    seen_classes = set()
    for idx in correct_candidates:
        c = int(test_labels[idx])
        if c not in seen_classes and len(chosen_correct_indices) < num_correct_samples:
            chosen_correct_indices.append(idx)
            seen_classes.add(c)

    # Pick diverse confusion samples (interesting pairs, e.g. Shirt vs T-shirt, Coat vs Pullover)
    chosen_confused_indices = []
    for idx in confused_candidates:
        if len(chosen_confused_indices) < num_confused_samples:
            chosen_confused_indices.append(idx)

    selected_indices = chosen_correct_indices + chosen_confused_indices

    for idx in selected_indices:
        img_raw = test_images[idx]
        true_label = int(test_labels[idx])

        # Prepare tensor
        img_norm = (img_raw.astype(np.float32) / 255.0 - train_mean) / train_std
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
        tensor.requires_grad = True

        heatmap, pred_label, conf = grad_cam.generate_heatmap(tensor, target_class=None)
        overlay = create_heatmap_overlay(img_raw, heatmap, alpha=0.55)

        samples_data.append(
            {
                "index": int(idx),
                "original_image": img_raw,
                "heatmap": heatmap,
                "overlay": overlay,
                "true_label": true_label,
                "pred_label": pred_label,
                "target_label": pred_label,
                "confidence": conf,
            }
        )

    grad_cam.remove_hooks()

    # Save Figure 10
    out_fig = Path(output_dir) / "figures" / "10_explainability_heatmaps.png"
    plot_explainability_heatmaps(samples_data, out_fig)
    print(f"Figure 10 successfully generated at {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explainability heatmaps.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/models/best_cnn.pt",
        help="Model checkpoint path",
    )
    args = parser.parse_args()

    run_explainability_study(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
