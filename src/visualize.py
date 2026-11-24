"""Visualization utilities for Fashion-MNIST classification and explainability."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.data import CLASS_NAMES

# Set consistent matplotlib styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_samples_per_class(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Union[str, Path],
    num_samples_per_class: int = 5,
) -> None:
    """Figure 01: Gallery of example images from each of the 10 classes."""
    fig, axes = plt.subplots(
        10, num_samples_per_class, figsize=(num_samples_per_class * 1.6, 16.0), dpi=300
    )
    fig.patch.set_facecolor("#ffffff")

    for c in range(10):
        class_indices = np.where(labels == c)[0]
        selected_idx = class_indices[:num_samples_per_class]

        for s_idx, idx in enumerate(selected_idx):
            ax = axes[c, s_idx]
            ax.imshow(images[idx], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#dddddd")

            if s_idx == 0:
                ax.set_ylabel(
                    f"{c}: {CLASS_NAMES[c]}",
                    fontsize=10,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                )

    plt.suptitle(
        "Fashion-MNIST Sample Gallery (10 Classes × 5 Samples)\nSource: dataset/train-images-idx3-ubyte.gz",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0.12, 0.01, 0.98, 0.98])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_class_distribution(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """Figure 02: Distribution of classes across train (54k), validation (6k), and test (10k)."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    train_counts = np.bincount(train_labels, minlength=10)
    val_counts = np.bincount(val_labels, minlength=10)
    test_counts = np.bincount(test_labels, minlength=10)

    x = np.arange(10)
    width = 0.26

    rects1 = ax.bar(
        x - width, train_counts, width, label=f"Train ({len(train_labels):,} total)", color="#2b5c8f"
    )
    rects2 = ax.bar(
        x, val_counts, width, label=f"Validation ({len(val_labels):,} total)", color="#d95f02"
    )
    rects3 = ax.bar(
        x + width, test_counts, width, label=f"Test ({len(test_labels):,} total)", color="#7570b3"
    )

    ax.set_xlabel("Fashion Category (Class Index & Name)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax.set_title(
        "Class Distribution Across Train (54,000), Validation (6,000), and Test (10,000)\nBalanced 10-Class Distribution Under Stratified Split (Seed=42)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}: {CLASS_NAMES[i]}" for i in range(10)], rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="#cccccc")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10)

    # Annotate bar values
    def autolabel(rects, dy=50):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_ylim(0, 6400)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_pixel_statistics(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """Figure 03: Pixel intensity histogram and class-average mean images."""
    fig = plt.figure(figsize=(15, 8), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # 1. Pixel intensity histogram on top-left
    ax_hist = plt.subplot2grid((3, 5), (0, 0), colspan=5)
    ax_hist.set_facecolor("#fafafa")
    flattened = images.flatten()
    n, bins, patches = ax_hist.hist(
        flattened, bins=50, range=(0, 255), color="#377eb8", edgecolor="#2b5c8f", alpha=0.8
    )
    ax_hist.set_title(
        "Pixel Intensity Distribution Across Train Split (54,000 Images × 784 Pixels)\nBackground Peak at 0 (Dark) and Foreground Product Distribution",
        fontsize=11,
        fontweight="bold",
    )
    ax_hist.set_xlabel("Pixel Value (0 = Black background, 255 = White foreground)", fontsize=10)
    ax_hist.set_ylabel("Pixel Count (Log scale)", fontsize=10)
    ax_hist.set_yscale("log")
    ax_hist.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

    # 2. Mean images per class (2 rows of 5)
    for c in range(10):
        class_images = images[labels == c].astype(np.float32)
        mean_img = class_images.mean(axis=0)

        row = 1 + (c // 5)
        col = c % 5
        ax = plt.subplot2grid((3, 5), (row, col))
        im = ax.imshow(mean_img, cmap="inferno", vmin=0, vmax=255)
        ax.set_title(f"{c}: {CLASS_NAMES[c]}", fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        "Pixel Statistics and Class-Wise Mean Heatmaps (Fashion-MNIST Train Split)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history_dict: Dict[str, Dict[str, List[float]]],
    output_path: Union[str, Path],
) -> None:
    """Figure 04: Loss and accuracy curves across epochs for compared models."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax_loss.set_facecolor("#fafafa")
    ax_acc.set_facecolor("#fafafa")

    colors = {
        "MLP Baseline": ("#e41a1c", "#fb8072"),
        "Standard CNN": ("#377eb8", "#80b1d3"),
        "Regularized CNN": ("#4daf4a", "#b3de69"),
    }

    for model_name, hist in history_dict.items():
        c_train, c_val = colors.get(model_name, ("#333333", "#888888"))
        epochs = range(1, len(hist["train_loss"]) + 1)

        ax_loss.plot(
            epochs, hist["train_loss"], label=f"{model_name} (Train)", color=c_train, linestyle="--"
        )
        ax_loss.plot(
            epochs, hist["val_loss"], label=f"{model_name} (Val)", color=c_train, linewidth=2.0
        )

        ax_acc.plot(
            epochs, hist["train_acc"], label=f"{model_name} (Train)", color=c_train, linestyle="--"
        )
        ax_acc.plot(
            epochs, hist["val_acc"], label=f"{model_name} (Val)", color=c_train, linewidth=2.0
        )

    ax_loss.set_title("Cross-Entropy Loss vs. Epoch", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.grid(True, linestyle="--", alpha=0.6, color="#cccccc")
    ax_loss.legend(fontsize=8.5, frameon=True, facecolor="#ffffff", edgecolor="#cccccc")

    ax_acc.set_title("Accuracy vs. Epoch (%)", fontsize=11, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=10)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.grid(True, linestyle="--", alpha=0.6, color="#cccccc")
    ax_acc.legend(fontsize=8.5, frameon=True, facecolor="#ffffff", edgecolor="#cccccc")

    plt.suptitle(
        "Training & Validation Curves Comparison Across Models\nTracked on 54,000 Train and 6,000 Stratified Validation Samples",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    comparison_metrics: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
) -> None:
    """Figure 05: Comparison of key metrics across models (Accuracy, Macro F1, Parameters)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#fafafa")
    ax2.set_facecolor("#fafafa")

    models = list(comparison_metrics.keys())
    val_accs = [comparison_metrics[m]["val_acc"] * 100 for m in models]
    test_accs = [comparison_metrics[m]["test_acc"] * 100 for m in models]
    macro_f1s = [comparison_metrics[m]["test_macro_f1"] * 100 for m in models]
    param_counts = [comparison_metrics[m]["param_count"] / 1000 for m in models]

    x = np.arange(len(models))
    width = 0.25

    ax1.bar(x - width, val_accs, width, label="Val Accuracy (%)", color="#377eb8")
    ax1.bar(x, test_accs, width, label="Test Accuracy (%)", color="#4daf4a")
    ax1.bar(x + width, macro_f1s, width, label="Test Macro F1 (%)", color="#984ea3")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Performance Metrics (Val & Test)", fontsize=11, fontweight="bold")
    ax1.set_ylim(75, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.6, color="#cccccc")
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", loc="lower right")

    for i in x:
        ax1.text(i - width, val_accs[i] + 0.5, f"{val_accs[i]:.2f}%", ha="center", fontsize=8)
        ax1.text(i, test_accs[i] + 0.5, f"{test_accs[i]:.2f}%", ha="center", fontsize=8)
        ax1.text(i + width, macro_f1s[i] + 0.5, f"{macro_f1s[i]:.2f}%", ha="center", fontsize=8)

    # Parameter count comparison
    bars = ax2.bar(x, param_counts, width=0.4, color="#ff7f00")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylabel("Model Parameters (in Thousands, K)", fontsize=10, fontweight="bold")
    ax2.set_title("Model Complexity (Parameter Count)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.6, color="#cccccc")

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}K", ha="center", fontsize=9)

    plt.suptitle(
        "Model Performance and Complexity Comparison\nEvaluated on Stratified Validation (6,000) and Test (10,000) Sets",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_counts(
    cm: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """Figure 06: Test set confusion matrix in raw sample counts."""
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        ax=ax,
        annot_kws={"size": 8.5},
    )

    ax.set_title(
        "Test Set Confusion Matrix (Raw Counts)\nTotal Test Samples = 10,000 (1,000 per class)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=10, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_normalized(
    cm: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """Figure 07: Normalized confusion matrix (recall per row / true class)."""
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        ax=ax,
        annot_kws={"size": 8.5},
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(
        "Normalized Confusion Matrix (Per-Row True Class Recall)\nValues Indicate Proportion of Actual Class Predicted as Target",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=10, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(
    report_dict: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
) -> None:
    """Figure 08: Per-class Precision, Recall, and F1-score on the test set."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    precisions = []
    recalls = []
    f1s = []
    for i in range(10):
        key = CLASS_NAMES[i] if CLASS_NAMES[i] in report_dict else str(i)
        precisions.append(report_dict[key]["precision"] * 100)
        recalls.append(report_dict[key]["recall"] * 100)
        f1s.append(report_dict[key]["f1-score"] * 100)

    x = np.arange(10)
    width = 0.26

    ax.bar(x - width, precisions, width, label="Precision (%)", color="#377eb8")
    ax.bar(x, recalls, width, label="Recall (%)", color="#4daf4a")
    ax.bar(x + width, f1s, width, label="F1-Score (%)", color="#e41a1c")

    ax.set_xlabel("Fashion Category", fontsize=10, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Per-Class Test Performance (Precision, Recall, F1-Score)\nEvaluation on 10,000 Unseen Test Samples",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}: {CLASS_NAMES[i]}" for i in range(10)], rotation=30, ha="right")
    ax.set_ylim(60, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="#cccccc")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", loc="lower right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_prediction_examples(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: Union[str, Path],
    num_examples: int = 5,
) -> None:
    """Figure 09: Gallery of correct vs incorrect test predictions with confidence."""
    fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 2.8, 6.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    correct_idx = np.where(true_labels == pred_labels)[0]
    incorrect_idx = np.where(true_labels != pred_labels)[0]

    # Select representative samples
    selected_correct = correct_idx[:num_examples]
    selected_incorrect = incorrect_idx[:num_examples]

    for col, idx in enumerate(selected_correct):
        ax = axes[0, col]
        ax.imshow(images[idx], cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        conf = probabilities[idx, pred_labels[idx]] * 100
        true_name = CLASS_NAMES[true_labels[idx]]
        pred_name = CLASS_NAMES[pred_labels[idx]]
        ax.set_title(
            f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)",
            fontsize=8.5,
            color="green",
            fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_color("green")
            spine.set_linewidth(2)

    for col, idx in enumerate(selected_incorrect):
        ax = axes[1, col]
        ax.imshow(images[idx], cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        conf = probabilities[idx, pred_labels[idx]] * 100
        true_name = CLASS_NAMES[true_labels[idx]]
        pred_name = CLASS_NAMES[pred_labels[idx]]
        true_conf = probabilities[idx, true_labels[idx]] * 100
        ax.set_title(
            f"True: {true_name} ({true_conf:.1f}%)\nPred: {pred_name} ({conf:.1f}%)",
            fontsize=8.5,
            color="red",
            fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_color("red")
            spine.set_linewidth(2)

    axes[0, 0].set_ylabel("CORRECT\nPREDICTIONS", fontsize=9.5, fontweight="bold", color="green")
    axes[1, 0].set_ylabel("INCORRECT\nPREDICTIONS", fontsize=9.5, fontweight="bold", color="red")

    plt.suptitle(
        "Sample Test Predictions (Correct Top Row vs. Misclassified Bottom Row)\nGreen borders indicate correct; red borders indicate confusion pairs",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_explainability_heatmaps(
    samples_data: List[Dict],
    output_path: Union[str, Path],
) -> None:
    """Figure 10: Input, Grad-CAM Attribution Heatmap, and Overlay across examples."""
    num_samples = len(samples_data)
    fig, axes = plt.subplots(num_samples, 3, figsize=(8.5, num_samples * 2.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample in enumerate(samples_data):
        img = sample["original_image"]  # (28, 28) uint8 or float [0, 1]
        heatmap = sample["heatmap"]  # (28, 28) float [0, 1]
        overlay = sample["overlay"]  # (28, 28, 3) or (28, 28, 4) float/uint8
        true_label = sample["true_label"]
        pred_label = sample["pred_label"]
        target_label = sample["target_label"]
        conf = sample["confidence"]
        status = "Correct" if true_label == pred_label else "Misclassified"

        # Column 0: Original input
        axes[row, 0].imshow(img, cmap="gray", interpolation="nearest")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("Input Image", fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel(
            f"[{status}]\nTrue: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]} ({conf:.1%})",
            fontsize=8.5,
            fontweight="bold",
            color="green" if status == "Correct" else "red",
        )

        # Column 1: Grad-CAM heatmap
        im_heat = axes[row, 1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        if row == 0:
            axes[row, 1].set_title("Grad-CAM Heatmap", fontsize=10, fontweight="bold")

        # Column 2: Overlay
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
        if row == 0:
            axes[row, 2].set_title("Heatmap Overlay", fontsize=10, fontweight="bold")

    plt.suptitle(
        "Explainability with Grad-CAM (Target Convolutional Activation)\nVisualizing Model Focus Regions for Correct and Confused Fashion Classes",
        fontsize=11.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
