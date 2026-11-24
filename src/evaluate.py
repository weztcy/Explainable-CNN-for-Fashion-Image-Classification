"""Final evaluation and error analysis for Fashion-MNIST classification."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES, get_dataloaders
from src.models import count_parameters, get_model
from src.visualize import (
    plot_confusion_matrix_counts,
    plot_confusion_matrix_normalized,
    plot_model_comparison,
    plot_per_class_metrics,
    plot_prediction_examples,
)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load model and metadata from saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    model = get_model(model_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def run_inference_on_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over dataloader and return true labels, predicted labels, and probabilities."""
    model.eval()
    all_trues = []
    all_preds = []
    all_probs = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_trues.extend(targets.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return (
        np.array(all_trues, dtype=np.int64),
        np.array(all_preds, dtype=np.int64),
        np.array(all_probs, dtype=np.float32),
    )


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    num_runs: int = 100,
    warmup: int = 15,
) -> Dict[str, float]:
    """Measure inference latency accurately on CPU/GPU."""
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Timing runs
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # milliseconds

    timings_np = np.array(timings)
    return {
        "batch_size": batch_size,
        "num_runs": num_runs,
        "mean_ms": float(np.mean(timings_np)),
        "std_ms": float(np.std(timings_np)),
        "p50_ms": float(np.percentile(timings_np, 50)),
        "p95_ms": float(np.percentile(timings_np, 95)),
    }


def find_top_confusion_pairs(cm: np.ndarray, top_k: int = 5) -> List[Dict]:
    """Identify the pairs of classes most frequently confused with each other."""
    pairs = []
    num_classes = cm.shape[0]
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true_class_idx": int(i),
                        "true_class_name": CLASS_NAMES[i],
                        "pred_class_idx": int(j),
                        "pred_class_name": CLASS_NAMES[j],
                        "count": int(cm[i, j]),
                        "percentage_of_true": float((cm[i, j] / cm[i].sum()) * 100.0),
                    }
                )
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def evaluate_all_models(
    data_dir: str = "dataset",
    output_dir: str = "outputs",
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """Run full evaluation on the test set for all trained models and generate figures 05-09."""
    device = torch.device("cpu")
    out_path = Path(output_dir)
    figures_dir = out_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_path / "split_indices.json"

    # Load dataloaders
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        val_ratio=0.1,
        seed=42,
        use_augmentation=False,
        split_path=split_path,
    )

    models_dir = out_path / "models"
    available_models = ["mlp", "standard_cnn", "regularized_cnn"]
    comparison_summary: Dict[str, Dict] = {}
    best_cnn_metrics = None
    best_model_name = None

    for m_name in available_models:
        ckpt_file = models_dir / f"{m_name}.pt"
        if not ckpt_file.exists():
            continue

        model, ckpt = load_model_from_checkpoint(str(ckpt_file), device)
        y_true, y_pred, y_prob = run_inference_on_loader(model, test_loader, device)

        acc = float((y_true == y_pred).mean())
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        param_count = count_parameters(model)

        display_name = {
            "mlp": "MLP Baseline",
            "standard_cnn": "Standard CNN",
            "regularized_cnn": "Regularized CNN",
        }[m_name]

        comparison_summary[display_name] = {
            "model_key": m_name,
            "val_loss": ckpt["val_loss"],
            "val_acc": ckpt["val_acc"] / 100.0,
            "test_acc": acc,
            "test_macro_precision": float(prec),
            "test_macro_recall": float(rec),
            "test_macro_f1": float(f1),
            "param_count": param_count,
            "best_epoch": ckpt["epoch"],
        }

    # Generate Figure 05: Model Comparison
    plot_model_comparison(comparison_summary, figures_dir / "05_model_comparison.png")
    print("Figure 05 generated at outputs/figures/05_model_comparison.png")

    # Select target model for detailed analysis: best_cnn or specified checkpoint
    target_ckpt = checkpoint_path if checkpoint_path else str(models_dir / "best_cnn.pt")
    model, ckpt = load_model_from_checkpoint(target_ckpt, device)
    best_model_name = ckpt["model_name"]
    print(f"\nPerforming deep evaluation on final best model: {best_model_name}")

    # Evaluate on test set
    y_true, y_pred, y_prob = run_inference_on_loader(model, test_loader, device)

    # Detailed test metrics
    test_accuracy = float((y_true == y_pred).mean())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    clf_report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    # Also add string format report for readability
    clf_report_str = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    print("\nTest Classification Report:\n", clf_report_str)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    top_confusion = find_top_confusion_pairs(cm, top_k=6)
    print("Top Confusion Pairs:")
    for p in top_confusion:
        print(f"  {p['true_class_name']} -> {p['pred_class_name']}: {p['count']} times ({p['percentage_of_true']:.1f}%)")

    # Latency benchmarks
    latency_b1 = measure_inference_latency(model, device, batch_size=1, num_runs=100)
    latency_b64 = measure_inference_latency(model, device, batch_size=64, num_runs=50)

    # Raw test images for visualization
    from src.data import find_dataset_files, read_idx_images
    file_map = find_dataset_files(data_dir)
    test_images_raw = read_idx_images(file_map["test_images"])

    # Generate Figures 06, 07, 08, 09
    plot_confusion_matrix_counts(cm, figures_dir / "06_confusion_matrix_counts.png")
    plot_confusion_matrix_normalized(cm, figures_dir / "07_confusion_matrix_normalized.png")
    plot_per_class_metrics(clf_report, figures_dir / "08_per_class_metrics.png")
    plot_prediction_examples(
        test_images_raw, y_true, y_pred, y_prob, figures_dir / "09_prediction_examples.png"
    )
    print("Figures 06, 07, 08, 09 successfully generated in outputs/figures/")

    # Compile comprehensive metrics.json
    final_metrics = {
        "final_model": best_model_name,
        "device": "CPU",
        "checkpoint_used": target_ckpt,
        "test_samples": len(y_true),
        "overall_metrics": {
            "test_accuracy": test_accuracy,
            "test_macro_precision": float(macro_p),
            "test_macro_recall": float(macro_r),
            "test_macro_f1": float(macro_f1),
            "test_weighted_precision": float(weighted_p),
            "test_weighted_recall": float(weighted_r),
            "test_weighted_f1": float(weighted_f1),
        },
        "per_class_report": clf_report,
        "confusion_matrix": cm.tolist(),
        "top_confusion_pairs": top_confusion,
        "model_comparison": comparison_summary,
        "inference_latency": {
            "single_item_batch_1": latency_b1,
            "batched_batch_64": latency_b64,
        },
    }

    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fashion-MNIST models.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/models/best_cnn.pt",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    evaluate_all_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint if Path(args.checkpoint).exists() else None,
    )


if __name__ == "__main__":
    main()
