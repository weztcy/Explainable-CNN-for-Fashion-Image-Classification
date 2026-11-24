"""Training script for Fashion-MNIST classification models."""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES, get_dataloaders
from src.models import count_parameters, get_model


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch. Returns (avg_loss, accuracy_percentage)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a dataloader. Returns (avg_loss, accuracy_percentage)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def train_model(
    model_name: str,
    data_dir: str = "dataset",
    output_dir: str = "outputs",
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 4,
    seed: int = 42,
    use_augmentation: bool = False,
    device_str: Optional[str] = None,
) -> Dict:
    """Train a single model with early stopping, tracking, and checkpointing."""
    set_seed(seed)
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    models_dir = out_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_path / "split_indices.json"

    # Dataloaders
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=0.1,
        seed=seed,
        use_augmentation=use_augmentation,
        split_path=split_path,
    )

    model = get_model(model_name).to(device)
    param_count = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler (reduces LR on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    no_improve_count = 0
    checkpoint_path = models_dir / f"{model_name}.pt"

    start_time = time.time()
    print(f"\n==========================================")
    print(f"Training: {model_name} (Params: {param_count:,})")
    print(f"Device: {device} | Augmentation: {use_augmentation} | Batch: {batch_size}")
    print(f"==========================================")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        t_elapsed = time.time() - t0

        scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["epoch_time"].append(float(t_elapsed))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"Time: {t_elapsed:.1f}s"
        )

        # Early stopping and checkpointing based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "param_count": param_count,
                    "train_mean": meta["train_mean"],
                    "train_std": meta["train_std"],
                    "class_names": CLASS_NAMES,
                    "seed": seed,
                    "hyperparameters": {
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "use_augmentation": use_augmentation,
                    },
                },
                checkpoint_path,
            )
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}."
                )
                break

    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time:.1f}s. Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")

    # Save training history
    history_file = out_path / f"history_{model_name}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "total_time_seconds": total_training_time,
                "history": history,
            },
            f,
            indent=2,
        )

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "param_count": param_count,
        "total_time_seconds": total_training_time,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST models.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["mlp", "standard_cnn", "regularized_cnn", "all"],
        help="Model to train or 'all' for comparative experiments",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    # Save run configuration
    run_config = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "model_arg": args.model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "patience": args.patience,
        "device": "cpu",
    }
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    from src.visualize import plot_training_curves

    if args.model == "all":
        # Run controlled comparison experiments (Task 4)
        models_to_run = [
            ("mlp", False, 1e-3),
            ("standard_cnn", False, 1e-3),
            ("regularized_cnn", True, 1e-3),
        ]
        all_histories = {}
        results = {}

        for m_name, aug, lr in models_to_run:
            display_name = {
                "mlp": "MLP Baseline",
                "standard_cnn": "Standard CNN",
                "regularized_cnn": "Regularized CNN",
            }[m_name]

            res = train_model(
                model_name=m_name,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=lr,
                patience=args.patience,
                seed=args.seed,
                use_augmentation=aug,
            )
            all_histories[display_name] = res["history"]
            results[m_name] = res

        # Generate Figure 4: Training curves
        plot_training_curves(all_histories, out_path / "figures" / "04_training_curves.png")
        print("\nFigure 04 generated at outputs/figures/04_training_curves.png")

        # Select best CNN based on validation loss/accuracy
        best_val_acc = -1.0
        best_cnn_name = "regularized_cnn"
        for m_name in ["standard_cnn", "regularized_cnn"]:
            if results[m_name]["best_val_acc"] > best_val_acc:
                best_val_acc = results[m_name]["best_val_acc"]
                best_cnn_name = m_name

        print(f"\nBest CNN selected by validation accuracy: {best_cnn_name} ({best_val_acc:.2f}%)")
        # Copy / save to best_cnn.pt
        import shutil
        src_best = out_path / "models" / f"{best_cnn_name}.pt"
        dst_best = out_path / "models" / "best_cnn.pt"
        shutil.copy2(src_best, dst_best)
        print(f"Copied {src_best} -> {dst_best}")

    else:
        aug = args.model == "regularized_cnn"
        res = train_model(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed,
            use_augmentation=aug,
        )


if __name__ == "__main__":
    main()
