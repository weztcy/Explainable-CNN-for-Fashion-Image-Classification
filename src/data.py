"""Dataset loading, verification, and preprocessing for Fashion-MNIST."""

import gzip
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T

# Official Fashion-MNIST class mapping
CLASS_NAMES: List[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Canonical suffixes for dataset files
CANONICAL_SUFFIXES: Dict[str, str] = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Official MD5 checksums (Zalando Research)
OFFICIAL_MD5: Dict[str, str] = {
    "train_images": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train_labels": "25c81989df183df01b3e8a0aad5dffbe",
    "test_images": "bef4ecab320f06d8554ea6380940ec79",
    "test_labels": "bb300cfdad3c16e7a12a480ee83cd310",
}


def find_dataset_files(data_dir: Union[str, Path]) -> Dict[str, Path]:
    """Find dataset files matching canonical suffixes in data_dir.
    
    Verifies that each canonical role matches exactly one file.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    matched_files: Dict[str, Path] = {}
    all_files = [p for p in data_path.iterdir() if p.is_file()]

    for role, suffix in CANONICAL_SUFFIXES.items():
        candidates = [p for p in all_files if p.name.endswith(suffix)]
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No file ending with '{suffix}' found in '{data_dir}' for role '{role}'."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple files match role '{role}' with suffix '{suffix}': {candidates}"
            )
        matched_files[role] = candidates[0]

    return matched_files


def calculate_md5(file_path: Union[str, Path]) -> str:
    """Calculate MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksums(file_map: Dict[str, Path]) -> Dict[str, Tuple[str, bool]]:
    """Verify MD5 checksums for all dataset files against official values."""
    results: Dict[str, Tuple[str, bool]] = {}
    for role, path in file_map.items():
        actual_md5 = calculate_md5(path)
        expected_md5 = OFFICIAL_MD5[role]
        matches = actual_md5.lower() == expected_md5.lower()
        results[role] = (actual_md5, matches)
    return results


def read_idx_images(file_path: Union[str, Path]) -> np.ndarray:
    """Read IDX3-ubyte gzipped image file.
    
    Returns np.ndarray of shape (N, 28, 28) with dtype uint8.
    """
    with gzip.open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(
                f"Invalid magic number {magic} in {file_path}. Expected 2051 for images."
            )
        raw_data = f.read()
        expected_bytes = num_images * rows * cols
        if len(raw_data) != expected_bytes:
            raise ValueError(
                f"Payload size mismatch in {file_path}: expected {expected_bytes} bytes, got {len(raw_data)}."
            )
        images = np.frombuffer(raw_data, dtype=np.uint8).reshape(num_images, rows, cols)
    return images


def read_idx_labels(file_path: Union[str, Path]) -> np.ndarray:
    """Read IDX1-ubyte gzipped label file.
    
    Returns np.ndarray of shape (N,) with dtype uint8.
    """
    with gzip.open(file_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(
                f"Invalid magic number {magic} in {file_path}. Expected 2049 for labels."
            )
        raw_data = f.read()
        if len(raw_data) != num_labels:
            raise ValueError(
                f"Payload size mismatch in {file_path}: expected {num_labels} bytes, got {len(raw_data)}."
            )
        labels = np.frombuffer(raw_data, dtype=np.uint8)
    return labels


def audit_dataset(data_dir: Union[str, Path]) -> Dict:
    """Perform comprehensive data audit and return audit summary."""
    file_map = find_dataset_files(data_dir)
    checksums = verify_checksums(file_map)

    train_images = read_idx_images(file_map["train_images"])
    train_labels = read_idx_labels(file_map["train_labels"])
    test_images = read_idx_images(file_map["test_images"])
    test_labels = read_idx_labels(file_map["test_labels"])

    # Check pair lengths
    assert len(train_images) == len(train_labels), "Train images/labels count mismatch"
    assert len(test_images) == len(test_labels), "Test images/labels count mismatch"

    # Check pixel value range
    train_min, train_max = int(train_images.min()), int(train_images.max())
    test_min, test_max = int(test_images.min()), int(test_images.max())

    # Check for all-zero images
    train_all_zeros = int((train_images.reshape(len(train_images), -1).sum(axis=1) == 0).sum())
    test_all_zeros = int((test_images.reshape(len(test_images), -1).sum(axis=1) == 0).sum())

    # Class distributions
    train_dist = {CLASS_NAMES[i]: int(c) for i, c in enumerate(np.bincount(train_labels, minlength=10))}
    test_dist = {CLASS_NAMES[i]: int(c) for i, c in enumerate(np.bincount(test_labels, minlength=10))}

    # Duplicate checking
    train_hashes = [hashlib.md5(img.tobytes()).hexdigest() for img in train_images]
    test_hashes = [hashlib.md5(img.tobytes()).hexdigest() for img in test_images]
    train_unique = len(set(train_hashes))
    test_unique = len(set(test_hashes))
    train_test_overlap = len(set(train_hashes).intersection(set(test_hashes)))

    return {
        "checksums": {k: {"md5": v[0], "valid": v[1]} for k, v in checksums.items()},
        "train": {
            "num_samples": len(train_images),
            "dimensions": list(train_images.shape[1:]),
            "pixel_min": train_min,
            "pixel_max": train_max,
            "all_zero_images": train_all_zeros,
            "unique_images": train_unique,
            "class_distribution": train_dist,
        },
        "test": {
            "num_samples": len(test_images),
            "dimensions": list(test_images.shape[1:]),
            "pixel_min": test_min,
            "pixel_max": test_max,
            "all_zero_images": test_all_zeros,
            "unique_images": test_unique,
            "class_distribution": test_dist,
        },
        "overlap_train_test": train_test_overlap,
    }


def get_stratified_split(
    labels: np.ndarray, val_ratio: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split labels stratified into train and validation indices.
    
    For 60,000 samples with val_ratio=0.1, returns 54,000 train indices
    and 6,000 validation indices, exactly preserving class balance.
    """
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    unique_classes = np.unique(labels)
    for c in sorted(unique_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        n_val = int(round(len(class_idx) * val_ratio))
        val_indices.extend(class_idx[:n_val])
        train_indices.extend(class_idx[n_val:])

    train_indices = np.array(sorted(train_indices), dtype=np.int64)
    val_indices = np.array(sorted(val_indices), dtype=np.int64)

    # Verification: no overlap and union equals original indices
    assert len(np.intersect1d(train_indices, val_indices)) == 0
    assert len(train_indices) + len(val_indices) == len(labels)

    return train_indices, val_indices


def save_split_indices(
    train_idx: np.ndarray, val_idx: np.ndarray, out_path: Union[str, Path]
) -> None:
    """Save split indices to JSON."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "num_train": len(train_idx),
                "num_val": len(val_idx),
            },
            f,
            indent=2,
        )


def load_split_indices(in_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load split indices from JSON."""
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["train_indices"], dtype=np.int64), np.array(
        data["val_indices"], dtype=np.int64
    )


def compute_train_mean_std(images: np.ndarray) -> Tuple[float, float]:
    """Compute mean and standard deviation from normalized [0, 1] images."""
    normalized = images.astype(np.float32) / 255.0
    mean = float(np.mean(normalized))
    std = float(np.std(normalized))
    return mean, std


class FashionMNISTDataset(Dataset):
    """PyTorch Dataset for Fashion-MNIST with strictly controlled normalization and augmentation."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        mean: float,
        std: float,
        augment: bool = False,
    ):
        """
        Args:
            images: np.ndarray of shape (N, 28, 28) uint8.
            labels: np.ndarray of shape (N,) uint8.
            mean: Precomputed mean from train split.
            std: Precomputed std from train split.
            augment: Whether to apply lightweight augmentation (train split only).
        """
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels.astype(np.int64)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_np = self.images[idx].astype(np.float32) / 255.0  # (28, 28) in [0, 1]
        tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, 28, 28)

        if self.augment:
            # Lightweight augmentation: RandomHorizontalFlip & slight translation/crop
            if torch.rand(1).item() > 0.5:
                tensor = torch.flip(tensor, dims=[2])
            # Random crop with 2-pixel reflection padding
            if torch.rand(1).item() > 0.5:
                padded = torch.nn.functional.pad(tensor, (2, 2, 2, 2), mode="reflect")
                top = int(torch.randint(0, 5, (1,)).item())
                left = int(torch.randint(0, 5, (1,)).item())
                tensor = padded[:, top : top + 28, left : left + 28]

        # Apply mean & std normalization
        tensor = (tensor - self.mean) / self.std
        target = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, target


def get_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 64,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_augmentation: bool = False,
    split_path: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Build train, validation, and test dataloaders with strictly segregated splits."""
    file_map = find_dataset_files(data_dir)
    all_train_images = read_idx_images(file_map["train_images"])
    all_train_labels = read_idx_labels(file_map["train_labels"])
    test_images = read_idx_images(file_map["test_images"])
    test_labels = read_idx_labels(file_map["test_labels"])

    # Load or create stratified train/val split
    if split_path and Path(split_path).exists():
        train_idx, val_idx = load_split_indices(split_path)
    else:
        train_idx, val_idx = get_stratified_split(
            all_train_labels, val_ratio=val_ratio, seed=seed
        )
        if split_path:
            save_split_indices(train_idx, val_idx, split_path)

    train_images = all_train_images[train_idx]
    train_labels = all_train_labels[train_idx]
    val_images = all_train_images[val_idx]
    val_labels = all_train_labels[val_idx]

    # Compute normalization statistics STRICTLY from train images
    train_mean, train_std = compute_train_mean_std(train_images)

    # Datasets
    train_dataset = FashionMNISTDataset(
        train_images, train_labels, mean=train_mean, std=train_std, augment=use_augmentation
    )
    val_dataset = FashionMNISTDataset(
        val_images, val_labels, mean=train_mean, std=train_std, augment=False
    )
    test_dataset = FashionMNISTDataset(
        test_images, test_labels, mean=train_mean, std=train_std, augment=False
    )

    # Generator for reproducible DataLoader shuffling
    gen = torch.Generator()
    gen.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    metadata = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_mean": train_mean,
        "train_std": train_std,
        "class_names": CLASS_NAMES,
        "seed": seed,
        "use_augmentation": use_augmentation,
    }

    return train_loader, val_loader, test_loader, metadata
