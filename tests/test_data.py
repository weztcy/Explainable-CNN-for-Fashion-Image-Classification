"""Unit tests for dataset loading, verification, and preprocessing."""

import numpy as np
import pytest
import torch

from src.data import (
    CLASS_NAMES,
    compute_train_mean_std,
    find_dataset_files,
    get_dataloaders,
    get_stratified_split,
    read_idx_images,
    read_idx_labels,
    verify_checksums,
)


def test_find_dataset_files_and_checksums():
    file_map = find_dataset_files("dataset")
    assert len(file_map) == 4
    for role in ["train_images", "train_labels", "test_images", "test_labels"]:
        assert role in file_map
        assert file_map[role].exists()

    checksum_results = verify_checksums(file_map)
    for role, (md5, valid) in checksum_results.items():
        assert valid is True, f"Checksum verification failed for {role}: {md5}"


def test_read_idx_dimensions_and_types():
    file_map = find_dataset_files("dataset")
    train_images = read_idx_images(file_map["train_images"])
    train_labels = read_idx_labels(file_map["train_labels"])
    test_images = read_idx_images(file_map["test_images"])
    test_labels = read_idx_labels(file_map["test_labels"])

    assert train_images.shape == (60000, 28, 28)
    assert train_images.dtype == np.uint8
    assert train_labels.shape == (60000,)
    assert train_labels.dtype == np.uint8

    assert test_images.shape == (10000, 28, 28)
    assert test_labels.shape == (10000,)

    # Value ranges
    assert train_images.min() >= 0 and train_images.max() <= 255
    assert set(np.unique(train_labels)) == set(range(10))
    assert set(np.unique(test_labels)) == set(range(10))


def test_stratified_split():
    labels = np.repeat(np.arange(10), 6000)  # 60,000 balanced labels
    train_idx, val_idx = get_stratified_split(labels, val_ratio=0.1, seed=42)

    assert len(train_idx) == 54000
    assert len(val_idx) == 6000
    assert len(np.intersect1d(train_idx, val_idx)) == 0

    # Verify per-class balance
    train_counts = np.bincount(labels[train_idx], minlength=10)
    val_counts = np.bincount(labels[val_idx], minlength=10)

    assert np.all(train_counts == 5400)
    assert np.all(val_counts == 600)


def test_dataloaders_pipeline():
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        data_dir="dataset", batch_size=32, seed=42
    )

    assert meta["train_size"] == 54000
    assert meta["val_size"] == 6000
    assert meta["test_size"] == 10000
    assert 0.0 < meta["train_mean"] < 1.0
    assert 0.0 < meta["train_std"] < 1.0

    # Check batch shape
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (32, 1, 28, 28)
    assert batch_y.shape == (32,)
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.int64
