from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

EMOTION_NAME_TO_LABEL = {name: label for label, name in EMOTION_LABELS.items()}


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for image transforms used by FER2013 datasets.

    Args:
        image_size: Target square image size after resizing.
        channels: Number of output channels, either grayscale ``1`` or RGB ``3``.
        augment: Whether to add stochastic training augmentations.
        imagenet_norm: Whether to apply ImageNet normalization statistics.
    """

    image_size: int = 48
    channels: int = 1
    augment: bool = False
    imagenet_norm: bool = False


def build_transforms(config: TransformConfig, input_is_pil: bool = False) -> transforms.Compose:
    """Create a torchvision transform pipeline from a transform configuration.

    Args:
        config: Image sizing, channel, augmentation, and normalization options.
        input_is_pil: Whether input images are already PIL images, as returned by
            image-folder datasets.

    Returns:
        Composed transforms that convert FER2013 pixel arrays into tensors.
    """
    ops: list[object] = []

    if not input_is_pil:
        ops.append(transforms.ToPILImage())

    if config.image_size != 48:
        ops.append(transforms.Resize((config.image_size, config.image_size)))

    if config.augment:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            ]
        )

    if config.channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    elif config.channels == 3:
        ops.append(transforms.Grayscale(num_output_channels=3))

    ops.append(transforms.ToTensor())

    if config.imagenet_norm:
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif config.channels == 1:
        ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        ops.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return transforms.Compose(ops)


def folder_class_to_fer_label(class_name: str) -> int:
    """Convert an image-folder class name into the official FER2013 label id.

    Args:
        class_name: Folder name, such as ``"angry"`` or ``"0"``.

    Returns:
        Integer FER2013 label id from 0 to 6.
    """
    normalized = class_name.lower().strip()
    if normalized.isdigit():
        label = int(normalized)
        if label in EMOTION_LABELS:
            return label
    if normalized in EMOTION_NAME_TO_LABEL:
        return EMOTION_NAME_TO_LABEL[normalized]
    raise ValueError(f"Unknown FER2013 folder class: {class_name}")


def _imagefolder_target_transform(dataset: ImageFolder):
    """Build a target transform that maps ImageFolder indices to FER2013 labels.

    Args:
        dataset: ImageFolder dataset with FER2013 class folders.

    Returns:
        Callable that converts ImageFolder's alphabetical class ids to FER labels.
    """
    index_to_fer_label = {
        class_index: folder_class_to_fer_label(class_name)
        for class_name, class_index in dataset.class_to_idx.items()
    }
    return index_to_fer_label.__getitem__


def dataset_labels(dataset: Dataset) -> np.ndarray:
    """Extract integer labels from ImageFolder datasets and subsets.

    Args:
        dataset: ImageFolder or Subset wrapping an ImageFolder.

    Returns:
        NumPy array of FER2013 labels in dataset order.
    """
    if isinstance(dataset, Subset):
        parent_labels = dataset_labels(dataset.dataset)
        return parent_labels[np.array(dataset.indices)]
    if isinstance(dataset, ImageFolder):
        target_transform = dataset.target_transform or (lambda target: target)
        return np.array([target_transform(target) for target in dataset.targets], dtype=int)
    raise TypeError(f"Cannot extract labels from dataset type: {type(dataset).__name__}")


def stratified_subset_indices(labels: np.ndarray, fraction: float = 1.0, seed: int = 42) -> list[int]:
    """Choose a reproducible class-stratified subset of dataset indices.

    Args:
        labels: Integer class labels in dataset order.
        fraction: Fraction of each class to keep, from 0 exclusive to 1 inclusive.
        seed: Random seed used to shuffle indices within each class.

    Returns:
        Sorted list of selected dataset indices.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and less than or equal to 1")
    if fraction == 1:
        return list(range(len(labels)))

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(np.unique(labels)):
        label_indices = np.flatnonzero(labels == label)
        keep_count = max(1, int(len(label_indices) * fraction))
        selected.extend(rng.choice(label_indices, size=keep_count, replace=False).tolist())
    return sorted(selected)


def class_weights(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """Compute inverse-frequency class weights normalized around one.

    Args:
        labels: Integer class labels for a dataset split.
        num_classes: Total number of expected emotion classes.

    Returns:
        Float tensor of per-class weights suitable for cross-entropy loss.
    """
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_imagefolder_dataloaders(
    data_dir: str | Path,
    train_config: TransformConfig,
    eval_config: TransformConfig,
    batch_size: int = 128,
    num_workers: int = 2,
    weighted_sampler: bool = False,
    val_fraction: float = 0.1,
    subset_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[dict[str, DataLoader], dict[str, Dataset]]:
    """Build dataloaders from FER2013 image folders split into train and test.

    Expected directory layout is ``data_dir/train/<emotion>/*.jpg`` and
    ``data_dir/test/<emotion>/*.jpg``. The training folder is split into train and
    validation subsets because this Kaggle format does not include a separate
    public validation folder.

    Args:
        data_dir: Root directory containing ``train`` and ``test`` folders.
        train_config: Transform settings used for the training subset.
        eval_config: Transform settings used for validation and test subsets.
        batch_size: Number of examples per dataloader batch.
        num_workers: Number of worker processes used by each dataloader.
        weighted_sampler: Whether to sample training examples with class-balanced
            probabilities.
        val_fraction: Fraction of the training folder held out for validation.
        subset_fraction: Class-stratified fraction of train and test images to
            keep for quick smoke tests.
        seed: Random seed used for the train/validation split.

    Returns:
        A pair containing dataloaders and their backing datasets keyed by split.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise ValueError(f"Expected train and test folders under {data_dir}")

    train_eval_dataset = ImageFolder(train_dir, transform=build_transforms(eval_config, input_is_pil=True))
    target_transform = _imagefolder_target_transform(train_eval_dataset)
    train_full_dataset = ImageFolder(
        train_dir,
        transform=build_transforms(train_config, input_is_pil=True),
        target_transform=target_transform,
    )
    train_eval_dataset.target_transform = target_transform

    train_candidate_indices = stratified_subset_indices(
        dataset_labels(train_full_dataset),
        fraction=subset_fraction,
        seed=seed,
    )

    val_size = int(len(train_candidate_indices) * val_fraction)
    train_size = len(train_candidate_indices) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("val_fraction must leave at least one train and one validation example")

    generator = torch.Generator().manual_seed(seed)
    shuffled_positions = torch.randperm(len(train_candidate_indices), generator=generator).tolist()
    indices = [train_candidate_indices[position] for position in shuffled_positions]
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_subset = Subset(train_full_dataset, train_indices)
    val_subset = Subset(train_eval_dataset, val_indices)

    test_dataset = ImageFolder(
        test_dir,
        transform=build_transforms(eval_config, input_is_pil=True),
    )
    test_dataset.target_transform = _imagefolder_target_transform(test_dataset)
    test_indices = stratified_subset_indices(
        dataset_labels(test_dataset),
        fraction=subset_fraction,
        seed=seed,
    )
    test_subset = Subset(test_dataset, test_indices)

    sampler = None
    shuffle = True
    if weighted_sampler:
        labels = dataset_labels(train_subset)
        weights = class_weights(labels).numpy()
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    datasets = {
        "train": train_subset,
        "val": val_subset,
        "test": test_subset,
    }
    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, datasets
