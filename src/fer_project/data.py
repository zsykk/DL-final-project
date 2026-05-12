from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}


@dataclass(frozen=True)
class TransformConfig:
    image_size: int = 48
    channels: int = 1
    augment: bool = False
    imagenet_norm: bool = False


class FER2013Dataset(Dataset):
    """FER2013 CSV dataset where pixels are stored as flattened strings."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform

        df = pd.read_csv(self.csv_path)
        required = {"emotion", "pixels", "Usage"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing FER2013 columns: {sorted(missing)}")

        usage_names = [name for name, mapped in USAGE_TO_SPLIT.items() if mapped == split]
        if not usage_names:
            raise ValueError(f"Unknown split '{split}'. Use train, val, or test.")

        self.frame = df[df["Usage"].isin(usage_names)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No rows found for split '{split}' in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        pixels = np.fromstring(row["pixels"], dtype=np.float32, sep=" ").reshape(48, 48)
        image = np.uint8(pixels)
        label = int(row["emotion"])

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(pixels / 255.0).unsqueeze(0)

        return image, label

    @property
    def labels(self) -> np.ndarray:
        return self.frame["emotion"].astype(int).to_numpy()


def build_transforms(config: TransformConfig) -> transforms.Compose:
    ops: list[object] = [transforms.ToPILImage()]

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

    if config.channels == 3:
        ops.append(transforms.Grayscale(num_output_channels=3))

    ops.append(transforms.ToTensor())

    if config.imagenet_norm:
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif config.channels == 1:
        ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        ops.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return transforms.Compose(ops)


def class_weights(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    csv_path: str | Path,
    train_config: TransformConfig,
    eval_config: TransformConfig,
    batch_size: int = 128,
    num_workers: int = 2,
    weighted_sampler: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, FER2013Dataset]]:
    datasets = {
        "train": FER2013Dataset(csv_path, "train", build_transforms(train_config)),
        "val": FER2013Dataset(csv_path, "val", build_transforms(eval_config)),
        "test": FER2013Dataset(csv_path, "test", build_transforms(eval_config)),
    }

    sampler = None
    shuffle = True
    if weighted_sampler:
        weights = class_weights(datasets["train"].labels).numpy()
        sample_weights = weights[datasets["train"].labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, datasets
