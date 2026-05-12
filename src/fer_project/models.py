from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Compact CNN for 48x48 grayscale FER2013 images."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported transfer model: {model_name}")
    return model


def build_transfer_model(
    model_name: str = "resnet18",
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError("Use one of: resnet18, mobilenet_v2, efficientnet_b0")

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model = _replace_classifier(model, model_name, num_classes)
    return model


def build_model(
    experiment: str,
    num_classes: int = 7,
    transfer_model: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    if experiment == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if experiment == "transfer":
        return build_transfer_model(
            model_name=transfer_model,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError("experiment must be 'baseline_cnn' or 'transfer'")
