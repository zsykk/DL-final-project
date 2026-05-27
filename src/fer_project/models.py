from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Compact convolutional network for 48x48 grayscale FER2013 images.

    Args:
        num_classes: Number of emotion classes predicted by the final layer.
        dropout: Dropout probability used before the final classifier layer.
        feature_channels: Output channels for each convolutional block.
        block_depths: Number of convolutional layers in each block.
        dropout2d: Spatial dropout used after each pooling block.
        classifier_hidden_layers: Hidden layer widths for the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout: float = 0.35,
        feature_channels: Sequence[int] = (32, 64, 128),
        block_depths: Sequence[int] = (2, 2, 1),
        dropout2d: Sequence[float] = (0.15, 0.20, 0.0),
        classifier_hidden_layers: Sequence[int] = (256,),
    ) -> None:
        """Initialize feature extraction and classification layers."""
        super().__init__()
        if len(feature_channels) != len(block_depths):
            raise ValueError("feature_channels and block_depths must have the same length")
        if len(dropout2d) != len(feature_channels):
            raise ValueError("dropout2d must have one value per convolutional block")

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, depth, block_dropout in zip(feature_channels, block_depths, dropout2d):
            for _ in range(depth):
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
            layers.append(nn.MaxPool2d(2))
            if block_dropout > 0:
                layers.append(nn.Dropout2d(block_dropout))

        spatial_size = 48 // (2 ** len(feature_channels))
        self.features = nn.Sequential(*layers)

        classifier_layers: list[nn.Module] = [nn.Flatten()]
        in_features = feature_channels[-1] * spatial_size * spatial_size
        for hidden_features in classifier_hidden_layers:
            classifier_layers.extend(
                [
                    nn.Linear(in_features, hidden_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_features
        classifier_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of image tensors.

        Args:
            x: Batch of FER2013 images shaped as ``(batch, 1, 48, 48)``.

        Returns:
            Unnormalized class logits shaped as ``(batch, num_classes)``.
        """
        return self.classifier(self.features(x))


def _build_mlp_classifier(
    in_features: int,
    num_classes: int,
    hidden_layers: Sequence[int] | None = None,
    dropout: float = 0.35,
) -> nn.Module:
    """Build either a linear or multi-layer classifier head."""
    hidden_layers = list(hidden_layers or [])
    if not hidden_layers:
        return nn.Linear(in_features, num_classes)

    layers: list[nn.Module] = []
    current_features = in_features
    for hidden_features in hidden_layers:
        layers.extend(
            [
                nn.Linear(current_features, hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        current_features = hidden_features
    layers.append(nn.Linear(current_features, num_classes))
    return nn.Sequential(*layers)


def _replace_classifier(
    model: nn.Module,
    model_name: str,
    num_classes: int,
    classifier_hidden_layers: Sequence[int] | None = None,
    classifier_dropout: float = 0.35,
) -> nn.Module:
    """Replace a torchvision model classifier with a FER2013 output layer.

    Args:
        model: Torchvision model instance to modify in place.
        model_name: Supported model family name used to locate the classifier.
        num_classes: Number of output emotion classes.

    Returns:
        The same model instance with its classifier replaced.
    """
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = _build_mlp_classifier(
            in_features,
            num_classes,
            hidden_layers=classifier_hidden_layers,
            dropout=classifier_dropout,
        )
    elif model_name.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = _build_mlp_classifier(
            in_features,
            num_classes,
            hidden_layers=classifier_hidden_layers,
            dropout=classifier_dropout,
        )
    elif model_name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = _build_mlp_classifier(
            in_features,
            num_classes,
            hidden_layers=classifier_hidden_layers,
            dropout=classifier_dropout,
        )
    else:
        raise ValueError(f"Unsupported transfer model: {model_name}")
    return model


def build_transfer_model(
    model_name: str = "resnet18",
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    classifier_hidden_layers: Sequence[int] | None = None,
    classifier_dropout: float = 0.35,
) -> nn.Module:
    """Build a supported pretrained architecture for transfer learning.

    Args:
        model_name: Torchvision architecture name to create.
        num_classes: Number of emotion classes predicted by the model.
        pretrained: Whether to initialize the backbone with default ImageNet weights.
        freeze_backbone: Whether to freeze existing backbone parameters before
            replacing the classifier.

    Returns:
        A torchvision model configured for FER2013 classification.
    """
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

    model = _replace_classifier(
        model,
        model_name,
        num_classes,
        classifier_hidden_layers=classifier_hidden_layers,
        classifier_dropout=classifier_dropout,
    )
    return model


def set_resnet18_trainable_layers(model: nn.Module, unfreeze_from: str = "fc") -> nn.Module:
    """Freeze ResNet18 then unfreeze a selected suffix of the network.

    ``unfreeze_from`` supports ``"fc"``, ``"layer4"``, ``"layer3"``,
    ``"layer2"``, ``"layer1"``, and ``"all"``.
    """
    order = ["fc", "layer4", "layer3", "layer2", "layer1", "all"]
    if unfreeze_from not in order:
        raise ValueError(f"unfreeze_from must be one of: {order}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    if unfreeze_from == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model

    trainable_modules = ["fc", "layer4", "layer3", "layer2", "layer1"]
    start = trainable_modules.index(unfreeze_from)
    for module_name in trainable_modules[start::-1]:
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True
    return model


def replace_resnet18_classifier(
    model: nn.Module,
    num_classes: int = 7,
    classifier_hidden_layers: Sequence[int] | None = None,
    classifier_dropout: float = 0.35,
) -> nn.Module:
    """Replace a ResNet18 classifier head while keeping the backbone weights."""
    in_features = model.fc.in_features
    model.fc = _build_mlp_classifier(
        in_features,
        num_classes,
        hidden_layers=classifier_hidden_layers,
        dropout=classifier_dropout,
    )
    return model


def build_model(
    experiment: str,
    num_classes: int = 7,
    transfer_model: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    baseline_feature_channels: Sequence[int] = (32, 64, 128),
    baseline_block_depths: Sequence[int] = (2, 2, 1),
    baseline_dropout2d: Sequence[float] = (0.15, 0.20, 0.0),
    baseline_classifier_hidden_layers: Sequence[int] = (256,),
    baseline_dropout: float = 0.35,
    classifier_hidden_layers: Sequence[int] | None = None,
    classifier_dropout: float = 0.35,
) -> nn.Module:
    """Create the model requested by an experiment configuration.

    Args:
        experiment: Either ``"baseline_cnn"`` or ``"transfer"``.
        num_classes: Number of emotion classes predicted by the model.
        transfer_model: Torchvision architecture used when ``experiment`` is
            ``"transfer"``.
        pretrained: Whether transfer models should use ImageNet weights.
        freeze_backbone: Whether transfer models should freeze their backbones.

    Returns:
        A PyTorch module ready for training or evaluation.
    """
    if experiment == "baseline_cnn":
        return BaselineCNN(
            num_classes=num_classes,
            dropout=baseline_dropout,
            feature_channels=baseline_feature_channels,
            block_depths=baseline_block_depths,
            dropout2d=baseline_dropout2d,
            classifier_hidden_layers=baseline_classifier_hidden_layers,
        )
    if experiment == "transfer":
        return build_transfer_model(
            model_name=transfer_model,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            classifier_hidden_layers=classifier_hidden_layers,
            classifier_dropout=classifier_dropout,
        )
    raise ValueError("experiment must be 'baseline_cnn' or 'transfer'")
