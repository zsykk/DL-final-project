from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm


class FocalLoss(nn.Module):
    """Cross-entropy focal loss for imbalanced multi-class classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def build_criterion(
    loss_name: str = "cross_entropy",
    class_weight: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Build the supervised loss used for an experiment."""
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weight)
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weight)
    raise ValueError("loss_name must be 'cross_entropy' or 'focal'")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one supervised training epoch and report average metrics.

    Args:
        model: PyTorch model to optimize.
        loader: Dataloader yielding image and label batches.
        criterion: Loss function used to compare logits with labels.
        optimizer: Optimizer responsible for parameter updates.
        device: Torch device where tensors and the model are placed.

    Returns:
        Dictionary containing mean loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def train_one_epoch_with_gradient_clip(model, loader, criterion, optimizer, device, grad_clip: float | None = None):
    """Run one training epoch and optionally clip gradients by value."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_loss_accuracy(model, loader, criterion, device):
    """Evaluate average loss and accuracy without updating model weights.

    Args:
        model: PyTorch model to evaluate.
        loader: Dataloader yielding image and label batches.
        criterion: Loss function used to compute evaluation loss.
        device: Torch device where tensors and the model are placed.

    Returns:
        Dictionary containing mean loss and accuracy for the dataloader.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_loss_accuracy_with_crops(model, loader, criterion, device):
    """Evaluate loss/accuracy, averaging logits across TenCrop batches if present."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        labels = labels.to(device)
        if images.ndim == 5:
            batch_size, num_crops, channels, height, width = images.shape
            images = images.view(-1, channels, height, width).to(device)
            logits = model(images).view(batch_size, num_crops, -1).mean(dim=1)
        else:
            images = images.to(device)
            logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def format_epoch_metrics(row: dict) -> dict:
    """Format epoch metrics for compact console logging."""
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()}


def fit(
    model,
    loaders,
    device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weight: torch.Tensor | None = None,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    checkpoint_path: str | Path | None = None,
):
    """Train a model for multiple epochs and optionally save the best checkpoint.

    Args:
        model: PyTorch model to train.
        loaders: Mapping containing at least ``"train"`` and ``"val"`` dataloaders.
        device: Torch device used for training and validation.
        epochs: Number of epochs to run.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay coefficient.
        class_weight: Optional per-class weights for cross-entropy loss.
        loss_name: Either ``"cross_entropy"`` or ``"focal"``.
        focal_gamma: Focusing parameter used when ``loss_name="focal"``.
        checkpoint_path: Optional path where the best validation-loss weights are
            saved.

    Returns:
        List of epoch metric dictionaries containing train and validation loss
        and accuracy.
    """
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = build_criterion(
        loss_name=loss_name,
        class_weight=class_weight.to(device) if class_weight is not None else None,
        focal_gamma=focal_gamma,
    )

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_loss_accuracy(model, loaders["val"], criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(format_epoch_metrics(row))

        if checkpoint_path is not None and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return history


def fit_adamw_two_stage(
    model,
    loaders,
    device,
    stage1_epochs: int = 10,
    stage2_epochs: int = 10,
    stage1_lr: float = 1e-4,
    stage2_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    class_weight: torch.Tensor | None = None,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    checkpoint_path: str | Path | None = None,
):
    """Train with AdamW in two stages, reloading the best stage-1 checkpoint.

    Stage 1 adapts the model at the main learning rate. Stage 2 reloads the best
    validation-loss checkpoint from stage 1 and continues with a lower learning
    rate, only overwriting the checkpoint if validation loss improves further.
    """
    history_stage1 = fit(
        model,
        loaders,
        device=device,
        epochs=stage1_epochs,
        lr=stage1_lr,
        weight_decay=weight_decay,
        class_weight=class_weight,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        checkpoint_path=checkpoint_path,
    )
    best_val_loss = min(row["val_loss"] for row in history_stage1)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=stage2_lr, weight_decay=weight_decay)
    criterion = build_criterion(
        loss_name=loss_name,
        class_weight=class_weight.to(device) if class_weight is not None else None,
        focal_gamma=focal_gamma,
    )

    history_stage2 = []
    for epoch in range(1, stage2_epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_loss_accuracy(model, loaders["val"], criterion, device)
        row = {
            "stage": "stage2_lower_lr",
            "epoch": stage1_epochs + epoch,
            "stage_epoch": epoch,
            "lr": stage2_lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history_stage2.append(row)
        print(format_epoch_metrics(row))

        if checkpoint_path is not None and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    normalized_stage1 = [
        {
            "stage": "stage1_adapt",
            "stage_epoch": row["epoch"],
            "lr": stage1_lr,
            **row,
        }
        for row in history_stage1
    ]
    return normalized_stage1 + history_stage2


def fit_sgd_schedule(
    model,
    loaders,
    device,
    epochs: int = 60,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    class_weight: torch.Tensor | None = None,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    grad_clip: float | None = 0.1,
    lr_decay_start: int = 20,
    lr_decay_every: int = 5,
    lr_decay_rate: float = 0.9,
    checkpoint_path: str | Path | None = None,
    initial_best_val_loss: float = float("inf"),
    epoch_offset: int = 0,
    stage_name: str = "stage",
):
    """Train with SGD, value clipping, LR decay, and best-checkpoint saving."""
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = build_criterion(
        loss_name=loss_name,
        class_weight=class_weight.to(device) if class_weight is not None else None,
        focal_gamma=focal_gamma,
    )

    history = []
    best_val_loss = initial_best_val_loss

    for local_epoch in range(1, epochs + 1):
        if local_epoch > lr_decay_start:
            frac = (local_epoch - lr_decay_start) // lr_decay_every
            current_lr = lr * (lr_decay_rate**frac)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
        else:
            current_lr = lr

        train_metrics = train_one_epoch_with_gradient_clip(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            grad_clip=grad_clip,
        )
        val_metrics = evaluate_loss_accuracy_with_crops(model, loaders["val"], criterion, device)

        row = {
            "stage": stage_name,
            "epoch": epoch_offset + local_epoch,
            "stage_epoch": local_epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(format_epoch_metrics(row))

        if checkpoint_path is not None and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return history, best_val_loss
