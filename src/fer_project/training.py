from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from .models import set_resnet18_trainable_layers


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


def _loss_f1_metrics(total_loss: float, total: int, y_true: list[int], y_pred: list[int]) -> dict:
    """Build loss/F1 metrics for an epoch or evaluation split."""
    return {
        "loss": total_loss / total,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def should_log_epoch(epoch: int, epochs: int, log_every: int = 5) -> bool:
    """Return whether an epoch should be printed to the console."""
    return epoch % log_every == 0 or epoch == epochs


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one supervised training epoch and report loss/F1 metrics.

    Args:
        model: PyTorch model to optimize.
        loader: Dataloader yielding image and label batches.
        criterion: Loss function used to compare logits with labels.
        optimizer: Optimizer responsible for parameter updates.
        device: Torch device where tensors and the model are placed.

    Returns:
        Dictionary containing mean loss, macro F1, and weighted F1 for the epoch.
    """
    model.train()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
        y_true.extend(labels.detach().cpu().tolist())

    return _loss_f1_metrics(total_loss, total, y_true, y_pred)


def train_one_epoch_with_gradient_clip(model, loader, criterion, optimizer, device, grad_clip: float | None = None):
    """Run one training epoch and optionally clip gradients by value."""
    model.train()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

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
        total += labels.size(0)
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
        y_true.extend(labels.detach().cpu().tolist())

    return _loss_f1_metrics(total_loss, total, y_true, y_pred)


@torch.no_grad()
def evaluate_loss_f1(model, loader, criterion, device):
    """Evaluate loss and F1 without updating model weights.

    Args:
        model: PyTorch model to evaluate.
        loader: Dataloader yielding image and label batches.
        criterion: Loss function used to compute evaluation loss.
        device: Torch device where tensors and the model are placed.

    Returns:
        Dictionary containing mean loss, macro F1, and weighted F1 for the dataloader.
    """
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.cpu().tolist())

    return _loss_f1_metrics(total_loss, total, y_true, y_pred)


@torch.no_grad()
def evaluate_loss_f1_with_crops(model, loader, criterion, device):
    """Evaluate loss/F1, averaging logits across TenCrop batches if present."""
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

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
        total += labels.size(0)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.cpu().tolist())

    return _loss_f1_metrics(total_loss, total, y_true, y_pred)


def format_epoch_metrics(row: dict) -> dict:
    """Format epoch metrics for compact console logging."""
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()}


def is_better_checkpoint(
    val_metrics: dict,
    best_val_macro_f1: float,
    best_val_loss: float,
) -> bool:
    """Select checkpoints by validation macro F1, using validation loss as a tie-breaker."""
    current_macro_f1 = val_metrics["macro_f1"]
    current_loss = val_metrics["loss"]
    return current_macro_f1 > best_val_macro_f1 or (
        current_macro_f1 == best_val_macro_f1 and current_loss < best_val_loss
    )


def best_checkpoint_metrics(history: list[dict]) -> tuple[float, float]:
    """Return the validation macro F1 and loss for the checkpoint-selected history row."""
    best_row = max(history, key=lambda row: (row["val_macro_f1"], -row["val_loss"]))
    return best_row["val_macro_f1"], best_row["val_loss"]


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
        checkpoint_path: Optional path where the best validation macro-F1 weights
            are saved, using validation loss as the tie-breaker.

    Returns:
        List of epoch metric dictionaries containing train/validation loss and F1.
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
    best_val_macro_f1 = float("-inf")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_loss_f1(model, loaders["val"], criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)
        if should_log_epoch(epoch, epochs):
            print(format_epoch_metrics(row))

        if checkpoint_path is not None and is_better_checkpoint(val_metrics, best_val_macro_f1, best_val_loss):
            best_val_macro_f1 = val_metrics["macro_f1"]
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
    validation macro-F1 checkpoint from stage 1 and continues with a lower
    learning rate, only overwriting the checkpoint if macro F1 improves, or if
    macro F1 ties and validation loss is lower.
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
    best_val_macro_f1, best_val_loss = best_checkpoint_metrics(history_stage1)

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
        val_metrics = evaluate_loss_f1(model, loaders["val"], criterion, device)
        row = {
            "stage": "stage2_lower_lr",
            "epoch": stage1_epochs + epoch,
            "stage_epoch": epoch,
            "lr": stage2_lr,
            "train_loss": train_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history_stage2.append(row)
        if should_log_epoch(epoch, stage2_epochs):
            print(format_epoch_metrics(row))

        if checkpoint_path is not None and is_better_checkpoint(val_metrics, best_val_macro_f1, best_val_loss):
            best_val_macro_f1 = val_metrics["macro_f1"]
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
    lr_decay_rate: float = 0.1,
    lr_plateau_patience: int = 5,
    lr_plateau_threshold: float = 1e-3,
    min_lr: float = 1e-6,
    checkpoint_path: str | Path | None = None,
    initial_best_val_macro_f1: float = float("-inf"),
    initial_best_val_loss: float = float("inf"),
    epoch_offset: int = 0,
    stage_name: str = "stage",
):
    """Train with SGD, value clipping, plateau LR decay, and best-checkpoint saving."""
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_decay_rate,
        patience=lr_plateau_patience,
        threshold=lr_plateau_threshold,
        threshold_mode="rel",
        min_lr=min_lr,
    )
    criterion = build_criterion(
        loss_name=loss_name,
        class_weight=class_weight.to(device) if class_weight is not None else None,
        focal_gamma=focal_gamma,
    )

    history = []
    best_val_macro_f1 = initial_best_val_macro_f1
    best_val_loss = initial_best_val_loss

    for local_epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch_with_gradient_clip(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            grad_clip=grad_clip,
        )
        val_metrics = evaluate_loss_f1_with_crops(model, loaders["val"], criterion, device)
        scheduler.step(val_metrics["loss"])

        row = {
            "stage": stage_name,
            "epoch": epoch_offset + local_epoch,
            "stage_epoch": local_epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)
        if should_log_epoch(local_epoch, epochs):
            print(format_epoch_metrics(row))

        if checkpoint_path is not None and is_better_checkpoint(val_metrics, best_val_macro_f1, best_val_loss):
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_loss = val_metrics["loss"]
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return history, best_val_loss


def fit_resnet18_gradual_unfreeze(
    model,
    loaders,
    device,
    unfreeze_stages: list[str] | tuple[str, ...] = ("fc", "layer4", "layer3", "layer2", "all"),
    epochs_per_stage: int = 5,
    lr_by_stage: dict[str, float] | None = None,
    weight_decay: float = 1e-4,
    class_weight: torch.Tensor | None = None,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    checkpoint_path: str | Path | None = None,
):
    """Gradually unfreeze ResNet18 stages and save the best validation macro-F1 model."""
    lr_by_stage = lr_by_stage or {
        "fc": 1e-3,
        "layer4": 5e-4,
        "layer3": 1e-4,
        "layer2": 5e-5,
        "layer1": 2e-5,
        "all": 1e-5,
    }
    model = model.to(device)
    criterion = build_criterion(
        loss_name=loss_name,
        class_weight=class_weight.to(device) if class_weight is not None else None,
        focal_gamma=focal_gamma,
    )
    history = []
    best_val_macro_f1 = float("-inf")
    best_val_loss = float("inf")
    global_epoch = 0

    for stage_name in unfreeze_stages:
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        set_resnet18_trainable_layers(model, stage_name)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr_by_stage.get(stage_name, 1e-5),
            weight_decay=weight_decay,
        )

        for stage_epoch in range(1, epochs_per_stage + 1):
            global_epoch += 1
            train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
            val_metrics = evaluate_loss_f1(model, loaders["val"], criterion, device)
            row = {
                "stage": f"unfreeze_{stage_name}",
                "epoch": global_epoch,
                "stage_epoch": stage_epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_weighted_f1": train_metrics["weighted_f1"],
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
            history.append(row)
            if should_log_epoch(stage_epoch, epochs_per_stage):
                print(format_epoch_metrics(row))

            if checkpoint_path is not None and is_better_checkpoint(val_metrics, best_val_macro_f1, best_val_loss):
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_val_loss = val_metrics["loss"]
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history, best_val_loss
