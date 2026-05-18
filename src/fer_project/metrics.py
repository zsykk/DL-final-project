from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data import EMOTION_LABELS


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Collect true and predicted labels from a model over a dataloader.

    Args:
        model: PyTorch model used for inference.
        loader: Dataloader yielding image and label batches.
        device: Torch device where image tensors and the model are placed.

    Returns:
        Tuple of NumPy arrays ``(y_true, y_pred)``.
    """
    model.eval()
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())

    return np.array(y_true), np.array(y_pred)


@torch.no_grad()
def collect_predictions_with_crops(model, loader, device):
    """Collect predictions, averaging logits across TenCrop batches if present."""
    model.eval()
    y_true = []
    y_pred = []

    for images, labels in loader:
        if images.ndim == 5:
            batch_size, num_crops, channels, height, width = images.shape
            images = images.view(-1, channels, height, width).to(device)
            logits = model(images).view(batch_size, num_crops, -1).mean(dim=1)
        else:
            images = images.to(device)
            logits = model(images)
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def classification_metrics(y_true, y_pred) -> dict:
    """Compute FER2013 F1 scores and a per-class classification report.

    Args:
        y_true: Ground-truth integer emotion labels.
        y_pred: Predicted integer emotion labels.

    Returns:
        Dictionary containing macro F1, weighted F1, and a scikit-learn report.
    """
    target_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_classification_report(y_true, y_pred, output_path: str | Path) -> pd.DataFrame:
    """Save a per-class classification report table.

    Args:
        y_true: Ground-truth integer emotion labels.
        y_pred: Predicted integer emotion labels.
        output_path: Destination path for the report table.

    Returns:
        DataFrame representation of the saved classification report.
    """
    target_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).transpose()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)
    return frame


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path: str | Path | None = None,
    normalize: bool = True,
    figsize: tuple[float, float] = (6, 4.5),
    ax=None,
):
    """Plot a FER2013 confusion matrix and optionally save it to disk.

    Args:
        y_true: Ground-truth integer emotion labels.
        y_pred: Predicted integer emotion labels.
        output_path: Optional image path where the plot is saved.
        normalize: Whether to row-normalize counts into per-class proportions.

    Returns:
        Matplotlib axes containing the rendered heatmap.
    """
    labels = list(EMOTION_LABELS.keys())
    names = [EMOTION_LABELS[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(output_path, dpi=180, bbox_inches="tight")

    return ax


def plot_loss_curve(history: pd.DataFrame, title: str):
    """Plot train and validation loss from an experiment history frame."""
    ax = history.plot(
        x="epoch",
        y=["train_loss", "val_loss"],
        marker="o",
        figsize=(5, 3),
        title=title,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.tight_layout()
    return ax


def plot_training_and_confusion(
    history: pd.DataFrame,
    y_true,
    y_pred,
    title: str,
    confusion_output_path: str | Path | None = None,
):
    """Plot loss curve and confusion matrix in one compact row."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    history.plot(
        x="epoch",
        y=["train_loss", "val_loss"],
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    plot_confusion_matrix(
        y_true,
        y_pred,
        output_path=confusion_output_path,
        figsize=(4.8, 3.8),
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return fig, axes


def top_confusions(y_true, y_pred, top_n: int = 10) -> pd.DataFrame:
    """Return the most common off-diagonal confusion pairs.

    Args:
        y_true: Ground-truth integer emotion labels.
        y_pred: Predicted integer emotion labels.
        top_n: Number of confusion pairs to report.

    Returns:
        DataFrame with true class, predicted class, raw count, and within-class
        confusion rate.
    """
    labels = list(EMOTION_LABELS.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    columns = ["true_emotion", "predicted_emotion", "count", "true_class_rate"]
    rows = []

    for true_label in labels:
        true_count = cm[true_label].sum()
        for pred_label in labels:
            if true_label == pred_label:
                continue
            count = int(cm[true_label, pred_label])
            if count == 0:
                continue
            rows.append(
                {
                    "true_emotion": EMOTION_LABELS[true_label],
                    "predicted_emotion": EMOTION_LABELS[pred_label],
                    "count": count,
                    "true_class_rate": count / max(true_count, 1),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["count", "true_class_rate"], ascending=False).head(top_n)
