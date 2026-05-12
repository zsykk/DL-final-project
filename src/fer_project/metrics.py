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
    model.eval()
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def classification_metrics(y_true, y_pred) -> dict:
    target_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "report": classification_report(y_true, y_pred, target_names=target_names, output_dict=True),
    }


def save_classification_report(y_true, y_pred, output_csv: str | Path) -> pd.DataFrame:
    target_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    frame = pd.DataFrame(report).transpose()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv)
    return frame


def plot_confusion_matrix(y_true, y_pred, output_path: str | Path | None = None, normalize: bool = True):
    labels = list(EMOTION_LABELS.keys())
    names = [EMOTION_LABELS[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180, bbox_inches="tight")

    return plt.gca()
