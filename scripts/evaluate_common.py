from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fer_project.data import (  # noqa: E402
    EMOTION_LABELS,
    _imagefolder_target_transform,
    baseline_eval_transform,
    baseline_tencrop_eval_transform,
    build_transforms,
    TransformConfig,
)
from fer_project.metrics import (  # noqa: E402
    collect_predictions,
    collect_predictions_with_crops,
    plot_confusion_matrix,
    top_confusions,
)
from fer_project.models import build_model  # noqa: E402


def ensure_fer_test_loader(data_dir: Path, transform, batch_size: int, num_workers: int) -> DataLoader:
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected FER2013 test folder at {test_dir}")
    print(f"Loading FER2013 test images from {test_dir} ...")
    dataset = ImageFolder(test_dir, transform=transform)
    dataset.target_transform = _imagefolder_target_transform(dataset)
    print(f"Found {len(dataset)} test images across {len(dataset.classes)} folders.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def transfer_eval_loader(data_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    transform = build_transforms(
        TransformConfig(image_size=224, channels=3, augment=False, imagenet_norm=True),
        input_is_pil=True,
    )
    return ensure_fer_test_loader(data_dir, transform, batch_size, num_workers)


def transfer_tencrop_transform(resize_size: int = 256, crop_size: int = 224):
    resize = transforms.Resize((resize_size, resize_size))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            resize,
            transforms.TenCrop(crop_size),
            transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(crop)) for crop in crops])),
        ]
    )


def transfer_tencrop_eval_loader(data_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    return ensure_fer_test_loader(data_dir, transfer_tencrop_transform(), batch_size, num_workers)


def baseline_eval_loader(data_dir: Path, batch_size: int, num_workers: int, ten_crop: bool) -> DataLoader:
    transform = (
        baseline_tencrop_eval_transform(image_size=48, crop_size=44)
        if ten_crop
        else baseline_eval_transform(image_size=48)
    )
    return ensure_fer_test_loader(data_dir, transform, batch_size, num_workers)


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def save_evaluation_outputs(
    name: str,
    y_true,
    y_pred,
    output_dir: Path,
    normalize_confusion: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_names = [EMOTION_LABELS[index] for index in range(len(EMOTION_LABELS))]
    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(target_names))),
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
    ).T
    report.to_csv(output_dir / f"{name}_classification_report.csv")

    confusions = top_confusions(y_true, y_pred, top_n=10)
    confusions.to_csv(output_dir / f"{name}_top_confusions.csv", index=False)

    ax = plot_confusion_matrix(
        y_true,
        y_pred,
        output_path=None,
        normalize=normalize_confusion,
        figsize=(7, 5.5),
    )
    ax.set_title(f"{name} confusion matrix")
    ax.figure.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(ax.figure)

    summary = report.loc[["macro avg", "weighted avg"], ["precision", "recall", "f1-score", "support"]]
    summary.to_csv(output_dir / f"{name}_summary.csv")

    per_class = report.loc[
        [EMOTION_LABELS[index] for index in range(len(EMOTION_LABELS))],
        ["precision", "recall", "f1-score", "support"],
    ]
    print(f"\n===== {name} =====")
    print("Test summary:")
    print(summary.round(4).to_string())
    print("\nPer-class F1:")
    print(per_class[["f1-score", "support"]].round(4).to_string())
    print("\nTop confusions:")
    print(confusions.head(10).round(4).to_string(index=False))
    print(f"\nSaved classification report: {output_dir / f'{name}_classification_report.csv'}")
    print(f"Saved confusion matrix: {output_dir / f'{name}_confusion_matrix.png'}")


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    name: str,
    output_dir: Path,
    use_crops: bool = False,
) -> None:
    print(f"\nStarting evaluation: {name}")
    print(f"Device: {device}; batches: {len(loader)}; crop averaging: {use_crops}")
    model = model.to(device)
    model.eval()
    if use_crops:
        y_true, y_pred = collect_predictions_with_crops(model, loader, device)
    else:
        y_true, y_pred = collect_predictions(model, loader, device)
    print(f"Finished inference for {name}; saving metrics and plots...")
    save_evaluation_outputs(name, y_true, y_pred, output_dir)
    print(f"Finished evaluation: {name}")


def build_transfer_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    transfer_model: str,
    classifier_hidden_layers: list[int] | None,
    classifier_dropout: float,
) -> torch.nn.Module:
    model = build_model(
        "transfer",
        transfer_model=transfer_model,
        pretrained=False,
        freeze_backbone=False,
        classifier_hidden_layers=classifier_hidden_layers,
        classifier_dropout=classifier_dropout,
    )
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    return model


def linears_from_state_dict(state_dict: dict) -> list[tuple[str, tuple[int, int]]]:
    return [
        (key, tuple(value.shape))
        for key, value in state_dict.items()
        if key.startswith("classifier.") and key.endswith(".weight") and value.ndim == 2
    ]


def infer_baseline_model_from_state_dict(state_dict: dict) -> torch.nn.Module:
    conv_shapes = [
        tuple(value.shape)
        for key, value in state_dict.items()
        if key.startswith("features.") and key.endswith(".weight") and value.ndim == 4
    ]
    out_channels = [shape[0] for shape in conv_shapes]
    feature_channels = []
    block_depths = []
    for channels in out_channels:
        if not feature_channels or feature_channels[-1] != channels:
            feature_channels.append(channels)
            block_depths.append(1)
        else:
            block_depths[-1] += 1

    classifier_layers = linears_from_state_dict(state_dict)
    hidden_layers = [shape[0] for _, shape in classifier_layers[:-1]]
    final_out = classifier_layers[-1][1][0]
    if final_out != 7:
        raise ValueError(f"Expected final classifier output size 7, got {final_out}")

    dropout_candidates = []
    if len(feature_channels) == 3:
        dropout_candidates = [(0.15, 0.20, 0.0), (0.0, 0.0, 0.0)]
    elif len(feature_channels) == 4:
        dropout_candidates = [
            (0.15, 0.20, 0.25, 0.0),
            (0.15, 0.20, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ]
    else:
        dropout_candidates = [tuple(0.0 for _ in feature_channels)]

    for dropout2d in dropout_candidates:
        model = build_model(
            "baseline_cnn",
            baseline_feature_channels=feature_channels,
            baseline_block_depths=block_depths,
            baseline_dropout2d=dropout2d,
            baseline_classifier_hidden_layers=hidden_layers,
        )
        model_state = model.state_dict()
        if model_state.keys() == state_dict.keys() and all(
            model_state[key].shape == value.shape for key, value in state_dict.items()
        ):
            model.load_state_dict(state_dict)
            return model

    raise ValueError("Could not infer matching BaselineCNN architecture from checkpoint.")


def save_comparison_table(output_dir: Path) -> None:
    rows = []
    for report_path in sorted(output_dir.glob("*_classification_report.csv")):
        name = report_path.name.removesuffix("_classification_report.csv")
        report = pd.read_csv(report_path, index_col=0)
        rows.append(
            {
                "experiment": name,
                "macro_f1": report.loc["macro avg", "f1-score"],
                "weighted_f1": report.loc["weighted avg", "f1-score"],
                "support": report.loc["weighted avg", "support"],
            }
        )
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)

    plt.figure(figsize=(max(7, 0.55 * len(summary)), 4.5))
    plot_frame = summary.melt(id_vars="experiment", value_vars=["macro_f1", "weighted_f1"])
    ax = sns.barplot(data=plot_frame, x="experiment", y="value", hue="variable")
    ax.set_xlabel("")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, min(1.0, max(0.75, plot_frame["value"].max() + 0.08)))
    ax.set_xticklabels([label.replace("_", "\n") for label in summary["experiment"]], fontsize=7)
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_f1_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
