from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_common import (  # noqa: E402
    infer_baseline_model_from_state_dict,
    load_state_dict,
    open_dashboard,
    transfer_tencrop_transform,
    write_evaluation_dashboard,
)
from fer_project.models import build_model  # noqa: E402


FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FER_NAME_TO_IDX = {name: idx for idx, name in enumerate(FER_LABELS)}
SHARED_EVAL_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise"]
CKPLUS_NAME_TO_FER_NAME = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
}
EXCLUDED_CKPLUS_NAMES = {"contempt"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CKPLUS_EVALUATIONS = [
    {
        "group": "baseline",
        "name": "baseline_cnn_deeper_weighted_sampler_focal_sgd_plateau_lr_80epochs_ckplus_external",
        "checkpoint": "baseline_cnn_deeper_weighted_sampler_focal_sgd_plateau_lr_80epochs.pt",
        "model_kind": "baseline",
        "preprocessing": "baseline_grayscale",
    },
    {
        "group": "resnet18",
        "name": "resnet18_layer3_class_weights_focal_sgd_plateau_lr_ckplus_external_rgb",
        "checkpoint": "resnet18_layer3_class_weights_focal_sgd_plateau_lr.pt",
        "model_kind": "transfer",
        "transfer_model": "resnet18",
        "classifier_hidden_layers": [256],
        "classifier_dropout": 0.4,
        "preprocessing": "rgb",
    },
    {
        "group": "resnet18",
        "name": "resnet18_layer3_class_weights_focal_sgd_plateau_lr_ckplus_external_grayscale_to_3ch",
        "checkpoint": "resnet18_layer3_class_weights_focal_sgd_plateau_lr.pt",
        "model_kind": "transfer",
        "transfer_model": "resnet18",
        "classifier_hidden_layers": [256],
        "classifier_dropout": 0.4,
        "preprocessing": "grayscale_to_3ch",
    },
    {
        "group": "resnet18",
        "name": "resnet18_layer3_class_weights_focal_sgd_plateau_lr_crop_tencrop_ckplus_external_rgb",
        "checkpoint": "resnet18_layer3_class_weights_focal_sgd_plateau_lr_crop_tencrop.pt",
        "model_kind": "transfer",
        "transfer_model": "resnet18",
        "classifier_hidden_layers": [256],
        "classifier_dropout": 0.4,
        "preprocessing": "rgb_tencrop",
        "use_crops": True,
    },
    {
        "group": "resnet18",
        "name": "resnet18_layer3_class_weights_focal_sgd_plateau_lr_crop_tencrop_ckplus_external_grayscale_to_3ch",
        "checkpoint": "resnet18_layer3_class_weights_focal_sgd_plateau_lr_crop_tencrop.pt",
        "model_kind": "transfer",
        "transfer_model": "resnet18",
        "classifier_hidden_layers": [256],
        "classifier_dropout": 0.4,
        "preprocessing": "grayscale_tencrop",
        "use_crops": True,
    },
    {
        "group": "efficientnet_b0",
        "name": "efficientnet_b0_last_three_blocks_weighted_sampler_sgd_plateau_lr_ckplus_external",
        "checkpoint": "efficientnet_b0_last_three_blocks_weighted_sampler_sgd_plateau_lr.pt",
        "model_kind": "transfer",
        "transfer_model": "efficientnet_b0",
        "classifier_hidden_layers": [256],
        "classifier_dropout": 0.4,
        "preprocessing": "grayscale_to_3ch",
    },
    {
        "group": "efficientnet_b0",
        "name": "efficientnet_b0_last_three_blocks_weighted_sampler_sgd_plateau_lr_crop_tencrop_ckplus_external",
        "checkpoint": "efficientnet_b0_last_three_blocks_weighted_sampler_sgd_plateau_lr_crop_tencrop.pt",
        "model_kind": "transfer",
        "transfer_model": "efficientnet_b0",
        "classifier_hidden_layers": [256],
        "classifier_dropout": 0.4,
        "preprocessing": "grayscale_tencrop",
        "use_crops": True,
    },
]


class CKPlusExternalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), int(row["label"])


def normalize_name(value) -> str:
    return str(value).lower().strip().replace(" ", "_").replace("-", "_")


def infer_label_from_path(path: Path, include_neutral: bool) -> tuple[int | None, str]:
    for parent in [path.parent, *path.parents]:
        name = normalize_name(parent.name)
        if name in EXCLUDED_CKPLUS_NAMES:
            return None, "excluded"
        if name in CKPLUS_NAME_TO_FER_NAME:
            fer_name = CKPLUS_NAME_TO_FER_NAME[name]
            if fer_name == "neutral" and not include_neutral:
                return None, "excluded"
            return FER_NAME_TO_IDX[fer_name], fer_name
    return None, "unmapped"


def discover_ckplus_images(ckplus_root: Path, include_neutral: bool) -> pd.DataFrame:
    print(f"Scanning CK+ images under {ckplus_root} ...")
    image_paths = sorted(path for path in ckplus_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    rows = []
    for path in image_paths:
        label, label_name = infer_label_from_path(path, include_neutral)
        if label is None:
            continue
        rows.append({"path": path, "label": label, "label_name": label_name, "source_parent": path.parent.name})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No CK+ images were mapped under {ckplus_root}")
    print(f"Found {len(image_paths)} image files; mapped {len(frame)} images for evaluation.")
    return frame


def build_transform(preprocessing: str):
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if preprocessing == "baseline_grayscale":
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    if preprocessing == "rgb":
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), imagenet_norm])
    if preprocessing == "grayscale_to_3ch":
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                imagenet_norm,
            ]
        )
    if preprocessing == "grayscale_tencrop":
        return transfer_tencrop_transform()
    if preprocessing == "rgb_tencrop":
        resize = transforms.Resize((256, 256))
        to_tensor = transforms.ToTensor()
        return transforms.Compose(
            [
                resize,
                transforms.TenCrop(224),
                transforms.Lambda(lambda crops: torch.stack([imagenet_norm(to_tensor(crop)) for crop in crops])),
            ]
        )
    raise ValueError(f"Unknown preprocessing: {preprocessing}")


def build_model_for_spec(spec: dict, checkpoint_dir: Path, device: torch.device):
    checkpoint_path = checkpoint_dir / spec["checkpoint"]
    print(f"Loading {spec['group']} checkpoint for {spec['name']} ...")
    state_dict = load_state_dict(checkpoint_path, device)
    if spec["model_kind"] == "baseline":
        model = infer_baseline_model_from_state_dict(state_dict)
    else:
        model = build_model(
            "transfer",
            transfer_model=spec["transfer_model"],
            pretrained=False,
            freeze_backbone=False,
            classifier_hidden_layers=spec.get("classifier_hidden_layers"),
            classifier_dropout=spec.get("classifier_dropout", 0.35),
        )
        model.load_state_dict(state_dict)
    return model.to(device)


@torch.no_grad()
def collect_predictions(model, loader, device, use_crops: bool):
    y_true = []
    y_pred = []
    model.eval()
    for images, labels in loader:
        if use_crops:
            batch_size, num_crops, channels, height, width = images.shape
            images = images.view(-1, channels, height, width).to(device)
            logits = model(images).view(batch_size, num_crops, -1).mean(dim=1)
        else:
            images = images.to(device)
            logits = model(images)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred)


def top_confusions_from_matrix(cm, row_labels, column_labels, top_n=15) -> pd.DataFrame:
    rows = []
    for true_idx, true_name in enumerate(row_labels):
        support = cm[true_idx].sum()
        if support == 0:
            continue
        for pred_idx, pred_name in enumerate(column_labels):
            count = int(cm[true_idx, pred_idx])
            if true_name == pred_name or count == 0:
                continue
            rows.append(
                {
                    "true_emotion": true_name,
                    "predicted_emotion": pred_name,
                    "count": count,
                    "true_class_rate": count / support,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["true_emotion", "predicted_emotion", "count", "true_class_rate"])
    return pd.DataFrame(rows).sort_values(["count", "true_class_rate"], ascending=False).head(top_n)


def save_ckplus_outputs(name: str, y_true, y_pred, output_dir: Path, shared_eval_labels: list[str]) -> dict:
    shared_eval_indices = [FER_NAME_TO_IDX[label] for label in shared_eval_labels]
    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=shared_eval_indices,
            target_names=shared_eval_labels,
            output_dict=True,
            zero_division=0,
        )
    ).T
    macro_f1 = f1_score(y_true, y_pred, labels=shared_eval_indices, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=shared_eval_indices, average="weighted", zero_division=0)
    report.to_csv(output_dir / f"{name}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(FER_LABELS))))
    cm_rows = cm[shared_eval_indices, :]
    top_confusions = top_confusions_from_matrix(cm_rows, shared_eval_labels, FER_LABELS)
    top_confusions.to_csv(output_dir / f"{name}_top_confusions.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(cm_rows, cmap="Blues")
    ax.set_xticks(range(len(FER_LABELS)), FER_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(shared_eval_labels)), shared_eval_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} confusion matrix")
    for row in range(cm_rows.shape[0]):
        for column in range(cm_rows.shape[1]):
            ax.text(column, row, int(cm_rows[row, column]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = report.loc[["macro avg", "weighted avg"], ["precision", "recall", "f1-score", "support"]]
    per_class = report.loc[shared_eval_labels, ["precision", "recall", "f1-score", "support"]]
    print(f"\n===== {name} =====")
    print("CK+ test summary:")
    print(summary.round(4).to_string())
    print("\nCK+ per-class F1:")
    print(per_class[["f1-score", "support"]].round(4).to_string())
    print("\nCK+ top confusions:")
    print(top_confusions.head(15).round(4).to_string(index=False))
    print(f"\nSaved classification report: {output_dir / f'{name}_classification_report.csv'}")
    print(f"Saved confusion matrix: {output_dir / f'{name}_confusion_matrix.png'}")

    return {"experiment": name, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "support": len(y_true)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final baseline, ResNet18, and EfficientNet-B0 checkpoints on CK+.")
    parser.add_argument("--group", choices=["baseline", "resnet18", "efficientnet_b0", "all"], default="all")
    parser.add_argument("--ckplus-root", type=Path, default=Path("data/raw/ckplus"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/evaluated_results/ckplus_external"))
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--open", action="store_true", help="Open the generated CK+ dashboard(s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shared_eval_labels = SHARED_EVAL_LABELS + (["neutral"] if args.include_neutral else [])
    frame = discover_ckplus_images(args.ckplus_root, include_neutral=args.include_neutral)

    specs = [spec for spec in CKPLUS_EVALUATIONS if args.group == "all" or spec["group"] == args.group]
    print(f"Running {len(specs)} CK+ evaluation(s) on device {device}.")
    group_summaries = {}
    summaries = []
    for spec in specs:
        output_dir = args.output_dir / spec["group"]
        output_dir.mkdir(parents=True, exist_ok=True)
        frame["label_name"].value_counts().to_csv(output_dir / "ckplus_class_counts.csv")
        print(f"\nStarting CK+ evaluation: {spec['name']}")
        print(f"Preprocessing: {spec['preprocessing']}; crop averaging: {bool(spec.get('use_crops', False))}")
        dataset = CKPlusExternalDataset(frame, build_transform(spec["preprocessing"]))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"CK+ batches: {len(loader)}; batch size: {args.batch_size}")
        model = build_model_for_spec(spec, args.checkpoint_dir, device)
        y_true, y_pred = collect_predictions(model, loader, device, use_crops=bool(spec.get("use_crops", False)))
        print(f"Finished inference for {spec['name']}; saving metrics and plots...")
        summary = save_ckplus_outputs(spec["name"], y_true, y_pred, output_dir, shared_eval_labels)
        summary["group"] = spec["group"]
        summaries.append(summary)
        group_summaries.setdefault(spec["group"], []).append(summary)
        print(f"Evaluated {spec['name']}")

    for group, rows in group_summaries.items():
        group_output_dir = args.output_dir / group
        pd.DataFrame(rows).to_csv(group_output_dir / "ckplus_external_evaluation_summary.csv", index=False)
        pd.DataFrame(rows).to_csv(group_output_dir / "evaluation_summary.csv", index=False)
        dashboard_path = write_evaluation_dashboard(group_output_dir, f"CK+ {group} Evaluated Results")
        print(f"Dashboard: {dashboard_path}")
        if args.open:
            open_dashboard(dashboard_path)
    pd.DataFrame(summaries).to_csv(args.output_dir / "ckplus_external_evaluation_summary.csv", index=False)
    print(f"Wrote CK+ external evaluation results to {args.output_dir}")


if __name__ == "__main__":
    main()
