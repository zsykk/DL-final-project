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

from fer_project.models import build_model  # noqa: E402


FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FER_NAME_TO_IDX = {name: idx for idx, name in enumerate(FER_LABELS)}
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
DEFAULT_CHECKPOINT = Path("results/checkpoints/resnet18_layer3_class_weights_focal_sgd_plateau_lr.pt")
DEFAULT_OUTPUT_PREFIX = "resnet18_layer3_class_weights_focal_sgd_plateau_lr_ckplus_external"


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
    return frame


@torch.no_grad()
def collect_predictions(model, loader, device):
    y_true = []
    y_pred = []
    model.eval()
    for images, labels in loader:
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


def evaluate_ckplus_variant(
    model,
    loader,
    device,
    preprocessing_name: str,
    output_prefix: str,
    output_dir: Path,
    shared_eval_labels: list[str],
) -> dict:
    shared_eval_indices = [FER_NAME_TO_IDX[name] for name in shared_eval_labels]
    y_true, y_pred = collect_predictions(model, loader, device)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{output_prefix}_{preprocessing_name}"
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

    return {
        "preprocessing": preprocessing_name,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "support": len(y_true),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the final ResNet18 checkpoint on CK+ external data.")
    parser.add_argument("--ckplus-root", type=Path, default=Path("data/raw/ckplus"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/evaluated_results/ckplus_external"))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_eval_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise"]
    if args.include_neutral:
        shared_eval_labels.append("neutral")

    frame = discover_ckplus_images(args.ckplus_root, include_neutral=args.include_neutral)
    frame["label_name"].value_counts().to_csv(args.output_dir / "ckplus_class_counts.csv")

    rgb_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    grayscale_to_3ch_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = build_model(
        "transfer",
        transfer_model="resnet18",
        pretrained=False,
        freeze_backbone=False,
        classifier_hidden_layers=[256],
        classifier_dropout=0.4,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    summaries = []
    for preprocessing_name, transform in [
        ("rgb", rgb_transform),
        ("grayscale_to_3ch", grayscale_to_3ch_transform),
    ]:
        dataset = CKPlusExternalDataset(frame, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        summaries.append(
            evaluate_ckplus_variant(
                model,
                loader,
                device,
                preprocessing_name,
                args.output_prefix,
                args.output_dir,
                shared_eval_labels,
            )
        )
        print(f"Evaluated CK+ {preprocessing_name}")

    pd.DataFrame(summaries).to_csv(args.output_dir / f"{args.output_prefix}_rgb_vs_grayscale_comparison.csv", index=False)
    print(f"Wrote evaluated CK+ results to {args.output_dir}")


if __name__ == "__main__":
    main()
