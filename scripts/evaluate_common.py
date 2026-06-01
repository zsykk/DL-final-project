from __future__ import annotations

import sys
import html
import os
import webbrowser
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


STATUS_ICONS = {
    "processing": "\u23f3",
    "done": "\u2705",
}

STATUS_FALLBACKS = {
    "processing": "[...]",
    "done": "[OK]",
}


def status(message: str, kind: str = "processing") -> None:
    """Print a short, human-readable status line with a consistent icon."""
    icon = STATUS_ICONS["done"] if kind == "done" else STATUS_ICONS["processing"]
    line = f"{icon} {message}"
    try:
        print(line)
    except UnicodeEncodeError:
        fallback = STATUS_FALLBACKS["done"] if kind == "done" else STATUS_FALLBACKS["processing"]
        print(f"{fallback} {message}")


def ensure_fer_test_loader(data_dir: Path, transform, batch_size: int, num_workers: int) -> DataLoader:
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected FER2013 test folder at {test_dir}")
    status(f"Loading FER2013 test images from {test_dir} ...", "processing")
    dataset = ImageFolder(test_dir, transform=transform)
    dataset.target_transform = _imagefolder_target_transform(dataset)
    status(f"Found {len(dataset)} test images across {len(dataset.classes)} folders.", "done")
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
    status(f"Loading checkpoint: {checkpoint_path}", "load")
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
    status(f"Saved classification report: {output_dir / f'{name}_classification_report.csv'}", "save")
    status(f"Saved confusion matrix: {output_dir / f'{name}_confusion_matrix.png'}", "save")


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    name: str,
    output_dir: Path,
    use_crops: bool = False,
) -> None:
    print()
    status(f"Starting evaluation: {name}", "processing")
    status(f"Device: {device}; batches: {len(loader)}; crop averaging: {use_crops}", "device")
    model = model.to(device)
    model.eval()
    if use_crops:
        y_true, y_pred = collect_predictions_with_crops(model, loader, device)
    else:
        y_true, y_pred = collect_predictions(model, loader, device)
    status(f"Finished inference for {name}; saving metrics and plots...", "save")
    save_evaluation_outputs(name, y_true, y_pred, output_dir)
    status(f"Finished evaluation: {name}", "done")


def build_transfer_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    transfer_model: str,
    classifier_hidden_layers: list[int] | None,
    classifier_dropout: float,
) -> torch.nn.Module:
    state_dict = load_state_dict(checkpoint_path, device)
    if classifier_hidden_layers is None:
        classifier_hidden_layers = infer_transfer_classifier_hidden_layers(state_dict, transfer_model)

    model = build_model(
        "transfer",
        transfer_model=transfer_model,
        pretrained=False,
        freeze_backbone=False,
        classifier_hidden_layers=classifier_hidden_layers,
        classifier_dropout=classifier_dropout,
    )
    model.load_state_dict(state_dict)
    return model


def infer_transfer_classifier_hidden_layers(state_dict: dict, transfer_model: str) -> list[int] | None:
    """Infer an MLP classifier head from transfer-learning checkpoint keys."""
    if transfer_model == "resnet18":
        if "fc.weight" in state_dict:
            return None
        prefix = "fc."
    elif transfer_model in {"mobilenet_v2", "efficientnet_b0"}:
        if "classifier.1.weight" in state_dict:
            return None
        prefix = "classifier.1."
    else:
        return None

    linear_layers = sorted(
        (
            (key, tuple(value.shape))
            for key, value in state_dict.items()
            if key.startswith(prefix) and key.endswith(".weight") and value.ndim == 2
        ),
        key=lambda item: [int(part) if part.isdigit() else part for part in item[0].split(".")],
    )
    if len(linear_layers) <= 1:
        return None
    hidden_layers = [shape[0] for _, shape in linear_layers[:-1]]
    status(f"Inferred classifier hidden layers from checkpoint: {hidden_layers}", "done")
    return hidden_layers


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


def _ordered_report_paths(output_dir: Path, experiment_order: list[str] | None = None) -> list[Path]:
    report_paths = list(output_dir.glob("*_classification_report.csv"))
    if not experiment_order:
        return sorted(report_paths)

    order_index = {name: index for index, name in enumerate(experiment_order)}
    report_paths = [
        path
        for path in report_paths
        if path.name.removesuffix("_classification_report.csv") in order_index
    ]
    return sorted(
        report_paths,
        key=lambda path: order_index[path.name.removesuffix("_classification_report.csv")],
    )


def save_comparison_table(output_dir: Path, experiment_order: list[str] | None = None) -> None:
    rows = []
    for report_path in _ordered_report_paths(output_dir, experiment_order):
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


def save_table_image(frame: pd.DataFrame, output_path: Path, title: str, max_rows: int = 16) -> None:
    display_frame = frame.head(max_rows).copy()
    for column in display_frame.columns:
        if pd.api.types.is_float_dtype(display_frame[column]):
            display_frame[column] = display_frame[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")

    fig_height = max(2.5, 0.42 * len(display_frame) + 1.2)
    fig_width = max(9, min(18, 1.2 * len(display_frame.columns) + 4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    table = ax.table(
        cellText=display_frame.values,
        colLabels=display_frame.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f3b52")
        else:
            cell.set_facecolor("#f6f7fb" if row % 2 == 0 else "white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_overview_assets(output_dir: Path, experiment_order: list[str] | None = None) -> None:
    summary_path = output_dir / "evaluation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        save_table_image(summary, output_dir / "summary_metrics_table.png", "Evaluation summary metrics")

    validation_rows = []
    heatmap_rows = []
    top_confusion_frames = []
    for report_path in _ordered_report_paths(output_dir, experiment_order):
        name = report_path.name.removesuffix("_classification_report.csv")
        report = pd.read_csv(report_path, index_col=0)
        history_path = PROJECT_ROOT / "results" / "metrics" / f"{name}_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            best_row = history.sort_values(["val_macro_f1", "val_loss"], ascending=[False, True]).iloc[0]
            validation_rows.append({"experiment": name, "best_val_macro_f1": best_row["val_macro_f1"]})

        for label in [EMOTION_LABELS[index] for index in range(len(EMOTION_LABELS))]:
            if label in report.index:
                heatmap_rows.append({"experiment": name, "emotion": label, "f1": report.loc[label, "f1-score"]})

        confusions_path = output_dir / f"{name}_top_confusions.csv"
        if confusions_path.exists():
            frame = pd.read_csv(confusions_path).head(8)
            frame.insert(0, "experiment", name)
            top_confusion_frames.append(frame)

    if validation_rows:
        validation = pd.DataFrame(validation_rows)
        plt.figure(figsize=(max(7, 0.55 * len(validation)), 4.5))
        ax = sns.barplot(data=validation, x="experiment", y="best_val_macro_f1", color="#4C78A8")
        ax.set_xlabel("")
        ax.set_ylabel("Best validation Macro F1")
        ax.set_ylim(0, min(1.0, max(0.75, validation["best_val_macro_f1"].max() + 0.08)))
        ax.set_xticklabels([label.replace("_", "\n") for label in validation["experiment"]], fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / "validation_macro_f1_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()

    if heatmap_rows:
        heatmap_frame = pd.DataFrame(heatmap_rows)
        pivot = heatmap_frame.pivot(index="experiment", columns="emotion", values="f1")
        plt.figure(figsize=(11, max(3.8, 0.55 * len(pivot))))
        ax = sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            annot_kws={"fontsize": 8},
        )
        ax.set_xlabel("Emotion")
        ax.set_ylabel("")
        ax.set_title("Per-class test F1")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "per_class_f1_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close()

    if top_confusion_frames:
        combined = pd.concat(top_confusion_frames, ignore_index=True)
        combined.to_csv(output_dir / "top_confusions.csv", index=False)
        save_table_image(
            combined,
            output_dir / "top_confusions_table.png",
            "Top Confusions",
            max_rows=min(24, len(combined)),
        )


def _relative_link(path: Path, base_dir: Path) -> str:
    return html.escape(os.path.relpath(path, base_dir).replace("\\", "/"))


def write_evaluation_dashboard(output_dir: Path, title: str, experiment_order: list[str] | None = None) -> Path:
    summary_html = ""
    summary_csv = output_dir / "evaluation_summary.csv"
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        summary_html = (
            "<section><h2>Evaluation Summary</h2>"
            + summary.round(4).to_html(index=False, classes="table", border=0)
            + "</section>"
        )

    comparison_html = ""
    plot_files = [
        path
        for path in [
            output_dir / "evaluation_f1_comparison.png",
            output_dir / "validation_macro_f1_comparison.png",
            output_dir / "per_class_f1_heatmap.png",
        ]
        if path.exists()
    ]
    for path in plot_files:
        label = path.stem.replace("_", " ").title()
        comparison_html += (
            f'<section><h2>{html.escape(label)}</h2><img src="{_relative_link(path, output_dir)}" '
            f'alt="{html.escape(label)}"></section>'
        )

    confusions_html = ""
    confusions_csv = output_dir / "top_confusions.csv"
    if confusions_csv.exists():
        confusions = pd.read_csv(confusions_csv)
        confusions_html = (
            "<section><h2>Top Confusions</h2>"
            + confusions.head(24).round(4).to_html(index=False, classes="table", border=0)
            + "</section>"
        )

    cards = []
    for report_path in _ordered_report_paths(output_dir, experiment_order):
        name = report_path.name.removesuffix("_classification_report.csv")
        report = pd.read_csv(report_path, index_col=0)
        saved_figure_path = PROJECT_ROOT / "results" / "figures" / f"{name}_confusion_matrix.png"
        matrix_path = saved_figure_path if saved_figure_path.exists() else output_dir / f"{name}_confusion_matrix.png"
        confusions_path = output_dir / f"{name}_top_confusions.csv"
        summary_rows = [row for row in ["macro avg", "weighted avg"] if row in report.index]
        emotion_rows = [row for row in [EMOTION_LABELS[index] for index in range(len(EMOTION_LABELS))] if row in report.index]
        parts = []
        if summary_rows:
            parts.append("<h4>Summary</h4>")
            parts.append(
                report.loc[summary_rows, ["precision", "recall", "f1-score", "support"]]
                .round(4)
                .to_html(classes="table compact", border=0)
            )
        if emotion_rows:
            parts.append("<h4>Per-Class F1</h4>")
            parts.append(report.loc[emotion_rows, ["f1-score", "support"]].round(4).to_html(classes="table compact", border=0))
        if confusions_path.exists():
            confusions = pd.read_csv(confusions_path).head(8)
            parts.append("<h4>Top Confusions</h4>")
            parts.append(confusions.round(4).to_html(index=False, classes="table compact", border=0))

        matrix_html = ""
        if matrix_path.exists():
            matrix_html = (
                f'<figure><img src="{_relative_link(matrix_path, output_dir)}" alt="{html.escape(name)} confusion matrix">'
                "<figcaption>Saved loss and confusion-matrix figure</figcaption></figure>"
            )

        cards.append(
            f"""
            <article class="experiment-card">
              <h3>{html.escape(name)}</h3>
              <div class="experiment-layout">
                <div>{''.join(parts)}</div>
                <div class="plot-stack">{matrix_html}</div>
              </div>
            </article>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f8fb; }}
    h1 {{ margin-bottom: 8px; }}
    section {{ background: white; border: 1px solid #d9dee8; border-radius: 8px; padding: 16px; margin: 18px 0; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    .table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .table th {{ background: #2f3b52; color: white; text-align: left; padding: 8px; }}
    .table td {{ border-bottom: 1px solid #e4e7ec; padding: 7px 8px; }}
    .table tr:nth-child(even) td {{ background: #f6f7fb; }}
    .experiment-card {{ border-top: 1px solid #e4e7ec; margin-top: 18px; padding-top: 16px; }}
    .experiment-layout {{ display: grid; grid-template-columns: minmax(320px, 1fr) minmax(320px, 0.9fr); gap: 18px; align-items: start; }}
    .plot-stack {{ display: grid; gap: 14px; }}
    .compact {{ font-size: 12px; }}
    h3 {{ overflow-wrap: anywhere; }}
    h4 {{ margin: 12px 0 6px; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; margin-top: 8px; color: #52606d; }}
    @media (max-width: 900px) {{ .experiment-layout {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Generated by checkpoint evaluation scripts.</p>
  {summary_html}
  {comparison_html}
  {confusions_html}
  <section><h2>Per-Experiment Details</h2>{''.join(cards)}</section>
</body>
</html>
"""
    dashboard_path = output_dir / "index.html"
    dashboard_path.write_text(html_text, encoding="utf-8")
    return dashboard_path


def open_dashboard(dashboard_path: Path) -> None:
    webbrowser.open(dashboard_path.resolve().as_uri())
