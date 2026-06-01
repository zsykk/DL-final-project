from __future__ import annotations

import argparse
import html
import math
import shutil
import webbrowser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EMOTION_ROWS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FINAL_BASELINE_EXPERIMENTS = [
    "baseline_cnn_aug",
    "baseline_cnn_aug_weighted_sampler",
    "baseline_cnn_aug_weighted_sampler_cross_entropy_sgd_plateau_lr",
    "baseline_cnn_crop_tencrop_weighted_sampler_cross_entropy_sgd_plateau_lr",
    "baseline_cnn_crop_tencrop_weighted_sampler_cross_entropy_sgd_plateau_lr_80epochs",
]
FINAL_RESNET18_EXPERIMENTS = [
    "resnet18_feature_extract_fc",
    "resnet18_unfreeze_layer4",
    "resnet18_unfreeze_layer3",
    "resnet18_layer3_class_weights",
    "resnet18_layer3_class_weights_focal",
    "resnet18_layer3_class_weights_focal_sgd_plateau_lr",
    "resnet18_layer3_class_weights_focal_sgd_plateau_lr_crop_tencrop",
]
FINAL_EFFICIENTNET_B0_EXPERIMENTS = [
    "efficientnet_b0_feature_extract_classifier",
    "efficientnet_b0_unfreeze_last_block",
    "efficientnet_b0_unfreeze_last_blocks",
    "efficientnet_b0_unfreeze_last_three_blocks",
    "efficientnet_b0_last_three_blocks_weighted_sampler",
    "efficientnet_b0_last_three_blocks_weighted_sampler_sgd_plateau_lr",
    "efficientnet_b0_last_three_blocks_weighted_sampler_sgd_plateau_lr_crop_tencrop",
]
FINAL_CKPLUS_EXTERNAL_EXPERIMENTS = [
    "baseline_cnn_crop_tencrop_weighted_sampler_cross_entropy_sgd_plateau_lr_80epochs_ckplus_external",
    "resnet18_layer3_class_weights_focal_sgd_plateau_lr_crop_tencrop_ckplus_external_grayscale_to_3ch",
    "efficientnet_b0_last_three_blocks_weighted_sampler_sgd_plateau_lr_crop_tencrop_ckplus_external",
]


def is_excluded_name(name: str) -> bool:
    lowered = name.lower()
    return "rafdb" in lowered or lowered.startswith("debug_") or lowered.startswith("panda_")


def experiment_name_from_metric_file(path: Path) -> str | None:
    suffixes = [
        "_classification_report.csv",
        "_history.csv",
        "_top_confusions.csv",
    ]
    for suffix in suffixes:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return None


def discover_experiments(metrics_dir: Path, group: str) -> list[str]:
    names = {
        name
        for path in metrics_dir.glob("*.csv")
        if (name := experiment_name_from_metric_file(path)) is not None
    }

    if group == "baseline":
        selected = [name for name in FINAL_BASELINE_EXPERIMENTS if name in names]
    elif group == "resnet18":
        selected = [name for name in FINAL_RESNET18_EXPERIMENTS if name in names]
    elif group == "efficientnet_b0":
        selected = [name for name in FINAL_EFFICIENTNET_B0_EXPERIMENTS if name in names]
    elif group == "ckplus_external":
        selected = [name for name in FINAL_CKPLUS_EXTERNAL_EXPERIMENTS if name in names]
    else:
        raise ValueError(f"Unknown group: {group}")

    return [name for name in selected if not is_excluded_name(name)]


def ckplus_model_group(name: str) -> str:
    if name.startswith("baseline_cnn_"):
        return "baseline"
    if name.startswith("resnet18_"):
        return "resnet18"
    if name.startswith("efficientnet_b0_"):
        return "efficientnet_b0"
    return "other"


def best_history_row(history: pd.DataFrame) -> pd.Series:
    return history.sort_values(["val_macro_f1", "val_loss"], ascending=[False, True]).iloc[0]


def short_label(name: str, group: str) -> str:
    prefixes = {
        "baseline": "baseline_cnn_",
        "resnet18": "resnet18_",
        "efficientnet_b0": "efficientnet_b0_",
    }
    label = name.removeprefix(prefixes.get(group, ""))
    label = label.replace("layer3_", "l3_")
    label = label.replace("weighted_sampler", "sampler")
    label = label.replace("class_weights", "weights")
    label = label.replace("feature_extract", "features")
    label = label.replace("classification_report", "report")
    label = label.replace("_", "\n")
    return label


def read_report(report_path: Path) -> pd.DataFrame | None:
    if not report_path.exists():
        return None
    return pd.read_csv(report_path, index_col=0)


def build_summary(metrics_dir: Path, experiment_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in experiment_names:
        row = {"experiment": name}
        history_path = metrics_dir / f"{name}_history.csv"
        report_path = metrics_dir / f"{name}_classification_report.csv"

        if history_path.exists():
            history = pd.read_csv(history_path)
            best_row = best_history_row(history)
            row.update(
                {
                    "best_epoch": best_row.get("epoch"),
                    "best_val_macro_f1": best_row.get("val_macro_f1"),
                    "best_val_weighted_f1": best_row.get("val_weighted_f1"),
                    "best_val_loss": best_row.get("val_loss"),
                }
            )

        report = read_report(report_path)
        if report is not None:
            if "macro avg" in report.index:
                row["test_macro_precision"] = report.loc["macro avg", "precision"]
                row["test_macro_recall"] = report.loc["macro avg", "recall"]
                row["test_macro_f1"] = report.loc["macro avg", "f1-score"]
            if "weighted avg" in report.index:
                row["test_weighted_f1"] = report.loc["weighted avg", "f1-score"]
            row["test_support"] = report["support"].max()

        rows.append(row)
    return pd.DataFrame(rows)


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


def save_metric_barplot(summary: pd.DataFrame, group: str, output_path: Path) -> None:
    metric_columns = [column for column in ["test_macro_f1", "test_weighted_f1"] if column in summary.columns]
    if not metric_columns:
        return
    plot_frame = summary[["experiment", *metric_columns]].melt(
        id_vars="experiment",
        var_name="metric",
        value_name="score",
    )
    plot_frame = plot_frame.dropna(subset=["score"])
    if plot_frame.empty:
        return

    plt.figure(figsize=(max(9, 0.55 * summary.shape[0]), 5))
    ax = sns.barplot(
        data=plot_frame,
        x="experiment",
        y="score",
        hue="metric",
        palette=["#2b6cb0", "#38a169"],
    )
    ax.set_title(f"{group} test F1 comparison")
    ax.set_xlabel("")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, min(1.0, max(0.75, plot_frame["score"].max() + 0.08)))
    labels = [short_label(name, group) for name in summary["experiment"]]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_validation_plot(summary: pd.DataFrame, group: str, output_path: Path) -> None:
    if "best_val_macro_f1" not in summary.columns or summary["best_val_macro_f1"].dropna().empty:
        return
    plot_frame = summary.dropna(subset=["best_val_macro_f1"]).copy()
    plt.figure(figsize=(max(9, 0.55 * plot_frame.shape[0]), 4.8))
    ax = sns.barplot(data=plot_frame, x="experiment", y="best_val_macro_f1", color="#805ad5")
    ax.set_title(f"{group} best validation macro F1")
    ax.set_xlabel("")
    ax.set_ylabel("Validation macro F1")
    ax.set_ylim(0, min(1.0, max(0.75, plot_frame["best_val_macro_f1"].max() + 0.08)))
    labels = [short_label(name, group) for name in plot_frame["experiment"]]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_per_class_heatmap(metrics_dir: Path, experiment_names: list[str], group: str, output_path: Path) -> None:
    rows = []
    for name in experiment_names:
        report = read_report(metrics_dir / f"{name}_classification_report.csv")
        if report is None:
            continue
        for emotion in EMOTION_ROWS:
            if emotion in report.index:
                rows.append(
                    {
                        "experiment": short_label(name, group).replace("\n", " "),
                        "emotion": emotion,
                        "f1": report.loc[emotion, "f1-score"],
                    }
                )

    plot_frame = pd.DataFrame(rows)
    if plot_frame.empty:
        return

    heatmap_frame = plot_frame.pivot(index="experiment", columns="emotion", values="f1")
    heatmap_frame = heatmap_frame[[emotion for emotion in EMOTION_ROWS if emotion in heatmap_frame.columns]]
    plt.figure(figsize=(9.5, max(3.5, 0.42 * len(heatmap_frame))))
    ax = sns.heatmap(heatmap_frame, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title(f"{group} per-class test F1")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_training_curves(metrics_dir: Path, experiment_names: list[str], group: str, output_path: Path) -> None:
    histories = []
    for name in experiment_names:
        history_path = metrics_dir / f"{name}_history.csv"
        if history_path.exists():
            histories.append((name, pd.read_csv(history_path)))
    if not histories:
        return

    cols = 2
    rows = math.ceil(len(histories) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.5, rows * 3.0)), squeeze=False)
    for ax, (name, history) in zip(axes.flatten(), histories):
        if "epoch" not in history.columns:
            ax.axis("off")
            continue
        if "train_loss" in history.columns:
            ax.plot(history["epoch"], history["train_loss"], label="train loss", color="#2b6cb0")
        if "val_loss" in history.columns:
            ax.plot(history["epoch"], history["val_loss"], label="val loss", color="#e53e3e")
        ax.set_title(short_label(name, group).replace("\n", " "), fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes.flatten()[len(histories) :]:
        ax.axis("off")
    fig.suptitle(f"{group} training curves", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_individual_loss_curves(metrics_dir: Path, experiment_names: list[str], output_dir: Path) -> None:
    loss_dir = output_dir / "loss_curves"
    loss_dir.mkdir(parents=True, exist_ok=True)
    for name in experiment_names:
        history_path = metrics_dir / f"{name}_history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        if "epoch" not in history.columns or not {"train_loss", "val_loss"}.intersection(history.columns):
            continue

        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        if "train_loss" in history.columns:
            ax.plot(history["epoch"], history["train_loss"], marker="o", linewidth=1.5, markersize=3, label="train loss")
        if "val_loss" in history.columns:
            ax.plot(history["epoch"], history["val_loss"], marker="o", linewidth=1.5, markersize=3, label="val loss")
        ax.set_title(f"{name} loss", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(loss_dir / f"{name}_loss_curve.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_top_confusions(metrics_dir: Path, experiment_names: list[str], output_dir: Path) -> None:
    frames = []
    for name in experiment_names:
        path = metrics_dir / f"{name}_top_confusions.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "experiment", name)
            frames.append(frame)
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "top_confusions.csv", index=False)
    save_table_image(
        combined,
        output_dir / "top_confusions_table.png",
        "Top Confusions",
        max_rows=min(24, len(combined)),
    )


def copy_confusion_matrices(
    figures_dir: Path,
    experiment_names: list[str],
    output_dir: Path,
    preferred_dirs: list[Path] | None = None,
) -> None:
    matrix_dir = output_dir / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for name in experiment_names:
        filename = f"{name}_confusion_matrix.png"
        source = next(
            (candidate for directory in preferred_dirs or [] if (candidate := directory / filename).exists()),
            figures_dir / filename,
        )
        if source.exists():
            shutil.copy2(source, matrix_dir / source.name)


def copy_source_reports(metrics_dir: Path, experiment_names: list[str], output_dir: Path) -> None:
    source_dir = output_dir / "source_reports"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in experiment_names:
        for suffix in ["classification_report.csv", "top_confusions.csv"]:
            source = metrics_dir / f"{name}_{suffix}"
            if source.exists():
                shutil.copy2(source, source_dir / source.name)


def relative_link(path: Path, base_dir: Path) -> str:
    return html.escape(path.relative_to(base_dir).as_posix())


def write_dashboard(output_dir: Path, title: str) -> Path:
    summary_html = ""
    summary_csv = output_dir / "summary_metrics.csv"
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        summary_html = (
            "<section><h2>Summary Metrics</h2>"
            + summary.round(4).to_html(index=False, classes="table", border=0)
            + "</section>"
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

    plot_files = [
        path
        for path in [
            output_dir / "test_f1_comparison.png",
            output_dir / "validation_macro_f1_comparison.png",
            output_dir / "per_class_f1_heatmap.png",
        ]
        if path.exists()
    ]
    confusion_files = sorted((output_dir / "confusion_matrices").glob("*.png")) if (output_dir / "confusion_matrices").exists() else []
    experiment_cards = []
    for matrix_path in confusion_files:
        name = matrix_path.name.removesuffix("_confusion_matrix.png")
        report_path = output_dir / "source_reports" / f"{name}_classification_report.csv"
        confusions_path = output_dir / "source_reports" / f"{name}_top_confusions.csv"
        loss_path = output_dir / "loss_curves" / f"{name}_loss_curve.png"
        detail_parts = []
        if report_path.exists():
            report = pd.read_csv(report_path, index_col=0)
            summary_rows = [row for row in ["macro avg", "weighted avg"] if row in report.index]
            emotion_rows = [row for row in EMOTION_ROWS if row in report.index]
            if summary_rows:
                detail_parts.append("<h4>Summary</h4>")
                detail_parts.append(
                    report.loc[summary_rows, ["precision", "recall", "f1-score", "support"]]
                    .round(4)
                    .to_html(classes="table compact", border=0)
                )
            if emotion_rows:
                detail_parts.append("<h4>Per-Class F1</h4>")
                detail_parts.append(
                    report.loc[emotion_rows, ["f1-score", "support"]]
                    .round(4)
                    .to_html(classes="table compact", border=0)
                )
        if confusions_path.exists():
            confusions = pd.read_csv(confusions_path).head(8)
            detail_parts.append("<h4>Top Confusions</h4>")
            detail_parts.append(confusions.round(4).to_html(index=False, classes="table compact", border=0))
        experiment_cards.append(
            f"""
            <article class="experiment-card">
              <h3>{html.escape(name)}</h3>
              <div class="experiment-layout">
                <div>{''.join(detail_parts)}</div>
                <div class="plot-stack">
                  {f'<figure><img src="{relative_link(loss_path, output_dir)}" alt="{html.escape(name)} loss curve"><figcaption>Training and validation loss</figcaption></figure>' if loss_path.exists() and not matrix_path.exists() else ''}
                  <figure>
                    <img src="{relative_link(matrix_path, output_dir)}" alt="{html.escape(name)} confusion matrix">
                    <figcaption>Saved loss and confusion-matrix figure</figcaption>
                  </figure>
                </div>
              </div>
            </article>
            """
        )

    cards = []
    for path in plot_files:
        label = path.stem.replace("_", " ").title()
        cards.append(
            f'<section><h2>{html.escape(label)}</h2><img src="{relative_link(path, output_dir)}" alt="{html.escape(label)}"></section>'
        )
    experiment_section = ""
    if experiment_cards:
        experiment_section = "<section><h2>Per-Experiment Details</h2>" + "".join(experiment_cards) + "</section>"

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
    .matrix-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .experiment-card {{ border-top: 1px solid #e4e7ec; margin-top: 18px; padding-top: 16px; }}
    .experiment-layout {{ display: grid; grid-template-columns: minmax(320px, 1fr) minmax(320px, 0.9fr); gap: 18px; align-items: start; }}
    .plot-stack {{ display: grid; gap: 14px; }}
    .compact {{ font-size: 12px; }}
    h3 {{ overflow-wrap: anywhere; }}
    h4 {{ margin: 12px 0 6px; }}
    @media (max-width: 900px) {{ .experiment-layout {{ grid-template-columns: 1fr; }} }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; margin-top: 8px; color: #52606d; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Generated from saved result files.</p>
  {summary_html}
  {''.join(cards)}
  {confusions_html}
  {experiment_section}
</body>
</html>
"""
    dashboard_path = output_dir / "index.html"
    dashboard_path.write_text(html_text, encoding="utf-8")
    return dashboard_path


def print_saved_experiment_details(name: str, metrics_dir: Path, figures_dir: Path) -> None:
    history_path = metrics_dir / f"{name}_history.csv"
    report_path = metrics_dir / f"{name}_classification_report.csv"
    confusions_path = metrics_dir / f"{name}_top_confusions.csv"
    figure_path = figures_dir / f"{name}_confusion_matrix.png"

    print(f"\n===== {name} =====")

    if history_path.exists():
        history = pd.read_csv(history_path)
        best_row = best_history_row(history)
        columns = [
            column
            for column in ["epoch", "stage", "stage_epoch", "lr", "val_loss", "val_macro_f1", "val_weighted_f1"]
            if column in best_row.index
        ]
        print("Best validation row:")
        print(best_row[columns].to_frame().T.round(4).to_string(index=False))
        print("\nRecent training history:")
        recent_columns = [
            column
            for column in ["epoch", "stage", "stage_epoch", "lr", "train_loss", "val_loss", "val_macro_f1"]
            if column in history.columns
        ]
        print(history[recent_columns].tail().round(4).to_string(index=False))

    if report_path.exists():
        report = pd.read_csv(report_path, index_col=0)
        summary_rows = [row for row in ["macro avg", "weighted avg"] if row in report.index]
        if summary_rows:
            print("\nTest summary:")
            print(report.loc[summary_rows, ["precision", "recall", "f1-score", "support"]].round(4).to_string())
        emotion_rows = [row for row in EMOTION_ROWS if row in report.index]
        if emotion_rows:
            print("\nPer-class F1:")
            print(report.loc[emotion_rows, ["f1-score", "support"]].round(4).to_string())

    if confusions_path.exists():
        print("\nTop confusions:")
        print(pd.read_csv(confusions_path).head(10).round(4).to_string(index=False))

    if figure_path.exists():
        print(f"\nSaved confusion matrix: {figure_path}")


def reproduce_group(group: str, results_dir: Path, output_root: Path) -> None:
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    ckplus_evaluated_dir = output_root.parent / "evaluated_results" / "ckplus_external"
    output_dir = output_root / group
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_names = discover_experiments(metrics_dir, group)
    if not experiment_names:
        raise ValueError(f"No saved experiments found for group '{group}' in {metrics_dir}")

    if group == "ckplus_external":
        for model_group in ["baseline", "resnet18", "efficientnet_b0"]:
            model_names = [name for name in experiment_names if ckplus_model_group(name) == model_group]
            if not model_names:
                continue
            model_output_dir = output_dir / model_group
            model_output_dir.mkdir(parents=True, exist_ok=True)
            summary = build_summary(metrics_dir, model_names)
            summary.to_csv(model_output_dir / "summary_metrics.csv", index=False)
            save_table_image(summary, model_output_dir / "summary_metrics_table.png", f"CK+ {model_group} summary metrics")
            save_metric_barplot(summary, model_group, model_output_dir / "test_f1_comparison.png")
            save_per_class_heatmap(metrics_dir, model_names, model_group, model_output_dir / "per_class_f1_heatmap.png")
            save_individual_loss_curves(metrics_dir, model_names, model_output_dir)
            save_top_confusions(metrics_dir, model_names, model_output_dir)
            copy_confusion_matrices(
                figures_dir,
                model_names,
                model_output_dir,
                preferred_dirs=[ckplus_evaluated_dir / model_group],
            )
            copy_source_reports(metrics_dir, model_names, model_output_dir)
            dashboard_path = write_dashboard(model_output_dir, f"CK+ {model_group} Saved Results")
            for name in model_names:
                print_saved_experiment_details(name, metrics_dir, figures_dir)
            print(f"Reproduced {len(model_names)} CK+ {model_group} experiment(s) into {model_output_dir}")
            print(f"Dashboard: {dashboard_path}")

        combined = build_summary(metrics_dir, experiment_names)
        combined.to_csv(output_dir / "summary_metrics.csv", index=False)
        save_table_image(combined, output_dir / "summary_metrics_table.png", "CK+ combined summary metrics")
        save_metric_barplot(combined, group, output_dir / "test_f1_comparison.png")
        save_top_confusions(metrics_dir, experiment_names, output_dir)
        copy_confusion_matrices(
            figures_dir,
            experiment_names,
            output_dir,
            preferred_dirs=[
                ckplus_evaluated_dir / "baseline",
                ckplus_evaluated_dir / "resnet18",
                ckplus_evaluated_dir / "efficientnet_b0",
            ],
        )
        copy_source_reports(metrics_dir, experiment_names, output_dir)
        dashboard_path = write_dashboard(output_dir, "CK+ Combined Saved Results")
        print(f"Reproduced {len(experiment_names)} ckplus_external experiment(s) into {output_dir}")
        print(f"Dashboard: {dashboard_path}")
        return

    if group == "resnet18":
        missing_expected = [name for name in FINAL_RESNET18_EXPERIMENTS if name not in experiment_names]
        if missing_expected:
            (output_dir / "missing_expected_artifacts.txt").write_text(
                "\n".join(missing_expected) + "\n",
                encoding="utf-8",
            )
            print(f"Warning: {len(missing_expected)} expected resnet18 experiment(s) were not found.")

    summary = build_summary(metrics_dir, experiment_names)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    save_table_image(summary, output_dir / "summary_metrics_table.png", f"{group} summary metrics")
    save_metric_barplot(summary, group, output_dir / "test_f1_comparison.png")
    save_validation_plot(summary, group, output_dir / "validation_macro_f1_comparison.png")
    save_per_class_heatmap(metrics_dir, experiment_names, group, output_dir / "per_class_f1_heatmap.png")
    save_training_curves(metrics_dir, experiment_names, group, output_dir / "training_loss_curves.png")
    save_individual_loss_curves(metrics_dir, experiment_names, output_dir)
    save_top_confusions(metrics_dir, experiment_names, output_dir)
    copy_confusion_matrices(figures_dir, experiment_names, output_dir)
    copy_source_reports(metrics_dir, experiment_names, output_dir)

    for name in experiment_names:
        print_saved_experiment_details(name, metrics_dir, figures_dir)

    dashboard_path = write_dashboard(output_dir, f"{group} Saved Results")
    print(f"Reproduced {len(experiment_names)} {group} experiment(s) into {output_dir}")
    print(f"Dashboard: {dashboard_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild report-ready tables and plots from saved experiment result files."
    )
    parser.add_argument(
        "--group",
        choices=["baseline", "resnet18", "efficientnet_b0", "ckplus_external", "all"],
        required=True,
        help="Saved result group to reproduce.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/generated/saved_results"))
    parser.add_argument("--open", action="store_true", help="Open generated dashboard(s) in the default browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = ["baseline", "resnet18", "efficientnet_b0", "ckplus_external"] if args.group == "all" else [args.group]
    for group in groups:
        reproduce_group(group, args.results_dir, args.output_root)
        if args.open:
            dashboard = args.output_root / group / "index.html"
            if dashboard.exists():
                webbrowser.open(dashboard.resolve().as_uri())


if __name__ == "__main__":
    main()
