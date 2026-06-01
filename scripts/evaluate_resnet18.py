from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_common import (
    build_transfer_checkpoint_model,
    evaluate_model,
    open_dashboard,
    save_comparison_table,
    save_evaluation_overview_assets,
    status,
    transfer_eval_loader,
    transfer_tencrop_eval_loader,
    write_evaluation_dashboard,
)


RESNET18_EXPERIMENTS = [
    "resnet18_feature_extract_fc",
    "resnet18_unfreeze_layer4",
    "resnet18_unfreeze_layer3",
    "resnet18_layer3_class_weights",
    "resnet18_layer3_class_weights_focal",
    "resnet18_layer3_class_weights_focal_sgd_plateau_lr",
    "resnet18_layer3_class_weights_focal_sgd_plateau_lr_crop_tencrop",
]


def load_classifier_config(metrics_dir: Path, name: str) -> tuple[list[int] | None, float]:
    config_path = metrics_dir / f"{name}_config.json"
    if not config_path.exists():
        return None, 0.35
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("classifier_hidden_layers"), config.get("classifier_dropout", 0.35)


def uses_tencrop(metrics_dir: Path, name: str) -> bool:
    config_path = metrics_dir / f"{name}_config.json"
    if not config_path.exists():
        return "tencrop" in name.lower()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return bool(config.get("ten_crop_eval", False)) or "tencrop" in name.lower()


def has_evaluation_outputs(output_dir: Path, name: str) -> bool:
    return (
        (output_dir / f"{name}_classification_report.csv").exists()
        and (output_dir / f"{name}_confusion_matrix.png").exists()
        and (output_dir / f"{name}_top_confusions.csv").exists()
    )


def refresh_dashboard(output_dir: Path) -> Path:
    save_comparison_table(output_dir, experiment_order=RESNET18_EXPERIMENTS)
    save_evaluation_overview_assets(output_dir, experiment_order=RESNET18_EXPERIMENTS)
    return write_evaluation_dashboard(
        output_dir,
        "ResNet18 Evaluated Results",
        experiment_order=RESNET18_EXPERIMENTS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved ResNet18 checkpoints on FER2013 test data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fer2013_images"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/evaluated_results/resnet18"))
    parser.add_argument("--experiment", action="append", choices=RESNET18_EXPERIMENTS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tencrop-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--include-tencrop-on-cpu",
        action="store_true",
        help="Include TenCrop checkpoints in this full ResNet18 run even when using CPU.",
    )
    parser.add_argument("--force", action="store_true", help="Re-evaluate checkpoints even when outputs already exist.")
    parser.add_argument("--open", action="store_true", help="Open the generated evaluation dashboard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    normal_loader = None
    tencrop_loader = None
    names = args.experiment or RESNET18_EXPERIMENTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status(f"Preparing to evaluate {len(names)} ResNet18 checkpoint(s).", "processing")
    status(f"Outputs will be written to {args.output_dir}", "output")
    dashboard_path = refresh_dashboard(args.output_dir)

    for name in names:
        use_crops = uses_tencrop(args.metrics_dir, name)
        if use_crops and device.type == "cpu" and not args.include_tencrop_on_cpu:
            status(
                f"Skipping CPU TenCrop checkpoint {name}; run scripts/evaluate_resnet18_tencrop.py for this one.",
                "done",
            )
            dashboard_path = refresh_dashboard(args.output_dir)
            continue

        if has_evaluation_outputs(args.output_dir, name) and not args.force:
            status(f"Skipping {name}; existing evaluation outputs found.", "done")
            dashboard_path = refresh_dashboard(args.output_dir)
            continue

        if use_crops:
            if tencrop_loader is None:
                tencrop_loader = transfer_tencrop_eval_loader(
                    args.data_dir,
                    min(args.batch_size, args.tencrop_batch_size),
                    args.num_workers,
                )
            loader = tencrop_loader
        else:
            if normal_loader is None:
                normal_loader = transfer_eval_loader(args.data_dir, args.batch_size, args.num_workers)
            loader = normal_loader
        checkpoint_path = args.checkpoint_dir / f"{name}.pt"
        hidden_layers, dropout = load_classifier_config(args.metrics_dir, name)
        model = build_transfer_checkpoint_model(
            checkpoint_path=checkpoint_path,
            device=device,
            transfer_model="resnet18",
            classifier_hidden_layers=hidden_layers,
            classifier_dropout=dropout,
        )
        evaluate_model(model, loader, device, name, args.output_dir, use_crops=use_crops)
        status(f"Evaluated {name}", "done")
        dashboard_path = refresh_dashboard(args.output_dir)

    dashboard_path = refresh_dashboard(args.output_dir)
    status(f"Wrote evaluated ResNet18 results to {args.output_dir}", "done")
    status(f"Dashboard: {dashboard_path}", "output")
    if args.open:
        open_dashboard(dashboard_path)


if __name__ == "__main__":
    main()
