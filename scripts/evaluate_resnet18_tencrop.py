from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate_common import (
    build_transfer_checkpoint_model,
    evaluate_model,
    open_dashboard,
    status,
    transfer_tencrop_eval_loader,
)
from evaluate_resnet18 import (
    RESNET18_EXPERIMENTS,
    has_evaluation_outputs,
    load_classifier_config,
    refresh_dashboard,
    uses_tencrop,
)


def discover_tencrop_experiments(metrics_dir: Path) -> list[str]:
    return [name for name in RESNET18_EXPERIMENTS if uses_tencrop(metrics_dir, name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 TenCrop checkpoints on FER2013 test data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fer2013_images"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/evaluated_results/resnet18"))
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Re-evaluate checkpoints even when outputs already exist.")
    parser.add_argument("--open", action="store_true", help="Open the generated evaluation dashboard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    names = args.experiment or discover_tencrop_experiments(args.metrics_dir)
    if not names:
        raise ValueError("No ResNet18 TenCrop checkpoints found to evaluate.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status(f"Preparing to evaluate {len(names)} ResNet18 TenCrop checkpoint(s).", "processing")
    status(f"Outputs will be written to {args.output_dir}", "processing")
    if device.type == "cpu":
        status("Running TenCrop on CPU; this is expected to be slow.", "processing")

    loader = transfer_tencrop_eval_loader(args.data_dir, args.batch_size, args.num_workers)
    dashboard_path = refresh_dashboard(args.output_dir)

    for name in names:
        if not uses_tencrop(args.metrics_dir, name):
            status(f"Skipping {name}; it is not configured for TenCrop evaluation.", "done")
            continue
        if has_evaluation_outputs(args.output_dir, name) and not args.force:
            status(f"Skipping {name}; existing evaluation outputs found.", "done")
            dashboard_path = refresh_dashboard(args.output_dir)
            continue

        checkpoint_path = args.checkpoint_dir / f"{name}.pt"
        hidden_layers, dropout = load_classifier_config(args.metrics_dir, name)
        model = build_transfer_checkpoint_model(
            checkpoint_path=checkpoint_path,
            device=device,
            transfer_model="resnet18",
            classifier_hidden_layers=hidden_layers,
            classifier_dropout=dropout,
        )
        evaluate_model(model, loader, device, name, args.output_dir, use_crops=True)
        status(f"Evaluated {name}", "done")
        dashboard_path = refresh_dashboard(args.output_dir)

    dashboard_path = refresh_dashboard(args.output_dir)
    status(f"Dashboard: {dashboard_path}", "done")
    if args.open:
        open_dashboard(dashboard_path)


if __name__ == "__main__":
    main()
