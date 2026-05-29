from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_common import (
    build_transfer_checkpoint_model,
    evaluate_model,
    save_comparison_table,
    transfer_eval_loader,
)


RESNET18_EXPERIMENTS = [
    "resnet18_feature_extract_fc",
    "resnet18_unfreeze_layer4",
    "resnet18_unfreeze_layer3",
    "resnet18_unfreeze_layer2",
    "resnet18_layer3_class_weights",
    "resnet18_layer3_weighted_sampler",
    "resnet18_layer3_focal",
    "resnet18_layer3_class_weights_focal",
    "resnet18_layer3_weighted_sampler_focal",
    "resnet18_layer3_class_weights_focal_sgd_plateau_lr",
]


def load_classifier_config(metrics_dir: Path, name: str) -> tuple[list[int] | None, float]:
    config_path = metrics_dir / f"{name}_config.json"
    if not config_path.exists():
        return None, 0.35
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("classifier_hidden_layers"), config.get("classifier_dropout", 0.35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved ResNet18 checkpoints on FER2013 test data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fer2013_images"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/evaluated_results/resnet18"))
    parser.add_argument("--experiment", action="append", choices=RESNET18_EXPERIMENTS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loader = transfer_eval_loader(args.data_dir, args.batch_size, args.num_workers)
    names = args.experiment or RESNET18_EXPERIMENTS

    for name in names:
        checkpoint_path = args.checkpoint_dir / f"{name}.pt"
        hidden_layers, dropout = load_classifier_config(args.metrics_dir, name)
        model = build_transfer_checkpoint_model(
            checkpoint_path=checkpoint_path,
            device=device,
            transfer_model="resnet18",
            classifier_hidden_layers=hidden_layers,
            classifier_dropout=dropout,
        )
        evaluate_model(model, loader, device, name, args.output_dir)
        print(f"Evaluated {name}")

    save_comparison_table(args.output_dir)
    print(f"Wrote evaluated ResNet18 results to {args.output_dir}")


if __name__ == "__main__":
    main()
