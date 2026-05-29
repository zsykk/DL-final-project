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


def discover_experiments(checkpoint_dir: Path, metrics_dir: Path) -> list[str]:
    names = []
    for checkpoint_path in sorted(checkpoint_dir.glob("efficientnet_b0_*.pt")):
        name = checkpoint_path.stem
        if "rafdb" in name.lower() or name.endswith("_summary"):
            continue
        if (metrics_dir / f"{name}_config.json").exists():
            names.append(name)
    return names


def load_classifier_config(metrics_dir: Path, name: str) -> tuple[list[int] | None, float]:
    config_path = metrics_dir / f"{name}_config.json"
    if not config_path.exists():
        return None, 0.35
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("classifier_hidden_layers"), config.get("classifier_dropout", 0.35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved EfficientNet-B0 checkpoints on FER2013 test data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fer2013_images"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/evaluated_results/efficientnet_b0"))
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loader = transfer_eval_loader(args.data_dir, args.batch_size, args.num_workers)
    names = args.experiment or discover_experiments(args.checkpoint_dir, args.metrics_dir)
    if not names:
        raise ValueError("No EfficientNet-B0 checkpoints found to evaluate.")

    for name in names:
        checkpoint_path = args.checkpoint_dir / f"{name}.pt"
        hidden_layers, dropout = load_classifier_config(args.metrics_dir, name)
        model = build_transfer_checkpoint_model(
            checkpoint_path=checkpoint_path,
            device=device,
            transfer_model="efficientnet_b0",
            classifier_hidden_layers=hidden_layers,
            classifier_dropout=dropout,
        )
        evaluate_model(model, loader, device, name, args.output_dir)
        print(f"Evaluated {name}")

    save_comparison_table(args.output_dir)
    print(f"Wrote evaluated EfficientNet-B0 results to {args.output_dir}")


if __name__ == "__main__":
    main()
