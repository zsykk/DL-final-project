from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate_common import (
    baseline_eval_loader,
    evaluate_model,
    infer_baseline_model_from_state_dict,
    load_state_dict,
    save_comparison_table,
)


def discover_experiments(checkpoint_dir: Path) -> list[str]:
    return sorted(path.stem for path in checkpoint_dir.glob("baseline_cnn_*.pt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved baseline CNN checkpoints on FER2013 test data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fer2013_images"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/evaluated_results/baseline"))
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    names = args.experiment or discover_experiments(args.checkpoint_dir)
    if not names:
        raise ValueError("No baseline CNN checkpoints found to evaluate.")
    print(f"Preparing to evaluate {len(names)} baseline checkpoint(s).")
    print(f"Outputs will be written to {args.output_dir}")

    normal_loader = None
    tencrop_loader = None
    for name in names:
        checkpoint_path = args.checkpoint_dir / f"{name}.pt"
        state_dict = load_state_dict(checkpoint_path, device)
        model = infer_baseline_model_from_state_dict(state_dict)
        use_crops = "tencrop" in name.lower() or "crop_" in name.lower()
        if use_crops:
            if tencrop_loader is None:
                tencrop_loader = baseline_eval_loader(args.data_dir, args.batch_size, args.num_workers, ten_crop=True)
            loader = tencrop_loader
        else:
            if normal_loader is None:
                normal_loader = baseline_eval_loader(args.data_dir, args.batch_size, args.num_workers, ten_crop=False)
            loader = normal_loader
        evaluate_model(model, loader, device, name, args.output_dir, use_crops=use_crops)
        print(f"Evaluated {name}")

    save_comparison_table(args.output_dir)
    print(f"Wrote evaluated baseline results to {args.output_dir}")


if __name__ == "__main__":
    main()
