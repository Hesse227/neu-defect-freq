"""Train one experiment configuration on NEU-DET and evaluate on the test split.

Every run uses the same seed, schedule and augmentation policy so that the
only intended difference between runs is the model yaml (see configs/).

Usage:
    python train.py --cfg configs/yolov8n-baseline.yaml --name a0-baseline
    python train.py --cfg configs/yolov8n-dct.yaml     --name a1-dct-p345
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules import register_yolo_modules  # noqa: E402


def metrics_to_dict(metrics, names: dict) -> dict:
    """Flatten a DetMetrics object into a json-friendly dict (per-class where available)."""
    out = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision_mean": float(metrics.box.mean_results()[0]) if hasattr(metrics.box, "mean_results") else None,
        "recall_mean": float(metrics.box.mean_results()[1]) if hasattr(metrics.box, "mean_results") else None,
    }
    all_ap = getattr(metrics.box, "all_ap", None)  # (nc, 10) AP curves
    per_class = {}
    if all_ap is not None:
        for i, name in names.items():
            per_class[str(name)] = {
                "ap50": float(all_ap[i, 0]),
                "ap50_95": float(all_ap[i].mean()),
            }
    out["per_class"] = per_class
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="model yaml")
    ap.add_argument("--data", default="configs/neu-det.yaml")
    ap.add_argument("--name", required=True, help="run name (runs/<name>)")
    ap.add_argument("--base", default="yolov8n.pt", help="pretrained weights to transfer (empty to disable)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--close-mosaic", type=int, default=15)
    args = ap.parse_args()

    register_yolo_modules()
    from ultralytics import YOLO

    data_yaml = str((ROOT / args.data).resolve())
    run_dir = ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.cfg)
    if args.base:
        model.load(args.base)  # transfer matching weights; new attention layers start fresh

    train_kwargs = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        patience=args.patience,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        device=args.device,
        workers=args.workers,
        cos_lr=True,
        plots=True,
        val=True,
        close_mosaic=args.close_mosaic,
        verbose=True,
    )
    model.train(**train_kwargs)

    # Final evaluation on the held-out test split with the best checkpoint.
    best = run_dir / "weights" / "best.pt"
    test_model = YOLO(str(best))
    metrics = test_model.val(
        data=data_yaml,
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        project=str(ROOT / "runs"),
        name=args.name + "-test",
        exist_ok=True,
    )
    summary = {
        "name": args.name,
        "cfg": args.cfg,
        "args": vars(args),
        "test": metrics_to_dict(metrics, metrics.names),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["test"], indent=2))

    (run_dir / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
