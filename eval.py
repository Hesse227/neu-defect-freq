"""Evaluate a trained checkpoint on the NEU-DET test split.

Writes runs/<name>/test_metrics.json (overall + per-class) and the
runs/<name>-test/ figures (PR curve, confusion matrix), so it can also be used
to re-generate the evaluation of a run whose training-time evaluation failed.

Usage:
    python eval.py --run a1-dct-p345            # uses runs/a1-dct-p345/weights/best.pt
    python eval.py --weights path/to/best.pt --name my-eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules import register_yolo_modules  # noqa: E402
from train import metrics_to_dict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory name under runs/ (uses its best.pt)")
    ap.add_argument("--weights", help="explicit path to a checkpoint")
    ap.add_argument("--name", help="output run name (defaults to --run)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    assert args.run or args.weights, "need --run or --weights"
    weights = Path(args.weights) if args.weights else ROOT / "runs" / args.run / "weights" / "best.pt"
    assert weights.is_file(), f"checkpoint not found: {weights}"
    name = args.name or (args.run or weights.parent.parent.name)

    register_yolo_modules()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(ROOT / "configs" / "neu-det.yaml"),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        project=str(ROOT / "runs"),
        name=name + "-test",
        exist_ok=True,
    )
    summary = {
        "name": name,
        "weights": str(weights),
        "split": args.split,
        "test": metrics_to_dict(metrics, metrics.names),
    }
    out_dir = ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
