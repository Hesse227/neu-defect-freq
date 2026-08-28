"""Aggregate all experiment results into the final tables and figures.

Reads runs/<name>/test_metrics.json for every finished run, measures parameter
count / inference speed, copies PR-curve & confusion-matrix plots from the
ultralytics validation outputs, and writes:

    results/ablation_table.md     main + per-class tables (markdown)
    results/<name>_pr_curve.png   test-split PR curves
    results/<name>_confusion.png  test-split confusion matrices
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import register_yolo_modules  # noqa: E402

RUNS = [  # (display name, run dir)
    ("YOLOv8n (baseline, seed 42)", "a0-baseline"),
    ("YOLOv8n (baseline, seed 123)", "a0-baseline-seed123"),
    ("YOLOv8n + DCT @ P3/P4/P5 (ours, seed 42)", "a1-dct-p345"),
    ("YOLOv8n + DCT @ P3/P4/P5 (ours, seed 123)", "a1-dct-p345-s123"),
    ("YOLOv8n + DCT @ P3 only", "a2-dct-p3"),
    ("YOLOv8n + SE @ P3/P4/P5 (control)", "a3-se-p345"),
    ("YOLOv8n + DCT @ P3/P4/P5, freeze first 5 layers", "a4-dct-frozen5"),
]
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def params_for(run: str) -> float:
    """Model parameter count in millions (builds the model from its yaml)."""
    cfg = {
        "a0-baseline": "configs/yolov8n-baseline.yaml",
        "a0-baseline-seed123": "configs/yolov8n-baseline.yaml",
        "a1-dct-p345": "configs/yolov8n-dct.yaml",
        "a1-dct-p345-s123": "configs/yolov8n-dct.yaml",
        "a2-dct-p3": "configs/yolov8n-dct-p3.yaml",
        "a3-se-p345": "configs/yolov8n-se.yaml",
        "a4-dct-frozen5": "configs/yolov8n-dct.yaml",
    }[run]
    from ultralytics import YOLO

    m = YOLO(cfg).model
    return sum(p.numel() for p in m.parameters()) / 1e6


def speed_for(weights: Path, device: str, n_warm: int = 8, n: int = 40) -> tuple[float, float]:
    """(mean inference ms, mean e2e ms) at imgsz=640 on test images."""
    from ultralytics import YOLO

    imgs = sorted((ROOT / "data" / "NEU-DET" / "images" / "test").glob("*.jpg"))[: n + n_warm]
    model = YOLO(str(weights))
    results = model.predict([str(p) for p in imgs], imgsz=640, device=device, verbose=False)
    inf, e2e = [], []
    for i, r in enumerate(results):
        if i < n_warm:
            continue
        s = r.speed  # {'preprocess', 'inference', 'postprocess'} in ms
        inf.append(s["inference"])
        e2e.append(s["preprocess"] + s["inference"] + s["postprocess"])
    return sum(inf) / len(inf), sum(e2e) / len(e2e)


def main() -> None:
    register_yolo_modules()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    rows = []
    for display, run in RUNS:
        mf = ROOT / "runs" / run / "test_metrics.json"
        if not mf.exists():
            print(f"SKIP {run}: no test_metrics.json")
            continue
        data = json.loads(mf.read_text())
        test = data["test"]
        w = ROOT / "runs" / run / "weights" / "best.pt"
        row = {
            "name": display,
            "run": run,
            "map50": test["map50"],
            "map50_95": test["map50_95"],
            "per_class": test.get("per_class", {}),
            "params_m": params_for(run),
        }
        if w.exists():
            row["gpu_inf_ms"], row["gpu_e2e_ms"] = speed_for(w, 0)
            try:
                row["cpu_inf_ms"], row["cpu_e2e_ms"] = speed_for(w, "cpu", n_warm=3, n=15)
            except Exception as e:  # CPU speed is nice-to-have
                print(f"CPU speed measurement failed: {e}")
        rows.append(row)

        # copy plots from the test-split validation output
        for src, dst in [
            (ROOT / "runs" / f"{run}-test" / "BoxPR_curve.png", out / f"{run}_pr_curve.png"),
            (ROOT / "runs" / f"{run}-test" / "confusion_matrix.png", out / f"{run}_confusion.png"),
            (ROOT / "runs" / f"{run}-test" / "confusion_matrix_normalized.png", out / f"{run}_confusion_norm.png"),
        ]:
            if src.exists():
                shutil.copy2(src, dst)

    if not rows:
        sys.exit("no finished runs found")

    md = ["# NEU-DET ablation results (test split, 180 images)", "",
          "| model | params (M) | mAP@50 | mAP@50:95 | GPU inf (ms) | GPU FPS | CPU inf (ms) |",
          "|---|---|---|---|---|---|---|"]
    base = next((r for r in rows if r["run"] == "a0-baseline"), None)
    for r in rows:
        d50 = f" ({r['map50'] - base['map50']:+.3f})" if base and r is not base else ""
        d95 = f" ({r['map50_95'] - base['map50_95']:+.3f})" if base and r is not base else ""
        gpu_fps = f"{1000 / r['gpu_inf_ms']:.1f}" if "gpu_inf_ms" in r else "-"
        cpu = f"{r.get('cpu_inf_ms', float('nan')):.1f}" if "cpu_inf_ms" in r else "-"
        md.append(
            f"| {r['name']} | {r['params_m']:.3f} | {r['map50']:.3f}{d50} | {r['map50_95']:.3f}{d95} "
            f"| {r.get('gpu_inf_ms', float('nan')):.1f} | {gpu_fps} | {cpu} |"
        )

    md += ["", "## Per-class AP@50 (test split)", "",
           "| model | " + " | ".join(CLASSES) + " |", "|---" * 7 + "|"]
    for r in rows:
        vals = [f"{r['per_class'].get(c, {}).get('ap50', float('nan')):.3f}" for c in CLASSES]
        md.append(f"| {r['name']} | " + " | ".join(vals) + " |")

    (out / "ablation_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
