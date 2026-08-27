"""Run the full experiment matrix sequentially (single GPU).

Ablations:
    a0-baseline : YOLOv8n
    a1-dct-p345 : + DCT multi-spectral channel attention @ P3/P4/P5
    a2-dct-p3   : + DCT attention @ P3 only
    a3-se-p345  : + SE (GAP) attention @ P3/P4/P5 (capacity-matched control)

All runs share seed / imgsz / epochs / augmentation. Completed runs (DONE
marker present) are skipped, so the script is safe to re-launch after an
interruption.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RUNS = [
    ("a0-baseline", "configs/yolov8n-baseline.yaml"),
    ("a1-dct-p345", "configs/yolov8n-dct.yaml"),
    ("a2-dct-p3", "configs/yolov8n-dct-p3.yaml"),
    ("a3-se-p345", "configs/yolov8n-se.yaml"),
]

EPOCHS = "100"

if __name__ == "__main__":
    for name, cfg in RUNS:
        if (ROOT / "runs" / name / "DONE").exists():
            print(f"skip {name} (DONE marker present)", flush=True)
            continue
        log_path = ROOT / "runs" / f"{name}.log"
        print(f"=== training {name} -> {log_path} ===", flush=True)
        with open(log_path, "w") as log:
            r = subprocess.run(
                [sys.executable, "-u", "train.py", "--cfg", cfg, "--name", name, "--epochs", EPOCHS],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print(f"=== {name} finished, exit={r.returncode} ===", flush=True)
    print("ALL RUNS COMPLETE", flush=True)
