"""Detection visualisations + error analysis for the NEU-DET experiments.

Produces (into results/):
  visualisations/detections_<class>_<i>.png : GT | baseline | DCT triplets
  visualisations/gain_cases.png             : cases missed by baseline, caught by DCT
  error_analysis.md                         : per-class AP ranking + miss analysis

Usage: python scripts/visualize.py [--conf 0.25]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import register_yolo_modules  # noqa: E402

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
COLORS = [(230, 120, 60), (60, 200, 60), (60, 120, 230), (200, 60, 200), (0, 190, 200), (200, 200, 60)]  # BGR
GREEN = (80, 220, 80)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def parse_boxes(stem: str, folder: str = "labels/test", size: int = 200):
    path = ROOT / "data" / "NEU-DET" / folder / f"{stem}.txt"
    boxes = []
    for ln in path.read_text().splitlines():
        parts = ln.split()
        if len(parts) != 5:
            continue
        c = int(parts[0])
        cx, cy, w, h = (float(v) * size for v in parts[1:])
        boxes.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes


def draw(img: np.ndarray, boxes, colors_by_class=True, color=(80, 220, 80), confs=None) -> np.ndarray:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for i, (c, (x1, y1, x2, y2)) in enumerate(boxes):
        col = COLORS[c] if colors_by_class else color
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 1)
        label = CLASSES[c] + (f" {confs[i]:.2f}" if confs else "")
        cv2.putText(vis, label, (int(x1), max(10, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
    return vis


def header(text: str, width: int = 200) -> np.ndarray:
    band = np.full((22, width, 3), 24, np.uint8)
    cv2.putText(band, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return band


def triptych(img, gt, base, dct, out_path: Path) -> None:
    cols = [
        np.vstack([header("ground truth"), draw(img, gt, colors_by_class=True)]),
        np.vstack([header("YOLOv8n baseline"), draw(img, base[0], confs=base[1])]),
        np.vstack([header("+ DCT attention (ours)"), draw(img, dct[0], confs=dct[1])]),
    ]
    gap = np.full((cols[0].shape[0], 6, 3), 255, np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.concatenate([cols[0], gap, cols[1], gap, cols[2]], axis=1))


def match(gt_boxes, pred_boxes, thr=0.5):
    """For each GT box: matched by pred? Returns (gt_matched, pred_matched)."""
    gt_m = [False] * len(gt_boxes)
    pr_m = [False] * len(pred_boxes)
    for gi, (gc, gb) in enumerate(gt_boxes):
        best, bi = 0.0, -1
        for pi, (pc, pb) in enumerate(pred_boxes):
            if pc != gc:
                continue
            v = iou(gb, pb)
            if v > best:
                best, bi = v, pi
        if best >= thr:
            gt_m[gi] = True
            if bi >= 0:
                pr_m[bi] = True
    return gt_m, pr_m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=ROOT / "runs" / "a0-baseline" / "weights" / "best.pt")
    ap.add_argument("--dct", default=ROOT / "runs" / "a1-dct-p345" / "weights" / "best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    register_yolo_modules()
    from ultralytics import YOLO

    out_dir = ROOT / "results" / "visualisations"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_imgs = sorted((ROOT / "data" / "NEU-DET" / "images" / "test").glob("*.jpg"))
    print(f"predicting {len(test_imgs)} test images with both models ...")

    def run(model_path):
        model = YOLO(str(model_path))
        preds = {}
        results = model.predict([str(p) for p in test_imgs], imgsz=640, conf=args.conf, device=args.device, verbose=False)
        for img_path, r in zip(test_imgs, results):
            boxes, confs = [], []
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].cpu().numpy() * 200 / r.orig_shape[1])
                boxes.append((int(int(b.cls[0])), (x1, y1, x2, y2)))
                confs.append(float(b.conf[0]))
            preds[img_path.stem] = (boxes, confs)
        return preds

    base_preds = run(args.baseline)
    dct_preds = run(args.dct)

    gt_all = {p.stem: parse_boxes(p.stem) for p in test_imgs}

    # ---- per-class AP comes from test_metrics.json; here: matched statistics ----
    gain_cases = []  # (stem, cls) GT boxes missed by baseline but caught by DCT
    cls_stats = {c: {"n": 0, "base_hit": 0, "dct_hit": 0} for c in range(6)}
    for stem, gt in gt_all.items():
        gb, _ = match(gt, base_preds[stem][0])
        gd, _ = match(gt, dct_preds[stem][0])
        for (c, _), mb, md in zip(gt, gb, gd):
            cls_stats[c]["n"] += 1
            cls_stats[c]["base_hit"] += mb
            cls_stats[c]["dct_hit"] += md
            if md and not mb:
                gain_cases.append((stem, c))

    # ---- triplet visualisations: 1 per class (a good DCT case) ----
    for c in range(6):
        cands = [s for s, cl in gain_cases if cl == c] or [s for s in gt_all if any(g[0] == c for g in gt_all[s])]
        if not cands:
            continue
        stem = cands[0]
        img = cv2.imread(str(ROOT / "data" / "NEU-DET" / "images" / "test" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        triptych(img, gt_all[stem], base_preds[stem], dct_preds[stem], out_dir / f"detections_{CLASSES[c]}.png")
    print(f"wrote per-class triplets -> {out_dir}")

    # ---- gain cases grid (missed by baseline, caught by DCT), up to 9 ----
    sel = gain_cases[:9]
    if sel:
        rows = []
        for stem, c in sel:
            img = cv2.imread(str(ROOT / "data" / "NEU-DET" / "images" / "test" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
            gb, _ = match(gt_all[stem], base_preds[stem][0])
            gd, _ = match(gt_all[stem], dct_preds[stem][0])
            gt_show = [g for g, m in zip(gt_all[stem], gd) if m]
            base_show = [g for g, m in zip(gt_all[stem], gb) if not m]  # the missed GT, drawn on baseline panel
            cols = [
                np.vstack([header("missed by baseline"), draw(img, base_show, colors_by_class=False, color=(60, 60, 220))]),
                np.vstack([header("caught by DCT (ours)"), draw(img, gt_show, colors_by_class=True)]),
            ]
            rows.append(np.concatenate(cols, axis=1))
        w = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 4, 4, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
        grid = np.concatenate(rows, axis=0)
        cv2.imwrite(str(out_dir / "gain_cases.png"), grid)
        print(f"wrote gain_cases.png ({len(sel)} cases)")
    else:
        print("no baseline-missed / dct-caught cases found at this conf threshold")

    # ---- error analysis markdown ----
    md = ["# Error analysis (baseline vs +DCT, test split, conf=%.2f, IoU 0.5)" % args.conf, "",
          "Recall per class (box-level matching):", "",
          "| class | GT boxes | baseline recall | +DCT recall |", "|---|---|---|---|"]
    for c in range(6):
        s = cls_stats[c]
        br = s["base_hit"] / max(s["n"], 1)
        dr = s["dct_hit"] / max(s["n"], 1)
        md.append(f"| {CLASSES[c]} | {s['n']} | {br:.3f} | {dr:.3f} |")
    md += ["", f"GT boxes missed by baseline but caught by +DCT: **{len(gain_cases)}**", ""]
    from collections import Counter

    cnt = Counter(c for _, c in gain_cases)
    for c, n in cnt.most_common():
        md.append(f"- {CLASSES[c]}: {n}")
    (ROOT / "results" / "error_analysis.md").write_text("\n".join(md) + "\n")
    print("wrote results/error_analysis.md")


if __name__ == "__main__":
    main()
