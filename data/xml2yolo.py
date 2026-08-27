"""NEU-DET: VOC XML -> YOLO format conversion + stratified 80/10/10 split.

Source dataset (Kaggle mirror, 1800 images + per-image VOC XML):
    sovitrath/neu-steel-surface-defect-detect-trainvalid-split
Original dataset:
    Song & Yan, "A noise robust method based on completed ensemble local binary
    patterns for hot-rolled steel strip surface defects", Applied Surface
    Science 285 (2013).  http://faculty.neu.edu.cn/songkechen/

We deliberately re-split the merged 1800 images ourselves (stratified by
defect class, fixed seed) instead of trusting the mirror's 1700/100 split, so
that train/val/test = 1440/180/180 with every class represented 240/30/30.

Usage:
    python data/xml2yolo.py                 # auto-download (cached) via kagglehub
    python data/xml2yolo.py --src <folder>  # local folder with *_images/*_annotations
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
# BGR colors for visualisation
COLORS = [(230, 120, 60), (60, 200, 60), (60, 120, 230), (200, 60, 200), (0, 190, 200), (200, 200, 60)]

SEED = 42
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def find_source(explicit: str | None) -> Path:
    """Locate the raw dataset (explicit path > local D-drive copy > kagglehub)."""
    local = Path(__file__).resolve().parent / "raw_neu_det"
    for cand in ([Path(explicit)] if explicit else []) + [local]:
        if (cand / "train_images").is_dir():
            return cand
    import os

    # keep any future kagglehub download on the D drive as well
    os.environ.setdefault("KAGGLEHUB_CACHE", str(Path(__file__).resolve().parents[1] / "data" / "kagglehub_cache"))
    import kagglehub

    return Path(kagglehub.dataset_download("sovitrath/neu-steel-surface-defect-detect-trainvalid-split"))


def collect_pairs(src: Path) -> list[tuple[Path, Path]]:
    """All (image, xml) pairs from the mirror's train/valid subfolders."""
    pairs = []
    for split in ("train", "valid"):
        imgs, xmls = src / f"{split}_images", src / f"{split}_annotations"
        if not imgs.is_dir():
            continue
        for img in sorted(imgs.glob("*.jpg")):
            xml = xmls / (img.stem + ".xml")
            assert xml.is_file(), f"missing annotation for {img.name}"
            pairs.append((img, xml))
    assert len(pairs) == 1800, f"expected 1800 image/xml pairs, got {len(pairs)}"
    return pairs


def image_class(stem: str) -> str:
    """'rolled-in_scale_105' -> 'rolled-in_scale' (NEU-DET files are named <class>_<id>.jpg)."""
    cls = stem.rsplit("_", 1)[0]
    assert cls in CLASS_ID, f"unknown class in filename: {stem}"
    return cls


def parse_voc(xml_path: Path) -> list[tuple[int, float, float, float, float]]:
    """VOC boxes -> YOLO (class, cx, cy, w, h) normalised to [0, 1], coords clipped."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w_img = float(size.find("width").text)
    h_img = float(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bb = obj.find("bndbox")
        x1 = max(0.0, min(float(bb.find("xmin").text), w_img))
        y1 = max(0.0, min(float(bb.find("ymin").text), h_img))
        x2 = max(0.0, min(float(bb.find("xmax").text), w_img))
        y2 = max(0.0, min(float(bb.find("ymax").text), h_img))
        if x2 - x1 < 1 or y2 - y1 < 1:  # degenerate box
            continue
        boxes.append(
            (CLASS_ID[name], (x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img, (x2 - x1) / w_img, (y2 - y1) / h_img)
        )
    return boxes


def stratified_split(pairs: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    """80/10/10 per class, fixed seed."""
    rng = random.Random(SEED)
    by_class: dict[str, list] = {c: [] for c in CLASSES}
    for pair in pairs:
        by_class[image_class(pair[0].stem)].append(pair)
    out = {"train": [], "val": [], "test": []}
    for cls in CLASSES:
        items = sorted(by_class[cls], key=lambda p: p[0].name)
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * SPLIT_FRACTIONS["train"])
        n_val = round(n * SPLIT_FRACTIONS["val"])
        out["train"] += items[:n_train]
        out["val"] += items[n_train : n_train + n_val]
        out["test"] += items[n_train + n_val :]
    for split, items in out.items():
        rng.shuffle(items)
    return out


def draw_boxes(img: np.ndarray, boxes, with_text=True) -> np.ndarray:
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    for cls, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[cls], 1)
        if with_text:
            cv2.putText(vis, CLASSES[cls], (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[cls], 1)
    return vis


def sanity_grid(split_dir: Path, out_png: Path, n_per_class: int = 1) -> None:
    """One example per class with GT boxes, as a conversion correctness check."""
    rng = random.Random(0)
    labels_dir = split_dir.parent.parent / "labels" / split_dir.name
    tiles = []
    for cls in CLASSES:
        cands = [p for p in split_dir.glob("*.jpg") if image_class(p.stem) == cls]
        for p in rng.sample(cands, n_per_class):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            boxes = parse_yolo_label(labels_dir / (p.stem + ".txt"))  # converted labels, not the XML
            tiles.append(draw_boxes(img, boxes))
    top = np.concatenate(tiles[:3], axis=1)
    bot = np.concatenate(tiles[3:], axis=1)
    gap = np.full((10, top.shape[1], 3), 255, np.uint8)
    cv2.imwrite(str(out_png), np.concatenate([top, gap, bot], axis=0))


def parse_yolo_label(path: Path) -> list:
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="folder containing *_images/*_annotations (default: kagglehub)")
    ap.add_argument("--out", default=str(ROOT / "data" / "NEU-DET"))
    args = ap.parse_args()

    src = find_source(args.src)
    pairs = collect_pairs(src)
    splits = stratified_split(pairs)

    out = Path(args.out)
    stats: dict[str, Counter] = {}
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        counter = Counter()
        for img, xml in items:
            boxes = parse_voc(xml)
            assert boxes, f"no valid boxes in {xml.name}"
            counter[img.stem.rsplit('_',1)[0]] += len(boxes)
            shutil.copy2(img, out / "images" / split / img.name)
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
            (out / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines) + "\n")
        stats[split] = counter
        print(f"{split:>5}: {len(items):4d} images, {sum(counter.values()):5d} boxes")

    yaml_path = ROOT / "configs" / "neu-det.yaml"
    yaml_path.parent.mkdir(exist_ok=True)
    yaml_path.write_text(
        f"# Auto-generated by data/xml2yolo.py (seed={SEED}, stratified 80/10/10)\n"
        f"path: {str(out).replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print(f"\nwrote {yaml_path}")

    sanity = ROOT / "data" / "sanity_check.png"
    sanity_grid(out / "images" / "train", sanity)
    print(f"wrote {sanity} (visual check of converted labels)")
    for split, counter in stats.items():
        print(f"  {split}: " + ", ".join(f"{c}={n}" for c, n in sorted(counter.items())))


if __name__ == "__main__":
    sys.exit(main())
