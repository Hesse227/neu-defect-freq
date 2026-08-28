"""Spectral analysis of NEU-DET defect classes (radial power spectra).

Motivated by DSF-Net's methodology (frequency-domain analysis *before* network
design), this script quantifies where each defect class's energy lives in the
spectrum. For every GT box we crop the defect region, resize to a fixed size,
compute the 2-D FFT power spectrum, azimuthally average it into a radial power
spectral density (PSD), and normalise per-image so classes are compared by the
*shape* of their energy distribution.

Outputs:
    results/spectral_analysis.png   six radial PSD curves (log scale)
    results/spectral_analysis.md    low/high-frequency energy shares per class

If fine classes (crazing, pitted_surface) skew high-frequency and blotch
classes (rolled-in_scale, patches) skew low-frequency, then the class-level
effects observed in the attention ablations (DCT hurts crazing, mildly helps
rolled-in_scale) were *predictable from the physics* — evidence chain:
spectral analysis (before) -> ablation outcome (after).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from xml2yolo import CLASSES, parse_yolo_label  # reuse the split-time label reader

IMG_DIR = ROOT / "data" / "NEU-DET" / "images" / "train"
LBL_DIR = ROOT / "data" / "NEU-DET" / "labels" / "train"
CROP = 128          # uniform crop size before FFT
K_SPLIT = 0.25      # low/high frequency split at 25% of Nyquist (k < 32/64 bins)
MAX_BOXES_PER_CLASS = 250


def radial_psd(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally-averaged power spectrum of a grayscale image."""
    f = np.fft.fftshift(np.fft.fft2(img.astype(np.float64) - img.mean()))
    psd2d = np.abs(f) ** 2
    h, w = psd2d.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    k = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    # integrate energy in each radial bin, normalise by bin area -> density
    tbin = np.bincount(k.ravel(), psd2d.ravel())
    nr = np.bincount(k.ravel())
    radial = tbin / np.maximum(nr, 1)
    radial = radial[: cx]  # up to Nyquist
    return radial / radial.sum()  # shape comparison, not magnitude


def main() -> None:
    rng = np.random.default_rng(42)
    curves = {}
    for cls_id, cls_name in enumerate(CLASSES):
        psds, boxes = [], 0
        imgs = sorted(p for p in IMG_DIR.glob("*.jpg") if p.stem.rsplit("_", 1)[0] == cls_name)
        for img_path in imgs:
            if boxes >= MAX_BOXES_PER_CLASS:
                break
            lbl = LBL_DIR / (img_path.stem + ".txt")
            if not lbl.is_file():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            H, W = img.shape
            for c, cx, cy, w, h in parse_yolo_label(lbl):
                if c != cls_id or boxes >= MAX_BOXES_PER_CLASS:
                    continue
                x1 = max(0, int((cx - w / 2) * W) - 2)
                y1 = max(0, int((cy - h / 2) * H) - 2)
                x2 = min(W, int((cx + w / 2) * W) + 2)
                y2 = min(H, int((cy + h / 2) * H) + 2)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                crop = cv2.resize(img[y1:y2, x1:x2], (CROP, CROP), interpolation=cv2.INTER_AREA)
                psds.append(radial_psd(crop))
                boxes += 1
        curves[cls_name] = np.mean(psds, axis=0)
        print(f"{cls_name:<15} n_boxes={boxes:4d}")

    # ---- figure ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    k = np.arange(1, curves[CLASSES[0]].size) / CROP  # normalised spatial frequency
    for cls_name in CLASSES:
        ax.plot(k, curves[cls_name][1:] * len(k), label=cls_name, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("normalised spatial frequency (cycles/pixel)")
    ax.set_ylabel("radial PSD (area-normalised, log)")
    ax.set_title("NEU-DET defect regions: radial power spectral density per class")
    ax.axvline(K_SPLIT, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(K_SPLIT + 0.005, ax.get_ylim()[1] * 0.5, "low | high split", fontsize=8, color="gray")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    out_png = ROOT / "results" / "spectral_analysis.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)

    # ---- table ----
    split_bin = int(K_SPLIT * CROP / 2)
    lines = [
        "# Spectral analysis of NEU-DET defect classes (GT-box crops, azimuthal-averaged FFT)",
        "",
        "Radial PSD per class, normalised per image (energy-distribution shape).",
        f"Low-frequency share = fraction of spectral energy below {K_SPLIT:.0%} of Nyquist.",
        "",
        "| class | low-freq share | high-freq share | low/high ratio |",
        "|---|---|---|---|",
    ]
    stats = {}
    for cls_name in CLASSES:
        cur = curves[cls_name]
        low = cur[1:split_bin].sum()
        high = cur[split_bin:].sum()
        stats[cls_name] = low / high
        lines.append(f"| {cls_name} | {low:.3f} | {high:.3f} | {low / high:.2f} |")
    (ROOT / "results" / "spectral_analysis.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_png} and results/spectral_analysis.md")
    for c, r in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<15} low/high energy ratio = {r:.2f}")


if __name__ == "__main__":
    main()
