"""
DCT Feature Extractor
---------------------
Extracts per-instance DCT texture features from real brain images using
YOLO segmentation masks (.txt) as region definitions.

For each instance region, per RGB channel:
  - freq  : dominant AC frequency (radial, normalised 0-1)
  - angle : dominant AC direction (degrees)
  - min   : pixel minimum within masked region
  - max   : pixel maximum within masked region

Output CSV columns:
  class, R_freq, R_angle, R_min, R_max,
         G_freq, G_angle, G_min, G_max,
         B_freq, B_angle, B_min, B_max

Usage (single pair):
    python dct_feature_extractor.py \
        --image mousebrain1640x640.png \
        --mask  000001.txt \
        --output 000001_features.csv

Usage (batch - auto-pairs by number in filename):
    python dct_feature_extractor.py --batch \
        --image_dir . --mask_dir . --output_dir .
"""

import argparse
import os
import csv
import re
import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# 2-D DCT-II via numpy FFT (no scipy required)
# ---------------------------------------------------------------------------

def dct1d(x: np.ndarray) -> np.ndarray:
    """1-D DCT-II using FFT, ortho-normalised."""
    N = x.shape[-1]
    # Reorder: [x0, x1, ..., xN-1, xN-1, ..., x0] trick via even extension
    v = np.concatenate([x, x[..., ::-1]], axis=-1)
    V = np.fft.rfft(v, axis=-1)[..., :N]
    k = np.arange(N)
    phase = np.exp(-1j * np.pi * k / (2 * N))
    X = np.real(V * phase)
    # Ortho normalisation
    X[..., 0] /= np.sqrt(4 * N)
    X[..., 1:] /= np.sqrt(2 * N)
    return X


def dct2d(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II (rows then columns)."""
    return dct1d(dct1d(block).T).T


def dominant_ac(dct_block: np.ndarray):
    """
    Find peak AC coefficient (DC zeroed), return (freq_normalised, angle_deg).
    """
    block = dct_block.copy()
    block[0, 0] = 0.0
    H, W = block.shape
    idx = np.unravel_index(np.argmax(np.abs(block)), block.shape)
    fu, fv = idx
    max_dist = np.sqrt((H - 1) ** 2 + (W - 1) ** 2)
    freq = float(np.sqrt(fu ** 2 + fv ** 2) / max_dist) if max_dist > 0 else 0.0
    angle = float(np.degrees(np.arctan2(fv, fu)))
    return freq, angle


# ---------------------------------------------------------------------------
# Mask parsing
# ---------------------------------------------------------------------------

def parse_yolo_masks(txt_path: str, W: int, H: int):
    """
    Parse YOLO segmentation .txt → list of (class_id, bool mask [H×W]).
    """
    instances = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = [(coords[i] * W, coords[i + 1] * H)
                   for i in range(0, len(coords) - 1, 2)]
            mask_img = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask_img).polygon(pts, fill=255)
            mask = np.array(mask_img, dtype=bool)
            if mask.sum() == 0:
                continue
            instances.append((cls, mask))
    return instances


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(real_image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Crop bounding box of masked region, zero outside mask, run 2-D DCT,
    extract dominant AC freq + angle, plus pixel min/max per channel.
    """
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    rmin, rmax = np.where(rows_any)[0][[0, -1]]
    cmin, cmax = np.where(cols_any)[0][[0, -1]]

    feats = {}
    for ch_idx, ch in enumerate(["R", "G", "B"]):
        ch_pixels = real_image[:, :, ch_idx][mask]
        pmin, pmax = int(ch_pixels.min()), int(ch_pixels.max())

        crop = real_image[rmin:rmax + 1, cmin:cmax + 1, ch_idx].astype(np.float64)
        crop_mask = mask[rmin:rmax + 1, cmin:cmax + 1]
        crop[~crop_mask] = 0.0

        if crop.size == 0:
            freq, angle = 0.0, 0.0
        else:
            dct_block = dct2d(crop)
            freq, angle = dominant_ac(dct_block)

        feats[f"{ch}_freq"]  = round(freq, 6)
        feats[f"{ch}_angle"] = round(angle, 4)
        feats[f"{ch}_min"]   = pmin
        feats[f"{ch}_max"]   = pmax

    return feats


# ---------------------------------------------------------------------------
# Overlay visualisation
# ---------------------------------------------------------------------------

def save_overlay(real_image: np.ndarray, instances: list, out_path: str):
    palette = [
        (255, 80,  80, 100), (80, 255,  80, 100), ( 80, 110, 255, 100),
        (255, 230, 80, 100), (255, 80,  220, 100), ( 80, 230, 230, 100),
    ]
    base = Image.fromarray(real_image).convert("RGBA")
    for cls, mask in instances:
        color = palette[cls % len(palette)]
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        mask_img = Image.fromarray((mask * color[3]).astype(np.uint8), "L")
        colored  = Image.new("RGBA", base.size, color)
        overlay.paste(colored, mask=mask_img)
        base = Image.alpha_composite(base, overlay)
    base.save(out_path)
    print(f"  Overlay  → {out_path}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_pair(image_path: str, mask_path: str, output_csv: str,
                 save_overlay_img: bool = True):
    print(f"\nProcessing: {os.path.basename(image_path)}")

    real = np.array(Image.open(image_path).convert("RGB"))
    H, W = real.shape[:2]
    print(f"  Image    : {W}×{H}")

    instances = parse_yolo_masks(mask_path, W, H)
    print(f"  Instances: {len(instances)}")

    if save_overlay_img:
        overlay_path = output_csv.replace(".csv", "_overlay.png")
        save_overlay(real, instances, overlay_path)

    rows = []
    for cls, mask in instances:
        feats = extract_features(real, mask)
        rows.append({"class": cls, **feats})

    fieldnames = [
        "class",
        "R_freq", "R_angle", "R_min", "R_max",
        "G_freq", "G_angle", "G_min", "G_max",
        "B_freq", "B_angle", "B_min", "B_max",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV      → {output_csv}")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DCT texture feature extractor")
    parser.add_argument("--image",  help="Path to real brain PNG")
    parser.add_argument("--mask",   help="Path to YOLO .txt mask file")
    parser.add_argument("--output", default="features.csv", help="Output CSV path")
    parser.add_argument("--batch",  action="store_true",
                        help="Batch mode: auto-pair images and masks by filename number")
    parser.add_argument("--image_dir",  default=".", help="Dir with real images (batch)")
    parser.add_argument("--mask_dir",   default=".", help="Dir with .txt masks (batch)")
    parser.add_argument("--output_dir", default=".", help="Dir for output CSVs (batch)")
    parser.add_argument("--no_overlay", action="store_true",
                        help="Skip saving overlay PNGs")
    args = parser.parse_args()

    if args.batch:
        imgs = sorted([f for f in os.listdir(args.image_dir)
                       if f.lower().endswith(".png")])
        paired = 0
        for img_name in imgs:
            m = re.search(r"(\d+)", img_name)
            if not m:
                continue
            num = m.group(1).zfill(6)
            txt_path = os.path.join(args.mask_dir, f"{num}.txt")
            if not os.path.exists(txt_path):
                print(f"  No mask for {img_name} (looked for {num}.txt), skipping")
                continue
            img_path = os.path.join(args.image_dir, img_name)
            out_csv  = os.path.join(args.output_dir, f"{num}_features.csv")
            process_pair(img_path, txt_path, out_csv,
                         save_overlay_img=not args.no_overlay)
            paired += 1
        print(f"\nBatch done — {paired} pair(s) processed.")
    else:
        if not args.image or not args.mask:
            parser.error("--image and --mask are required in single-file mode")
        process_pair(args.image, args.mask, args.output,
                     save_overlay_img=not args.no_overlay)


if __name__ == "__main__":
    main()
