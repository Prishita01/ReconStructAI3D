#!/usr/bin/env python3
"""
evaluate_depths.py – Bulk compare COLMAP vs RTAB depth maps when run
from inside the RTAB_Colamp/ folder.

Assumes this structure:

    RTAB_Colamp/
    ├── evaluate_depths.py      ← you drop it here
    ├── depth/                  ← RTAB‑Map “ground‑truth” PNGs (e.g. 1.png, 2.png…)
    └── depth_pngs_colmap/      ← COLMAP depth PNGs (e.g. 1.jpg.geometric.png)
"""
import os, csv
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)

def eval_depth(colmap_path, gt_path):
    col = load_gray(colmap_path)
    gt  = load_gray(gt_path)

    # resize COLMAP → GT resolution
    col_rs = cv2.resize(col, (gt.shape[1], gt.shape[0]),
                       interpolation=cv2.INTER_NEAREST)

    # mask out invalid (zero) pixels
    mask = (gt > 0) & (col_rs > 0)
    if mask.sum() == 0:
        raise ValueError(f"No overlap between {os.path.basename(colmap_path)} & {os.path.basename(gt_path)}")

    # align scales by median
    scale = np.median(gt[mask]) / (np.median(col_rs[mask]) + 1e-8)
    col_aligned = col_rs * scale

    # compute metrics
    gt_v  = gt[mask]
    col_v = col_aligned[mask]
    mse   = np.mean((col_v - gt_v) ** 2)
    rmse  = np.sqrt(mse)
    psnr  = peak_signal_noise_ratio(gt_v, col_v,
                                    data_range=gt_v.max() - gt_v.min())

    # SSIM on full images (invalids zeroed)
    gt_full  = gt.copy();  gt_full[~mask] = 0
    col_full = col_aligned.copy(); col_full[~mask] = 0
    ssim = structural_similarity(gt_full, col_full,
                                 data_range=gt_full.max() - gt_full.min())

    return {"scale": scale, "RMSE": rmse, "PSNR": psnr, "SSIM": ssim}

def main():
    root    = os.path.dirname(os.path.abspath(__file__))
    col_dir = os.path.join(root, "depth_pngs_colmap")
    gt_dir  = os.path.join(root, "depth")
    out_csv = os.path.join(root, "depth_metrics.csv")

    # find matching pairs
    pairs = []
    for fn in os.listdir(col_dir):
        if not fn.endswith(".geometric.png"):
            continue
        # strip ".geometric.png"
        core = fn[:-len(".geometric.png")]   # e.g. "1.jpg" or "10.jpg"
        # strip the trailing ".jpg"
        if core.lower().endswith(".jpg"):
            base = core[:-4]                  # e.g. "1" or "10"
        else:
            base = core
        gt_path = os.path.join(gt_dir, base + ".png")
        col_path = os.path.join(col_dir, fn)
        if os.path.isfile(gt_path):
            pairs.append((col_path, gt_path, base))
        else:
            print(f"⚠ warning: no GT for {fn} → expected {base}.png")

    if not pairs:
        print("No matching PNG pairs found in depth_pngs_colmap/ vs depth/")
        return

    # evaluate and write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","scale","RMSE","PSNR","SSIM"])
        writer.writeheader()
        for col, gt, key in pairs:
            try:
                m = eval_depth(col, gt)
                writer.writerow({"image": key, **m})
                print(f"✔ {key}: PSNR={m['PSNR']:.2f}, SSIM={m['SSIM']:.3f}")
            except Exception as e:
                print(f"⚠ {key} error: {e}")

    print(f"\nDone! Metrics saved to {out_csv}")

if __name__ == "__main__":
    main()