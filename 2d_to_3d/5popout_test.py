# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from PIL import Image

# =========================
# CONFIG
# =========================
IMG_PATH   = "/raid/ron/multimodality_final/work/composited_selected.png"
DEPTH_PATH = "/raid/ron/multimodality_final/work/output/depth_with_alpha.npy"  # float depth (H,W)
OUT_DIR    = "/raid/ron/multimodality_final/work/popout_output_cutout100"
TARGET = 512

K = 100          # ✅ 切 100 層
MIN_ALPHA = 1    # alpha > MIN_ALPHA 視為物體
SAVE_MASK = True # 是否輸出每層 mask PNG

os.makedirs(OUT_DIR, exist_ok=True)
layers_dir = os.path.join(OUT_DIR, "layers")
masks_dir  = os.path.join(OUT_DIR, "masks")
os.makedirs(layers_dir, exist_ok=True)
if SAVE_MASK:
    os.makedirs(masks_dir, exist_ok=True)

# =========================
# 1) Load RGBA (keep alpha!)
# =========================
img = Image.open(IMG_PATH).convert("RGBA").resize((TARGET, TARGET), Image.LANCZOS)
rgba = np.array(img)  # H,W,4
a = rgba[..., 3].astype(np.uint8)
obj_mask = (a > MIN_ALPHA)

H, W = a.shape
if obj_mask.sum() < 50:
    raise RuntimeError("Alpha 前景太少，確認輸入是否真的是透明 cutout PNG。")

# =========================
# 2) Load depth & resize
# =========================
depth = np.load(DEPTH_PATH).astype(np.float32)
depth = cv2.resize(depth, (TARGET, TARGET), interpolation=cv2.INTER_CUBIC)

# =========================
# 3) Normalize depth ONLY on object region
# =========================
vals = depth[obj_mask]
dmin, dmax = float(vals.min()), float(vals.max())
den = max(dmax - dmin, 1e-6)

depth_n = (depth - dmin) / den
depth_n = np.clip(depth_n, 0.0, 1.0)

# 物體外不參與分層（設 NaN）
depth_n[~obj_mask] = np.nan

# =========================
# 4) Split into K layers by percentiles (object-only)
# =========================
vals_n = depth_n[obj_mask]
edges = np.percentile(vals_n, np.linspace(0, 100, K + 1))

# =========================
# 5) Save RGBA layers: alpha = original_alpha * layer_mask
# =========================
def save_layer(layer_mask_bool, out_png_path):
    m = (layer_mask_bool.astype(np.uint8) * 255)  # 0/255 mask
    out = rgba.copy().astype(np.float32)
    out[..., 3] = out[..., 3] * (m.astype(np.float32) / 255.0)  # 保留原本邊緣 alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(out_png_path)
    return m

for i in range(K):
    lo, hi = edges[i], edges[i + 1]

    if i < K - 1:
        layer_mask = (depth_n >= lo) & (depth_n < hi) & obj_mask
    else:
        layer_mask = (depth_n >= lo) & (depth_n <= hi) & obj_mask

    out_path = os.path.join(layers_dir, f"layer_{i:03d}.png")  # ✅ 3 位數：000~099
    m = save_layer(layer_mask, out_path)

    if SAVE_MASK:
        cv2.imwrite(os.path.join(masks_dir, f"mask_{i:03d}.png"), m)

    if i % 10 == 0:
        print(f"Saved {i:03d}/{K-1:03d}")

# =========================
# Debug depth visualization (object only)
# =========================
vis = np.zeros((H, W), np.uint8)
vis[obj_mask] = (vals_n * 255).astype(np.uint8)
Image.fromarray(vis, "L").save(os.path.join(OUT_DIR, "depth_obj_vis.png"))

print("\n✅ Done. Output saved to:", OUT_DIR)
print(" - layers/layer_000.png ~ layer_099.png (RGBA, transparent kept)")
if SAVE_MASK:
    print(" - masks/mask_000.png ~ mask_099.png")
print(" - depth_obj_vis.png (debug)")
