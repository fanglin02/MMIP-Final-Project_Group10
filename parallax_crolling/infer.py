import os
import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from tqdm import tqdm

# =====================================================
# Config
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_LIST = [
    "/ssd6/fang/MMIP_final/parallax_scroll/image/rgb.png"
]

EXPORT_ROOT = "output_da3_real"
os.makedirs(EXPORT_ROOT, exist_ok=True)

P_LOW = 2     # robust normalization
P_HIGH = 98

# =====================================================
# Load model
# =====================================================
model = DepthAnything3.from_pretrained(
    "depth-anything/da3nested-giant-large"
).to(device)

model.eval()

# =====================================================
# Inference
# =====================================================
with torch.no_grad():
    pred = model.inference(
        IMAGE_LIST,
        export_dir=None,
        export_format=""
    )

depth_batch = pred.depth  # [N, H, W]

# =====================================================
# Process each image
# =====================================================
for idx, img_path in enumerate(tqdm(IMAGE_LIST)):
    name = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(EXPORT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------
    # Load RGB
    # -------------------------------------------------
    rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert rgb is not None, f"Failed to load {img_path}"
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    H, W = rgb.shape[:2]

    cv2.imwrite(
        os.path.join(out_dir, "rgb.png"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    )

    # -------------------------------------------------
    # Raw depth
    # 模型直接輸出的深度(未正規化、未修正方向)
    # -------------------------------------------------
    depth_raw = depth_batch[idx].astype(np.float32)
    np.save(os.path.join(out_dir, "depth_raw.npy"), depth_raw)

    # -------------------------------------------------
    # Robust normalization (percentile)
    # 將深度映射到 [0,1] (用於後續分層、parallax、warping)
    # -------------------------------------------------
    d_min = np.percentile(depth_raw, P_LOW)
    d_max = np.percentile(depth_raw, P_HIGH)

    depth_norm = (depth_raw - d_min) / (d_max - d_min + 1e-8)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    np.save(os.path.join(out_dir, "depth_norm.npy"), depth_norm)

    # -------------------------------------------------
    # Depth direction unification
    # Convention: bright = near
    # DA3 的深度輸出方向統一為 亮 = 近、暗 = 遠 (確保後續分層或 3D 運算一致)
    # -------------------------------------------------
    depth_norm_inv = 1.0 - depth_norm
    np.save(os.path.join(out_dir, "depth_norm_inv.npy"), depth_norm_inv)

    # -------------------------------------------------
    # Visualization
    # 將深度轉成彩色圖，方便檢查層次
    # -------------------------------------------------
    depth_vis = (depth_norm_inv * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    cv2.imwrite(
        os.path.join(out_dir, "depth_vis.png"),
        depth_vis
    )

    # -------------------------------------------------
    # Depth histogram (research-friendly)
    # 記錄深度分佈，方便研究、判斷 FG/MG/BG 門檻(可用於後續自適應分層)
    # -------------------------------------------------
    hist, bins = np.histogram(depth_raw.flatten(), bins=256)

    np.savez(
        os.path.join(out_dir, "depth_hist.npz"),
        hist=hist,
        bins=bins
    )

    # -------------------------------------------------
    # Heuristic depth confidence
    # Idea: smooth region = more reliable
    # 平滑區域 → 高可信度，物體邊界 → 低可信度(可用於 inpainting / warping / 修補)
    # -------------------------------------------------
    gx = cv2.Sobel(depth_norm_inv, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_norm_inv, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)

    depth_conf = np.exp(-grad * 10.0)
    depth_conf = np.clip(depth_conf, 0.0, 1.0)

    np.save(os.path.join(out_dir, "depth_conf.npy"), depth_conf)

    # -------------------------------------------------
    # Quick sanity visualization (optional but useful)
    # 快速檢查可信度分佈(可視化邊界和可靠區域)
    # -------------------------------------------------
    conf_vis = (depth_conf * 255).astype(np.uint8)
    conf_vis = cv2.applyColorMap(conf_vis, cv2.COLORMAP_TURBO)

    cv2.imwrite(
        os.path.join(out_dir, "depth_conf_vis.png"),
        conf_vis
    )

    print(f"[✓] Finished processing: {name}")

