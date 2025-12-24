# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# ==========================================
# CONFIG
# ==========================================
IMAGE_PATH = "/raid/ron/multimodality_final/work/composited_selected.png"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "LiheYoung/depth-anything-small-hf"

# ==========================================
# 1) Load RGBA + keep alpha
# ==========================================
img = Image.open(IMAGE_PATH)

if img.mode != "RGBA":
    # 如果不是 RGBA，也照樣做，但 alpha 會變成全不透明
    img = img.convert("RGBA")

rgba = np.array(img)                    # (H,W,4)
rgb  = rgba[..., :3].astype(np.float32) # (H,W,3)
a    = rgba[..., 3].astype(np.uint8)    # (H,W) 0~255

h, w = a.shape
alpha_mask = (a > 0).astype(np.uint8)

# 用中性灰把透明背景補起來再送模型（避免黑底干擾深度）
bg = np.ones_like(rgb) * 127.0
a_f = (a.astype(np.float32) / 255.0)[..., None]
comp = rgb * a_f + bg * (1.0 - a_f)
image_rgb_for_depth = Image.fromarray(comp.astype(np.uint8), mode="RGB")

# ==========================================
# 2) Depth inference
# ==========================================
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(task="depth-estimation", model=MODEL_NAME, device=device)

result = pipe(image_rgb_for_depth)
depth_pil = result["depth"].resize((w, h), resample=Image.BICUBIC)

depth_np = np.array(depth_pil).astype(np.float32)

# ==========================================
# 3) Normalize ONLY on foreground (alpha>0)
# ==========================================
if alpha_mask.sum() > 10:
    vals = depth_np[alpha_mask > 0]
    dmin, dmax = float(vals.min()), float(vals.max())
else:
    dmin, dmax = float(depth_np.min()), float(depth_np.max())

den = max(dmax - dmin, 1e-6)
depth_u8 = np.clip((depth_np - dmin) / den * 255.0, 0, 255).astype(np.uint8)

# 背景區不重要，但我們保持透明；RGB 可以隨便填（這裡填 0）
depth_u8[alpha_mask == 0] = 0

# ==========================================
# 4) Build output: depth as grayscale + original alpha
# ==========================================
depth_L = Image.fromarray(depth_u8, mode="L")  # 灰階深度
depth_L.putalpha(Image.fromarray(a, mode="L")) # 套回原 alpha

out_path = os.path.join(OUTPUT_DIR, "depth_with_alpha.png")
depth_L.save(out_path)
print(f"✅ Saved: {out_path}")
