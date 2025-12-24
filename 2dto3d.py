# -*- coding: utf-8 -*-
"""
SAM3 → Cutout → Depth → Pop-out → Parallax Video
ONE FILE PIPELINE (no intermediate visualization files)
Support multiple prompts → each prompt produces Ken Burns + Parallax
All outputs go to OUTPUT_ROOT
White background + shadow effect
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os, re
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import pipeline

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ============================================================
# 1. CONFIG
# ============================================================
IMAGE_PATH = "image/9_gt.png"
TEXT_PROMPTS = "train,church,house,model"  # 多 prompt 用逗號分隔
CONF_TH = 0.3

DEPTH_MODEL = "LiheYoung/depth-anything-small-hf"

TARGET = 512
K_LAYERS = 100

FPS = 30
DURATION = 5
MAX_DX = 40
MAX_DY = 20

REF_MOVE_SCALE = 0.25
NEAR_IS_HIGH_INDEX = True

OUTPUT_ROOT = "output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ============================================================
# 2. UTILS
# ============================================================
def _safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "prompt"

# ============================================================
# 3. SAM3 SEGMENTATION
# ============================================================
def run_sam3(image, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model().to(device).eval()
    processor = Sam3Processor(model, confidence_threshold=CONF_TH)

    state = processor.set_image(image)
    out = processor.set_text_prompt(state=state, prompt=text)

    masks = out.get("masks", None)
    scores = out.get("scores", None)

    if masks is None or scores is None:
        raise RuntimeError("No SAM3 masks found")

    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores)

    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)

    if masks.size == 0:
        raise RuntimeError(f"No masks for prompt: {text}")

    best = int(np.argmax(scores))
    mask = masks[best]
    if mask.ndim == 3:
        mask = mask[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

# ============================================================
# 4. CUTOUT RGBA
# ============================================================
def cutout_rgba(image, mask):
    rgb = np.array(image)
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = mask
    return rgba

# ============================================================
# 4.1 ADD SHADOW
# ============================================================
def add_shadow(rgba, offset=(10,10), shadow_strength=0.3, blur=15):
    H, W = rgba.shape[:2]
    
    shadow = np.zeros((H, W, 4), np.uint8)
    shadow[..., 3] = (rgba[..., 3] * shadow_strength).astype(np.uint8)
    
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    shadow_rgb = cv2.warpAffine(np.zeros((H, W, 3), np.uint8), M, (W, H), borderValue=0)
    shadow_alpha = cv2.warpAffine(shadow[..., 3], M, (W, H), borderValue=0)
    
    shadow_alpha = cv2.GaussianBlur(shadow_alpha, (blur|1, blur|1), 0)
    
    shadow_out = np.zeros_like(rgba, np.uint8)
    shadow_out[..., :3] = 0  
    shadow_out[..., 3] = shadow_alpha
    
    rgba_shadow = rgba.copy().astype(np.float32)
    a_shadow = shadow_out[..., 3:4] / 255.0
    rgba_shadow[..., :3] = (1 - a_shadow) * rgba_shadow[..., :3] + a_shadow * shadow_out[..., :3]
    rgba_shadow[..., 3] = np.clip(rgba[..., 3], 0, 255)
    
    return rgba_shadow.astype(np.uint8)

# ============================================================
# 5. DEPTH ESTIMATION
# ============================================================
def estimate_depth(rgba):
    h, w = rgba.shape[:2]
    a = rgba[..., 3]
    fg = a > 0

    bg = np.ones((h, w, 3), np.float32) * 255  # 白背景
    comp = rgba[..., :3] * (a[..., None] / 255.0) + bg * (1 - a[..., None] / 255.0)
    comp = comp.astype(np.uint8)

    pipe = pipeline("depth-estimation", model=DEPTH_MODEL,
                    device=0 if torch.cuda.is_available() else -1)
    depth = pipe(Image.fromarray(comp))["depth"].resize((w, h))
    depth = np.array(depth, np.float32)

    vals = depth[fg]
    dmin, dmax = vals.min(), vals.max()
    depth_n = (depth - dmin) / max(dmax - dmin, 1e-6)
    depth_n[~fg] = np.nan
    return depth_n

# ============================================================
# 6. POP-OUT LAYERS
# ============================================================
def build_layers(rgba, depth_n):
    rgba = cv2.resize(rgba, (TARGET, TARGET), cv2.INTER_LANCZOS4)
    depth_n = cv2.resize(depth_n, (TARGET, TARGET), cv2.INTER_CUBIC)

    obj = ~np.isnan(depth_n)
    vals = depth_n[obj]
    edges = np.percentile(vals, np.linspace(0, 100, K_LAYERS + 1))

    layers = []
    for i in range(K_LAYERS):
        lo, hi = edges[i], edges[i + 1]
        if i < K_LAYERS - 1:
            m = (depth_n >= lo) & (depth_n < hi) & obj
        else:
            m = (depth_n >= lo) & (depth_n <= hi) & obj
        out = rgba.copy().astype(np.float32)
        out[..., 3] *= m.astype(np.float32)
        layers.append(out.astype(np.uint8))
    return layers, rgba

# ============================================================
# 7. COMPOSITING HELPERS
# ============================================================
def shift_rgba_premult(img, dx, dy):
    H, W = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    img = img.astype(np.float32)
    a = img[..., 3:4] / 255.0
    prem = img[..., :3] * a

    prem_w = cv2.warpAffine(prem, M, (W, H), borderValue=0)
    a_w = cv2.warpAffine(a, M, (W, H), borderValue=0)[..., None]

    rgb = np.zeros_like(prem_w)
    m = a_w > 1e-6
    rgb[m[..., 0]] = prem_w[m[..., 0]] / a_w[m[..., 0]]

    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = np.clip(a_w[..., 0] * 255, 0, 255)
    return out

def alpha_blend(base, top):
    a = top[..., 3:4] / 255.0
    return base * (1 - a) + top[..., :3] * a

# ============================================================
# 8. VIDEO RENDERING
# ============================================================
def render_videos(base, ref, layers, prompt_name):
    H, W = base.shape[:2]
    K = len(layers)

    order = list(range(K))
    depth_f = np.linspace(0, 1, K)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    kb_name = os.path.join(OUTPUT_ROOT, f"kenburns_{prompt_name}.mp4")
    px_name = os.path.join(OUTPUT_ROOT, f"parallax_{prompt_name}.mp4")

    def composite(dx, dy):
        out = base.astype(np.float32)
        ref_s = shift_rgba_premult(ref, int(dx * REF_MOVE_SCALE), int(dy * REF_MOVE_SCALE))
        out = alpha_blend(out, ref_s)
        for i, idx in enumerate(order):
            lay = shift_rgba_premult(layers[idx], int(dx * depth_f[i]), int(dy * depth_f[i]))
            out = alpha_blend(out, lay)
        return np.clip(out, 0, 255).astype(np.uint8)

    # Ken Burns
    kb = cv2.VideoWriter(kb_name, fourcc, FPS, (W, H))
    for i in range(FPS * DURATION):
        t = i / (FPS * DURATION)
        frame = composite(0, int(MAX_DY * np.sin(2 * np.pi * t)))
        kb.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    kb.release()
    print(f"[OK] Video successfully saved:  {kb_name}")

    # Parallax
    px = cv2.VideoWriter(px_name, fourcc, FPS, (W, H))
    for i in range(FPS * DURATION):
        t = i / (FPS * DURATION)
        frame = composite(int(MAX_DX * np.sin(2 * np.pi * t)), 0)
        px.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    px.release()
    print(f"[OK] Video successfully saved:  {px_name}")

# ============================================================
# 9. MAIN
# ============================================================
def main():
    image = Image.open(IMAGE_PATH).convert("RGB")
    prompts = [p.strip() for p in TEXT_PROMPTS.split(",") if p.strip()]

    for p in prompts:
        print(f"\n=== Processing prompt: {p} ===")
        prompt_name = _safe_name(p)

        # 1) SAM3 → mask
        mask = run_sam3(image, p)

        # 2) Cutout RGBA + Shadow
        rgba = cutout_rgba(image, mask)
        rgba = add_shadow(rgba, offset=(10, 10), shadow_strength=0.3)

        # 3) Depth estimation
        depth_n = estimate_depth(rgba)

        # 4) Pop-out layers
        layers, ref = build_layers(rgba, depth_n)

        # 5) Base image with white background
        alpha = rgba[..., 3:4] / 255.0
        white_bg = np.ones_like(rgba[..., :3], dtype=np.float32) * 255
        base = white_bg * (1 - alpha) + rgba[..., :3] * alpha
        base = base.astype(np.uint8)
        base = cv2.resize(base, (TARGET, TARGET), cv2.INTER_LANCZOS4)

        # 6) Render videos
        render_videos(base, ref, layers, prompt_name)

if __name__ == "__main__":
    main()
