# -*- coding: utf-8 -*-
import os, glob, re
import cv2
import numpy as np
from PIL import Image

# ============================================================
# CONFIG  (✅ 100 層版本：修白線、底圖不動、不放大)
# ============================================================
LAYER_DIR = "/raid/ron/multimodality_final/work/popout_output_cutout100/layers"  # layer_000.png ~ layer_099.png
BASE_PATH = "/raid/ron/multimodality_final/work/image.png"                      # 底圖（不動）
REF_PATH  = "/raid/ron/multimodality_final/work/cutouts_json_out/train/object_id000_s0.928.png"  # 完整物件cutout（補縫）

OUT_KB = "kenburns_100layers_fix_stripes.mp4"
OUT_PX = "parallax_100layers_fix_stripes.mp4"

FPS = 30
DURATION = 5
TARGET = 512

MAX_DX = 40
MAX_DY = 20

# True：layer_099 最靠近(移動最多)；False：layer_000 最靠近
NEAR_IS_HIGH_INDEX = True

# 參考層跟著動的比例（0=不動；0.2~0.4 通常最好）
REF_MOVE_SCALE = 0.25

# ============================================================
# LOAD BASE + REF + LAYERS
# ============================================================
print("[1] Loading base/ref/layers...")

for p in [BASE_PATH, REF_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"❌ Not found: {p}")

base = Image.open(BASE_PATH).convert("RGBA").resize((TARGET, TARGET), Image.LANCZOS)
base_np = np.array(base)  # RGBA

ref = Image.open(REF_PATH).convert("RGBA").resize((TARGET, TARGET), Image.LANCZOS)
ref_np = np.array(ref)    # RGBA

H, W = TARGET, TARGET

paths = glob.glob(os.path.join(LAYER_DIR, "layer_*.png"))
if len(paths) == 0:
    raise FileNotFoundError(f"❌ No layer_*.png in: {LAYER_DIR}")

def layer_index(p):
    m = re.search(r"layer_(\d+)\.png$", os.path.basename(p))
    return int(m.group(1)) if m else -1

paths = sorted(paths, key=layer_index)
K = len(paths)
print(f"✅ Found {K} layers (expect 100)")

layers = []
for p in paths:
    im = Image.open(p).convert("RGBA").resize((TARGET, TARGET), Image.LANCZOS)
    layers.append(np.array(im))
layers = np.stack(layers, axis=0)  # (K,H,W,4)

# 遠->近 合成順序 + 深度位移係數
if NEAR_IS_HIGH_INDEX:
    order = list(range(K))                 # 0(遠) -> K-1(近)
    depth_factor = np.linspace(0.0, 1.0, K)
else:
    order = list(reversed(range(K)))       # K-1(遠) -> 0(近)
    depth_factor = np.linspace(0.0, 1.0, K)

# ============================================================
# HELPERS
# ============================================================
def shift_rgba_premult(img_rgba, dx, dy):
    """warpAffine with premultiplied alpha to avoid white halos"""
    if dx == 0 and dy == 0:
        return img_rgba

    M = np.float32([[1, 0, dx], [0, 1, dy]])

    img = img_rgba.astype(np.float32)
    a = img[..., 3:4] / 255.0
    prem_rgb = img[..., :3] * a

    prem_w = cv2.warpAffine(
        prem_rgb, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    a_w = cv2.warpAffine(
        a, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )[..., None]

    rgb = np.zeros_like(prem_w)
    valid = a_w > 1e-6
    rgb[valid[..., 0]] = prem_w[valid[..., 0]] / a_w[valid[..., 0]]

    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3]  = np.clip(a_w[..., 0] * 255.0, 0, 255).astype(np.uint8)
    return out

def alpha_blend(base_rgb, top_rgba):
    a = top_rgba[..., 3:4].astype(np.float32) / 255.0
    return base_rgb * (1 - a) + top_rgba[..., :3].astype(np.float32) * a

def composite_frame(dx_global=0, dy_global=0):
    # 1) 底圖固定
    out = base_np[..., :3].astype(np.float32)

    # 2) ✅ 先墊「完整物件參考層」在切片下面：補縫防白條
    ref_dx = int(dx_global * REF_MOVE_SCALE)
    ref_dy = int(dy_global * REF_MOVE_SCALE)
    ref_s  = shift_rgba_premult(ref_np, ref_dx, ref_dy)
    out = alpha_blend(out, ref_s)

    # 3) 再疊切片（遠->近）
    for j, idx in enumerate(order):
        f = depth_factor[j]
        dx = int(dx_global * f)
        dy = int(dy_global * f)
        lay = shift_rgba_premult(layers[idx], dx, dy)
        out = alpha_blend(out, lay)

    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# A) 上下微擺動（NO zoom）
# ============================================================
print("[2] Rendering KenBurns-like (NO zoom)...")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
kb_writer = cv2.VideoWriter(OUT_KB, fourcc, FPS, (W, H))

frames = FPS * DURATION
for i in range(frames):
    t = i / frames
    dy = int(MAX_DY * np.sin(2 * np.pi * t))
    frame = composite_frame(dx_global=0, dy_global=dy)
    kb_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

kb_writer.release()
print("Saved:", OUT_KB)

# ============================================================
# B) 左右視差（NO zoom）
# ============================================================
print("[3] Rendering Parallax (NO zoom)...")

px_writer = cv2.VideoWriter(OUT_PX, fourcc, FPS, (W, H))

for i in range(frames):
    t = i / frames
    dx = int(MAX_DX * np.sin(2 * np.pi * t))
    frame = composite_frame(dx_global=dx, dy_global=0)
    px_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

px_writer.release()
print("Saved:", OUT_PX)

print("\n✅ Done:", OUT_KB, OUT_PX)
