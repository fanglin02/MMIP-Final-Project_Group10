import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

# =====================================================
# Config
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = "image/rgb.png"          # input image
OUTPUT_MP4 = "output/parallax_final.mp4"

output_dir = os.path.dirname(OUTPUT_MP4)
os.makedirs(output_dir, exist_ok=True)

P_LOW, P_HIGH = 2, 98

# preview / video
SCALE = 1.0
N = 300
fps = 30

speed_bg = 1.5
speed_mg = 9.0
speed_fg = 180.0

pad = 20

# =====================================================
# Load model
# =====================================================
model = DepthAnything3.from_pretrained(
    "depth-anything/da3nested-giant-large"
).to(device).eval()

# =====================================================
# Load RGB
# =====================================================
rgb = cv2.imread(IMAGE_PATH)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]

# =====================================================
# Depth inference
# =====================================================
with torch.no_grad():
    pred = model.inference([IMAGE_PATH], export_dir=None, export_format="")
depth_raw = pred.depth[0].astype(np.float32)

# =====================================================
# Depth normalize + invert (bright = near)
# =====================================================
d_min = np.percentile(depth_raw, P_LOW)
d_max = np.percentile(depth_raw, P_HIGH)
depth_norm = np.clip((depth_raw - d_min) / (d_max - d_min + 1e-8), 0, 1)
depth = 1.0 - depth_norm

depth = cv2.resize(depth, (W, H))

# =====================================================
# Layer masks
# =====================================================
t_fg = np.percentile(depth, 53)
t_bg = np.percentile(depth, 5)

mask_fg = depth >= t_fg
mask_bg = depth <= t_bg
mask_mg = (~mask_fg) & (~mask_bg)

# =====================================================
# Morph helpers
# =====================================================
def morph_close(mask, k=21):
    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((k, k), np.uint8)
    ).astype(bool)

def dilate(mask, k=5):
    return cv2.dilate(mask.astype(np.uint8),
                      np.ones((k,k), np.uint8), 1).astype(bool)

def feather(alpha, k=9):
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(alpha, (k,k), 0)

def make_rgba(img, mask):
    m = dilate(mask)
    a = feather(m.astype(np.float32) * 255)
    return np.dstack([img, a.astype(np.uint8)])

# =====================================================
# Stable semantic MG (NO holes)
# =====================================================
mask_mg_sem = morph_close(mask_mg | mask_fg)
mask_mg_sem = mask_mg_sem & (~mask_fg)

# =====================================================
# RGBA layers
# =====================================================
FG = make_rgba(rgb, mask_fg)
MG = make_rgba(rgb, mask_mg_sem | mask_fg)   # visual MG = MG ∪ FG
BG = rgb.copy()

# =====================================================
# Seamless tiling
# =====================================================
def make_seamless_mirror_loop(layer, blend_w=64):
    H, W = layer.shape[:2]
    mirror = layer[:, ::-1].copy()
    right = layer[:, W-blend_w:W]
    left = mirror[:, :blend_w]

    alpha = np.linspace(0,1,blend_w)[None,:,None]
    seam = (1-alpha)*right + alpha*left

    return np.concatenate(
        [layer[:, :W-blend_w], seam.astype(layer.dtype), mirror[:, blend_w:]],
        axis=1
    )

def make_strip(layer, repeat=6):
    base = make_seamless_mirror_loop(layer)
    return np.concatenate([base]*repeat, axis=1)

BG_strip = make_strip(BG)
MG_strip = make_strip(MG)
FG_strip = make_strip(FG)

# =====================================================
# Motion blur (FG only)
# =====================================================
FG_strip = cv2.GaussianBlur(FG_strip, (51,1), 0)

# =====================================================
# Scroll + alpha composite
# =====================================================
def scroll_crop(layer, x, out_w):
    W_big = layer.shape[1]
    x = x % (W_big - out_w)
    return layer[:, x:x+out_w]

def alpha_comp(bg, fg):
    out = bg.astype(np.float32)
    a = fg[...,3:4] / 255.0
    out[...,:3] = fg[...,:3]*a + out[...,:3]*(1-a)
    return out.astype(np.uint8)

# =====================================================
# Prepare video
# =====================================================
Hp, Wp = int(H*SCALE), int(W*SCALE)
out_h, out_w = Hp-2*pad, Wp-2*pad

BG_p = cv2.resize(BG_strip, None, fx=SCALE, fy=SCALE)
MG_p = cv2.resize(MG_strip, None, fx=SCALE, fy=SCALE)
FG_p = cv2.resize(FG_strip, None, fx=SCALE, fy=SCALE)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, fps, (out_w, out_h))

# =====================================================
# Smooth lighting curve
# =====================================================
light_ctrl = np.random.uniform(0.8, 1.05, (1,10)).astype(np.float32)
light_curve = cv2.resize(light_ctrl, (N,1), interpolation=cv2.INTER_CUBIC).flatten()
light_curve = np.clip(light_curve, 0.6, 1.1)

# =====================================================
# Render
# =====================================================
for i in range(N):
    bg_x = int(i * speed_bg)
    mg_x = int(i * speed_mg)
    fg_x = int(i * speed_fg)

    shake_y = int(2.5*np.sin(i*0.5) + 1.5*np.cos(i*1.2))
    shake_x = int(1.0*np.sin(i*0.8))

    frame = scroll_crop(BG_p, bg_x, Wp)
    frame = alpha_comp(frame, scroll_crop(MG_p, mg_x, Wp))
    frame = alpha_comp(frame, scroll_crop(FG_p, fg_x, Wp))

    frame = cv2.convertScaleAbs(frame, alpha=light_curve[i])

    y0 = pad + shake_y
    x0 = pad + shake_x
    frame = frame[y0:y0+out_h, x0:x0+out_w]

    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

writer.release()

if os.path.exists(OUTPUT_MP4) and os.path.getsize(OUTPUT_MP4) > 0:
    print(f"[OK] Video successfully saved: {os.path.abspath(OUTPUT_MP4)} ")
else:
    print("[ERROR] Video file was not created correctly.")