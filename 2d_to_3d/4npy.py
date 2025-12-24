import cv2
import numpy as np
import os

png_path = "/raid/ron/multimodality_final/work/output/depth_with_alpha.png"
npy_path = os.path.splitext(png_path)[0] + ".npy"

depth = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)  # ✅ 保留原始 bit-depth / channel
if depth is None:
    raise FileNotFoundError(f"Cannot read: {png_path}")

# 如果你的 depth_raw.png 不小心是彩色(3/4通道)，可改成取單通道或轉灰階
# 這裡先保守處理：若是多通道就取第 0 通道
if depth.ndim == 3:
    depth = depth[..., 0]

np.save(npy_path, depth)
print("Saved:", npy_path, "shape=", depth.shape, "dtype=", depth.dtype)
