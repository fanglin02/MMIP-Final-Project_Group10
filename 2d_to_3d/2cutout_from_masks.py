# -*- coding: utf-8 -*-
import os, json, glob, argparse
import numpy as np
from PIL import Image

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_mask(path, size_wh):
    m = Image.open(path).convert("L")
    if m.size != size_wh:
        m = m.resize(size_wh, resample=Image.NEAREST)
    m = np.array(m, dtype=np.uint8)
    return np.where(m > 127, 255, 0).astype(np.uint8)

def make_rgba(rgb_np, alpha_np):
    return np.concatenate([rgb_np, alpha_np[..., None]], axis=-1)

def clamp_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, W-1)); x2 = int(np.clip(x2, 0, W-1))
    y1 = int(np.clip(y1, 0, H-1)); y2 = int(np.clip(y2, 0, H-1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def find_mask(mask_root, prompt, mid):
    # 預設檔名：mask_000.png
    p = os.path.join(mask_root, prompt, f"mask_{mid:03d}.png")
    if os.path.exists(p):
        return p
    # 退而求其次：mask_*{id}*.png
    cand = sorted(glob.glob(os.path.join(mask_root, prompt, f"mask_*{mid}*.png")))
    return cand[0] if cand else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/raid/ron/multimodality_final/work/9_gt.png")
    ap.add_argument("--json_path", default="/raid/ron/multimodality_final/work/sam3_out/results.json")
    ap.add_argument("--mask_root", default="/raid/ron/multimodality_final/work/sam3_out")
    ap.add_argument("--outdir", default="cutouts_json_out")
    ap.add_argument("--score_th", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--save_bg_rgba", action="store_true", help="另存背景 RGBA（物件透明）")
    ap.add_argument("--crop_by_box", action="store_true", help="用 box 裁 ROI（輸出會是 ROI 尺寸）")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    rgb = np.array(img, dtype=np.uint8)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    union = np.zeros((H, W), dtype=np.uint8)

    for entry in data:
        prompt = entry["prompt"]
        items = entry.get("items", [])

        # score filter
        items = [it for it in items if float(it.get("score", 0.0)) >= args.score_th]

        # top-k by score
        if args.topk and args.topk > 0:
            items = sorted(items, key=lambda it: float(it.get("score", 0.0)), reverse=True)[:args.topk]

        if not items:
            print(f"[SKIP] {prompt}: no items after filtering")
            continue

        p_dir = os.path.join(args.outdir, prompt)
        ensure_dir(p_dir)

        for it in items:
            mid = int(it["id"])
            score = float(it.get("score", 0.0))
            box = it.get("box_xyxy", None)

            mask_path = find_mask(args.mask_root, prompt, mid)
            if mask_path is None:
                print(f"[WARN] mask not found: prompt={prompt} id={mid}")
                continue

            m = read_mask(mask_path, (W, H))   # (H,W) 0/255
            union = np.maximum(union, m)
            inv = (255 - m).astype(np.uint8)

            tag = f"id{mid:03d}_s{score:.3f}"

            if args.crop_by_box and box is not None:
                x1, y1, x2, y2 = clamp_box_xyxy(box, W, H)
                # 加 1px 邊界避免裁太緊
                x2 = min(W-1, x2+1); y2 = min(H-1, y2+1)

                rgb_roi = rgb[y1:y2+1, x1:x2+1]
                m_roi   = m[y1:y2+1, x1:x2+1]
                inv_roi = inv[y1:y2+1, x1:x2+1]

                obj_rgba = make_rgba(rgb_roi, m_roi)
                Image.fromarray(obj_rgba, mode="RGBA").save(os.path.join(p_dir, f"object_{tag}_roi.png"))

                bg_rgb = rgb_roi.copy()
                bg_rgb[m_roi > 0] = 0
                Image.fromarray(bg_rgb, mode="RGB").save(os.path.join(p_dir, f"bg_{tag}_roi.png"))

                if args.save_bg_rgba:
                    bg_rgba = make_rgba(rgb_roi, inv_roi)
                    Image.fromarray(bg_rgba, mode="RGBA").save(os.path.join(p_dir, f"bg_{tag}_roi_rgba.png"))

            else:
                obj_rgba = make_rgba(rgb, m)
                Image.fromarray(obj_rgba, mode="RGBA").save(os.path.join(p_dir, f"object_{tag}.png"))

                bg_rgb = rgb.copy()
                bg_rgb[m > 0] = 0
                Image.fromarray(bg_rgb, mode="RGB").save(os.path.join(p_dir, f"bg_{tag}.png"))

                if args.save_bg_rgba:
                    bg_rgba = make_rgba(rgb, inv)
                    Image.fromarray(bg_rgba, mode="RGBA").save(os.path.join(p_dir, f"bg_{tag}_rgba.png"))

        print(f"[OK] {prompt}: exported {len(items)} items")

    # union 背景：移除所有物件
    bg_all = rgb.copy()
    bg_all[union > 0] = 0
    Image.fromarray(bg_all, mode="RGB").save(os.path.join(args.outdir, "background_all_removed.png"))

    bg_all_rgba = make_rgba(rgb, (255 - union).astype(np.uint8))
    Image.fromarray(bg_all_rgba, mode="RGBA").save(os.path.join(args.outdir, "background_all_removed_rgba.png"))

    print(f"[DONE] saved to {args.outdir}")

if __name__ == "__main__":
    main()
