# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ✅ 官方 sam3 API（不是 transformers）
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "prompt"


def _get_colors(n: int):
    rng = np.random.default_rng(123)
    colors = rng.integers(0, 255, size=(max(n, 1), 3), dtype=np.uint8)
    return [tuple(map(int, c.tolist())) for c in colors]


def normalize_mask_2d(mi: np.ndarray) -> np.ndarray:
    """
    壓成 (H,W) 的 uint8 {0,1}
    """
    mi = np.asarray(mi)
    mi = np.squeeze(mi)

    if mi.ndim != 2:
        raise ValueError(f"Mask cannot be converted to 2D. Got shape={mi.shape}, dtype={mi.dtype}")

    if mi.dtype == np.uint8:
        mi = (mi > 0).astype(np.uint8)
    else:
        mi = (mi > 0.5).astype(np.uint8)

    return mi


def normalize_masks_stack(masks: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    回傳 (N,H,W) uint8 {0,1}
    - 若 masks 為 None / 空 list / 空 array，回傳 (0,H,W)
    """
    if masks is None:
        return np.zeros((0, H, W), dtype=np.uint8)

    masks = np.asarray(masks)

    # ✅ 空結果：例如 [] -> shape (0,)
    if masks.size == 0:
        return np.zeros((0, H, W), dtype=np.uint8)

    # (H,W) -> (1,H,W)
    if masks.ndim == 2:
        masks = masks[None, ...]

    # (N,1,H,W) -> (N,H,W)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    # 仍然不是 (N,H,W) 就 squeeze
    if masks.ndim != 3:
        masks = np.squeeze(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.size == 0:
            return np.zeros((0, H, W), dtype=np.uint8)
        if masks.ndim != 3:
            raise ValueError(f"Unexpected masks stack shape after squeeze: {masks.shape}")

    # N==0 也要處理
    if masks.shape[0] == 0:
        return np.zeros((0, H, W), dtype=np.uint8)

    out = []
    for i in range(masks.shape[0]):
        out.append(normalize_mask_2d(masks[i]))

    # ✅ 如果 out 仍為空（保險）
    if len(out) == 0:
        return np.zeros((0, H, W), dtype=np.uint8)

    return np.stack(out, axis=0)


def overlay_masks(image: Image.Image, masks_any: np.ndarray, alpha: int = 120) -> Image.Image:
    base = image.convert("RGBA")
    H, W = image.size[1], image.size[0]

    masks = normalize_masks_stack(masks_any, H=H, W=W)
    if masks.shape[0] == 0:
        return base

    colors = _get_colors(masks.shape[0])

    for i in range(masks.shape[0]):
        m = masks[i]
        if int(m.sum()) == 0:
            continue

        overlay = Image.new("RGBA", base.size, colors[i] + (0,))
        alpha_img = Image.fromarray((m * alpha).astype(np.uint8), mode="L")
        overlay.putalpha(alpha_img)
        base = Image.alpha_composite(base, overlay)

    return base


def draw_boxes(image_rgba: Image.Image, boxes_xyxy: np.ndarray, scores: np.ndarray) -> Image.Image:
    out = image_rgba.copy()
    draw = ImageDraw.Draw(out)
    colors = _get_colors(len(boxes_xyxy))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    W, H = out.size
    for i, (box, sc) in enumerate(zip(boxes_xyxy, scores)):
        x1, y1, x2, y2 = box
        x1 = int(np.clip(x1, 0, W - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        y2 = int(np.clip(y2, 0, H - 1))
        draw.rectangle([x1, y1, x2, y2], outline=colors[i], width=3)
        draw.text((x1, max(0, y1 - 14)), f"{i}:{float(sc):.3f}", fill=(255, 255, 255, 255), font=font)
    return out


def parse_prompts(text: str) -> List[str]:
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    text = text.strip()
    return [text] if text else []


def to_numpy(x, default: np.ndarray) -> np.ndarray:
    if x is None:
        return default
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/raid/ron/multimodality_final/work/9_gt.png")
    ap.add_argument("--text", default="train, church, house,model")
    ap.add_argument("--outdir", default="./sam3_out")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--alpha", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (device == "cuda" and args.fp16) else torch.float32

    image = Image.open(args.image).convert("RGB")
    H, W = image.size[1], image.size[0]

    if args.ckpt.strip():
        model = build_sam3_image_model(checkpoint_path=args.ckpt.strip())
    else:
        model = build_sam3_image_model()

    model = model.to(device=device)
    if dtype == torch.float16:
        model = model.half()
    model.eval()

    processor = Sam3Processor(model, confidence_threshold=args.conf)
    state = processor.set_image(image)

    prompts = parse_prompts(args.text)
    if not prompts:
        raise ValueError("Empty --text")

    all_meta = []

    for p in prompts:
        out = processor.set_text_prompt(state=state, prompt=p)

        masks_np_raw = to_numpy(out.get("masks", None), default=np.zeros((0, H, W), dtype=np.uint8))
        boxes_np = to_numpy(out.get("boxes", None), default=np.zeros((0, 4), dtype=np.float32))
        scores_np = to_numpy(out.get("scores", None), default=np.zeros((0,), dtype=np.float32))

        masks_np = normalize_masks_stack(masks_np_raw, H=H, W=W)

        n = int(masks_np.shape[0])
        print(f"[PROMPT] {p} -> {n} masks   masks_shape={masks_np.shape} dtype={masks_np.dtype}")

        p_name = _safe_name(p)
        p_dir = os.path.join(args.outdir, p_name)
        os.makedirs(p_dir, exist_ok=True)

        items = []
        for i in range(n):
            m = (masks_np[i] * 255).astype(np.uint8)
            Image.fromarray(m, mode="L").save(os.path.join(p_dir, f"mask_{i:03d}.png"))
            items.append({
                "id": i,
                "score": float(scores_np[i]) if i < len(scores_np) else None,
                "box_xyxy": [float(x) for x in (boxes_np[i].tolist() if i < len(boxes_np) else [0, 0, 0, 0])],
            })

        overlay = overlay_masks(image, masks_np, alpha=int(np.clip(args.alpha, 0, 255)))
        if len(boxes_np) > 0 and len(scores_np) > 0:
            overlay = draw_boxes(overlay, boxes_np, scores_np)
        overlay.save(os.path.join(args.outdir, f"overlay_{p_name}.png"))

        all_meta.append({"prompt": p, "count": n, "items": items})

    with open(os.path.join(args.outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
