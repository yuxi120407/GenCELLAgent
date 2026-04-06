import os
import json
import re
import time
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SAM3_CHECKPOINT = "/home/idies/workspace/Storage/xyu1/persistent/GenCELLAgent/src/sam3/checkpoints/sam3/sam3.pt"
GEMINI_MODEL = "gemini-3-flash-preview"

BBX_PROMPT_TEMPLATE = """You are a cell biology expert analyzing microscopy images.
Your task is to locate and draw bounding boxes around: {prompt}

Carefully examine the image and identify all visible instances of the target structure.

Return a JSON object with the following keys:
- "positive_boxes": list of bounding boxes, each as [ymin, xmin, ymax, xmax] in normalized coordinates (0-1000 scale, where 1000 = full image dimension)
- "positive_points": list of [y, x] center points inside target structures (0-1000 scale)
- "negative_points": list of [y, x] points clearly in the background or non-target regions (0-1000 scale)

Guidelines:
- Provide 1-8 bounding boxes covering the main target instances
- Include 2-6 positive points inside clear examples of the target structure
- Include 1-3 negative points in background regions
- Use the full 0-1000 scale for all coordinates

Return ONLY a valid JSON object. No explanation, no markdown fences, no extra text."""


def _clean_json(response_text: str) -> str:
    """Robustly extract JSON from Gemini response."""
    cleaned = re.sub(r'```(?:json)?\s*', '', response_text).strip().rstrip('`').strip()
    if not cleaned.startswith('{'):
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
    return cleaned


def _to_pixel_coords(prompts: dict, img_width: int, img_height: int) -> dict:
    """Convert normalized (0-1000) prompts to pixel coordinates."""
    pixel_prompts = {}

    if "positive_boxes" in prompts:
        pixel_boxes = []
        for box in prompts["positive_boxes"]:
            ymin, xmin, ymax, xmax = box
            if max(xmax, ymax) > max(img_width, img_height):
                left   = (xmin / 1000) * img_width
                top    = (ymin / 1000) * img_height
                right  = (xmax / 1000) * img_width
                bottom = (ymax / 1000) * img_height
            else:
                left, top, right, bottom = xmin, ymin, xmax, ymax
            pixel_boxes.append([left, top, right, bottom])
        pixel_prompts["positive_boxes"] = pixel_boxes

    for key in ("positive_points", "negative_points"):
        if key in prompts:
            pixel_pts = []
            for point in prompts[key]:
                y, x = point
                if max(x, y) > max(img_width, img_height):
                    px = (x / 1000) * img_width
                    py = (y / 1000) * img_height
                else:
                    px, py = x, y
                pixel_pts.append([px, py])
            pixel_prompts[key] = pixel_pts

    return pixel_prompts


def gemini_sam3_segment(
    prompt: str,
    image_path: str,
    save_dir: str = None,
    retry_count: int = 1,
) -> str:
    """
    Segment a cell image using Gemini-generated bounding boxes fed into SAM3.

    Args:
        prompt:      Text description of the target structure to segment.
        image_path:  Path to the input microscopy image.
        save_dir:    Directory to save results. Falls back to output/segment_results/.
        retry_count: Used only for versioned filenames (v1, v2, ...).

    Returns:
        String reporting segment_save_path and segment_mask_path.
    """
    # ── Setup save paths ────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        save_path      = os.path.join(save_dir, f"{name}_segmentation_v{retry_count}{ext}")
        mask_save_path = os.path.join(save_dir, f"{name}_segmentation_v{retry_count}_mask.png")
    else:
        fallback_dir = "/home/idies/workspace/Storage/xyu1/persistent/GenCELLAgent/output/segment_results"
        os.makedirs(fallback_dir, exist_ok=True)
        ts             = int(time.time())
        save_path      = f"{fallback_dir}/results_{ts}.png"
        mask_save_path = f"{fallback_dir}/results_{ts}_mask.png"

    # ── Step 1: Gemini generates bounding boxes ──────────────────────────────
    print(f"[gemini_sam3_segment] Calling Gemini ({GEMINI_MODEL}) for bounding boxes...")
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini = genai.GenerativeModel(GEMINI_MODEL)

    image_pil = Image.open(image_path)
    img_width, img_height = image_pil.size

    bbx_prompt = BBX_PROMPT_TEMPLATE.format(prompt=prompt)
    response = gemini.generate_content([bbx_prompt, image_pil])
    prompts = json.loads(_clean_json(response.text))

    prompts.setdefault("positive_points", [])
    prompts.setdefault("negative_points", [])

    n_boxes  = len(prompts.get("positive_boxes", []))
    n_pos_pt = len(prompts.get("positive_points", []))
    n_neg_pt = len(prompts.get("negative_points", []))
    print(f"  Positive boxes: {n_boxes}, Positive points: {n_pos_pt}, Negative points: {n_neg_pt}")

    if n_boxes == 0:
        return f"Error: Gemini did not generate any bounding boxes for prompt: '{prompt}'"

    # ── Step 2: Convert to pixel coordinates ────────────────────────────────
    pixel_prompts = _to_pixel_coords(prompts, img_width, img_height)
    input_boxes   = np.array(pixel_prompts["positive_boxes"])

    # ── Step 3: Load SAM3 ────────────────────────────────────────────────────
    print("[gemini_sam3_segment] Loading SAM3 model...")
    sam3_model = build_sam3_image_model(
        checkpoint_path=SAM3_CHECKPOINT,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        device="cuda",
        eval_mode=True,
    )
    processor = Sam3Processor(sam3_model)

    # ── Step 4: SAM3 inference ───────────────────────────────────────────────
    print("[gemini_sam3_segment] Running SAM3 inference...")
    inference_state = processor.set_image(image_pil)

    all_masks = []
    for box in input_boxes:
        masks, scores, _ = sam3_model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        all_masks.append(masks[0])

    # Merge masks
    merged_mask = np.zeros((img_height, img_width), dtype=bool)
    for mask in all_masks:
        merged_mask = np.logical_or(merged_mask, mask)

    # ── Step 5: Save overlay + binary mask ──────────────────────────────────
    image_np = np.array(image_pil.convert("RGB"))
    overlay  = image_np.copy().astype(np.float32)
    overlay[merged_mask] = (
        image_np[merged_mask].astype(np.float32) * 0.5
        + np.array([255, 0, 0], dtype=np.float32) * 0.5
    )
    overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, overlay_bgr)

    mask_uint8 = merged_mask.astype(np.uint8) * 255
    cv2.imwrite(mask_save_path, mask_uint8)

    print(f"[gemini_sam3_segment] Overlay saved: {save_path}")
    print(f"[gemini_sam3_segment] Mask saved:    {mask_save_path}")

    return (
        f"Segmentation completed successfully in segment_save_path:{save_path}, "
        f"the corresponding mask saved in segment_mask_path:{mask_save_path}"
    )
