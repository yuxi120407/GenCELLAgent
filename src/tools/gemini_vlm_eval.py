import os
import sys
import json
import re
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Import organelle-specific feedback prompts from Prompts library
from src.config.paths import REPO_ROOT
sys.path.insert(0, os.path.join(REPO_ROOT, "Scenario_Automatic"))
from Prompts.Gemini_prompts_new import ER_FEEDBACK, MITO_FEEDBACK, GOLGI_FEEDBACK

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-3-flash-preview"

# Fallback generic prompt if organelle is not recognised
GENERIC_FEEDBACK = """You are an expert cell biologist evaluating a cell segmentation result.

IMAGE LAYOUT:
- LEFT: Original microscopy image
- RIGHT: Current segmentation result (WHITE = detected structure, BLACK = background)

TASK: Evaluate segmentation quality by visual comparison of LEFT vs RIGHT images.

Evaluate across five criteria (each 0.0-1.0):
1. MORPHOLOGICAL COVERAGE (25%): Are all target structures captured?
2. BOUNDARY ACCURACY (25%): Do boundaries precisely follow structure edges?
3. SPATIAL ACCURACY (20%): Are correct spatial locations identified?
4. CONTINUITY (15%): Are connected structures preserved as continuous regions?
5. SPECIFICITY (15%): Is background/non-target correctly excluded?

quality_score = 0.25*morphological_coverage + 0.25*boundary_accuracy + 0.20*spatial_accuracy + 0.15*continuity + 0.15*specificity

OUTPUT FORMAT (JSON only, no markdown, no backticks):
{{
  "quality_score": <float 0.0-1.0>,
  "criteria_scores": {{
    "morphological_coverage": <float 0.0-1.0>,
    "boundary_accuracy": <float 0.0-1.0>,
    "spatial_accuracy": <float 0.0-1.0>,
    "continuity": <float 0.0-1.0>,
    "specificity": <float 0.0-1.0>
  }},
  "visual_guidance": "<3-5 sentences on what the target looks like and how to improve detection>",
  "SummaryOfReasons": "<2-3 sentence overall evaluation summary>"
}}"""


def _select_feedback_prompt(segmentation_prompt: str) -> str:
    """Pick the organelle-specific feedback prompt based on keywords in the segmentation prompt."""
    text = segmentation_prompt.lower()
    if any(k in text for k in ("golgi", "golgi apparatus", "cisternae", "trans-golgi")):
        return GOLGI_FEEDBACK
    if any(k in text for k in ("mitochond", "mito", "cristae")):
        return MITO_FEEDBACK
    if any(k in text for k in ("endoplasmic reticulum", " er ", "er network", "cisternae", "tubular er", "nuclear envelope")):
        return ER_FEEDBACK
    return GENERIC_FEEDBACK

REFINE_PROMPT_TEMPLATE = """You are revising a cell segmentation instruction based on visual evaluation feedback.

Original Segmentation Prompt:
"{original_prompt}"

Evaluation Summary:
"{summary}"

Visual Guidance for Improvement:
"{visual_guidance}"

Instructions:
- Rewrite the prompt to be clearer and more specific about identifying the target structure.
- Incorporate the visual guidance to help the next segmentation attempt find missed or incorrect regions.
- Limit your revision to 3 concise sentences.
- Return ONLY the revised prompt text. No JSON, no explanation, no markdown."""


def _clean_json(response_text: str) -> str:
    """Robustly extract JSON from Gemini response."""
    cleaned = re.sub(r'```(?:json)?\s*', '', response_text).strip().rstrip('`').strip()
    if not cleaned.startswith('{'):
        start = cleaned.find('{')
        end   = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
    return cleaned


def _create_comparison_image(original_image_path: str, mask_path: str) -> Image.Image:
    """Create side-by-side: LEFT = original | RIGHT = binary WHITE/BLACK mask.
    This matches the format expected by ER_FEEDBACK, MITO_FEEDBACK, GOLGI_FEEDBACK."""
    original = Image.open(original_image_path).convert("L")  # grayscale for EM images
    mask     = Image.open(mask_path).convert("L")

    if original.size != mask.size:
        mask = mask.resize(original.size, Image.NEAREST)

    # Threshold mask to clean binary (0 or 255)
    mask_binary = Image.fromarray((np.array(mask) > 127).astype(np.uint8) * 255)

    w, h = original.size
    comparison = Image.new("L", (w * 2, h))
    comparison.paste(original, (0, 0))
    comparison.paste(mask_binary, (w, 0))
    return comparison


def gemini_vlm_eval(
    image_path: str,
    mask_path: str,
    segmentation_prompt: str,
    save_dir: str = None,
) -> str:
    """
    Evaluate a segmentation result with Gemini VLM and generate a refined prompt.

    Args:
        image_path:           Path to the original microscopy image.
        mask_path:            Path to the binary mask PNG (output of gemini_sam3_segment).
        segmentation_prompt:  The prompt that produced this segmentation.
        save_dir:             Optional directory to save evaluation JSON and comparison image.

    Returns:
        JSON string with keys: evaluation, overall_score (0-100), refined_segmentation_prompt.
    """
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini = genai.GenerativeModel(GEMINI_MODEL)

    # ── Step 1: Build comparison image ──────────────────────────────────────
    comparison = _create_comparison_image(image_path, mask_path)

    # ── Step 2: Select organelle-specific feedback prompt ───────────────────
    feedback_prompt = _select_feedback_prompt(segmentation_prompt)
    print(f"[gemini_vlm_eval] Using feedback prompt for: "
          f"{'Golgi' if feedback_prompt is GOLGI_FEEDBACK else 'Mito' if feedback_prompt is MITO_FEEDBACK else 'ER' if feedback_prompt is ER_FEEDBACK else 'Generic'}")

    # ── Step 3: Evaluate segmentation ───────────────────────────────────────
    print(f"[gemini_vlm_eval] Calling Gemini ({GEMINI_MODEL}) for evaluation...")
    eval_response = gemini.generate_content(
        [feedback_prompt, comparison],
        generation_config={"temperature": 0.2},
    )

    try:
        eval_json = json.loads(_clean_json(eval_response.text.strip()))
    except json.JSONDecodeError:
        eval_json = {"raw_response": eval_response.text.strip()}

    quality_score   = eval_json.get("quality_score", 0.0)
    visual_guidance = eval_json.get("visual_guidance", "")
    # SummaryOfReasons is only in GENERIC_FEEDBACK; organelle prompts use visual_guidance instead
    summary         = eval_json.get("SummaryOfReasons") or visual_guidance or eval_response.text.strip()

    print(f"  Quality score: {quality_score:.3f} ({round(quality_score * 100, 1)}/100)")

    # ── Step 3: Generate refined segmentation prompt ─────────────────────────
    print("[gemini_vlm_eval] Generating refined segmentation prompt...")
    refine_input = REFINE_PROMPT_TEMPLATE.format(
        original_prompt=segmentation_prompt,
        summary=summary,
        visual_guidance=visual_guidance,
    )
    refine_response = gemini.generate_content(refine_input)
    refined_prompt  = refine_response.text.strip()

    # ── Step 4: Always save comparison image (UI needs the path) ────────────
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = save_dir if save_dir else os.path.dirname(image_path)
    os.makedirs(out_dir, exist_ok=True)
    comparison_path = os.path.join(out_dir, f"{base_name}_comparison.png")
    comparison.save(comparison_path)
    print(f"  Saved comparison image: {comparison_path}")

    if save_dir:
        eval_path = os.path.join(save_dir, f"{base_name}_eval.json")
        with open(eval_path, "w") as f:
            json.dump({"evaluation": eval_json, "refined_segmentation_prompt": refined_prompt}, f, indent=2)
        print(f"  Saved evaluation: {eval_path}")

    return json.dumps({
        "evaluation": eval_json,
        "overall_score": round(quality_score * 100, 1),  # 0-100 scale
        "refined_segmentation_prompt": refined_prompt,
        "comparison_image_path": comparison_path,
    }, indent=2)
