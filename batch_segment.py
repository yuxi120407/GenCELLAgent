"""
Batch Segmentation Pipeline
Three modes based on input and prompt:
  1. Cell Mode     → Style-based tool selection (micro-SAM, cellpose, cellSAM)
  2. Organelle Mode → Gemini + SAM3 auto segmentation (golgi, ER, mitochondria)
  3. Reference Mode → SegGPT one-shot segmentation (with reference image + mask)
"""
import os
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)
import sys
import argparse
import json
from glob import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import matplotlib
matplotlib.use('Agg')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['NAPARI_ASYNC'] = '0'
from unittest.mock import MagicMock
# Mock napari before any micro_sam imports
for mod_name in ['napari', 'napari.utils', 'napari.utils.colormaps',
                 'napari.utils.colormaps.colormap', 'napari.utils.color',
                 'napari.utils.colormaps.standardize_color']:
    sys.modules.setdefault(mod_name, MagicMock())

import numpy as np
import cv2
from PIL import Image
import imageio.v3 as imageio

from src.tools.gemini_sam3_segment import gemini_sam3_segment
from src.tools.oneshot_segGPT import seggpt_inference_img
from src.config.logging import logger

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"


# ── Direct cell segmentation functions (no subprocess) ──────────

def _save_overlay_and_mask(image_path, mask, save_dir, tool_name):
    """Save instance overlay (each cell = different color) + mask PNGs."""
    stem = Path(image_path).stem
    image_dir = os.path.join(save_dir, stem)
    os.makedirs(image_dir, exist_ok=True)

    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        import tifffile
        raw = tifffile.imread(image_path)
        if raw.ndim == 2:
            img = cv2.cvtColor((raw / raw.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif raw.ndim == 3:
            if raw.shape[0] < raw.shape[2]:
                raw = np.transpose(raw, (1, 2, 0))
            img = (raw / raw.max() * 255).astype(np.uint8)
            if img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)

    # Create instance overlay — each object gets a unique random color
    overlay = img.copy().astype(np.float32)
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]

    rng = np.random.RandomState(42)
    colors = rng.randint(50, 255, size=(len(instance_ids) + 1, 3)).astype(np.float32)

    for idx, inst_id in enumerate(instance_ids):
        inst_mask = mask == inst_id
        overlay[inst_mask] = overlay[inst_mask] * 0.4 + colors[idx] * 0.6

    n_instances = len(instance_ids)
    print(f"  {n_instances} instances detected")

    overlay_path = os.path.join(image_dir, f"{stem}_{tool_name}.png")
    mask_path = os.path.join(image_dir, f"{stem}_{tool_name}_mask.png")

    cv2.imwrite(overlay_path, overlay.astype(np.uint8))
    cv2.imwrite(mask_path, mask.astype(np.uint16))

    return overlay_path, mask_path


def cellpose_segment_direct(image_path: str, save_dir: str = None, save_directory: str = None, model_type: str = "cyto2", **_) -> str:
    """Run cellpose directly in current env using local weights."""
    save_dir = save_dir or save_directory or os.path.join("output", "batch_results")
    from cellpose import models
    import tifffile

    device, gpu = models.assign_device(True, True)
    # Use local weights if available
    local_model_path = os.path.join(_REPO_ROOT, "models", "cellpose", "cpsam")
    if os.path.exists(local_model_path):
        model = models.CellposeModel(gpu=gpu, pretrained_model=local_model_path, device=device)
    else:
        model = models.CellposeModel(gpu=gpu, model_type=model_type, device=device)

    image = tifffile.imread(image_path)
    channels = [0, 0]
    if image.ndim == 3:
        fname = os.path.basename(image_path).lower()
        if "tissuenet" in fname:
            channels = [2, 3]
        else:
            image = image.mean(axis=-1) if image.shape[-1] <= 4 else image.mean(axis=0)

    seg = model.eval(image, diameter=None, flow_threshold=None, channels=channels)[0]
    overlay_path, mask_path = _save_overlay_and_mask(image_path, seg, save_dir, "cellpose")

    return (f"Cellpose segmentation completed successfully in "
            f"segment_save_path:{overlay_path}, "
            f"the corresponding mask saved in segment_mask_path:{mask_path}")


def micro_sam_segment_direct(image_path: str, save_dir: str = None, save_directory: str = None, **_) -> str:
    """Run micro-SAM directly in current env."""
    save_dir = save_dir or save_directory or os.path.join("output", "batch_results")
    import tifffile
    from micro_sam import util
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder,
        get_predictor_and_decoder,
    )

    # Use local weights if available — copy to micro_sam cache so it finds them
    local_sam = os.path.join(_REPO_ROOT, "models", "micro_sam", "vit_l_lm")
    local_decoder = os.path.join(_REPO_ROOT, "models", "micro_sam", "vit_l_lm_decoder")
    if os.path.exists(local_sam):
        from micro_sam import util
        cache_dir = util.microsam_cachedir()
        os.makedirs(os.path.join(cache_dir, "models"), exist_ok=True)
        cache_sam = os.path.join(cache_dir, "models", "vit_l_lm")
        cache_dec = os.path.join(cache_dir, "models", "vit_l_lm_decoder")
        if not os.path.exists(cache_sam):
            import shutil
            shutil.copy2(local_sam, cache_sam)
        if os.path.exists(local_decoder) and not os.path.exists(cache_dec):
            import shutil
            shutil.copy2(local_decoder, cache_dec)

    predictor, decoder = get_predictor_and_decoder(model_type="vit_l_lm", checkpoint_path=None)
    image = tifffile.imread(image_path)
    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]

    image_embeddings = util.precompute_image_embeddings(predictor=predictor, input_=image, ndim=2)
    ais = InstanceSegmentationWithDecoder(predictor, decoder)
    ais.initialize(image=image, image_embeddings=image_embeddings)
    seg = ais.generate()

    if seg is None or (isinstance(seg, np.ndarray) and seg.max() == 0):
        seg = np.zeros(image.shape[:2], dtype=np.uint32)

    overlay_path, mask_path = _save_overlay_and_mask(image_path, seg, save_dir, "micro_sam")

    return (f"Micro-SAM segmentation completed successfully in "
            f"segment_save_path:{overlay_path}, "
            f"the corresponding mask saved in segment_mask_path:{mask_path}")


def cellsam_segment_direct(image_path: str, save_dir: str = None, save_directory: str = None, **_) -> str:
    """Run CellSAM directly in current env using local weights."""
    save_dir = save_dir or save_directory or os.path.join("output", "batch_results")
    import tifffile
    from cellSAM import cellsam_pipeline

    image = tifffile.imread(image_path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] > 3:
        image = image[..., :3]

    # Use local weights if available
    local_model_path = os.path.join(_REPO_ROOT, "models", "cellsam", "cellsam_general.pt")
    if os.path.exists(local_model_path):
        seg = cellsam_pipeline(image, model_path=local_model_path, use_wsi=False)
    else:
        seg = cellsam_pipeline(image, use_wsi=False)
    overlay_path, mask_path = _save_overlay_and_mask(image_path, seg, save_dir, "cellsam")

    return (f"CellSAM segmentation completed successfully in "
            f"segment_save_path:{overlay_path}, "
            f"the corresponding mask saved in segment_mask_path:{mask_path}")


TOOL_MAP = {
    "cellpose": cellpose_segment_direct,
    "micro_sam": micro_sam_segment_direct,
    "cellsam": cellsam_segment_direct,
}

MODE_DETECTION_PROMPT = """You are a microscopy image analysis assistant. Based on the user's request, classify it into one of three modes:

1. "cell" — The user wants to segment cells, nuclei, or detect individual cell instances.
   Examples: "segment all cells", "find nuclei", "help me detect cells in this image", "instance segmentation"

2. "organelle" — The user wants to segment a specific sub-cellular structure (organelle).
   Also extract the organelle name.
   Examples: "segment golgi", "find mitochondria", "detect endoplasmic reticulum", "segment ER"

3. "reference" — The user mentions using a reference image, example, template, or one-shot learning.
   Examples: "use this example to segment", "segment like the reference", "one-shot segmentation"

User request: "{prompt}"

Respond with ONLY a JSON object (no markdown, no backticks):
{{"mode": "cell" or "organelle" or "reference", "organelle": "name if organelle mode else null"}}"""


def detect_mode(prompt: str, reference_image: str = None, reference_mask: str = None) -> tuple:
    """
    Detect which mode to use based on prompt and inputs.
    Uses Gemini LLM to understand natural language prompts.

    Returns:
        (mode, organelle_name) — organelle_name is None for cell/reference modes
    """
    # Reference mode takes priority if files are provided
    if reference_image and reference_mask:
        return "reference", None

    try:
        from google import genai
        import json
        import re

        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=GOOGLE_API_KEY)

        from src.config.setup import config

        response = client.models.generate_content(
            model=config.MODEL_AGENT,
            contents=MODE_DETECTION_PROMPT.format(prompt=prompt)
        )
        text = response.text.strip()

        # Clean JSON
        text = re.sub(r'```(?:json)?\s*', '', text).strip()
        if not text.startswith('{'):
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                text = text[start:end+1]

        result = json.loads(text)
        mode = result.get("mode", "cell")
        organelle = result.get("organelle", None)

        logger.info(f"LLM mode detection: '{prompt}' → mode={mode}, organelle={organelle}")
        return mode, organelle

    except Exception as e:
        logger.warning(f"LLM mode detection failed: {e}, falling back to keyword matching")
        return _detect_mode_fallback(prompt), None


def _detect_mode_fallback(prompt: str) -> str:
    """Fallback keyword-based mode detection if LLM fails."""
    prompt_lower = prompt.lower()
    organelle_keywords = ["golgi", "er", "endoplasmic reticulum", "mitochondria", "mito", "lysosome", "vesicle"]
    for keyword in organelle_keywords:
        if keyword in prompt_lower:
            return "organelle"
    return "cell"


import sys
from pathlib import Path
_REPO_ROOT = str(Path(__file__).resolve().parent)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "tools"))

# Cached VGG model and val features (loaded once)
_vgg_cache = {}

STYLE_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "vgg", "vgg_conv.pth")
STYLE_LAYERS = ['r11', 'r21', 'r31', 'r41', 'r51']

BEST_TOOL = {
    "LiveCELL": "cellpose",
    "TissueNet": "cellpose",
    "PlantSeg": "micro_sam",
    "Lizard": "cellpose",
    "2018_Data_Science": "micro_sam",
    "Mouse_Brain": "micro_sam",
    "Damond": "cellsam",
    "Yeast_Z2": "cellsam",
}

NORM_FOR_DATASET = {
    "LiveCELL": "cellpose",
    "TissueNet": "cellpose",
    "PlantSeg": "microsam",
    "Lizard": "cellpose",
    "2018_Data_Science": "microsam",
    "Mouse_Brain": "microsam",
    "Damond": "cellsam",
    "Yeast_Z2": "cellsam",
}

ALL_NORMS = ["cellpose", "microsam", "cellsam"]

_VAL_ROOT = os.path.join(_REPO_ROOT, "data", "val")
VAL_FOLDERS = {
    "LiveCELL": os.path.join(_VAL_ROOT, "LiveCELL"),
    "TissueNet": os.path.join(_VAL_ROOT, "TissueNet"),
    "PlantSeg": os.path.join(_VAL_ROOT, "PlantSeg"),
    "Lizard": os.path.join(_VAL_ROOT, "Lizard"),
    "2018_Data_Science": os.path.join(_VAL_ROOT, "2018_Data_Science"),
    "Mouse_Brain": os.path.join(_VAL_ROOT, "Mouse_Brain"),
    "Damond": os.path.join(_VAL_ROOT, "Damond"),
    "Yeast_Z2": os.path.join(_VAL_ROOT, "Yeast_Z2"),
}

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg")


def _load_vgg_and_val_features():
    """Load VGG model and extract val features once, cache for reuse."""
    if _vgg_cache:
        return _vgg_cache

    from style_utils import load_vgg_model, load_image_with_norm, GramMatrix, layerwise_pearson_corr
    import numpy as np

    logger.info("Loading VGG model and val features (one-time)...")
    vgg = load_vgg_model(STYLE_MODEL_PATH)
    gram = GramMatrix()

    # Extract val features under each dataset's normalization
    val_features = {}
    for ds, folder in VAL_FOLDERS.items():
        norm = NORM_FOR_DATASET[ds]
        ds_name = ds if ds == "Mouse_Brain" else None
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])
        feats = []
        for p in files:
            tensor = load_image_with_norm(p, norm, dataset_name=ds_name)
            feats.append([gram(f).detach() for f in vgg(tensor, STYLE_LAYERS)])
        val_features[ds] = feats
        logger.info(f"  {ds}: {len(feats)} val images")

    _vgg_cache['vgg'] = vgg
    _vgg_cache['gram'] = gram
    _vgg_cache['val_features'] = val_features
    return _vgg_cache


def select_cell_tool(image_path: str, tool_name: str = None) -> str:
    """
    Select the best cell segmentation tool using VGG style similarity.
    Compares input image to all val datasets, finds most similar,
    returns that dataset's best tool.
    """
    if tool_name:
        return tool_name

    try:
        from style_utils import load_image_with_norm, layerwise_pearson_corr
        import numpy as np

        cache = _load_vgg_and_val_features()
        vgg = cache['vgg']
        gram = cache['gram']
        val_features = cache['val_features']

        # Extract input features under all norms
        input_feats = {}
        for norm in ALL_NORMS:
            tensor = load_image_with_norm(image_path, norm)
            input_feats[norm] = [gram(f).detach() for f in vgg(tensor, STYLE_LAYERS)]

        # Compare to each val dataset using that dataset's norm
        avg_sims = {}
        for ds in BEST_TOOL:
            norm = NORM_FOR_DATASET[ds]
            ds_feats = val_features[ds]
            sims = []
            for vf in ds_feats:
                layer_sims = [layerwise_pearson_corr(input_feats[norm][k], vf[k]) for k in range(5)]
                sims.append(np.mean(layer_sims))
            avg_sims[ds] = np.mean(sims)

        # Find most similar dataset
        best_ds = max(avg_sims, key=avg_sims.get)
        selected_tool = BEST_TOOL[best_ds]

        logger.info(f"Tool selection for {os.path.basename(image_path)}:")
        for ds, sim in sorted(avg_sims.items(), key=lambda x: -x[1]):
            marker = " ← SELECTED" if ds == best_ds else ""
            logger.info(f"  {ds:<22} ({BEST_TOOL[ds]:<12}): {sim:.4f}{marker}")

        return selected_tool

    except Exception as e:
        logger.warning(f"Style-based selection failed: {e}, defaulting to cellpose")
        return "cellpose"


def segment_cell(image_path: str, save_dir: str, tool_name: str = None) -> dict:
    """Mode 1: Cell segmentation with tool selection."""
    selected_tool = select_cell_tool(image_path, tool_name)
    logger.info(f"Cell mode: using {selected_tool} for {os.path.basename(image_path)}")

    tool_fn = TOOL_MAP.get(selected_tool, cellpose_segment_direct)
    result = tool_fn(image_path=image_path, save_dir=save_dir)

    return {
        "mode": "cell",
        "tool": selected_tool,
        "image": os.path.basename(image_path),
        "result": result,
    }


def segment_organelle(
    image_path: str,
    prompt: str,
    save_dir: str,
    max_iterations: int = 1,
    quality_threshold: float = 0.85,
    evaluation_mode: str = "boxes_only",
    gemini_model: str = None,
) -> dict:
    """
    Mode 2: Organelle segmentation with Gemini + SAM3.

    If max_iterations=1: single pass (no feedback loop)
    If max_iterations>1: iterative refinement with feedback
    """
    logger.info(f"Organelle mode: segmenting '{prompt}' in {os.path.basename(image_path)}")

    if max_iterations <= 1:
        # Simple single-pass mode
        result = gemini_sam3_segment(
            prompt=prompt,
            image_path=image_path,
            save_dir=save_dir,
            retry_count=1,
        )
        return {
            "mode": "organelle",
            "target": prompt,
            "image": os.path.basename(image_path),
            "iterations": 1,
            "result": result,
        }

    # Multi-pass with feedback loop
    import sys
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "sam3", "test_gemini_sam3"))
    from test_all_feedback_2_16 import (
        generate_sam3_prompts_with_gemini,
        process_image_with_iterations_new,
    )
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from src.config.setup import config

    model_name = gemini_model or config.MODEL_SAM3_SEGMENT
    sam3_checkpoint = config.SAM3_CHECKPOINT

    # Map prompt to organelle prompt template
    prompt_lower = prompt.lower()
    organelle_prompts = {
        "golgi": "golgi",
        "er": "er",
        "endoplasmic reticulum": "er",
        "mito": "mito",
        "mitochondria": "mito",
        "lysosome": "lysosome",
        "vesicle": "vesicle",
    }
    organelle_key = None
    for keyword, key in organelle_prompts.items():
        if keyword in prompt_lower:
            organelle_key = key
            break

    # Load organelle-specific prompt
    try:
        from Prompts.Gemini_prompts import GOLGI_PROMPT, ER_PROMPT, MITO_PROMPT
        PROMPT_MAP = {
            "golgi": GOLGI_PROMPT,
            "er": ER_PROMPT,
            "mito": MITO_PROMPT,
        }
        text_prompt = PROMPT_MAP.get(organelle_key, prompt)
    except ImportError:
        text_prompt = prompt

    # Build SAM3 model
    logger.info(f"Loading SAM3 from {sam3_checkpoint}")
    sam3_model = build_sam3_image_model(
        checkpoint_path=sam3_checkpoint,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        device="cuda",
        eval_mode=True,
    )
    sam3_processor = Sam3Processor(sam3_model)

    # Generate initial prompts with Gemini
    logger.info(f"Generating initial prompts with {model_name}")
    initial_prompts = generate_sam3_prompts_with_gemini(
        image_path=image_path,
        text_prompt=text_prompt,
        model_name=model_name,
    )

    # Run iterative refinement
    os.makedirs(save_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(save_dir, image_name)

    summary = process_image_with_iterations_new(
        image_path=image_path,
        initial_prompts=initial_prompts,
        model=sam3_model,
        processor=sam3_processor,
        output_dir=output_dir,
        gemini_model_name=model_name,
        enable_refinement=True,
        quality_threshold=quality_threshold,
        max_iterations=max_iterations,
        evaluation_mode=evaluation_mode,
    )

    return {
        "mode": "organelle",
        "target": prompt,
        "image": os.path.basename(image_path),
        "iterations": max_iterations,
        "best_iteration": summary.get("best_iteration", {}),
        "quality_progression": summary.get("quality_progression", {}),
        "output_dir": output_dir,
    }


def segment_reference(image_path: str, reference_image: str, reference_mask: str, save_dir: str) -> dict:
    """Mode 3: One-shot segmentation with SegGPT."""
    logger.info(f"Reference mode: SegGPT for {os.path.basename(image_path)}")

    result = seggpt_inference_img(
        image_path=image_path,
        prompt_image_path=reference_image,
        prompt_mask_path=reference_mask,
        save_dir=save_dir,
    )

    return {
        "mode": "reference",
        "image": os.path.basename(image_path),
        "reference": os.path.basename(reference_image),
        "result": result,
    }


def segment(
    image_path: str,
    prompt: str = "segment cells",
    save_dir: str = None,
    tool_name: str = None,
    reference_image: str = None,
    reference_mask: str = None,
    max_iterations: int = 1,
    quality_threshold: float = 0.85,
    evaluation_mode: str = "boxes_only",
) -> dict:
    """
    Main entry point: auto-selects mode and runs segmentation.

    Args:
        image_path:        Path to input image
        prompt:            Text description (e.g., "segment cells", "golgi", "mitochondria")
        save_dir:          Output directory
        tool_name:         Force specific tool (cellpose/micro_sam/cellsam), optional
        reference_image:   Path to reference image for one-shot mode
        reference_mask:    Path to reference mask for one-shot mode
        max_iterations:    Number of feedback iterations for organelle mode (1=no feedback)
        quality_threshold: Stop early if quality exceeds this (organelle mode)
        evaluation_mode:   SAM3 mode for feedback evaluation

    Returns:
        dict with mode, tool, and result paths
    """
    if save_dir is None:
        save_dir = os.path.join("output", "batch_results")
    os.makedirs(save_dir, exist_ok=True)

    mode, organelle = detect_mode(prompt, reference_image, reference_mask)
    logger.info(f"Detected mode: {mode} | organelle: {organelle} | prompt: '{prompt}' | image: {os.path.basename(image_path)}")

    if mode == "cell":
        return segment_cell(image_path, save_dir, tool_name)
    elif mode == "organelle":
        organelle_prompt = organelle if organelle else prompt
        return segment_organelle(
            image_path, organelle_prompt, save_dir,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            evaluation_mode=evaluation_mode,
        )
    elif mode == "reference":
        return segment_reference(image_path, reference_image, reference_mask, save_dir)


def batch_segment(
    image_dir: str,
    prompt: str = "segment cells",
    save_dir: str = None,
    tool_name: str = None,
    reference_image: str = None,
    reference_mask: str = None,
    max_iterations: int = 1,
    quality_threshold: float = 0.85,
    evaluation_mode: str = "boxes_only",
) -> list:
    """
    Run segmentation on all images in a directory.

    Args:
        image_dir:         Directory containing input images
        prompt:            Text description for all images
        save_dir:          Output directory
        tool_name:         Force specific tool for cell mode
        reference_image:   Reference image for one-shot mode
        reference_mask:    Reference mask for one-shot mode
        max_iterations:    Number of feedback iterations for organelle mode
        quality_threshold: Stop early if quality exceeds this
        evaluation_mode:   SAM3 mode for feedback evaluation

    Returns:
        List of result dicts
    """
    if save_dir is None:
        save_dir = os.path.join("output", "batch_results")

    image_paths = sorted(
        glob(os.path.join(image_dir, "*.tif")) +
        glob(os.path.join(image_dir, "*.tiff")) +
        glob(os.path.join(image_dir, "*.png")) +
        glob(os.path.join(image_dir, "*.jpg"))
    )

    print(f"Found {len(image_paths)} images in {image_dir}")
    mode, organelle = detect_mode(prompt, reference_image, reference_mask)
    print(f"Mode: {mode}" + (f" (organelle: {organelle})" if organelle else ""))
    print(f"Prompt: '{prompt}'")
    if tool_name:
        print(f"Tool: {tool_name}")
    print()

    results = []
    for i, path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(path)}...")
        try:
            result = segment(
                image_path=path,
                prompt=prompt,
                save_dir=save_dir,
                tool_name=tool_name,
                reference_image=reference_image,
                reference_mask=reference_mask,
                max_iterations=max_iterations,
                quality_threshold=quality_threshold,
                evaluation_mode=evaluation_mode,
            )
            results.append(result)
            print(f"  ✓ {result['mode']} → {result.get('tool', result.get('target', 'seggpt'))}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({"image": os.path.basename(path), "error": str(e)})

    # Save summary
    summary_path = os.path.join(save_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Segmentation Pipeline")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image_dir", type=str, help="Directory of images")
    parser.add_argument("--prompt", type=str, default="segment cells",
                        help="What to segment (e.g., 'cells', 'golgi', 'mitochondria')")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--tool", type=str, default=None,
                        choices=["cellpose", "micro_sam", "cellsam"],
                        help="Force specific tool for cell mode")
    parser.add_argument("--reference_image", type=str, default=None,
                        help="Reference image for one-shot mode")
    parser.add_argument("--reference_mask", type=str, default=None,
                        help="Reference mask for one-shot mode")
    parser.add_argument("--max_iterations", type=int, default=1,
                        help="Number of feedback iterations for organelle mode (1=no feedback)")
    parser.add_argument("--quality_threshold", type=float, default=0.85,
                        help="Stop early if quality exceeds this (organelle mode)")
    parser.add_argument("--evaluation_mode", type=str, default="boxes_only",
                        choices=["boxes_only", "boxes_all_points", "boxes_positive_points_only",
                                 "boxes_negative_points_only", "boxes_filtered_points"],
                        help="SAM3 evaluation mode for feedback")
    args = parser.parse_args()

    if args.image:
        result = segment(
            image_path=args.image,
            prompt=args.prompt,
            save_dir=args.save_dir,
            tool_name=args.tool,
            reference_image=args.reference_image,
            reference_mask=args.reference_mask,
            max_iterations=args.max_iterations,
            quality_threshold=args.quality_threshold,
            evaluation_mode=args.evaluation_mode,
        )
        print(json.dumps(result, indent=2))

    elif args.image_dir:
        batch_segment(
            image_dir=args.image_dir,
            prompt=args.prompt,
            save_dir=args.save_dir,
            tool_name=args.tool,
            reference_image=args.reference_image,
            reference_mask=args.reference_mask,
            max_iterations=args.max_iterations,
            quality_threshold=args.quality_threshold,
            evaluation_mode=args.evaluation_mode,
        )
    else:
        parser.print_help()
