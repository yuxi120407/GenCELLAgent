import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


BASE_WORKSPACE = "/home/idies/workspace/Storage/xyu1/persistent"
SCENARIO_MODEL_SELECTION_DIR = os.path.join(
    BASE_WORKSPACE, "GenCELLAgent_new", "Scenario_Model_selection"
)

if SCENARIO_MODEL_SELECTION_DIR not in sys.path:
    sys.path.insert(0, SCENARIO_MODEL_SELECTION_DIR)


def infer_micro_sam_dataset(image_path: str) -> Optional[str]:
    text = image_path.lower()
    dataset_map = {
        "livecell": "livecell",
        "tissuenet": "tissuenet",
        "plantseg": "plantseg",
        "lizard": None,
        "2018_data_science": None,
        "2018-data-science": None,
        "mouse_brain": None,
        "mouse-brain": None,
        "damond": None,
    }
    for key, value in dataset_map.items():
        if key in text:
            return value
    return None


def load_image_rgb(image_path: str):
    import imageio.v3 as imageio
    import numpy as np

    image = imageio.imread(image_path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]

    image = image.astype(np.float32)
    if image.max() > 0:
        image = image / image.max() * 255.0
    return image.astype(np.uint8)


def load_mask_binary(mask_path: str):
    import imageio.v3 as imageio
    import numpy as np

    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8) * 255


def save_overlay_and_mask(image_path: str, raw_mask_path: str, save_dir: str, suffix: str):
    import imageio.v3 as imageio
    import numpy as np

    stem = Path(image_path).stem
    overlay_path = os.path.join(save_dir, f"{stem}_{suffix}.png")
    mask_path = os.path.join(save_dir, f"{stem}_{suffix}_mask.png")

    image = load_image_rgb(image_path)
    mask = load_mask_binary(raw_mask_path)

    overlay = image.copy().astype(np.float32)
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([255, 0, 0], dtype=np.float32) * 0.5

    imageio.imwrite(overlay_path, overlay.astype(np.uint8))
    imageio.imwrite(mask_path, mask)
    return overlay_path, mask_path


def run_tool(args):
    os.makedirs(args.save_dir, exist_ok=True)

    if args.tool == "cellpose":
        from util_cellpose import run_cellpose_segmentation_single

        prediction_folder = run_cellpose_segmentation_single(
            path=args.image_path,
            experiment_root=args.save_dir,
            model_type=args.model_type,
        )
        raw_mask_path = os.path.join(prediction_folder, os.path.basename(args.image_path))
        overlay_path, mask_path = save_overlay_and_mask(
            args.image_path, raw_mask_path, args.save_dir, "cellpose"
        )

    elif args.tool == "cellsam":
        from util_cellsam import run_cellsam_segmentation_single

        raw_mask_path = run_cellsam_segmentation_single(
            path=args.image_path,
            experiment_root=args.save_dir,
            python_path=sys.executable,
            bbox_threshold=args.bbox_threshold,
            device=args.device,
        )
        overlay_path, mask_path = save_overlay_and_mask(
            args.image_path, raw_mask_path, args.save_dir, "cellsam"
        )

    elif args.tool == "micro_sam":
        from util_micro_sam import run_automatic_instance_segmentation

        dataset_name = args.dataset_name
        if dataset_name == "auto":
            dataset_name = infer_micro_sam_dataset(args.image_path)

        prediction_folder = run_automatic_instance_segmentation(
            image_path=args.image_path,
            dataset_name=dataset_name,
            experiment_root=args.save_dir,
            model_type=args.model_type,
        )
        raw_mask_path = os.path.join(prediction_folder, os.path.basename(args.image_path))
        overlay_path, mask_path = save_overlay_and_mask(
            args.image_path, raw_mask_path, args.save_dir, "micro_sam"
        )

    else:
        raise ValueError(f"Unsupported tool: {args.tool}")

    return {
        "tool": args.tool,
        "overlay_path": overlay_path,
        "mask_path": mask_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, choices=["cellpose", "cellsam", "micro_sam"])
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_type", default="cyto2")
    parser.add_argument("--bbox_threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset_name", default="auto")
    args = parser.parse_args()

    result = run_tool(args)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
