import json
import os
import subprocess
from typing import Optional


BASE_WORKSPACE = "/home/idies/workspace/Storage/xyu1/persistent"
RUNNER_PATH = os.path.join(
    BASE_WORKSPACE, "GenCELLAgent_new", "src", "tools", "cell_segmentation_env_runner.py"
)
MICRO_SAM_PYTHON = os.path.join(
    BASE_WORKSPACE, "pytorch_env", "micro-sam", "bin", "python"
)
CELLPOSE_PYTHON = MICRO_SAM_PYTHON
CELLSAM_PYTHON = os.path.join(
    BASE_WORKSPACE, "pytorch_env", "sam3_gcloud", "bin", "python"
)


def _resolve_save_dir(save_directory: Optional[str] = None, save_dir: Optional[str] = None) -> str:
    out_dir = save_directory or save_dir
    if not out_dir:
        out_dir = os.path.join(BASE_WORKSPACE, "GenCELLAgent_new", "output", "cell_segmentation_models")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _run_env_tool(python_path: str, args: list[str]) -> dict:
    result = subprocess.run(
        [python_path, RUNNER_PATH, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not stdout:
        raise RuntimeError(f"No output returned from tool runner. stderr: {result.stderr}")
    return json.loads(stdout[-1])


def cellpose_segment(
    image_path: str,
    save_directory: Optional[str] = None,
    save_dir: Optional[str] = None,
    model_type: str = "cyto2",
    **_: object,
) -> str:
    out_dir = _resolve_save_dir(save_directory=save_directory, save_dir=save_dir)
    result = _run_env_tool(
        CELLPOSE_PYTHON,
        [
            "--tool", "cellpose",
            "--image_path", image_path,
            "--save_dir", out_dir,
            "--model_type", model_type,
        ],
    )
    return (
        f"Cellpose segmentation completed successfully in segment_save_path:{result['overlay_path']}, "
        f"the corresponding mask saved in segment_mask_path:{result['mask_path']}"
    )


def cellsam_segment(
    image_path: str,
    save_directory: Optional[str] = None,
    save_dir: Optional[str] = None,
    bbox_threshold: float = 0.3,
    device: str = "cuda",
    **_: object,
) -> str:
    out_dir = _resolve_save_dir(save_directory=save_directory, save_dir=save_dir)
    result = _run_env_tool(
        CELLSAM_PYTHON,
        [
            "--tool", "cellsam",
            "--image_path", image_path,
            "--save_dir", out_dir,
            "--bbox_threshold", str(bbox_threshold),
            "--device", device,
        ],
    )
    return (
        f"CellSAM segmentation completed successfully in segment_save_path:{result['overlay_path']}, "
        f"the corresponding mask saved in segment_mask_path:{result['mask_path']}"
    )


def micro_sam_segment(
    image_path: str,
    save_directory: Optional[str] = None,
    save_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    model_type: str = "vit_l_lm",
    **_: object,
) -> str:
    out_dir = _resolve_save_dir(save_directory=save_directory, save_dir=save_dir)
    result = _run_env_tool(
        MICRO_SAM_PYTHON,
        [
            "--tool", "micro_sam",
            "--image_path", image_path,
            "--save_dir", out_dir,
            "--dataset_name", dataset_name if dataset_name is not None else "auto",
            "--model_type", model_type,
        ],
    )
    return (
        f"Micro-SAM segmentation completed successfully in segment_save_path:{result['overlay_path']}, "
        f"the corresponding mask saved in segment_mask_path:{result['mask_path']}"
    )
