#!/usr/bin/env python3
"""
CellSAM wrapper function for integration into segmentation pipeline.
Uses subprocess to run in separate conda environment.
"""
import subprocess
import os
from pathlib import Path


def run_cellsam_segmentation_single(
    path: str,
    experiment_root: str,
    python_path: str = "/home/idies/workspace/Storage/xyu1/persistent/pytorch_env/sam3_gcloud/bin/python",
    model_path: str = "/home/idies/workspace/Storage/xyu1/persistent/cellSAM/model_weights/cellsam_base_v1.1.pt",
    bbox_threshold: float = 0.3,
    device: str = "cuda",
    script_path: str = None
):
    """
    Run CellSAM segmentation on a single image using subprocess.
    
    This function wraps CellSAM in a subprocess call to allow running in a separate
    Python environment, following the same pattern as micro_sam and cellpose in your pipeline.
    
    Args:
        path: Path to the input image file
        experiment_root: Root directory for saving predictions (creates predictions/ subdirectory)
        python_path: Path to Python executable in the environment with CellSAM installed
                     (default: "/home/idies/workspace/Storage/xyu1/persistent/pytorch_env/sam3_gcloud/bin/python")
        model_path: Path to CellSAM model weights
        bbox_threshold: Bounding box threshold for segmentation (default: 0.3)
        device: Device to use - "cuda" or "cpu" (default: "cuda")
        script_path: Path to cellsam_single_inference.py script (default: auto-detect in same directory)
    
    Returns:
        Path to the generated segmentation mask
        
    Example:
        >>> mask_path = run_cellsam_segmentation_single(
        ...     path="/path/to/image.tif",
        ...     experiment_root="./results"
        ... )
    """
    # Create predictions folder
    prediction_folder = os.path.join(experiment_root, "predictions")
    os.makedirs(prediction_folder, exist_ok=True)
    
    # Get filename and construct output path
    fname = os.path.basename(path)
    out_path = os.path.join(prediction_folder, fname)
    
    # Skip if already exists
    if os.path.exists(out_path):
        print(f"CellSAM prediction already exists: {out_path}")
        return out_path
    
    # Path to the inference script
    if script_path is None:
        # Auto-detect: assume script is in same directory as this wrapper
        wrapper_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(wrapper_dir, "cellsam_single_inference.py")
    
    # Verify script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"CellSAM inference script not found at: {script_path}\n"
            f"Please ensure cellsam_single_inference.py exists at this location.\n"
            f"You can specify a custom path using the script_path parameter."
        )
    
    # Mouse_Brain preprocessing: 4× upscaling + percentile normalization
    input_path = str(path)
    temp_path = None
    if fname.lower().startswith("mouse_brain"):
        import numpy as np
        import tifffile
        from PIL import Image as PILImage
        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 3:
            gray = arr.mean(axis=2)
        else:
            gray = arr
        h, w = gray.shape[:2]
        pil_img = PILImage.fromarray(gray)
        pil_upscaled = pil_img.resize((w * 4, h * 4), PILImage.BILINEAR)
        img_upscaled = np.array(pil_upscaled, dtype=np.float32)
        p1, p99 = np.percentile(img_upscaled, [1, 99])
        if p99 > p1:
            img_normalized = np.clip((img_upscaled - p1) / (p99 - p1) * 255, 0, 255)
        else:
            img_normalized = np.zeros_like(img_upscaled)
        img_uint8 = img_normalized.astype(np.uint8)
        rgb = np.stack([img_uint8] * 3, axis=-1)
        temp_path = os.path.join(prediction_folder, f"_temp_mouse_brain_{fname}")
        tifffile.imwrite(temp_path, rgb)
        input_path = temp_path
        print(f"  Mouse_Brain preprocessed: {h}x{w} → {rgb.shape[0]}x{rgb.shape[1]}")

    # Construct the command using direct Python path
    cmd = [
        python_path,
        str(script_path),
        "--input_image", input_path,
        "--output_mask", str(out_path),
        "--model_path", model_path,
        "--bbox_threshold", str(bbox_threshold),
        "--device", device
    ]
    
    print(f"Running CellSAM segmentation on {fname}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Downscale Mouse_Brain prediction back to original size
        if fname.lower().startswith("mouse_brain"):
            import numpy as np
            import tifffile
            from skimage.transform import resize
            pred = tifffile.imread(out_path)
            raw = tifffile.imread(path)
            orig_h, orig_w = raw.shape[:2]
            if pred.shape[:2] != (orig_h, orig_w):
                pred = resize(pred, (orig_h, orig_w), order=0, preserve_range=True, anti_aliasing=False).astype(pred.dtype)
                tifffile.imwrite(out_path, pred)
                print(f"  Mouse_Brain prediction downscaled to: {orig_h}x{orig_w}")

        print(f"✓ CellSAM segmentation complete: {out_path}")
        return out_path

    except subprocess.CalledProcessError as e:
        print(f"✗ CellSAM segmentation failed for {fname}")
        print(f"Error: {e.stderr}")
        raise
    except Exception as e:
        print(f"✗ Unexpected error running CellSAM: {str(e)}")
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# Example integration into your pipeline
def integrate_into_pipeline_example():
    """
    Example showing how to integrate CellSAM into your existing pipeline.
    
    Add this to your pipeline code where you have micro_sam and cellpose:
    """
    example_code = '''
    # In your pipeline code, add this elif block:
    
    elif parts[0] == "cellsam":
        run_cellsam_segmentation_single(
            path, 
            experiment_root=experiment_root,
            conda_env="cellsam",  # Your CellSAM conda environment name
            bbox_threshold=0.3,
            device="cuda"
        )
    '''
    print(example_code)


if __name__ == "__main__":
    # Example usage
    print("CellSAM Pipeline Wrapper")
    print("="*60)
    
    # Show integration example
    integrate_into_pipeline_example()
    
    print("\n" + "="*60)
    print("Example direct call:")
    print("="*60)
    
    example_usage = """
from cellsam_pipeline_wrapper import run_cellsam_segmentation_single

# Run CellSAM on a single image
mask_path = run_cellsam_segmentation_single(
    image_path="/path/to/livecell/images/image.tif",
    experiment_root="./results",
    conda_env="cellsam",
    bbox_threshold=0.3
)

print(f"Mask saved at: {mask_path}")
    """
    print(example_usage)