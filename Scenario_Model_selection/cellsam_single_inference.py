#!/usr/bin/env python3
"""
CellSAM Single Image Inference Script
Supports both single image and batch processing modes.
"""
import numpy as np
from PIL import Image
import torch
import argparse
import os
import tifffile
from pathlib import Path
import imageio.v3 as imageio

from cellSAM.model import get_local_model, segment_cellular_image


def segment_single_image(
    input_image_path: str,
    output_mask_path: str,
    model_path: str = '/home/idies/workspace/Storage/xyu1/persistent/cellSAM/model_weights/cellsam_base_v1.1.pt',
    bbox_threshold: float = 0.3,
    device: str = 'cuda'
):
    """
    Segment a single image using CellSAM.
    
    Args:
        input_image_path: Path to input image file
        output_mask_path: Path where the segmentation mask will be saved
        model_path: Path to CellSAM model weights
        bbox_threshold: Bounding box threshold for segmentation
        device: Device to use (cuda or cpu)
    
    Returns:
        Number of cells detected
    """
    # Load model
    print(f"Loading CellSAM model from {model_path}...")
    model = get_local_model(model_path=model_path)
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    
    # Load image
    print(f"Loading image from {input_image_path}...")
    img = imageio.imread(input_image_path)
    
    # Handle different image formats
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # Take only first 3 channels (RGB)
    elif img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]  # Take first 3 channels
    elif img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)  # Convert grayscale to RGB
    
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    # Segment
    print("Running segmentation...")
    mask, _, _ = segment_cellular_image(
        img,
        model=model,
        bbox_threshold=bbox_threshold,
        normalize=True,
        device=device
    )
    
    # Save mask
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    Image.fromarray(mask.astype(np.uint16)).save(output_mask_path)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    num_cells = int(mask.max())
    print(f"Segmentation complete: {num_cells} cells detected")
    
    return num_cells


def main():
    parser = argparse.ArgumentParser(
        description='CellSAM inference - supports single image or batch processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image mode
  python cellsam_single_inference.py --input_image image.tif --output_mask mask.tif
  
  # Batch mode
  python cellsam_single_inference.py --input_dir ./data --output_dir ./results
        """
    )
    
    # Single image mode arguments
    parser.add_argument('--input_image', type=str,
                        help='Single input image path (for single image mode)')
    parser.add_argument('--output_mask', type=str,
                        help='Output mask path for single image (for single image mode)')
    
    # Batch mode arguments
    parser.add_argument('--input_dir', type=str,
                        help='Input directory containing images/ subdirectory (for batch mode)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for saving results (for batch mode)')
    
    # Common arguments
    parser.add_argument('--model_path', type=str, 
                        default='/home/idies/workspace/Storage/xyu1/persistent/cellSAM/model_weights/cellsam_base_v1.1.pt',
                        help='Path to CellSAM model weights')
    parser.add_argument('--bbox_threshold', type=float, default=0.3,
                        help='Bounding box threshold for segmentation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Determine mode
    single_mode = args.input_image is not None and args.output_mask is not None
    batch_mode = args.input_dir is not None and args.output_dir is not None
    
    if not single_mode and not batch_mode:
        parser.error("Must provide either (--input_image and --output_mask) for single mode "
                     "or (--input_dir and --output_dir) for batch mode")
    
    if single_mode and batch_mode:
        parser.error("Cannot use both single mode and batch mode arguments simultaneously")
    
    # Single image mode
    if single_mode:
        print("="*60)
        print("Running CellSAM in SINGLE IMAGE mode")
        print("="*60)
        
        if not os.path.exists(args.input_image):
            raise FileNotFoundError(f"Input image not found: {args.input_image}")
        
        num_cells = segment_single_image(
            input_image_path=args.input_image,
            output_mask_path=args.output_mask,
            model_path=args.model_path,
            bbox_threshold=args.bbox_threshold,
            device=args.device
        )
        
        print("="*60)
        print(f"✓ SUCCESS: Mask saved to {args.output_mask}")
        print("="*60)
        return
    
    # Batch mode (original functionality)
    if batch_mode:
        print("="*60)
        print("Running CellSAM in BATCH mode")
        print("="*60)
        
        from tqdm import tqdm
        
        # Construct paths
        input_base = Path(args.input_dir)
        test_images_dir = input_base / "images"
        output_dir = Path(args.output_dir)
        
        # Validate input directories
        if not test_images_dir.exists():
            raise ValueError(f"Images directory not found: {test_images_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model once
        print("Loading CellSAM model...")
        model = get_local_model(model_path=args.model_path)
        model = model.to(args.device)
        print(f"Model loaded. Device: {next(model.parameters()).device}")
        
        # Get all TIF files
        tif_files = sorted(list(test_images_dir.glob("*.tif")))
        print(f"Found {len(tif_files)} TIF files in {test_images_dir}")
        
        # Process each image
        for img_path in tqdm(tif_files, desc="Processing images"):
            mask_tif_path = output_dir / f"{img_path.name}"
            if mask_tif_path.exists():
                print(f"  Skipping {img_path.name}: prediction already exists")
                continue
            
            try:
                # Load image
                img = tifffile.imread(img_path)
                
                if img.ndim == 3 and img.shape[-1] == 4:
                    img = img[..., :3]  # Take only first 3 channels (RGB)
                elif img.ndim == 3 and img.shape[-1] > 3:
                    img = img[..., :3]  # Take first 3 channels
                elif img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)  # Convert grayscale to RGB
                
                # Segment
                mask, _, _ = segment_cellular_image(
                    img,
                    model=model,
                    bbox_threshold=args.bbox_threshold,
                    normalize=True,
                    device=args.device
                )
                
                # Save mask
                Image.fromarray(mask.astype(np.uint16)).save(mask_tif_path)
                
                print(f"✓ {img_path.name}: {mask.max()} cells detected")
                
            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {str(e)}")
                continue
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n✓ Batch processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()