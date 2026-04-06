import os
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from google import genai
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from Prompts.Gemini_prompts_new import ER_DEFAULT_DETECTION, ER_PROMPT_TEMPLATE, ER_FEEDBACK, MITO_PROMPT_TEMPLATE, MITO_DEFAULT_DETECTION, MITO_FEEDBACK, GOLGI_PROMPT_TEMPLATE, GOLGI_DEFAULT_DETECTION, GOLGI_FEEDBACK
from tqdm import tqdm
import sys
import time
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SAM3_CHECKPOINT = "/home/idies/workspace/Storage/xyu1/persistent/GenCELLAgent/src/sam3/checkpoints/sam3/sam3.pt"

# ============================================================================
# GEMINI PROMPT GENERATION
# ============================================================================

def create_comparison_image(original_image_path, segmentation_mask):
    """
    Create side-by-side comparison: Original | Binary Mask
    
    Args:
        original_image_path: Path to original image
        segmentation_mask: Binary mask from SAM3 (numpy array, boolean or 0/1)
    
    Returns:
        PIL Image with side-by-side comparison
    """
    from PIL import Image
    import numpy as np
    
    # Load original
    original = Image.open(original_image_path)
    
    # Convert mask to binary image (0 or 255)
    mask_binary = (segmentation_mask > 0).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_binary)
    
    # Ensure same size
    if original.size != mask_image.size:
        mask_image = mask_image.resize(original.size)
    
    # Create side-by-side (grayscale)
    w, h = original.size
    comparison = Image.new('L', (w * 2, h))
    comparison.paste(original, (0, 0))
    comparison.paste(mask_image, (w, 0))
    
    return comparison
    
def clean_gemini_json_response(response_text):
    """Robustly extract JSON from Gemini response"""
    cleaned = re.sub(r'```(?:json)?\s*', '', response_text).strip()
    
    if not cleaned.startswith('{'):
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]
    
    return cleaned




def generate_sam3_prompts_with_gemini(image_path, text_prompt, model_name="gemini-3-pro-preview"):
    """Generate SAM3 prompts using Gemini vision model"""
    import google.generativeai as genai
    
    genai.configure(api_key=GOOGLE_API_KEY)
    image = Image.open(image_path)
    model = genai.GenerativeModel(model_name)
    

    
    response = model.generate_content([text_prompt, image])

    # Extract tokens from response
    tokens = {
        'input': getattr(response.usage_metadata, 'prompt_token_count', 0),
        'output': getattr(response.usage_metadata, 'candidates_token_count', 0)
    }
    
    clean_json_str = clean_gemini_json_response(response.text)
    
    prompts = json.loads(clean_json_str)
    

    if "negative_points" not in prompts:
        prompts["negative_points"] = []
    
    print(f"✓ Gemini prompts generated:")
    print(f"  Positive boxes: {len(prompts['positive_boxes'])}")
    print(f"  Positive points: {len(prompts['positive_points'])}")
    print(f"  Negative points: {len(prompts.get('negative_points', []))}")
    
    return prompts, tokens



# ============================================================================
# COORDINATE CONVERSION
# ============================================================================

def convert_prompts_to_pixel_coordinates(prompts, img_width, img_height):
    """Convert normalized prompts to pixel coordinates"""
    pixel_prompts = {}
    
    if "positive_boxes" in prompts:
        pixel_boxes = []
        for box in prompts["positive_boxes"]:
            ymin, xmin, ymax, xmax = box
            if max(xmax, ymax) > max(img_width, img_height):
                left = (xmin / 1000) * img_width
                top = (ymin / 1000) * img_height
                right = (xmax / 1000) * img_width
                bottom = (ymax / 1000) * img_height
            else:
                left, top, right, bottom = xmin, ymin, xmax, ymax
            pixel_boxes.append([left, top, right, bottom])
        pixel_prompts["positive_boxes"] = pixel_boxes
    
    if "positive_points" in prompts:
        pixel_points = []
        for point in prompts["positive_points"]:
            y, x = point
            if max(x, y) > max(img_width, img_height):
                px = (x / 1000) * img_width
                py = (y / 1000) * img_height
            else:
                px, py = x, y
            pixel_points.append([px, py])
        pixel_prompts["positive_points"] = pixel_points
    
    if "negative_points" in prompts:
        pixel_points = []
        for point in prompts["negative_points"]:
            y, x = point
            if max(x, y) > max(img_width, img_height):
                px = (x / 1000) * img_width
                py = (y / 1000) * img_height
            else:
                px, py = x, y
            pixel_points.append([px, py])
        pixel_prompts["negative_points"] = pixel_points
    
    return pixel_prompts


def get_points_in_box(points, labels, box):
    """Get only points that fall within the bounding box"""
    x0, y0, x1, y1 = box
    inside_mask = (
        (points[:, 0] >= x0) & (points[:, 0] <= x1) &
        (points[:, 1] >= y0) & (points[:, 1] <= y1)
    )
    return points[inside_mask], labels[inside_mask]


# ============================================================================
# GROUND TRUTH LOADING
# ============================================================================

def load_ground_truth_label(image_path):
    """Load ground truth label by replacing /images/ with /labels/ and image_ with label_"""
    label_path = str(image_path).replace('/images/', '/labels/')
    label_path = label_path.replace('image_', 'label_')
    
    if not Path(label_path).exists():
        return None
    
    gt_label = np.array(Image.open(label_path))
    return gt_label


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def show_points(coords, labels, ax, marker_size=375):
    """Show positive and negative points"""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)


# ============================================================================
# MAIN SAM3 INFERENCE AND VISUALIZATION
# ============================================================================

def sam3_inference_and_visualize(
    image_path,
    prompts,
    model,
    processor,
    output_dir="./sam3_results",
    use_boxes=True,
    use_points=True,
    points_in_box_only=False,
    positive_points_only=False,
    negative_points_only=False
):
    """Run SAM3 inference, visualize results, and save metrics to JSON"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # Determine mode name
    if positive_points_only:
        mode_suffix = 'positive_points_only'
    elif negative_points_only:
        mode_suffix = 'negative_points_only'
    elif points_in_box_only:
        mode_suffix = 'filtered_points'
    elif use_points:
        mode_suffix = 'all_points'
    else:
        mode_suffix = 'only'
    
    if use_boxes:
        mode_name = f"boxes_{mode_suffix}"
    else:
        mode_name = "points_only"
    
    # Create filename with mode
    filename_base = f"{image_name}_{mode_name}"
    
    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Convert prompts to pixel coordinates
    pixel_prompts = convert_prompts_to_pixel_coordinates(prompts, img_width, img_height)
    
    # Save prompts JSON (only once per image, not per mode)
    prompts_json_path = output_dir.parent / f"{image_name}_prompts.json"
    if not prompts_json_path.exists():
        prompts_data = {
            'image_name': image_name,
            'image_size': {'width': img_width, 'height': img_height},
            'normalized_prompts': prompts,
            'pixel_prompts': {
                'positive_boxes': [[float(v) for v in box] for box in pixel_prompts.get("positive_boxes", [])],
                'positive_points': [[float(v) for v in pt] for pt in pixel_prompts.get("positive_points", [])],
                'negative_points': [[float(v) for v in pt] for pt in pixel_prompts.get("negative_points", [])]
            }
        }
        
        with open(prompts_json_path, 'w') as f:
            json.dump(prompts_data, f, indent=2)
        
        print(f"✓ Saved prompts: {prompts_json_path}")
    
    # Extract prompts
    input_boxes = np.array(pixel_prompts.get("positive_boxes", []))
    
    # Prepare points based on mode
    input_point = None
    input_label = None
    
    if use_points:
        positive_points_pixel = np.array(pixel_prompts.get("positive_points", []))
        negative_points_pixel = np.array(pixel_prompts.get("negative_points", []))
        
        if positive_points_only and len(positive_points_pixel) > 0:
            input_point = positive_points_pixel
            input_label = np.ones(len(positive_points_pixel))
        elif negative_points_only and len(negative_points_pixel) > 0:
            input_point = negative_points_pixel
            input_label = np.zeros(len(negative_points_pixel))
        elif len(positive_points_pixel) > 0 and len(negative_points_pixel) > 0:
            input_point = np.concatenate([positive_points_pixel, negative_points_pixel], axis=0)
            input_label = np.concatenate([
                np.ones(len(positive_points_pixel)), 
                np.zeros(len(negative_points_pixel))
            ])
    
    # Set image
    inference_state = processor.set_image(image)
    
    # Collect results
    all_masks = []
    all_scores = []
    all_boxes = []
    per_box_info = []
    
    # Load ground truth
    gt_label = load_ground_truth_label(image_path)
    
    # Process each box
    if use_boxes and len(input_boxes) > 0:
        for i, input_box in enumerate(input_boxes):
            if points_in_box_only and input_point is not None:
                box_points, box_labels = get_points_in_box(input_point, input_label, input_box)
                use_points_for_box = box_points if len(box_points) > 0 else None
                use_labels_for_box = box_labels if len(box_labels) > 0 else None
            elif use_points and input_point is not None:
                use_points_for_box = input_point
                use_labels_for_box = input_label
            else:
                use_points_for_box = None
                use_labels_for_box = None
            
            masks, scores, logits = model.predict_inst(
                inference_state,
                point_coords=use_points_for_box,
                point_labels=use_labels_for_box,
                box=input_box,
                multimask_output=False,
            )
            
            all_masks.append(masks[0])
            all_scores.append(scores[0])
            all_boxes.append(input_box)
            
            box_info = {
                'box_id': i + 1,
                'box_coords': [float(v) for v in input_box],
                'sam3_score': float(scores[0]),
                'num_points_used': len(use_points_for_box) if use_points_for_box is not None else 0,
                'num_positive_points': int((use_labels_for_box == 1).sum()) if use_labels_for_box is not None else 0,
                'num_negative_points': int((use_labels_for_box == 0).sum()) if use_labels_for_box is not None else 0
            }
            per_box_info.append(box_info)
    
    # Calculate overall metrics
    overall_metrics = {
        'mode': mode_name,
        'num_boxes': len(all_boxes),
        'avg_sam3_score': float(np.mean(all_scores)) if all_scores else 0.0
    }
    
    if gt_label is not None:
        gt_binary = (gt_label > 0).astype(bool)
        merged_mask = np.zeros_like(gt_label, dtype=bool)
        for mask in all_masks:
            merged_mask = np.logical_or(merged_mask, mask)
        
        TP = np.logical_and(merged_mask, gt_binary).sum()
        FP = np.logical_and(merged_mask, ~gt_binary).sum()
        FN = np.logical_and(~merged_mask, gt_binary).sum()
        TN = np.logical_and(~merged_mask, ~gt_binary).sum()
        
        overall_metrics['dice'] = float((2 * TP) / (2 * TP + FP + FN)) if (2 * TP + FP + FN) > 0 else 0.0
        overall_metrics['precision'] = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        overall_metrics['recall'] = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        overall_metrics['iou'] = float(TP / (TP + FP + FN)) if (TP + FP + FN) > 0 else 0.0
        overall_metrics['f1'] = overall_metrics['dice']
        overall_metrics['accuracy'] = float((TP + TN) / (TP + TN + FP + FN)) if (TP + TN + FP + FN) > 0 else 0.0
        
        overall_metrics['pixel_stats'] = {
            'true_positive': int(TP),
            'false_positive': int(FP),
            'false_negative': int(FN),
            'true_negative': int(TN),
            'total_gt_pixels': int(gt_binary.sum()),
            'total_pred_pixels': int(merged_mask.sum())
        }
    
    # Visualization
    num_boxes = len(all_boxes)
    has_gt = gt_label is not None
    num_panels = num_boxes + (2 if has_gt else 1)
    
    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))
    if num_panels == 1:
        axes = [axes]
    
    colors_map = [
        [1.0, 0.2, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 0.2, 1.0],
        [1.0, 0.8, 0.0],
    ]
    
    # Individual masks
    for idx, (mask, score, box) in enumerate(zip(all_masks, all_scores, all_boxes)):
        ax = axes[idx]
        ax.imshow(image, cmap='gray')
        
        mask_np = mask.astype(np.uint8)
        color = np.concatenate([colors_map[idx % len(colors_map)], [0.6]])
        h, w = mask_np.shape
        mask_img = mask_np.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_img)
        
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, 
                                    edgecolor='yellow', facecolor='none', linewidth=2))
        
        if input_point is not None:
            box_points, box_labels = get_points_in_box(input_point, input_label, box)
            if len(box_points) > 0:
                show_points(box_points, box_labels, ax, marker_size=200)
        
        ax.set_title(f'Box #{idx+1}\nScore: {score:.3f}', fontsize=10, weight='bold')
        ax.axis('off')
    
    # Merged view
    ax_merged = axes[num_boxes]
    ax_merged.imshow(image, cmap='gray')
    
    for idx, mask in enumerate(all_masks):
        mask_np = mask.astype(np.uint8)
        color = np.concatenate([colors_map[idx % len(colors_map)], [0.5]])
        h, w = mask_np.shape
        mask_img = mask_np.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax_merged.imshow(mask_img)
    
    for idx, box in enumerate(all_boxes):
        x0, y0, x1, y1 = box
        ax_merged.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, 
                                           edgecolor=colors_map[idx % len(colors_map)], 
                                           facecolor='none', linewidth=2))
    
    if input_point is not None:
        show_points(input_point, input_label, ax_merged, marker_size=200)
    
    ax_merged.set_title(f'All Merged\nAvg: {overall_metrics["avg_sam3_score"]:.3f}', 
                        fontsize=10, weight='bold')
    ax_merged.axis('off')
    
    # Ground truth
    if has_gt:
        ax_gt = axes[num_boxes + 1]
        gt_binary_vis = (gt_label > 0).astype(np.uint8)
        ax_gt.imshow(gt_binary_vis, cmap='gray', vmin=0, vmax=1)
        
        ax_gt.set_title(
            f'Ground Truth\nDice: {overall_metrics["dice"]:.3f} | Prec: {overall_metrics["precision"]:.3f}\nRec: {overall_metrics["recall"]:.3f} | IoU: {overall_metrics["iou"]:.3f}',
            fontsize=10, weight='bold'
        )
        ax_gt.axis('off')
    
    # Save visualization with mode in filename
    viz_path = output_dir / f"{filename_base}_visualization.png"
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results JSON with mode in filename
    results_data = {
        'image_path': str(image_path),
        'image_name': image_name,
        'image_size': {'width': img_width, 'height': img_height},
        'mode': mode_name,
        'overall_metrics': overall_metrics,
        'per_box_info': per_box_info,
        'visualization_path': str(viz_path),
        'prompts_json_path': str(prompts_json_path)
    }
    
    json_path = output_dir / f"{filename_base}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Image: {image_name}")
    print(f"Mode: {mode_name}")
    print(f"Boxes: {len(all_boxes)}, Avg SAM3 score: {overall_metrics['avg_sam3_score']:.3f}")
    if has_gt:
        print(f"Dice: {overall_metrics['dice']:.3f} | Prec: {overall_metrics['precision']:.3f}")
        print(f"Recall: {overall_metrics['recall']:.3f} | IoU: {overall_metrics['iou']:.3f}")
    print(f"Saved: {viz_path}")
    print(f"JSON: {json_path}")
    print(f"{'='*60}\n")
    
    return {
        'merged_mask': merged_mask,  
        'image_path': str(image_path),
        'image_name': image_name,
        'image_size': {'width': img_width, 'height': img_height},
        'mode': mode_name,
        'overall_metrics': overall_metrics,
        'per_box_info': per_box_info,
        'visualization_path': str(viz_path),
        'prompts_json_path': str(prompts_json_path)
    }



# ============================================================================
# MAIN EXECUTION
# ============================================================================

from pathlib import Path
from tqdm import tqdm
import traceback


def generate_visual_feedback_with_gemini(
    original_image_path,
    segmentation_mask,
    previous_prompts,
    feedback_prompt,
    model_name="gemini-2.0-flash-exp",
):
    """
    Stage 1: Generate visual feedback based on ER-specific criteria
    Improved prompt for better correlation with actual metrics
    """
    from google import genai
    from PIL import Image
    import json
    
    # Create comparison image
    comparison_image = create_comparison_image(original_image_path, segmentation_mask)
    

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=model_name,
            contents=[feedback_prompt, comparison_image]
        )
        
        # Extract tokens
        tokens = {
            'input': getattr(response.usage_metadata, 'prompt_token_count', 0),
            'output': getattr(response.usage_metadata, 'candidates_token_count', 0)
        }
        
        print("\n" + "="*80)
        print("STAGE 1: ER VISUAL FEEDBACK (IMPROVED)")
        print("="*80)
        
        # Clean and parse
        clean_json_str = clean_gemini_json_response(response.text)
        feedback_data = json.loads(clean_json_str)
        
        # Validate
        if 'quality_score' not in feedback_data:
            print("Warning: No quality_score in feedback")
            if return_tokens:
                return None, tokens
            return None
        
        if 'criteria_scores' not in feedback_data:
            print("Warning: No criteria_scores in feedback")
            feedback_data['criteria_scores'] = {}
        
        if 'visual_guidance' not in feedback_data:
            print("Warning: No visual_guidance in feedback")
            if return_tokens:
                return None, tokens
            return None
        
        quality_score = feedback_data['quality_score']
        criteria = feedback_data['criteria_scores']
        visual_guidance = feedback_data['visual_guidance']
        
        print(f"\nOverall Quality Score: {quality_score:.3f}")
        print(f"Tokens: {tokens['input']} in, {tokens['output']} out")
        
        print(f"\nCriteria Breakdown:")
        print(f"  Morphological Coverage: {criteria.get('morphological_coverage', 0):.3f} (25%)")
        print(f"  Boundary Accuracy:      {criteria.get('boundary_accuracy', 0):.3f} (25%)")
        print(f"  Spatial Accuracy:       {criteria.get('spatial_accuracy', 0):.3f} (20%)")
        print(f"  Continuity:             {criteria.get('continuity', 0):.3f} (15%)")
        print(f"  Specificity:            {criteria.get('specificity', 0):.3f} (15%)")
        
        # Verify weighted calculation
        calculated_score = (
            criteria.get('morphological_coverage', 0) * 0.25 +
            criteria.get('boundary_accuracy', 0) * 0.25 +
            criteria.get('spatial_accuracy', 0) * 0.20 +
            criteria.get('continuity', 0) * 0.15 +
            criteria.get('specificity', 0) * 0.15
        )
        print(f"  Calculated (verification): {calculated_score:.3f}")
        
        if abs(quality_score - calculated_score) > 0.05:
            print(f"  ⚠ Warning: Score mismatch! Using calculated value.")
            feedback_data['quality_score'] = calculated_score
        
        print(f"\n{'='*80}")
        print("VISUAL DETECTION GUIDANCE:")
        print(f"{'='*80}")
        print(visual_guidance)
        print(f"{'='*80}")

        return feedback_data, tokens

        
    except json.JSONDecodeError as e:
        print(f"\n✗ Failed to parse feedback: {e}")
        print(f"Raw response:\n{response.text[:1000]}")
        return None, {'input': 0, 'output': 0}

    except Exception as e:
        print(f"\n✗ Error generating feedback: {e}")
        import traceback
        traceback.print_exc()

        return None, {'input': 0, 'output': 0}



def generate_refined_prompts_from_feedback(
    original_image_path,
    feedback_data,
    prompt_template,
    model_name="gemini-2.0-flash-exp"
):
    """
    Stage 2: Generate refined prompts using visual_guidance as 'what_to_detect'
    """
    from google import genai
    from PIL import Image
    import json
    
    # Extract visual guidance from feedback
    visual_guidance = feedback_data['visual_guidance']
    quality_score = feedback_data['quality_score']
    
    if not visual_guidance:
        print("⚠ No visual_guidance in feedback_data, using default")
        visual_guidance = "Endoplasmic Reticulum: interconnected network of flattened sacs (cisternae) and tubular structures. Look for reticular patterns, branching tubules, and sheet-like structures."
    
    # Create refined prompt by replacing {what_to_detect} in ER_PROMPT_TEMPLATE
    refinement_prompt = prompt_template.format(what_to_detect=visual_guidance)
    
    print(f"\n{'='*80}")
    print("STAGE 2: GENERATING REFINED PROMPTS")
    print(f"{'='*80}")
    print(f"Quality Score from Feedback: {quality_score:.2f}")
    print(f"\nUpdated 'WHAT TO DETECT' section:")
    print(f"{'-'*80}")
    print(visual_guidance)
    print(f"{'-'*80}\n")
    
    # Load image
    original_image = Image.open(original_image_path)
    
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=model_name,
            contents=[refinement_prompt, original_image]
        )


        tokens = {
            'input': getattr(response.usage_metadata, 'prompt_token_count', 0),
            'output': getattr(response.usage_metadata, 'candidates_token_count', 0)
        }
        
        print(f"{'='*80}")
        print("GEMINI RESPONSE:")
        print(f"{'='*80}")
        print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        print(f"{'='*80}")
        
        # Clean and parse JSON
        clean_json_str = clean_gemini_json_response(response.text)
        refined_prompts = json.loads(clean_json_str)
        
        # Validate structure
        if 'positive_boxes' not in refined_prompts:
            print("Warning: No 'positive_boxes' in refined prompts")
            return None
        
        # Add default keys if missing
        if 'negative_points' not in refined_prompts:
            refined_prompts['negative_points'] = []
        if 'positive_points' not in refined_prompts:
            refined_prompts['positive_points'] = []
        
        print(f"\n✓ Refined prompts generated:")
        print(f"  Positive boxes: {len(refined_prompts['positive_boxes'])}")
        print(f"  Positive points: {len(refined_prompts.get('positive_points', []))}")
        print(f"  Negative points: {len(refined_prompts.get('negative_points', []))}")
        
        return refined_prompts, tokens
        
    except json.JSONDecodeError as e:
        print(f"\n✗ Failed to parse refined prompts: {e}")
        print(f"Response text: {response.text[:500]}")
        return None
    except Exception as e:
        print(f"\n✗ Error generating refined prompts: {e}")
        import traceback
        traceback.print_exc()
        return None



# ============================================================================
# Precess with feedback iteration
# ============================================================================


def process_image_with_iterations_new(
    image_path,
    initial_prompts,
    model,
    processor,
    output_dir,
    gemini_model_name,
    feedback_prompt,
    prompt_template,
    enable_refinement=True,
    quality_threshold=0.85,
    max_iterations=4,  # Number of SAM3 iterations (will have max_iterations+1 feedbacks)
    evaluation_mode='boxes_all_points'

):
    """
    Process image with iterative refinement
    - Runs max_iterations of SAM3 segmentation
    - Each SAM3 run followed by feedback generation
    - Final extra feedback after last SAM3 (no more SAM3)
    
    Example: max_iterations=4
    - Iteration 0: SAM3 + Feedback 0 → Refine
    - Iteration 1: SAM3 + Feedback 1 → Refine
    - Iteration 2: SAM3 + Feedback 2 → Refine
    - Iteration 3: SAM3 + Feedback 3 → Refine
    - Final Feedback 4 (no SAM3)
    """
    output_dir = Path(output_dir)
    image_name = Path(image_path).stem
    
    iterations_data = {}
    iteration_prompts = initial_prompts
    stopped_reason = 'completed_all_iterations'
    
    # All SAM3 modes to run
    modes = [
        {'name': 'boxes_all_points', 'use_boxes': True, 'use_points': True, 
         'points_in_box_only': False, 'positive_points_only': False, 'negative_points_only': False},
        {'name': 'boxes_positive_points_only', 'use_boxes': True, 'use_points': True,
         'points_in_box_only': False, 'positive_points_only': True, 'negative_points_only': False},
        {'name': 'boxes_negative_points_only', 'use_boxes': True, 'use_points': True,
         'points_in_box_only': False, 'positive_points_only': False, 'negative_points_only': True},
        {'name': 'boxes_filtered_points', 'use_boxes': True, 'use_points': True,
         'points_in_box_only': True, 'positive_points_only': False, 'negative_points_only': False},
        {'name': 'boxes_only', 'use_boxes': True, 'use_points': False,
         'points_in_box_only': False, 'positive_points_only': False, 'negative_points_only': False}
    ]
    
    # Validate evaluation_mode
    valid_modes = [m['name'] for m in modes]
    if evaluation_mode not in valid_modes:
        raise ValueError(f"evaluation_mode must be one of {valid_modes}, got '{evaluation_mode}'")
    
    print(f"\nUsing '{evaluation_mode}' mode for Gemini evaluation")
    print(f"Will run {max_iterations} SAM3 iterations + 1 final feedback")

    # Track best iteration
    best_iteration = 0
    best_quality_score = 0


    sam3_times = []
    gemini_eval_times = []
    gemini_refine_times = []
    gemini_eval_tokens = {'input': 0, 'output': 0}
    gemini_refine_tokens = {'input': 0, 'output': 0}
    iteration_times = []
    
    # ========================================================================
    # MAIN LOOP: Run SAM3 for max_iterations
    # ========================================================================
    iteration_times = []
    iteration_tokens = {'input': 0, 'output': 0}
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} - SAM3 SEGMENTATION")
        print(f"{'='*80}")
        
        # Create iteration directory
        iter_dir = output_dir / f"iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prompts for this iteration
        prompts_path = iter_dir / f"{image_name}_prompts_iter{iteration}.json"
        prompts_to_save = {
            'positive_boxes': iteration_prompts.get('positive_boxes', []),
            'positive_points': iteration_prompts.get('positive_points', []),
            'negative_points': iteration_prompts.get('negative_points', []),
        }
        
        with open(prompts_path, 'w') as f:
            json.dump(prompts_to_save, f, indent=2)
        
        # --------------------------------------------------------------------
        # Run SAM3 for all modes
        # --------------------------------------------------------------------
        
        sam3_start = time.time()
        iteration_results = {}

        
        for mode_idx, mode_config in enumerate(modes):
            mode_name = mode_config['name']
            mode_params = {k: v for k, v in mode_config.items() if k != 'name'}
            
            eval_marker = " ← EVALUATION MODE" if mode_name == evaluation_mode else ""
            print(f"\n[{mode_idx+1}/{len(modes)}] Running mode: {mode_name}{eval_marker}")
            
            result = sam3_inference_and_visualize(
                image_path=image_path,
                prompts=iteration_prompts,
                model=model,
                processor=processor,
                output_dir=str(iter_dir),
                **mode_params
            )
            
            iteration_results[mode_name] = result

        sam3_time = time.time() - sam3_start
        sam3_times.append(sam3_time)
        print(f"✓ SAM3 inference time: {sam3_time:.1f}s")
        
        # Store iteration data
        iterations_data[f'iteration_{iteration}'] = iteration_results
        
        # --------------------------------------------------------------------
        # Generate Feedback for this iteration
        # --------------------------------------------------------------------
        
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} - FEEDBACK GENERATION")
        print(f"{'='*80}")
        
        eval_result = iteration_results.get(evaluation_mode)
        
        if not eval_result or 'merged_mask' not in eval_result:
            print(f"⚠ No merged_mask in {evaluation_mode} mode, skipping feedback")
            # Continue with next iteration using same prompts
            continue
        
        merged_mask = eval_result['merged_mask']
        
        # Save comparison image
        comparison_image = create_comparison_image(image_path, merged_mask)
        comparison_path = output_dir / f"comparison_iteration_{iteration}_{evaluation_mode}.png"
        comparison_image.save(comparison_path)
        print(f"✓ Saved comparison image: {comparison_path}")

        gemini_eval_start = time.time()
        # Generate visual feedback
        feedback_data, feedback_tokens = generate_visual_feedback_with_gemini(
            original_image_path=image_path,
            segmentation_mask=merged_mask,
            previous_prompts=iteration_prompts,
            feedback_prompt = feedback_prompt,
            model_name=gemini_model_name
        )

        gemini_eval_time = time.time() - gemini_eval_start
        gemini_eval_times.append(gemini_eval_time)
        gemini_eval_tokens['input'] += feedback_tokens.get('input', 0)
        gemini_eval_tokens['output'] += feedback_tokens.get('output', 0)
        
        if feedback_data is None:
            print("⚠ Failed to generate feedback, continuing with same prompts")
            continue
        
        # Extract quality score
        quality_score = feedback_data.get('quality_score', 0)
        
        # Update eval_result with feedback
        eval_result['quality_score'] = quality_score
        eval_result['missed_regions'] = feedback_data.get('missed_regions', [])
        eval_result['false_positives'] = feedback_data.get('false_positives', [])
        eval_result['recommendations'] = feedback_data.get('recommendations', [])
        
        # Update stored iteration data
        iterations_data[f'iteration_{iteration}'][evaluation_mode] = eval_result
        
        # Save feedback
        feedback_path = iter_dir / f"{image_name}_feedback_iter{iteration}.json"
        with open(feedback_path, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        print(f"✓ Saved feedback: {feedback_path}")
        
        print(f"\nQuality Score: {quality_score:.2f}")
        
        # Track best iteration
        if quality_score > best_quality_score:
            best_quality_score = quality_score
            best_iteration = iteration
            print(f"✓ New best quality score: {quality_score:.2f}")
        else:
            print(f"  Previous best: {best_quality_score:.2f} (iteration {best_iteration})")
        
        # --------------------------------------------------------------------
        # Generate refined prompts for NEXT iteration (if not last)
        # --------------------------------------------------------------------
        
        if iteration < max_iterations - 1:  # Not the last SAM3 iteration
            print(f"\n{'='*80}")
            print(f"GENERATING PROMPTS FOR ITERATION {iteration + 1}")
            print(f"{'='*80}")

            gemini_refine_start = time.time()
            
            refined_prompts, refine_tokens = generate_refined_prompts_from_feedback(
                original_image_path=image_path,
                feedback_data=feedback_data,
                prompt_template=prompt_template, 
                model_name=gemini_model_name
            )

            gemini_refine_time = time.time() - gemini_refine_start
            gemini_refine_times.append(gemini_refine_time)
            gemini_refine_tokens['input'] += refine_tokens.get('input', 0)
            gemini_refine_tokens['output'] += refine_tokens.get('output', 0)

            print(f"✓ Gemini refinement time: {gemini_refine_time:.1f}s")
            print(f"✓ Tokens: {refine_tokens.get('input', 0)} in, {refine_tokens.get('output', 0)} out")
            
            if refined_prompts is None:
                print("⚠ Failed to generate refined prompts, reusing previous prompts")
            else:
                iteration_prompts = refined_prompts
                print(f"✓ Refined prompts ready for iteration {iteration + 1}")
                
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
    
    # ========================================================================
    # FINAL FEEDBACK (after last SAM3 iteration, no more SAM3)
    # ========================================================================
    
    final_iteration_num = max_iterations
    print(f"\n{'='*80}")
    print(f"FINAL FEEDBACK (Iteration {final_iteration_num}) - NO SAM3")
    print(f"{'='*80}")
    
    # Use last SAM3 result for final feedback
    last_iter_key = f'iteration_{max_iterations - 1}'
    if last_iter_key in iterations_data:
        last_eval_result = iterations_data[last_iter_key].get(evaluation_mode)
        
        if last_eval_result and 'merged_mask' in last_eval_result:
            merged_mask = last_eval_result['merged_mask']
            
            # Save comparison image
            comparison_image = create_comparison_image(image_path, merged_mask)
            comparison_path = output_dir / f"comparison_final_{evaluation_mode}.png"
            comparison_image.save(comparison_path)
            print(f"✓ Saved final comparison image: {comparison_path}")


            final_eval_start = time.time()
            
            # Generate final feedback
            final_feedback_data, final_feedback_tokens = generate_visual_feedback_with_gemini(
                original_image_path=image_path,
                segmentation_mask=merged_mask,
                previous_prompts=iteration_prompts,
                feedback_prompt=feedback_prompt,
                model_name=gemini_model_name
            )

            final_eval_time = time.time() - final_eval_start
            gemini_eval_times.append(final_eval_time)
            gemini_eval_tokens['input'] += final_feedback_tokens.get('input', 0)
            gemini_eval_tokens['output'] += final_feedback_tokens.get('output', 0)

            print(f"✓ Final evaluation time: {final_eval_time:.1f}s")
            
            if final_feedback_data:
                final_quality_score = final_feedback_data.get('quality_score', 0)
                
                # Create a feedback-only entry
                iterations_data[f'iteration_{final_iteration_num}'] = {
                    evaluation_mode: {
                        'quality_score': final_quality_score,
                        'missed_regions': final_feedback_data.get('missed_regions', []),
                        'false_positives': final_feedback_data.get('false_positives', []),
                        'recommendations': final_feedback_data.get('recommendations', []),
                        'feedback_only': True  # Mark as feedback-only iteration
                    }
                }
                
                # Save final feedback
                final_feedback_path = output_dir / f"{image_name}_feedback_final.json"
                with open(final_feedback_path, 'w') as f:
                    json.dump(final_feedback_data, f, indent=2)
                print(f"✓ Saved final feedback: {final_feedback_path}")
                
                print(f"\nFinal Quality Score: {final_quality_score:.2f}")
                
                # Check if final is best
                if final_quality_score > best_quality_score:
                    # Can't use this as best since no SAM3 was run
                    print(f"⚠ Final feedback score ({final_quality_score:.2f}) higher than best, but no SAM3 to use")
            else:
                print("⚠ Failed to generate final feedback")
    
    # ========================================================================
    # CREATE SUMMARY
    # ========================================================================
    
    summary = {
        'image_name': image_name,
        'num_sam3_iterations': max_iterations,
        'num_feedback_iterations': max_iterations + 1,  # Including final feedback
        'stopped_reason': stopped_reason,
        'evaluation_mode': evaluation_mode,
        'best_iteration': {
            'iteration_number': best_iteration,
            'quality_score': best_quality_score
        },
        'iterations': {}
    }
    
    quality_scores = []
    
    for iter_key in sorted(iterations_data.keys()):
        if iter_key.startswith('iteration_'):
            iter_num = int(iter_key.split('_')[1])
            iter_summary = {}
            
            iter_results = iterations_data[iter_key]
            
            for mode_name, mode_result in iter_results.items():
                if isinstance(mode_result, dict):
                    mode_summary = {}
                    
                    # Check if this is a feedback-only iteration
                    if mode_result.get('feedback_only', False):
                        mode_summary['feedback_only'] = True
                        mode_summary['quality_score'] = mode_result.get('quality_score')
                        mode_summary['missed_regions'] = mode_result.get('missed_regions', [])
                        mode_summary['false_positives'] = mode_result.get('false_positives', [])
                    else:
                        # Regular iteration with SAM3 + feedback
                        mode_summary['num_boxes'] = len(mode_result.get('boxes', []))
                        
                        if 'overall_metrics' in mode_result:
                            mode_summary['metrics'] = mode_result['overall_metrics']
                        
                        # Add feedback data for evaluation mode
                        if mode_name == evaluation_mode:
                            if 'quality_score' in mode_result:
                                mode_summary['quality_score'] = mode_result['quality_score']
                                quality_scores.append(mode_result['quality_score'])
                            if 'missed_regions' in mode_result:
                                mode_summary['missed_regions'] = mode_result['missed_regions']
                            if 'false_positives' in mode_result:
                                mode_summary['false_positives'] = mode_result['false_positives']
                    
                    iter_summary[mode_name] = mode_summary
            
            summary['iterations'][iter_key] = iter_summary
    
    # Extract best iteration metrics
    if f'iteration_{best_iteration}' in iterations_data:
        best_iter_data = iterations_data[f'iteration_{best_iteration}']
        
        if evaluation_mode in best_iter_data:
            best_mode_result = best_iter_data[evaluation_mode]
            
            if 'overall_metrics' in best_mode_result:
                summary['best_iteration']['final_metrics'] = best_mode_result['overall_metrics']
    
    # Quality score progression
    if len(quality_scores) > 0:
        summary['quality_progression'] = {
            'scores': quality_scores,
            'initial_score': quality_scores[0],
            'final_score': quality_scores[-1],
            'improvement': quality_scores[-1] - quality_scores[0],
            'best_score': max(quality_scores),
            'worst_score': min(quality_scores)
        }
        
        print(f"\n{'='*80}")
        print("QUALITY SCORE PROGRESSION")
        print(f"{'='*80}")
        for i, score in enumerate(quality_scores):
            marker = " ← BEST" if i == best_iteration else ""
            print(f"Iteration {i}: {score:.3f}{marker}")
        print(f"\nInitial → Final: {quality_scores[0]:.3f} → {quality_scores[-1]:.3f}")
        print(f"Change: {quality_scores[-1] - quality_scores[0]:+.3f}")
        print(f"Best: {max(quality_scores):.3f} (iteration {best_iteration})")
        print(f"{'='*80}\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"SAM3 Iterations: {max_iterations}")
    print(f"Total Feedbacks: {max_iterations + 1} (including final)")
    print(f"Best Iteration: {best_iteration} (Quality: {best_quality_score:.2f})")
    if 'final_metrics' in summary['best_iteration']:
        metrics = summary['best_iteration']['final_metrics']
        print(f"Best Dice: {metrics.get('dice', 0):.3f}")
        print(f"Best IoU:  {metrics.get('iou', 0):.3f}")
    print(f"{'='*80}")
    
    # Save summary
    summary_path = output_dir / "iteration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Iteration summary saved: {summary_path}")
    
    # Save overlays
    print("\nSaving iteration overlay visualizations...")
    
    gt_path = str(image_path).replace('/images/', '/labels/').replace('image_', 'label_')
    try:
        gt_label = np.array(Image.open(gt_path))
    except:
        gt_label = None
        print("Ground truth not found")
    
    save_iteration_overlays(
        image_path=image_path,
        iterations_data=iterations_data,
        gt_label=gt_label,
        output_dir=output_dir,
        evaluation_mode=evaluation_mode
    )
    
    save_individual_iteration_masks(
        image_path=image_path,
        iterations_data=iterations_data,
        gt_label=gt_label,
        output_dir=output_dir,
        evaluation_mode=evaluation_mode
    )

    stats = {
            'iteration_times': iteration_times,
            'avg_iteration_time': sum(iteration_times) / len(iteration_times) if iteration_times else 0,
            'sam3_times': sam3_times,
            'gemini_eval_times': gemini_eval_times,
            'gemini_refine_times': gemini_refine_times,
            'gemini_eval_tokens': gemini_eval_tokens,
            'gemini_refine_tokens': gemini_refine_tokens,
            # Total tokens for backward compatibility
            'total_tokens': {
                'input': gemini_eval_tokens['input'] + gemini_refine_tokens['input'],
                'output': gemini_eval_tokens['output'] + gemini_refine_tokens['output']
            }
        }
    
    return summary, stats
# ============================================================================
# BATCH PROCESSING FOR ALL IMAGES
# ============================================================================


def save_iteration_overlays(
    image_path,
    iterations_data,
    gt_label,
    output_dir,
    evaluation_mode='boxes_all_points'
):
    """
    Save overlay visualizations for each iteration + ground truth
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original image
    original = Image.open(image_path)
    image_array = np.array(original)
    
    # Get all iterations that have actual SAM3 results (not feedback-only)
    all_iters = []
    for k in sorted(iterations_data.keys()):
        if k.startswith('iteration_'):
            iter_data = iterations_data[k].get(evaluation_mode, {})
            # Only include if it has a merged_mask (i.e., SAM3 was run)
            if 'merged_mask' in iter_data:
                all_iters.append(k)
    
    if len(all_iters) == 0:
        print("No iterations with masks found")
        return None
    
    # Create figure: original + all SAM3 iterations + GT
    n_panels = 1 + len(all_iters) + 1  # original + iterations + GT
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    
    if n_panels == 1:
        axes = [axes]
    
    # Panel 0: Original image
    axes[0].imshow(image_array, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # Panels 1 to N: Each iteration with SAM3 results
    for idx, iter_key in enumerate(all_iters):
        ax = axes[idx + 1]
        iter_num = iter_key.split('_')[1]
        
        # Get merged mask from this iteration
        iter_result = iterations_data[iter_key].get(evaluation_mode, {})
        
        merged_mask = iter_result['merged_mask']
        
        # Show original with overlay
        ax.imshow(image_array, cmap='gray')
        
        # Overlay mask in red with transparency
        mask_overlay = np.zeros((*merged_mask.shape, 4))
        mask_overlay[merged_mask] = [1, 0, 0, 0.5]  # Red, 50% transparent
        ax.imshow(mask_overlay)
        
        # Get metrics
        metrics = iter_result.get('overall_metrics', {})
        dice = metrics.get('dice', 0)
        iou = metrics.get('iou', 0)
        
        # Get quality score if available
        quality_score = iter_result.get('quality_score')
        
        # Build title
        title = f"Iteration {iter_num}\nDice: {dice:.3f} | IoU: {iou:.3f}"
        if quality_score is not None:
            title = f"Iteration {iter_num} (Q: {quality_score:.2f})\nDice: {dice:.3f} | IoU: {iou:.3f}"
        
        ax.set_title(title, fontsize=11, weight='bold')
        ax.axis('off')
        
        # Draw boxes if available
        if 'per_box_info' in iter_result:
            for box_info in iter_result['per_box_info']:
                box_coords = box_info['box_coords']
                x0, y0, x1, y1 = box_coords
                
                # Draw box
                rect = mpatches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=2, edgecolor='yellow', facecolor='none'
                )
                ax.add_patch(rect)
        
        # Draw points if mode includes points
        if 'points' in evaluation_mode or evaluation_mode == 'boxes_all_points':
            # Try to get prompts from iteration directory
            iter_dir = output_dir / f"iteration_{iter_num}"
            image_name = Path(image_path).stem
            prompts_path = iter_dir / f"{image_name}_prompts_iter{iter_num}.json"
            
            if prompts_path.exists():
                with open(prompts_path, 'r') as f:
                    prompts = json.load(f)
                
                # Convert normalized points to pixels
                h, w = image_array.shape
                
                # Draw positive points
                if 'positive_points' in prompts and prompts['positive_points']:
                    pos_points = np.array(prompts['positive_points'])
                    if pos_points.shape[0] > 0:
                        # Convert from [0,1000] to pixels
                        pos_y = (pos_points[:, 0] / 1000) * h
                        pos_x = (pos_points[:, 1] / 1000) * w
                        ax.scatter(pos_x, pos_y, c='lime', marker='*', s=200, 
                                 edgecolors='white', linewidths=1, label='Positive')
                
                # Draw negative points
                if 'negative_points' in prompts and prompts['negative_points']:
                    neg_points = np.array(prompts['negative_points'])
                    if neg_points.shape[0] > 0:
                        # Convert from [0,1000] to pixels
                        neg_y = (neg_points[:, 0] / 1000) * h
                        neg_x = (neg_points[:, 1] / 1000) * w
                        ax.scatter(neg_x, neg_y, c='red', marker='*', s=200, 
                                 edgecolors='white', linewidths=1, label='Negative')
    
    # Last panel: Ground Truth
    ax_gt = axes[-1]
    
    if gt_label is not None:
        gt_binary = (gt_label > 0).astype(bool)
        
        # Show original with GT overlay
        ax_gt.imshow(image_array, cmap='gray')
        
        # Overlay GT in green with transparency
        gt_overlay = np.zeros((*gt_binary.shape, 4))
        gt_overlay[gt_binary] = [0, 1, 0, 0.5]  # Green, 50% transparent
        ax_gt.imshow(gt_overlay)
        
        ax_gt.set_title('Ground Truth', fontsize=12, weight='bold')
        ax_gt.axis('off')
    else:
        ax_gt.text(0.5, 0.5, 'No GT', ha='center', va='center')
        ax_gt.set_title('Ground Truth\n(Not Available)', fontsize=12, weight='bold', color='red')
        ax_gt.axis('off')
    
    plt.tight_layout()
    
    # Save
    overlay_path = output_dir / f"all_iterations_overlay_{evaluation_mode}.png"
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved iteration overlays: {overlay_path}")
    
    return overlay_path


def save_individual_iteration_masks(
    image_path,
    iterations_data,
    gt_label,
    output_dir,
    evaluation_mode='boxes_all_points'
):
    """
    Save individual binary mask images for each iteration
    """
    from PIL import Image
    import numpy as np
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all iterations with SAM3 results (not feedback-only)
    all_iters = []
    for k in sorted(iterations_data.keys()):
        if k.startswith('iteration_'):
            iter_data = iterations_data[k].get(evaluation_mode, {})
            # Only include if it has a merged_mask
            if 'merged_mask' in iter_data:
                all_iters.append(k)
    
    for iter_key in all_iters:
        iter_num = iter_key.split('_')[1]
        
        # Get merged mask
        iter_result = iterations_data[iter_key].get(evaluation_mode, {})
        merged_mask = iter_result['merged_mask']
        
        # Convert to binary image (0 or 255)
        mask_binary = (merged_mask > 0).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_binary)
        
        # Save
        mask_path = output_dir / f"mask_iteration_{iter_num}_{evaluation_mode}.png"
        mask_image.save(mask_path)
        print(f"✓ Saved mask: {mask_path}")
    
    # Save GT
    if gt_label is not None:
        gt_binary = (gt_label > 0).astype(np.uint8) * 255
        gt_image = Image.fromarray(gt_binary)
        gt_path = output_dir / "mask_ground_truth.png"
        gt_image.save(gt_path)
        print(f"✓ Saved GT mask: {gt_path}")




# ============================================================================
# TEST: All Images with Iterations
# ============================================================================

import argparse
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Process all ER images with SAM3 iterative refinement')
    
    # Paths
    parser.add_argument('--input_dir', type=str,
                       default='/home/idies/workspace/Storage/xyu1/persistent/cellmap-segmentation-challenge/src/cellmap_segmentation_challenge/utils/new_data/saved_data_multiscale_er_new/256_8nm/images',
                       help='Input directory containing images')
    parser.add_argument('--output_dir', type=str,
                       default='./test_refinement_all_images',
                       help='Output directory for results')
    parser.add_argument('--sam3_checkpoint', type=str,
                       default='/home/idies/workspace/Storage/xyu1/persistent/GenCELLAgent/src/sam3/checkpoints/sam3/sam3.pt',
                       help='Path to SAM3 checkpoint')
    
    # Model settings
    parser.add_argument('--gemini_model', type=str,
                       default='gemini-3-flash-preview',
                       help='Gemini model name')
    
    # Refinement settings
    parser.add_argument('--enable_refinement', action='store_true', default=True,
                       help='Enable iterative refinement')
    parser.add_argument('--quality_threshold', type=float, default=0.85,
                       help='Quality threshold to stop refinement')
    parser.add_argument('--max_iterations', type=int, default=5,
                       help='Maximum number of iterations')
    parser.add_argument('--evaluation_mode', type=str, default='boxes_only',
                       choices=['boxes_only', 'boxes_and_positive_points', 'boxes_and_all_points'],
                       help='Evaluation mode for SAM3')
    
    # Processing settings
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start processing from this image index')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End processing at this image index (exclusive)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip images that already have results')

    parser.add_argument('--organelle', type=str, 
                        default="er",
                        choices=['er', 'mito', 'mitochondria', 'golgi'],
                        help='Organelle type to segment (default: er)')
    
    return parser.parse_args()

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    args = parse_args()


    # Select prompt based on organelle type
    if args.organelle == 'er':
        text_prompt = ER_PROMPT_TEMPLATE.format(what_to_detect=ER_DEFAULT_DETECTION)
        prompt_template = ER_PROMPT_TEMPLATE
        feedback_prompt = ER_FEEDBACK
    elif args.organelle in ['mito', 'mitochondria']:
        text_prompt = MITO_PROMPT_TEMPLATE.format(what_to_detect=MITO_DEFAULT_DETECTION)
        prompt_template = MITO_PROMPT_TEMPLATE
        feedback_prompt = MITO_FEEDBACK
    elif args.organelle == 'golgi':
        text_prompt = GOLGI_PROMPT_TEMPLATE.format(what_to_detect=GOLGI_DEFAULT_DETECTION)
        prompt_template = GOLGI_PROMPT_TEMPLATE
        feedback_prompt = GOLGI_FEEDBACK
    else:
        raise ValueError(f"Unknown organelle: {args.organelle}")
        
    
    print("="*80)
    print("BATCH PROCESSING: All Images with Iterative Refinement")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  SAM3 checkpoint: {args.sam3_checkpoint}")
    print(f"  Gemini model: {args.gemini_model}")
    print(f"  Quality threshold: {args.quality_threshold}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Evaluation mode: {args.evaluation_mode}")
    print(f"  Skip existing: {args.skip_existing}")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 1: Load SAM3 Model
    # ========================================================================
    
    print("="*80)
    print("STEP 1: Loading SAM3 Model")
    print("="*80)
    
    model = build_sam3_image_model(
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        device="cuda",
        eval_mode=True,
    )
    processor = Sam3Processor(model)
    
    print("✓ SAM3 model loaded\n")
    
    # ========================================================================
    # STEP 2: Get All Images
    # ========================================================================
    
    print("="*80)
    print("STEP 2: Finding Images")
    print("="*80)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get all PNG images
    all_images = sorted(list(input_dir.glob("*.png")))
    
    # Apply start/end index
    if args.end_idx is not None:
        all_images = all_images[args.start_idx:args.end_idx]
    else:
        all_images = all_images[args.start_idx:]
    
    print(f"\nFound {len(all_images)} images to process")
    print(f"  (Index range: {args.start_idx} to {args.start_idx + len(all_images)})")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 3: Process Each Image
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: Processing Images")
    print("="*80 + "\n")
    
    results_summary = []
    failed_images = []
    skipped_images = []
    
    # Timing and token tracking
    # timing_stats = {
    #     'per_image_times': [],
    #     'per_iteration_times': [],
    #     'total_tokens': {'input': 0, 'output': 0},
    #     'per_image_tokens': []
    # }'

    timing_stats = {
    'per_image_times': [],
    'gemini_bbox_times': [],           # Time for initial bbox generation
    'sam3_inference_times': [],        # Time for SAM3 inference
    'gemini_eval_times': [],           # Time for Gemini evaluation
    'gemini_refine_times': [],         # ADD - Time for Gemini prompt refinement
    'per_iteration_times': [],
    'total_tokens': {
        'gemini_bbox_input': 0,
        'gemini_bbox_output': 0,
        'gemini_eval_input': 0,
        'gemini_eval_output': 0,
        'gemini_refine_input': 0,      # ADD
        'gemini_refine_output': 0      # ADD
    },
    'per_image_tokens': []
}
    
    start_time = datetime.now()
    
    # Create progress bar
    pbar = tqdm(
        enumerate(all_images, start=args.start_idx),
        total=len(all_images),
        desc="Overall Progress",
        unit="image",
        ncols=120
    )
    
    for idx, image_path in pbar:
        # Track image processing time
        image_start = time.time()
        
        # Update progress bar description
        pbar.set_description(f"Processing {image_path.stem}")
        
        # Print header
        pbar.write(f"\n{'='*80}")
        pbar.write(f"Image {idx+1}/{args.start_idx + len(all_images)}: {image_path.name}")
        pbar.write(f"{'='*80}")
        
        # Check if already processed
        image_output_dir = output_dir / image_path.stem
        summary_file = image_output_dir / "iteration_summary.json"
        
        if args.skip_existing and summary_file.exists():
            pbar.write(f"⊘ Skipping - already processed")
            skipped_images.append(image_path.name)
            pbar.set_postfix({
                'Success': len(results_summary),
                'Failed': len(failed_images),
                'Skipped': len(skipped_images)
            })
            continue
        
        try:
            # Initialize image token counter
            image_tokens = {
                'gemini_bbox_input': 0,
                'gemini_bbox_output': 0,
                'gemini_eval_input': 0,
                'gemini_eval_output': 0,
                'gemini_refine_input': 0,
                'gemini_refine_output': 0
            }
            
            # Generate initial prompts
            pbar.write(f"[1/3] Generating initial prompts with Gemini...")
            prompt_start = time.time()
            
            initial_prompts, prompt_tokens = generate_sam3_prompts_with_gemini(
                image_path=image_path,
                text_prompt=ER_PROMPT_TEMPLATE.format(what_to_detect=ER_DEFAULT_DETECTION),
                model_name=args.gemini_model,
            )
            
            prompt_time = time.time() - prompt_start

            prompt_time = time.time() - prompt_start
            timing_stats['gemini_bbox_times'].append(prompt_time)  # ADD
            image_tokens['gemini_bbox_input'] = prompt_tokens.get('input', 0)    # MODIFY
            image_tokens['gemini_bbox_output'] = prompt_tokens.get('output', 0)  # MODIFY
            #image_tokens['input'] += prompt_tokens.get('input', 0)
            # image_tokens['output'] += prompt_tokens.get('output', 0)
            
            pbar.write(f"  ✓ Positive boxes: {len(initial_prompts['positive_boxes'])}")
            pbar.write(f"  ✓ Positive points: {len(initial_prompts['positive_points'])}")
            pbar.write(f"  ✓ Negative points: {len(initial_prompts['negative_points'])}")
            pbar.write(f"  ✓ Time: {prompt_time:.1f}s | Tokens: {prompt_tokens.get('input', 0)} in, {prompt_tokens.get('output', 0)} out")
            
            # Process with iterations
            pbar.write(f"[2/3] Running SAM3 with iterative refinement...")
            iteration_start = time.time()
            
            result, iteration_stats = process_image_with_iterations_new(
                image_path=str(image_path),
                initial_prompts=initial_prompts,
                model=model,
                processor=processor,
                output_dir=image_output_dir,
                gemini_model_name=args.gemini_model,
                enable_refinement=args.enable_refinement,
                quality_threshold=args.quality_threshold,
                max_iterations=args.max_iterations,
                evaluation_mode=args.evaluation_mode,
                feedback_prompt=feedback_prompt,
                prompt_template=prompt_template
            )
            
            iteration_time = time.time() - iteration_start

            timing_stats['sam3_inference_times'].extend(iteration_stats.get('sam3_times', []))
            timing_stats['gemini_eval_times'].extend(iteration_stats.get('gemini_eval_times', []))
            timing_stats['gemini_refine_times'].extend(iteration_stats.get('gemini_refine_times', []))
            
            # Accumulate tokens from iterations
            image_tokens['gemini_eval_input'] = iteration_stats.get('gemini_eval_tokens', {}).get('input', 0)
            image_tokens['gemini_eval_output'] = iteration_stats.get('gemini_eval_tokens', {}).get('output', 0)
            image_tokens['gemini_refine_input'] = iteration_stats.get('gemini_refine_tokens', {}).get('input', 0)
            image_tokens['gemini_refine_output'] = iteration_stats.get('gemini_refine_tokens', {}).get('output', 0)
            
            # Track per-iteration times
            if 'iteration_times' in iteration_stats:
                timing_stats['per_iteration_times'].extend(iteration_stats['iteration_times'])
            
            # Save summary
            pbar.write(f"[3/3] Saving results...")
            
            # Calculate total image time
            image_time = time.time() - image_start
            timing_stats['per_image_times'].append(image_time)

            
            timing_stats['total_tokens']['gemini_bbox_input'] += image_tokens['gemini_bbox_input']
            timing_stats['total_tokens']['gemini_bbox_output'] += image_tokens['gemini_bbox_output']
            timing_stats['total_tokens']['gemini_eval_input'] += image_tokens['gemini_eval_input']
            timing_stats['total_tokens']['gemini_eval_output'] += image_tokens['gemini_eval_output']
            timing_stats['total_tokens']['gemini_refine_input'] += image_tokens['gemini_refine_input']
            timing_stats['total_tokens']['gemini_refine_output'] += image_tokens['gemini_refine_output']
            timing_stats['per_image_tokens'].append(image_tokens)
            
            # Add to summary
            best_info = result['best_iteration']
            summary_entry = {
                'image_name': result['image_name'],
                'num_sam3_iterations': result['num_sam3_iterations'],
                'num_feedback_iterations': result['num_feedback_iterations'],
                'stopped_reason': result['stopped_reason'],
                'best_iteration': best_info['iteration_number'],
                'best_quality': best_info['quality_score'],
                'best_metrics': best_info.get('final_metrics', {}),
                'output_dir': str(image_output_dir),
                'timing': {
                    'total_time': image_time,
                    'prompt_generation_time': prompt_time,
                    'iteration_time': iteration_time,
                    'avg_iteration_time': iteration_stats.get('avg_iteration_time', 0),

                     # ADD DETAILED BREAKDOWN:
                    'sam3_times': iteration_stats.get('sam3_times', []),
                    'gemini_eval_times': iteration_stats.get('gemini_eval_times', []),
                    'gemini_refine_times': iteration_stats.get('gemini_refine_times', []),
                    'avg_sam3_time': np.mean(iteration_stats.get('sam3_times', [0])),
                    'avg_gemini_eval_time': np.mean(iteration_stats.get('gemini_eval_times', [0])),
                    'avg_gemini_refine_time': np.mean(iteration_stats.get('gemini_refine_times', [0])) if iteration_stats.get('gemini_refine_times') else 0
                },
                'tokens': image_tokens
            }
            results_summary.append(summary_entry)
            
            # Get best metrics
            best_dice = best_info.get('final_metrics', {}).get('dice', 0)
            best_iou = best_info.get('final_metrics', {}).get('iou', 0)
            
            # Print quick summary
            pbar.write(f"  ✓ Completed:")
            pbar.write(f"    SAM3 iterations: {result['num_sam3_iterations']}")
            pbar.write(f"    Best quality: {best_info['quality_score']:.3f} (iteration {best_info['iteration_number']})")
            pbar.write(f"    Best Dice: {best_dice:.3f} | IoU: {best_iou:.3f}")
            pbar.write(f"    Total time: {image_time:.1f}s | Avg iter: {iteration_stats.get('avg_iteration_time', 0):.1f}s")
            pbar.write(f"    Tokens: {image_tokens['input']} in, {image_tokens['output']} out")
            
            # Calculate running averages
            avg_time = sum(timing_stats['per_image_times']) / len(timing_stats['per_image_times'])
            avg_tokens_in = timing_stats['total_tokens']['input'] / len(results_summary)
            avg_tokens_out = timing_stats['total_tokens']['output'] / len(results_summary)
            
            # Update progress bar with latest metrics
            pbar.set_postfix({
                'Quality': f"{best_info['quality_score']:.2f}",
                'Dice': f"{best_dice:.3f}",
                'Time': f"{image_time:.0f}s",
                'AvgTime': f"{avg_time:.0f}s",
                'Success': len(results_summary),
                'Failed': len(failed_images)
            })
            
        except Exception as e:
            pbar.write(f"  ✗ Failed: {str(e)}")
            failed_images.append({
                'image_name': image_path.name,
                'error': str(e)
            })
            
            pbar.set_postfix({
                'Success': len(results_summary),
                'Failed': len(failed_images),
                'Skipped': len(skipped_images)
            })
    
    pbar.close()
    
    # ========================================================================
    # STEP 4: Save Batch Summary with Timing and Token Stats
    # ========================================================================
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("STEP 4: Saving Batch Summary")
    print("="*80)
    
    # Calculate comprehensive statistics
    if timing_stats['per_image_times']:
        avg_image_time = sum(timing_stats['per_image_times']) / len(timing_stats['per_image_times'])
        min_image_time = min(timing_stats['per_image_times'])
        max_image_time = max(timing_stats['per_image_times'])
    else:
        avg_image_time = min_image_time = max_image_time = 0
    
    if timing_stats['per_iteration_times']:
        avg_iter_time = sum(timing_stats['per_iteration_times']) / len(timing_stats['per_iteration_times'])
        min_iter_time = min(timing_stats['per_iteration_times'])
        max_iter_time = max(timing_stats['per_iteration_times'])
    else:
        avg_iter_time = min_iter_time = max_iter_time = 0
    
    batch_summary = {
        'processing_date': start_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'configuration': {
            'input_dir': str(args.input_dir),
            'output_dir': str(args.output_dir),
            'gemini_model': args.gemini_model,
            'quality_threshold': args.quality_threshold,
            'max_iterations': args.max_iterations,
            'evaluation_mode': args.evaluation_mode,
        },
        'statistics': {
            'total_images': len(all_images),
            'processed': len(results_summary),
            'failed': len(failed_images),
            'skipped': len(skipped_images),
        },
        'timing_statistics': {
            'total_duration_seconds': duration.total_seconds(),
            'avg_image_time_seconds': avg_image_time,
            'min_image_time_seconds': min_image_time,
            'max_image_time_seconds': max_image_time,
            'avg_iteration_time_seconds': avg_iter_time,
            'min_iteration_time_seconds': min_iter_time,
            'max_iteration_time_seconds': max_iter_time,

            'avg_gemini_bbox_time': sum(timing_stats['gemini_bbox_times']) / len(timing_stats['gemini_bbox_times']) if timing_stats['gemini_bbox_times'] else 0,
            'avg_sam3_inference_time': sum(timing_stats['sam3_inference_times']) / len(timing_stats['sam3_inference_times']) if timing_stats['sam3_inference_times'] else 0,
            'avg_gemini_eval_time': sum(timing_stats['gemini_eval_times']) / len(timing_stats['gemini_eval_times']) if timing_stats['gemini_eval_times'] else 0,
            'avg_gemini_refine_time': sum(timing_stats['gemini_refine_times']) / len(timing_stats['gemini_refine_times']) if timing_stats['gemini_refine_times'] else 0,
            'total_sam3_time': sum(timing_stats['sam3_inference_times']),
            'total_gemini_eval_time': sum(timing_stats['gemini_eval_times']),
            'total_gemini_refine_time': sum(timing_stats['gemini_refine_times']),
                
            },
        'token_statistics': {
        'gemini_bbox_input_tokens': timing_stats['total_tokens']['gemini_bbox_input'],
        'gemini_bbox_output_tokens': timing_stats['total_tokens']['gemini_bbox_output'],
        'gemini_eval_input_tokens': timing_stats['total_tokens']['gemini_eval_input'],
        'gemini_eval_output_tokens': timing_stats['total_tokens']['gemini_eval_output'],
        'gemini_refine_input_tokens': timing_stats['total_tokens']['gemini_refine_input'],
        'gemini_refine_output_tokens': timing_stats['total_tokens']['gemini_refine_output'],
        'total_input_tokens': (timing_stats['total_tokens']['gemini_bbox_input'] + 
                              timing_stats['total_tokens']['gemini_eval_input'] + 
                              timing_stats['total_tokens']['gemini_refine_input']),
        'total_output_tokens': (timing_stats['total_tokens']['gemini_bbox_output'] + 
                               timing_stats['total_tokens']['gemini_eval_output'] + 
                               timing_stats['total_tokens']['gemini_refine_output']),
        'avg_input_tokens_per_image': (timing_stats['total_tokens']['gemini_bbox_input'] + 
                                       timing_stats['total_tokens']['gemini_eval_input'] + 
                                       timing_stats['total_tokens']['gemini_refine_input']) / max(len(results_summary), 1),
        'avg_output_tokens_per_image': (timing_stats['total_tokens']['gemini_bbox_output'] + 
                                        timing_stats['total_tokens']['gemini_eval_output'] + 
                                        timing_stats['total_tokens']['gemini_refine_output']) / max(len(results_summary), 1),
    },
        'results': results_summary,
        'failed': failed_images,
        'skipped': skipped_images
    }
    
    summary_path = output_dir / f"batch_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    print(f"\n✓ Batch summary saved to: {summary_path}")
    
    # ========================================================================
    # STEP 5: Final Report with Timing and Token Info
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    
    print(f"\nProcessing Duration: {duration}")
    print(f"\nImages:")
    print(f"  Total: {len(all_images)}")
    print(f"  Processed: {len(results_summary)}")
    print(f"  Failed: {len(failed_images)}")
    print(f"  Skipped: {len(skipped_images)}")
    
    # Timing statistics
    print(f"\nTiming Statistics:")
    print(f"  Average per image: {avg_image_time:.1f}s")
    print(f"  Min per image: {min_image_time:.1f}s")
    print(f"  Max per image: {max_image_time:.1f}s")
    print(f"  Average per iteration: {avg_iter_time:.1f}s")
    print(f"  Total iterations: {len(timing_stats['per_iteration_times'])}")
    
    # Token statistics
    print(f"\nToken Statistics (Breakdown):")
    total_in = (timing_stats['total_tokens']['gemini_bbox_input'] + 
                timing_stats['total_tokens']['gemini_eval_input'] + 
                timing_stats['total_tokens']['gemini_refine_input'])
    total_out = (timing_stats['total_tokens']['gemini_bbox_output'] + 
                 timing_stats['total_tokens']['gemini_eval_output'] + 
                 timing_stats['total_tokens']['gemini_refine_output'])
    
    print(f"  Gemini Bbox - Input: {timing_stats['total_tokens']['gemini_bbox_input']:,}, "
          f"Output: {timing_stats['total_tokens']['gemini_bbox_output']:,}")
    print(f"  Gemini Eval - Input: {timing_stats['total_tokens']['gemini_eval_input']:,}, "
          f"Output: {timing_stats['total_tokens']['gemini_eval_output']:,}")
    print(f"  Gemini Refine - Input: {timing_stats['total_tokens']['gemini_refine_input']:,}, "
          f"Output: {timing_stats['total_tokens']['gemini_refine_output']:,}")
    print(f"  TOTAL - Input: {total_in:,}, Output: {total_out:,}")
    
    if results_summary:
        avg_in = total_in / len(results_summary)
        avg_out = total_out / len(results_summary)
        print(f"  Average per image - Input: {avg_in:,.0f}, Output: {avg_out:,.0f}")
    
    if results_summary:
        print(f"\nQuality Statistics:")
        qualities = [r['best_quality'] for r in results_summary]
        print(f"  Mean: {sum(qualities)/len(qualities):.3f}")
        print(f"  Min:  {min(qualities):.3f}")
        print(f"  Max:  {max(qualities):.3f}")
        
        # Metrics statistics
        dices = [r['best_metrics'].get('dice', 0) for r in results_summary if r['best_metrics']]
        if dices:
            print(f"\nDice Statistics:")
            print(f"  Mean: {sum(dices)/len(dices):.3f}")
            print(f"  Min:  {min(dices):.3f}")
            print(f"  Max:  {max(dices):.3f}")
        
        print(f"\nIteration Statistics:")
        sam3_iters = [r['num_sam3_iterations'] for r in results_summary]
        print(f"  Mean SAM3 iterations: {sum(sam3_iters)/len(sam3_iters):.1f}")
        print(f"  Min:  {min(sam3_iters)}")
        print(f"  Max:  {max(sam3_iters)}")
        
        # Best and worst performing images
        sorted_by_quality = sorted(results_summary, key=lambda x: x['best_quality'], reverse=True)
        print(f"\nTop 5 Best Quality:")
        for i, r in enumerate(sorted_by_quality[:5], 1):
            dice = r['best_metrics'].get('dice', 0) if r['best_metrics'] else 0
            img_time = r['timing']['total_time']
            print(f"  {i}. {r['image_name']}: Quality={r['best_quality']:.3f}, Dice={dice:.3f}, Time={img_time:.0f}s")
        
        print(f"\nTop 5 Worst Quality:")
        for i, r in enumerate(sorted_by_quality[-5:], 1):
            dice = r['best_metrics'].get('dice', 0) if r['best_metrics'] else 0
            img_time = r['timing']['total_time']
            print(f"  {i}. {r['image_name']}: Quality={r['best_quality']:.3f}, Dice={dice:.3f}, Time={img_time:.0f}s")
    
    if failed_images:
        print(f"\nFailed Images:")
        for fail in failed_images:
            print(f"  • {fail['image_name']}: {fail['error'][:80]}")
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Batch summary: {summary_path}")
    print("="*80 + "\n")