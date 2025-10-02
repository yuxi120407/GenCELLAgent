import os
import argparse
from tqdm import tqdm

from util_micro_sam import run_automatic_instance_segmentation, get_paths
from util import get_pred_paths 
from micro_sam.evaluation.evaluation import run_evaluation


def run_micro_sam_segmentation(dataset_name, experiment_root, model_type="vit_l_lm", split="test"):
    """
    Run MicroSAM automatic instance segmentation on all images in the dataset
    
    Args:
        dataset_name: Name of the dataset to process
        experiment_root: Root directory for experiments
        model_type: Model type to use (default: vit_l_lm)
        split: Dataset split to use (default: test)
    
    Returns:
        gt_paths: List of ground truth paths
    """
    print("=" * 80)
    print(f"Starting MicroSAM segmentation for dataset: {dataset_name}")
    print(f"Experiment root: {experiment_root}")
    print(f"Model type: {model_type}")
    print("=" * 80)
    
    # Get image and ground truth paths
    image_paths, gt_paths = get_paths(dataset_name, split=split)
    
    print(f"\nFound {len(image_paths)} images to process\n")
    
    # Run segmentation on each image
    for path in tqdm(image_paths, desc="Running MicroSAM segmentation"):
        run_automatic_instance_segmentation(
            path, 
            dataset_name=None,  # Use None for default model
            experiment_root=experiment_root,
            model_type=model_type
        )
    
    print("\nSegmentation completed!")
    return gt_paths


def evaluate_results(gt_paths, experiment_root, results_filename="amg_default_2.csv"):
    """
    Evaluate the segmentation results
    
    Args:
        gt_paths: List of ground truth paths
        experiment_root: Root directory for experiments
        results_filename: Name of the results CSV file
    
    Returns:
        results: Evaluation results
    """
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80)
    
    prediction_folder = os.path.join(experiment_root, "predictions")
    pred_paths = get_pred_paths(prediction_folder)
    
    results_path = os.path.join(experiment_root, "results", results_filename)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    print(f"\nPrediction folder: {prediction_folder}")
    print(f"Found {len(pred_paths)} predictions")
    print(f"Found {len(gt_paths)} ground truth masks")
    
    results = run_evaluation(gt_paths, pred_paths, save_path=results_path)
    
    print("\nEvaluation Results:")
    print(results)
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run MicroSAM automatic instance segmentation")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined_dataset_new",
        help="Dataset name to process"
    )
    
    parser.add_argument(
        "--experiment_root",
        type=str,
        default="/home/idies/workspace/Temporary/xyu1/scratch/ours_model_selection_7_21/miro_sam_all",
        help="Root directory for experiments"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_l_lm",
        help="Model type to use for segmentation"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)"
    )
    
    parser.add_argument(
        "--results_filename",
        type=str,
        default="amg_default_2.csv",
        help="Name of the results CSV file"
    )
    
    args = parser.parse_args()
    
    # Run segmentation
    gt_paths = run_micro_sam_segmentation(
        dataset_name=args.dataset,
        experiment_root=args.experiment_root,
        model_type=args.model_type,
        split=args.split
    )
    
    # Run evaluation
    evaluate_results(
        gt_paths=gt_paths,
        experiment_root=args.experiment_root,
        results_filename=args.results_filename
    )


if __name__ == "__main__":
    main()