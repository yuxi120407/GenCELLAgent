import os
import argparse
import pandas as pd
from tqdm import tqdm

from util_micro_sam import run_automatic_instance_segmentation, get_paths
from util_cellpose import run_cellpose_segmentation_single
from util import get_pred_paths 
from micro_sam.evaluation.evaluation import run_evaluation


def get_model_for_image(df, image_name):
    """Get the selected model for a given image from the CSV file"""
    row = df[df["image_name"] == image_name]
    if not row.empty:
        return row.iloc[0]["selected_model"]
    else:
        return f"Image '{image_name}' not found in the CSV."


def run_model_selection(dataset_name, model_selection_csv, experiment_root, use_default=True):
    """
    Run model selection based on CSV file
    
    Args:
        dataset_name: Name of the dataset to process
        model_selection_csv: Path to the CSV file containing model selections
        experiment_root: Root directory for experiments
        use_default: If True, use default dataset (None) for micro models
    """
    # Get all the data paths from combined test set
    image_paths, _ = get_paths(dataset_name=dataset_name, split="test")
    
    # Get the model selection csv file
    df = pd.read_csv(model_selection_csv)
    
    # Process each image
    for path in tqdm(image_paths, desc="Processing images"):
        fname = os.path.basename(path)
        print(f"\nProcessing: {fname}")
        
        model_name = get_model_for_image(df, fname)
        print(f"Selected model: {model_name}")
        
        parts = model_name.split("_")
        
        if parts[0] == "micro":
            if not use_default:
                if parts[-1] == "livecell":
                    run_automatic_instance_segmentation(
                        path, 
                        dataset_name="livecell", 
                        experiment_root=experiment_root,
                        model_type="vit_l_lm"
                    )
                elif parts[-1] == "TissueNet":
                    run_automatic_instance_segmentation(
                        path, 
                        dataset_name="tissuenet", 
                        experiment_root=experiment_root,
                        model_type="vit_l_lm"
                    )
                elif parts[-1] == "plantseg":
                    run_automatic_instance_segmentation(
                        path, 
                        dataset_name="plantseg", 
                        experiment_root=experiment_root,
                        model_type="vit_l_lm"
                    )
            else:
                run_automatic_instance_segmentation(
                    path, 
                    dataset_name=None, 
                    experiment_root=experiment_root,
                    model_type="vit_l_lm"
                )
        
        elif parts[0] == "cellpose":
            run_cellpose_segmentation_single(
                path, 
                experiment_root=experiment_root,
                model_type="cyto2"
            )


def evaluate_results(dataset_name, experiment_root, results_filename="amg_default_test_1.csv"):
    """
    Evaluate the segmentation results
    
    Args:
        dataset_name: Name of the dataset
        experiment_root: Root directory for experiments
        results_filename: Name of the results CSV file
    """
    prediction_folder = os.path.join(experiment_root, "predictions")
    
    _, gt_paths = get_paths(dataset_name, split="test")
    pred_paths = get_pred_paths(prediction_folder)
    
    results_path = os.path.join(experiment_root, "results", results_filename)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    results = run_evaluation(gt_paths, pred_paths, save_path=results_path)
    print("\nEvaluation Results:")
    print(results)
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run model selection based on CSV file")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined_dataset_new",
        help="Dataset name to process"
    )
    
    parser.add_argument(
        "--model_selection_csv",
        type=str,
        default="model_selection.csv",
        help="Path to the CSV file containing model selections"
    )
    
    parser.add_argument(
        "--experiment_root",
        type=str,
        default="/home/idies/workspace/Temporary/xyu1/scratch/ours_model_selection_7_21/Ours_all",
        help="Root directory for experiments"
    )
    
    parser.add_argument(
        "--use_default",
        action="store_true",
        default=True,
        help="If set, use default dataset (None) for micro models"
    )
    
    parser.add_argument(
        "--no_default",
        action="store_false",
        dest="use_default",
        help="If set, use specific datasets for micro models"
    )
    
    parser.add_argument(
        "--results_filename",
        type=str,
        default="amg_model_selection.csv",
        help="Name of the results CSV file"
    )
    
    parser.add_argument(
        "--skip_segmentation",
        action="store_true",
        help="Skip segmentation and only run evaluation"
    )
    
    args = parser.parse_args()
    
    # Run segmentation
    if not args.skip_segmentation:
        print("=" * 80)
        print("Starting model selection segmentation...")
        print("=" * 80)
        run_model_selection(
            dataset_name=args.dataset,
            model_selection_csv=args.model_selection_csv,
            experiment_root=args.experiment_root,
            use_default=args.use_default
        )
    
    # Run evaluation
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80)
    evaluate_results(
        dataset_name=args.dataset,
        experiment_root=args.experiment_root,
        results_filename=args.results_filename
    )


if __name__ == "__main__":
    main()