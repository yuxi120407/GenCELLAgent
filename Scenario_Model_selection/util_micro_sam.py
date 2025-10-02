import os
import argparse
from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_amg, run_instance_segmentation_with_decoder
from util import get_pred_paths
from glob import glob
import imageio.v3 as imageio

import h5py
import matplotlib.pyplot as plt
from skimage.measure import label as connected_components

import pandas as pd
from typing import Dict, Any

from torch_em.util.util import get_random_colors
from torch_em.data.datasets.light_microscopy.covid_if import get_covid_if_data

from micro_sam import util
from micro_sam.evaluation.model_comparison import _enhance_image
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    AutomaticMaskGenerator,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)
ROOT = "/home/idies/workspace/Temporary/xyu1/scratch/cellpose_subset"
Results_ROOT = "/home/idies/workspace/Temporary/xyu1/scratch/micro_sam_subset/experiment_dir/"
EXPERIMENT_ROOT = "/home/idies/workspace/Temporary/xyu1/scratch/ours_model_selection_test/ours_all_LTPL_OOD"

def get_paths(dataset_name, split="test"):
    file_search_specs = "*"
    raw_dir = os.path.join(ROOT, dataset_name, split, "images", file_search_specs)
    labels_dir = os.path.join(ROOT, dataset_name, split, "masks", file_search_specs)
    
    
    return sorted(glob(os.path.join(raw_dir))), sorted(glob(os.path.join(labels_dir)))



def prepare_generate_kwargs_from_csv(csv_path: str, row_index: int = 0) -> Dict[str, Any]:
    """
    Reads parameter overrides from a CSV file and returns a dictionary of kwargs
    to pass into the `generate()` function.

    Parameters:
        csv_path (str): Path to the CSV file containing parameter values.
        row_index (int): Which row to use (default is 0 — first data row).

    Returns:
        Dict[str, Any]: Dictionary of keyword arguments for generate().
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Define mapping from CSV columns to function parameters
    param_map = {
        'center_distance_threshold': 'center_distance_threshold',
        'boundary_distance_threshold': 'boundary_distance_threshold',
        'distance_smoothing': 'distance_smoothing',
        'min_size': 'min_size',
        # Add more mappings here if your CSV includes them
    }

    # Build kwargs dict
    overrides = {}
    for csv_col, param_name in param_map.items():
        if csv_col in df.columns:
            val = df.iloc[row_index][csv_col]
            if pd.notna(val):  # skip NaN values
                overrides[param_name] = val

    return overrides


def run_ais_inference(dataset_name, model_type, checkpoint, experiment_folder):
    """
    Run automatic mask generation (AIS) inference.

    Args:
        dataset_name (str): Name of the dataset.
        model_type (str): Model type (e.g., vit_l).
        checkpoint (str): Path to the model checkpoint.
        experiment_folder (str): Folder to save predictions.

    Returns:
        str: Path to the folder containing predictions.
    """
    val_image_paths, val_gt_paths = get_paths(dataset_name, split="val")
    test_image_paths, _ = get_paths(dataset_name, split="test")

    prediction_folder = run_instance_segmentation_with_decoder(
        checkpoint=checkpoint,
        model_type=model_type,
        experiment_folder=experiment_folder,
        val_image_paths=val_image_paths,
        val_gt_paths=val_gt_paths,
        test_image_paths=test_image_paths,
    )
    return prediction_folder

# def run_automatic_instance_segmentation(image_path, dataset_name, model_type="vit_l_lm"):

#     """Automatic Instance Segmentation by training an additional instance decoder in SAM.

#     NOTE: It is supported only for `µsam` models.

#     Args:
#         image: The input image.
#         model_type: The choice of the `µsam` model.

#     Returns:
#         The instance segmentation.
#     """
#     # Step 1: Initialize the model attributes using the pretrained µsam model weights.
#     #   - the 'predictor' object for generating predictions using the Segment Anything model.
#     #   - the 'decoder' backbone (for AIS).
    
#     prediction_folder = os.path.join(EXPERIMENT_ROOT, "predictions")
#     os.makedirs(prediction_folder, exist_ok=True)
#     fname = os.path.basename(image_path)
#     out_path = os.path.join(prediction_folder, fname)
    
#     image = imageio.imread(image_path)
#     predictor, decoder = get_predictor_and_decoder(
#         model_type=model_type,  # choice of the Segment Anything model
#         checkpoint_path=None,  # overwrite to pass our own finetuned model /home/idies/workspace/Temporary/xyu1/scratch/micro_sam_model/LM/vit_l.pt"
#     )

#     # Step 2: Computation of the image embeddings from the vision transformer-based image encoder.
#     if dataset_name == "tissuenet":
#         image_embeddings = util.precompute_image_embeddings(
#         predictor=predictor,  # the predictor object responsible for generating predictions
#         input_=image,  # the input image
#         ndim=2,  # number of input dimensions
#     )
        
#     else:
#         image_embeddings = util.precompute_image_embeddings(
#             predictor=predictor,  # the predictor object responsible for generating predictions
#             input_=image,  # the input image
#             ndim=2,  # number of input dimensions
#         )

#     # Step 3: Combining the decoder with the Segment Anything backbone for automatic instance segmentation.
#     ais = InstanceSegmentationWithDecoder(predictor, decoder)

#     # Step 4: Initializing the precomputed image embeddings to perform faster automatic instance segmentation.
#     if dataset_name == "livecell":
#         csv_file = Results_ROOT + "/livecell/vit_l_lm/results/grid_search_params_instance_segmentation_with_decoder.csv"
#         kwargs = prepare_generate_kwargs_from_csv(csv_file, row_index=0)
#         prediction = ais.generate(**kwargs)
#     elif dataset_name == "plantseg":
#         csv_file = Results_ROOT + "/plantseg/root/vit_l_lm/results/grid_search_params_instance_segmentation_with_decoder.csv"
#         kwargs = prepare_generate_kwargs_from_csv(csv_file, row_index=0)
#         prediction = ais.generate(**kwargs)
#     elif dataset_name == "tissuenet":
#         csv_file = Results_ROOT + "/tissuenet/slices/multi_chan/vit_l_lm/results/grid_search_params_instance_segmentation_with_decoder.csv"
#         kwargs = prepare_generate_kwargs_from_csv(csv_file, row_index=0)
#         prediction = ais.generate(**kwargs)
#     elif dataset_name == None:
#         prediction = ais.generate()


#     ais.initialize(
#         image=image,  # the input image
#         image_embeddings=image_embeddings,  # precomputed image embeddings
#     )

#     # Step 5: Getting automatic instance segmentations for the given image and applying the relevant post-processing steps.

#     prediction = mask_data_to_segmentation(prediction, with_background=True)
#     imageio.imwrite(out_path, prediction, compression=5)
#     return prediction_folder


def run_automatic_instance_segmentation(
    image_path: str,
    dataset_name: str,
    experiment_root: str,
    model_type: str = "vit_l_lm",
    results_root: str = "/home/idies/workspace/Temporary/xyu1/scratch/micro_sam_subset/experiment_dir/"
) -> str:
    """
    Perform automatic instance segmentation using a µSAM-based model with an additional decoder.

    Args:
        image_path (str): Path to the input image file.
        dataset_name (str): Name of the dataset (e.g., "livecell", "plantseg", "tissuenet").
        model_type (str, optional): Type of µSAM model to use. Defaults to "vit_l_lm".
        results_root (str, optional): Root directory where configuration CSVs are stored.
        experiment_root (str, optional): Root directory where predictions will be saved.

    Returns:
        str: Path to the directory containing the prediction output.
    """
    import os
    import imageio

    # Step 1: Set up prediction output path
    prediction_folder = os.path.join(experiment_root, "predictions")
    os.makedirs(prediction_folder, exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join(prediction_folder, filename)

    # Step 2: Load input image and initialize predictor and decoder
    image = imageio.imread(image_path)
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,
        checkpoint_path=None
    )

    # Step 3: Compute image embeddings
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor,
        input_=image,
        ndim=2
    )

    # Step 4: Initialize automatic instance segmentation module
    ais = InstanceSegmentationWithDecoder(predictor, decoder)
    ais.initialize(
        image=image,
        image_embeddings=image_embeddings
    )

    # Step 5: Load dataset-specific configuration and generate prediction
    if dataset_name == "livecell":
        config_path = os.path.join(results_root, "livecell/vit_l_lm/results/grid_search_params_instance_segmentation_with_decoder.csv")
    elif dataset_name == "plantseg":
        config_path = os.path.join(results_root, "plantseg/root/vit_l_lm/results/grid_search_params_instance_segmentation_with_decoder.csv")
    elif dataset_name == "tissuenet":
        config_path = os.path.join(results_root, "tissuenet/slices/multi_chan/vit_l_lm/results/grid_search_params_instance_segmentation_with_decoder.csv")
    elif dataset_name is None:
        config_path = None
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    if config_path:
        kwargs = prepare_generate_kwargs_from_csv(config_path, row_index=0)
        prediction = ais.generate(**kwargs)
    else:
        prediction = ais.generate()

    # Step 6: Post-process and save segmentation mask
    prediction_mask = mask_data_to_segmentation(prediction, with_background=True)
    imageio.imwrite(output_path, prediction_mask, compression=5)

    return prediction_folder



def eval_amg(dataset_name, prediction_folder, experiment_folder):
    """
    Evaluate AMG results against ground truth.

    Args:
        dataset_name (str): Dataset name.
        prediction_folder (str): Path to predicted masks.
        experiment_folder (str): Folder to save evaluation results.
    """
    print(f"Evaluating predictions in: {prediction_folder}")
    _, gt_paths = get_paths(dataset_name, split="test")
    pred_paths = get_pred_paths(prediction_folder)

    results_path = os.path.join(experiment_folder, "results", "amg.csv")
    results = run_evaluation(gt_paths, pred_paths, save_path=results_path)
    print("Evaluation Results:\n", results)