#test cellpose
import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from micro_sam.evaluation.evaluation import run_evaluation


# from util import get_paths   # for hlrn
from util import get_pred_paths 

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from style_utils import load_vgg_model, load_and_preprocess_images, GramMatrix, weighted_style_correlation
from glob import glob

ROOT = "/home/idies/workspace/Temporary/xyu1/scratch/cellpose_subset"
EXPERIMENT_ROOT = "/home/idies/workspace/Temporary/xyu1/scratch/cellpose_test_subset_single"

def get_paths(dataset_name, split="test"):
    file_search_specs = "*"
    raw_dir = os.path.join(ROOT, dataset_name, split, "images", file_search_specs)
    labels_dir = os.path.join(ROOT, dataset_name, split, "masks", file_search_specs)
    
    
    return sorted(glob(os.path.join(raw_dir))), sorted(glob(os.path.join(labels_dir)))
    
    

def load_cellpose_model(model_type):
    from cellpose import models
    device, gpu = models.assign_device(True, True)

    if model_type in ["cyto", "cyto2", "cyto3", "nuclei"]:
        model = models.CellposeModel(gpu=gpu, model_type=model_type, device=device)
    elif model_type in ["livecell", "tissuenet", "livecell_cp3"]:
        model = models.CellposeModel(gpu=gpu, model_type=model_type, device=device)
    else:
        raise ValueError(model_type)

    return model


def run_cellpose_segmentation(dataset, model_type):
    prediction_folder = os.path.join(EXPERIMENT_ROOT, dataset, model_type, "predictions")
    os.makedirs(prediction_folder, exist_ok=True)

    image_paths, _ = get_paths(dataset, split="test")
    model = load_cellpose_model(model_type)
    

    time_per_image = []
    for path in tqdm(image_paths, desc=f"Segmenting {dataset} with cellpose ({model_type})"):
        fname = os.path.basename(path)
        prefix = fname.split("_")[0]
        out_path = os.path.join(prediction_folder, fname)
        if os.path.exists(out_path):
            continue
        image = imageio.imread(path)
        channels = [0, 0]  # it's assumed to use one-channel, unless overwritten by logic below

        if image.ndim == 3:
            assert image.shape[-1] == 3
            if prefix == "TissueNet":
                channels = [2, 3]
            else:
                image = image.mean(axis=-1)

        start_time = time.time()

        seg = model.eval(image, diameter=None, flow_threshold=None, channels=channels)[0]

        end_time = time.time()
        time_per_image.append(end_time - start_time)

        assert seg.shape == image.shape[:2]
        imageio.imwrite(out_path, seg, compression=5)

    n_images = len(image_paths)
    print(f"The mean time over {n_images} images is:", np.mean(time_per_image), f"({np.std(time_per_image)})")

    return prediction_folder

def run_cellpose_segmentation_single(path, dataset, model_type):
    prediction_folder = os.path.join(EXPERIMENT_ROOT, dataset, model_type, "predictions")
    os.makedirs(prediction_folder, exist_ok=True)

    model = load_cellpose_model(model_type)
    

    time_per_image = []
    fname = os.path.basename(path)
    prefix = fname.split("_")[0]
    out_path = os.path.join(prediction_folder, fname)

    image = imageio.imread(path)
    channels = [0, 0]  # it's assumed to use one-channel, unless overwritten by logic below

    if image.ndim == 3:
        assert image.shape[-1] == 3
        if prefix == "TissueNet":
            channels = [2, 3]
        else:
            image = image.mean(axis=-1)

    start_time = time.time()

    seg = model.eval(image, diameter=None, flow_threshold=None, channels=channels)[0]

    end_time = time.time()
    time_per_image = end_time - start_time

    assert seg.shape == image.shape[:2]
    imageio.imwrite(out_path, seg, compression=5)

    print(f"The mean time over single images is:", time_per_image)

    return prediction_folder


def evaluate_dataset(prediction_folder, dataset, model_type):
    _, gt_paths = get_paths(dataset, split="test")
    pred_paths = get_pred_paths(prediction_folder)
    assert len(gt_paths) == len(pred_paths), f"{len(gt_paths)}, {len(pred_paths)}"
    result_path = os.path.join(EXPERIMENT_ROOT, dataset, "results", f"cellpose-{model_type}.csv")
    if os.path.exists(result_path):
        print(pd.read_csv(result_path))
        print(f"Results are already saved at {result_path}")
        return result_path

    results = run_evaluation(gt_paths, pred_paths, result_path)
    print(results)
    print(f"Results are saved at {result_path}")


def run_cellpose_baseline(datasets, model_types):
    if isinstance(datasets, str):
        datasets = [datasets]

    if isinstance(model_types, str):
        model_types = [model_types]

    for dataset in datasets:
        for model_type in model_types:
            prediction_folder = run_cellpose_segmentation(dataset, model_type)
            print(prediction_folder)
            evaluate_dataset(prediction_folder, dataset, model_type)


def main(datasets, model_types):
    """Main function to run cellpose baseline evaluation
    
    Args:
        datasets: str or list of str - dataset name(s) to process
        model_types: str or list of str - cellpose model type(s) to use
    """
    run_cellpose_baseline(datasets=datasets, model_types=model_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cellpose baseline evaluation")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined_dataset_new",
        help="Dataset name to process."
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="cyto2",
        help="Cellpose model type to use. Options: cyto, cyto2, cyto3, nuclei, livecell, tissuenet, livecell_cp3"
    )
    
    args = parser.parse_args()
    
    main(datasets=args.dataset, model_types=args.model_type)