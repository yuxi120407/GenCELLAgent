# Cell Segmentation: Model Selection Pipeline

## Setup
```bash
conda env create -f environment.yml -n micro-sam
conda activate micro-sam
```

## Usage

### 1. Run Cellpose
```bash
python cellpose_test.py --dataset combined_dataset_new --model_type cyto2
```

### 2. Run Micro-SAM
```bash
python run_micro_sam.py \
    --dataset combined_dataset_new \
    --experiment_root /path/to/micro_sam_results
```

### 3. Generate Model Selection CSV
```bash
jupyter notebook model_selection.ipynb
```
**Output:** `model_selection.csv` with per-image model choices

### 4. Apply Model Selection
```bash
python run_model_selection.py \
    --dataset combined_dataset_new \
    --model_selection_csv model_selection.csv \
    --experiment_root /path/to/final_results
```
---

## Example: IMC Dataset (Damond_2019_Pancreas)

### Dataset Acquisition
Obtain the Damond_2019_Pancreas IMC dataset from:
- **Source:** [imcdatasets - Damond_2019_Pancreas](https://bodenmillergroup.github.io/imcdatasets/reference/Damond_2019_Pancreas.html)

### Processing Workflow

#### 1. IMC Data Processing
Run the preprocessing notebook:
```bash
jupyter notebook IMC_processing.ipynb
```
**Purpose:** Converts raw IMC data to format compatible with segmentation pipeline

#### 2. Model Selection for IMC
Run the IMC-specific model selection notebook:
```bash
jupyter notebook model_selection_IMC.ipynb
```
**Purpose:** Analyzes and selects optimal segmentation model for IMC pancreas images

### Notes
- IMC (Imaging Mass Cytometry) data requires specialized preprocessing
- The two notebooks handle the complete workflow from raw IMC to model selection`
    

