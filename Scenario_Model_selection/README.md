## Scenario_1 Model Selction 


### Installation

1. Set up a virtual environment:
   ```
    conda env create -f environment.yml -n micro-sam
   ```

### Usage

1. Run cellpose:
   ```
   python cellpose_test.py --dataset combined_dataset_new --model_type cyto2
   ```

2. Run Micro-SAM:
   ```
python run_micro_sam.py \
    --dataset combined_dataset_new \
    --experiment_root /home/idies/workspace/Temporary/xyu1/scratch/ours_model_selection_7_21/miro_sam_all
   ```

3. Run Jupyter notebook (model_selection,ipynb)  get figure and model_selection.csv:

    
4. Run Model selection:

```
python run_model_selection.py \
    --dataset combined_dataset_new \
    --model_selection_csv model_selection.csv \
    --experiment_root /home/idies/workspace/Temporary/xyu1/scratch/ours_model_selection_7_21/Ours_all
    ```
    

