#!/bin/bash

# Define input variables
IMAGE_INDEX=53

# Define paths
SCRIPT_PATH="/home/idies/workspace/Storage/xyu1/persistent/GenCELLAgent/src/tools/sam_correction_tool_micro_sam_time.py"
IMAGE_PATH="/home/idies/workspace/Storage/xyu1/persistent/GenCELLAgent/Scenario_Human/golgi/images/image_${IMAGE_INDEX}.png"
MASK_PATH="/home/idies/workspace/Storage/xyu1/persistent/Scenario_Human/golgi/images/best_mask_step_5_iter5.png"
SAVE_PATH="/home/idies/workspace/Storage/xyu1/persistent/Scenario_Human/golgi/correct_img/image_${IMAGE_INDEX}"

# Run Streamlit app
streamlit run "$SCRIPT_PATH" \
  -- \
  --image_path "$IMAGE_PATH" \
  --mask_path "$MASK_PATH" \
  --save_path "$SAVE_PATH"