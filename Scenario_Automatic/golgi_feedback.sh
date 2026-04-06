#!/bin/bash

# Configuration
IMAGE_DIR="/home/idies/workspace/Storage/xyu1/persistent/Langchain/ours_test/test_images/golgi/images" 
OUTPUT_DIR="./sam3_results_all_golgi_seggpt"
CHECKPOINT="/home/idies/workspace/Storage/xyu1/persistent/GenCELLAgent/src/sam3/checkpoints/sam3/sam3.pt"
GEMINI_MODEL="gemini-3-flash-preview"  #gemini-3-pro-preview
ANALYSIS_DIR="./analysis_all_golgi_seggpt"
QUALITY_THRESHOLD=0.85
IOU_THRESHOLD=0.00
MAX_ITERATIONS=5
EVALUATION_MODE="boxes_only"
ORGANELLE="golgi"

python -u test_all_feedback.py \
    --input_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --sam3_checkpoint "$CHECKPOINT" \
    --gemini_model "$GEMINI_MODEL" \
    --quality_threshold $QUALITY_THRESHOLD \
    --max_iterations $MAX_ITERATIONS \
    --evaluation_mode $EVALUATION_MODE \
    --skip_existing \
    --organelle "$ORGANELLE"


python -u analyze_feedback.py \
    --iou-threshold "$IOU_THRESHOLD" \
    --input "$OUTPUT_DIR" \
    --output "$ANALYSIS_DIR"