#!/bin/bash
# GenCELLAgent Batch Segmentation Examples
# Usage: bash batch_example.sh

cd "$(dirname "$0")"

echo "=========================================="
echo "GenCELLAgent Batch Segmentation Examples"
echo "=========================================="

# --- Cell Mode: Auto Tool Selection ---
echo ""
echo "[1/5] Cell Mode - Cellpose (auto-selected for LiveCELL)"
echo "=========================================="
python batch_segment_new.py --image examples/cells/A172_Phase_A7_2_01d00h00m_4.tif --prompt "segment all cells"

echo ""
echo "[2/5] Cell Mode - CellSAM (auto-selected for Yeast)"
echo "=========================================="
python batch_segment_new.py --image examples/yeast/im051.tif --prompt "segment all cells"

echo ""
echo "[3/5] Cell Mode - micro-SAM (auto-selected for PlantSeg)"
echo "=========================================="
python batch_segment_new.py --image examples/plantseg/plantseg_root_val_Movie1_t00004_crop_gt_00013.tif --prompt "segment all cells"

# --- Organelle Mode: Gemini + SAM3 with Feedback ---
echo ""
echo "[4/5] Organelle Mode - Golgi with iterative feedback"
echo "=========================================="
python batch_segment_new.py --image examples/golgi/images/sample_0000.png --prompt "segment the golgi" --max_iterations 3

# --- Reference Mode: SegGPT One-Shot ---
echo ""
echo "[5/5] Reference Mode - SegGPT one-shot segmentation"
echo "=========================================="
python batch_segment_new.py --image examples/er/images/image_101.png --prompt "segment using reference" --reference_image examples/er/images/image_1.png --reference_mask examples/er/labels/label_1.png

echo ""
echo "=========================================="
echo "All examples complete! Results in output/batch_results/"
echo "=========================================="
