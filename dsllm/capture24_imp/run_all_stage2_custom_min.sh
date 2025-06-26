#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Clear all files in the output folder before running, using config_stage2.yaml
CONFIG_PATH="$(dirname "$0")/config_stage2.yaml"
WINDOW_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['window_size_seconds'])")
DOWNSAMPLE_FACTOR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['downsample_factor'])")
OUTPUT_TAG="${WINDOW_SIZE}seconds_${DOWNSAMPLE_FACTOR}DS"
SOURCE_DIR="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare_buffer/${OUTPUT_TAG}"
OUTPUT_DIR="/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/${OUTPUT_TAG}"
# echo "Clearing all files in $OUTPUT_DIR..."
# rm -rf "$OUTPUT_DIR"/*

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "Updating ts_backbone.yaml sample_rate to match config..."
python update_ts_backbone_sample_rate.py

# ---- Stage 1 ----
TRAIN_DATA="$SOURCE_DIR/train/capture24_train_data_stage2_${OUTPUT_TAG}.pkl"
TRAIN_LABELS="$SOURCE_DIR/train/capture24_train_labels_stage2_${OUTPUT_TAG}.pkl"
TRAIN_SETTINGS="$SOURCE_DIR/train/settings.json"
TEST_DATA="$SOURCE_DIR/test/capture24_test_data_stage2_${OUTPUT_TAG}.pkl"
TEST_LABELS="$SOURCE_DIR/test/capture24_test_labels_stage2_${OUTPUT_TAG}.pkl"
TEST_SETTINGS="$SOURCE_DIR/test/settings.json"

# if [[ -f "$TRAIN_DATA" && -f "$TRAIN_LABELS" && -f "$TRAIN_SETTINGS" && -f "$TEST_DATA" && -f "$TEST_LABELS" && -f "$TEST_SETTINGS" ]]; then
#     echo "[1/4] Stage 2.1 outputs already exist, skipping process_capture24_stage2_1_custom_mins.py"
# else
# echo "[1/4] Running process_capture24_stage2_1_custom_mins.py..."
# python process_capture24_stage2_1_custom_mins.py
# fi

# echo "[2/4] Running process_capture24_stage2_2_custom_mins.py..."
# python process_capture24_stage2_2_custom_mins.py

# # ---- Stage 3 ----
# QA_TRAIN="$SOURCE_DIR/train/capture24_train_qa_stage2_cls.json"
# if [[ -f "$QA_TRAIN" ]]; then
#     echo "[3/4] Stage 2.3 output already exists, skipping process_capture24_stage2_3_custom_mins.py"
# else
# echo "[3/4] Running process_capture24_stage2_3_custom_mins.py..."
# python process_capture24_stage2_3_custom_mins.py
# fi

# # ---- Stage 4 ----
# QA_TEST="$SOURCE_DIR/test/capture24_test_qa_stage2_cls.json"
# if [[ -f "$QA_TEST" ]]; then
#     echo "[4/4] Stage 2.4 output already exists, skipping process_capture24_stage2_4_custom_mins.py"
# else
# echo "[4/4] Running process_capture24_stage2_4_custom_mins.py..."
# python process_capture24_stage2_4_custom_mins.py
# fi

# echo "All stage 2 custom mins scripts completed successfully!"

# echo "Starting training script..."
bash train_capture24_stage2_custom.sh 