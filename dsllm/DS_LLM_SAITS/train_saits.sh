#!/bin/bash

# Usage: ./train_saits.sh <input_ds> <target_ds>
# Example: ./train_saits.sh 2000 100

# Path to SAITS main directory
SAITS_DIR="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/SAITS-main"
# Path to config file
CONFIG_PATH="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/DS_LLM_SAITS/Capture24_SAITS_best.ini"
# Path to make_saits_dataset.py
MAKE_SAITS_DATASET="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/DS_LLM_SAITS/make_saits_dataset.py"

# Parse arguments
INPUT_DS=${1:-1000}
TARGET_DS=${2:-100}

# Uncomment and edit the following line if you need to activate a conda or virtualenv environment
# source activate sensorllm-flash-attn

# Update dataset_name and model_name in the .ini config
sed -i "s/^dataset_name = .*/dataset_name = capture24_${TARGET_DS}DS_impute_from_${INPUT_DS}DS/" "$CONFIG_PATH"
sed -i "s/^model_name = .*/model_name = Capture24_SAITS_enhanced_from_${INPUT_DS}x/" "$CONFIG_PATH"

# Update [test] section paths in the .ini config
MODEL_DIR="\/project\/cc-20250120231604\/ssd\/users\/kwsu\/data\/trained_model\/saits_models\/Capture24_SAITS_enhanced_from_${INPUT_DS}x"
sed -i "s|^model_path = .*|model_path = ${MODEL_DIR}/models/REPLACE_WITH_MODEL_FILENAME|" "$CONFIG_PATH"
sed -i "s|^result_saving_path = .*|result_saving_path = ${MODEL_DIR}/test_results|" "$CONFIG_PATH"

# Step 1: Make SAITS dataset
python "$MAKE_SAITS_DATASET" --input_ds "$INPUT_DS" --target_ds "$TARGET_DS"

# Step 2: Run SAITS training
python "$SAITS_DIR/run_models.py" --config_path "$CONFIG_PATH" 