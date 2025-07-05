#!/bin/bash

# Path to SAITS main directory
SAITS_DIR="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/SAITS-main"
# Path to config file
CONFIG_PATH="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/DS_LLM_SAITS/Capture24_SAITS_best.ini"

# Uncomment and edit the following line if you need to activate a conda or virtualenv environment
# source activate sensorllm-flash-attn

# Run SAITS in test_mode for imputation
python "$SAITS_DIR/run_models.py" --config_path "$CONFIG_PATH" --test_mode 