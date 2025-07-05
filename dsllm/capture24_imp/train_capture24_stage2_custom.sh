#!/bin/bash

# Usage: bash train_capture24_stage2_custom.sh [FINETUNE_NAME]
# If FINETUNE_NAME is not provided, a timestamped name will be used.

# Load OUTPUT_TAG from config_stage2.yaml (based on window size and downsample factor)
CONFIG_PATH="$(dirname "$0")/config_stage2.yaml"
WINDOW_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['window_size_seconds'])")
DOWNSAMPLE_FACTOR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['downsample_factor'])")
OUTPUT_TAG="${WINDOW_SIZE}seconds_${DOWNSAMPLE_FACTOR}DS"

# Get timestamp for unique folder names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Allow user to override FINETUNE_NAME as the first argument
FINETUNE_NAME=${1:-capture24_stage2_${OUTPUT_TAG}_${TIMESTAMP}}

# Set paths
DATA_ROOT="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_upsampled_saits/300seconds_100DS_upsampled_from_1000DS"
MODEL_ROOT="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/downloaded_models/llama_3_2_1b_instruct_cache"
CHRONOS_PATH="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/downloaded_models/chronos/chronos-t5-large"
OUTPUT_BASE="/project/cc-20250120231604/ssd/users/kwsu/data/trained_model"

# Set up Python environment
export PYTHONPATH="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm:${PYTHONPATH}"

# Create logs directory if it doesn't exist
mkdir -p "${OUTPUT_BASE}/SensorLLM_train_stage2/${FINETUNE_NAME}/logs"

# Set up logging with timestamps
LOG_FILE="${OUTPUT_BASE}/SensorLLM_train_stage2/${FINETUNE_NAME}/logs/training_${TIMESTAMP}.log"
ERROR_LOG="${OUTPUT_BASE}/SensorLLM_train_stage2/${FINETUNE_NAME}/logs/error_${TIMESTAMP}.log"

# Ensure unbuffered output for Python
export PYTHONUNBUFFERED=1

# Set up logging with tee and proper line buffering
exec 1> >(stdbuf -oL tee -a "${LOG_FILE}")
exec 2> >(stdbuf -oL tee -a "${ERROR_LOG}" >&2)

# Generate random port for distributed training (49152-65535 range)
master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
export PYTHONPATH=/project/cc-20250120231604/ssd/users/kwsu/research/dsllm:$PYTHONPATH
# Set CUDA_VISIBLE_DEVICES to ensure both GPUs are used (0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

# Print system information
echo "=== System Information ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "========================="

# Print training configuration
echo "=== Training Configuration ==="
echo "FINETUNE_NAME: ${FINETUNE_NAME}"
echo "Master Port: ${master_port}"
echo "Log File: ${LOG_FILE}"
echo "========================="

# Resume functionality: check for latest checkpoint in output dir
OUTPUT_DIR="${OUTPUT_BASE}/SensorLLM_train_stage2/${FINETUNE_NAME}"
LATEST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
RESUME_ARG=""
if [ -n "$LATEST_CKPT" ]; then
  echo "Resuming training from checkpoint: $LATEST_CKPT"
  RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
else
  echo "No checkpoint found. Starting training from scratch."
fi

torchrun --nproc_per_node=2 --master_port=$master_port ../train/train_mem.py   \
--model_name_or_path "${MODEL_ROOT}" \
--pt_encoder_backbone_ckpt "${CHRONOS_PATH}"   \
--model_type "SequenceClassification" \
--num_labels 10  \
--use_weighted_loss True  \
--tokenize_method 'StanNormalizeUniformBins'    \
--dataset "capture24" \
--data_path "${DATA_ROOT}/train/capture24_train_data_stage2_300seconds_100DS_upsampled.pkl"    \
--qa_path "${DATA_ROOT}/train/capture24_train_qa_stage2_cls.json"     \
--eval_data_path "${DATA_ROOT}/val/capture24_val_data_stage2_300seconds_100DS_upsampled.pkl"   \
--eval_qa_path "${DATA_ROOT}/val/capture24_val_qa_stage2_cls.json"    \
--preprocess_type "smry+trend+corr+Q" \
--output_dir "${OUTPUT_DIR}"    \
--model_max_length 4096    \
--num_train_epochs 8    \
--per_device_train_batch_size 4    \
--gradient_accumulation_steps 8    \
--per_device_eval_batch_size 4    \
--eval_strategy 'steps'    \
--save_strategy 'steps'    \
--do_train True    \
--do_eval True    \
--save_steps 200    \
--eval_steps 200    \
--save_total_limit 1    \
--load_best_model_at_end True    \
--metric_for_best_model f1_macro    \
--greater_is_better True    \
--learning_rate 2e-3    \
--weight_decay 0.0    \
--warmup_ratio 0.03    \
--lr_scheduler_type cosine    \
--logging_steps 1    \
--bf16 True      \
--fix_llm True  \
--fix_cls_head False  \
--fix_ts_encoder True    \
--gradient_checkpointing True    \
--only_stage2 True	\
--stage_2 True  \
--shuffle True \
--report_to tensorboard \
--use_cache False \
--logging_dir "${OUTPUT_DIR}/logs" \
--logging_first_step True \
--logging_strategy "steps" \
--logging_nan_inf_filter False \
--ddp_find_unused_parameters False \
$RESUME_ARG

# Print completion message
echo "=== Training Complete ==="
echo "Logs saved to: ${LOG_FILE}"
echo "Errors saved to: ${ERROR_LOG}"
echo "========================="

