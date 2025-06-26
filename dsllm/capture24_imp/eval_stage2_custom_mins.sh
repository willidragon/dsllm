#!/bin/bash

# Usage: bash eval_stage2_custom_mins.sh [MODEL_DIR] [DATA_TAG] [OUTPUT_FILE_NAME]
# If DATA_TAG is not provided, it will be inferred from MODEL_DIR.
# Example: bash eval_stage2_custom_mins.sh \
#   ../outputs/SensorLLM_train_stage2/capture24_stage2_30S_10DS_COMPLETE \
#   30S_10DS_COMPLETE \
#   eval_capture24.json

# Set defaults to match the new model directory and data tag
MODEL_DIR=${1:-"../outputs/SensorLLM_train_stage2/capture24_stage2_300seconds_100DS_20250509_231419/checkpoint-900"}

# If DATA_TAG is provided, use it; otherwise, extract from MODEL_DIR
if [ -n "$2" ]; then
  DATA_TAG="$2"
else
  # Traverse up to find the directory starting with 'capture24_stage2_'
  SEARCH_DIR="$MODEL_DIR"
  while [ ! -z "$SEARCH_DIR" ]; do
    BASENAME=$(basename "$SEARCH_DIR")
    if [[ "$BASENAME" == capture24_stage2_* ]]; then
      DATA_TAG=${BASENAME#capture24_stage2_}
      # Remove trailing _complete, _COMPLETE, or _YYYYMMDD_HHMMSS if present
      DATA_TAG=$(echo "$DATA_TAG" | sed -E 's/(_complete|_COMPLETE|_[0-9]{8}_[0-9]{6})$//')
      break
    fi
    SEARCH_DIR=$(dirname "$SEARCH_DIR")
    # Stop if we reach root
    if [[ "$SEARCH_DIR" == "/" || "$SEARCH_DIR" == "." ]]; then
      echo "Error: Could not find a parent directory starting with 'capture24_stage2_' in path $MODEL_DIR" >&2
      exit 1
    fi
  done
fi

OUTPUT_FILE_NAME=${3:-"eval_capture24_${DATA_TAG}.json"}

# Determine data directory (stage_2 or stage_2_upsampled)
if [[ "$DATA_TAG" == *upsampled* ]]; then
  DATA_DIR="../data/stage_2_upsampled"
else
  DATA_DIR="../data/stage_2"
fi

DATA_PATH="$DATA_DIR/${DATA_TAG}/test/capture24_test_data_stage2_${DATA_TAG}.pkl"
QA_PATH="$DATA_DIR/${DATA_TAG}/test/capture24_test_qa_stage2_cls.json"
PT_ENCODER_BACKBONE_CKPT="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/downloaded_models/chronos/chronos-t5-large"  # Updated to correct path

# Clear the evaluation folder before running evaluation
# echo "Clearing evaluation folder: $MODEL_DIR/evaluation"
# rm -rf "$MODEL_DIR/evaluation"/*

export PYTHONPATH=../../

# Print out the resolved variables for confirmation

echo "MODEL_DIR: $MODEL_DIR"
echo "DATA_TAG: $DATA_TAG"
echo "DATA_PATH: $DATA_PATH"
echo "QA_PATH: $QA_PATH"
echo "OUTPUT_FILE_NAME: $OUTPUT_FILE_NAME"

data_path_msg="Using data from: $DATA_PATH"
echo $data_path_msg

python ../eval/eval_capture24_stage2.py \
  --model_name_or_path "$MODEL_DIR" \
  --pt_encoder_backbone_ckpt "$PT_ENCODER_BACKBONE_CKPT" \
  --data_path "$DATA_PATH" \
  --qa_path "$QA_PATH" \
  --output_file_name "$OUTPUT_FILE_NAME" \
  --tokenize_method "MeanScaleUniformBins" \
  --torch_dtype "bfloat16" \
  --batch_size 8 \
  --num_workers 2 \
  --preprocess_type "smry+trend+corr+Q" 