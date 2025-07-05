#!/bin/bash

# Set the exact model directory and data tag
MODEL_DIR="/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/SensorLLM_train_stage2/capture24_stage2_300seconds_2000DS_with_val"
DATA_TAG="300seconds_2000DS"
OUTPUT_FILE_NAME="eval_capture24_${DATA_TAG}.json"

# Use only the upsampled data directory
DATA_DIR="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare_buffer/300seconds_2000DS/test"
DATA_PATH="$DATA_DIR/capture24_test_data_stage2_300seconds_2000DS.pkl"
QA_PATH="$DATA_DIR/capture24_test_qa_stage2_cls.json"
PT_ENCODER_BACKBONE_CKPT="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/downloaded_models/chronos/chronos-t5-large"

# Debug prints for paths
echo "DATA_DIR: $DATA_DIR"
echo "DATA_PATH: $DATA_PATH"
echo "QA_PATH: $QA_PATH"

export PYTHONPATH=../../

# Print out the resolved variables for confirmation
echo "MODEL_DIR: $MODEL_DIR"
echo "DATA_TAG: $DATA_TAG"
echo "OUTPUT_FILE_NAME: $OUTPUT_FILE_NAME"

data_path_msg="Using upsampled data from: $DATA_PATH"
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