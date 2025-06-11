#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_stage2_custom_mins.sh"
MODEL_ROOT="$SCRIPT_DIR/../outputs/SensorLLM_train_stage2"

shopt -s nullglob

model_dirs=("$MODEL_ROOT"/*/)
if [ ${#model_dirs[@]} -eq 0 ]; then
    echo "No model directories found in $MODEL_ROOT"
    exit 1
fi

for model_dir in "${model_dirs[@]}"; do
    echo "Checking $model_dir"
    # Check if evaluation folder exists in the model_dir
    if [ ! -d "${model_dir}evaluation" ]; then
        echo "Running eval on $model_dir"
        bash "$EVAL_SCRIPT" "$model_dir"
    else
        echo "Skipping $model_dir (evaluation exists)"
    fi

    # Now check for checkpoints
    for ckpt_dir in "$model_dir"checkpoint-*/; do
        if [ -d "$ckpt_dir" ]; then
            if [ ! -d "${ckpt_dir}evaluation" ]; then
                echo "Running eval on $ckpt_dir"
                bash "$EVAL_SCRIPT" "$ckpt_dir"
            else
                echo "Skipping $ckpt_dir (evaluation exists)"
            fi
        fi
    done
done 