#!/bin/bash

# Path to the imputations.h5 file (updated for enhanced experiment)
H5_PATH="/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/saits_models/Capture24_SAITS_enhanced_from_1000x/test_results/imputations.h5"
# Output base directory for extracted pkl files
OUT_BASE="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_upsampled_saits"
# Data tag and DS for naming (edit if needed)
DATA_TAG="300seconds"
TARGET_DS="100"
SPLITS=(train val test)

# Path to h5_to_pkl.py script
H5_TO_PKL="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/DS_LLM_SAITS/h5_to_pkl.py"

for SPLIT in "${SPLITS[@]}"; do
    # Try to find the label file for this split
    LABEL_SRC="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare_buffer/${DATA_TAG}_${TARGET_DS}DS/${SPLIT}/capture24_${SPLIT}_labels_stage2_${DATA_TAG}_${TARGET_DS}DS.pkl"
    if [ -f "$LABEL_SRC" ]; then
        LABEL_ARG="--label_src $LABEL_SRC"
    else
        LABEL_ARG=""
    fi
    # Run extraction
    python "$H5_TO_PKL" --h5_path "$H5_PATH" --split "$SPLIT" $LABEL_ARG
    echo "Extracted $SPLIT split to pkl."
done 