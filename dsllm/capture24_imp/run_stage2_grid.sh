#!/bin/bash

set -e

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="$CONFIG_DIR/run_all_stage2_custom_min.sh"

CONFIGS=(
    config_stage2_300s_100ds.yaml
)

for CFG in "${CONFIGS[@]}"; do
    # Skip commented lines (starting with # or empty)
    [[ "$CFG" =~ ^#.*$ || -z "$CFG" ]] && continue
    echo "=============================================="
    echo "Running pipeline with config: $CFG"
    echo "=============================================="
    # Copy the config to the expected name
    cp "$CONFIG_DIR/$CFG" "$CONFIG_DIR/config_stage2.yaml"
    # Run the pipeline
    bash "$PIPELINE_SCRIPT"
    echo "=============================================="
    echo "Finished run with config: $CFG"
    echo "=============================================="
done 