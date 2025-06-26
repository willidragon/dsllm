#!/bin/bash

# Run both processing scripts sequentially and log their outputs (also print to terminal)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# You can change python3 to your preferred Python executable if needed
PYTHON=python3

# Log files
LOG1="$SCRIPT_DIR/process_stage2_1.log"
LOG2="$SCRIPT_DIR/process_stage2_3.log"

# Run the first script and wait for it to finish
$PYTHON "$SCRIPT_DIR/process_capture24_stage2_1_custom_mins_multiple.py" 2>&1 | tee "$LOG1"

# After the first script finishes, run the second script
$PYTHON "$SCRIPT_DIR/process_capture24_stage2_3_custom_mins_multiple.py" 2>&1 | tee "$LOG2"

echo "Both processing scripts have completed." 