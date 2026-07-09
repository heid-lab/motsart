#!/bin/bash
# Wrapper for summarize_telemetry.py ... aggregate per-reaction timings + per-stage failure rates across a results tree (writes *_timing/*_failures CSVs).
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate motsart

python "$(dirname "$0")/summarize_telemetry.py" \
    --results-folder results_goflow/results_goflow \
    --out-prefix results_goflow/telemetry
    # --rxn-ids 74,121
