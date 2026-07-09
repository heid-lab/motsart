#!/bin/bash
# Wrapper for compare_engines.py — compare optimization cost across engines and/or
# TS-guess sources, with reductions vs the first series.
# Each --series is  LABEL:RESULTS_FOLDER:VALIDATOR:TS_METHOD
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate motsart

python "$(dirname "$0")/compare_engines.py" \
    --paired \
    --series baseline:results_goflow/results_goflow:MLIPValidator:racer_ts \
    --series tsoptnet:results_goflow/results_goflow:MLIPValidator:learning \
    --out-csv results_goflow/compare_mlip.csv
