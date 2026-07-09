#!/bin/bash
# Wrapper for summarize_cycles.py — saddle-point optimization cost (mean/median/MAD
# cycles, convergence) for ONE (results folder, validator, TS method).
# Edit the args below, then: bash experiments/summarize_cycles.sh
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate motsart

python "$(dirname "$0")/summarize_cycles.py" \
    --results-folder results_goflow/results_goflow \
    --validator MLIPValidator \
    --ts-method racer_ts
    # --rxn-ids 74,121 \
    # --out-csv results_goflow/summary_racer_ts.csv
