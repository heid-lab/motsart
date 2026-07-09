#!/bin/bash
# Wrapper for e2_periplanar_analysis.py — E2 H-Cb-Ca-LG dihedral distribution
# (all attempted vs. IRC-validated TSs), with plot + syn/anti energy comparison.
# Runs both mechanisms: E2 (full periplanar/LG-base analysis) and SN2 (LG/Nu-only outcome table).
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate motsart

SCRIPT="$(dirname "$0")/e2_periplanar_analysis.py"
RESULTS_FOLDER=results_sn2e2_mlfsm

echo "=================== E2 ==================="
python "$SCRIPT" \
    --results-folder "$RESULTS_FOLDER" --ts-method ml_fsm --validator MLIPValidator --mechanism e2

echo
echo "=================== SN2 ==================="
python "$SCRIPT" \
    --results-folder "$RESULTS_FOLDER" --ts-method ml_fsm --validator MLIPValidator --mechanism sn2
