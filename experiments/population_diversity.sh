#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate motsart

N_RXNS=16
GENERATIONS=300
SUBSAMPLE=256
OUT_DIR=results_diversity

python experiments/population_diversity.py \
    --rxn-csv data/cyclo32_atom_mapped_small.csv \
    --n-rxns "$N_RXNS" \
    --generations "$GENERATIONS" \
    --subsample "$SUBSAMPLE" \
    --selections tournament,truncation \
    --out-dir "$OUT_DIR"

echo "===== Done. See $OUT_DIR/diversity_fig.pdf ====="
