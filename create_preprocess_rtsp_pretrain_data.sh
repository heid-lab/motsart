#!/bin/bash
# Preprocessing script to create PyG data objects for multi-head flow matching
# Creates train/val/test pickle files with R, TS, P geometries and conformers

set -e

RESULTS_FOLDER="results_musica/results_musica"
RXN_CSV="data/cyclo32_small_rand.csv"
OUT_DIR="/Users/leo/Documents/motsart_project/goflow/data/CYCLO_PRE/processed_data"
TS_METHOD="racer_ts"
N_CONFORMERS=32
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
SEED=42

python -m motsart.learning.results_to_data_pkl_pre \
    --results_folder "$RESULTS_FOLDER" \
    --rxn_csv "$RXN_CSV" \
    --out_dir "$OUT_DIR" \
    --ts_method "$TS_METHOD" \
    --n_conformers "$N_CONFORMERS" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --test_ratio "$TEST_RATIO" \
    --seed "$SEED" \
    --generate_conformers \
    --group_by_rxn
