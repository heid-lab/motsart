#!/bin/bash
# Preprocessing script to create PyG data objects for fine-tuning with DFT guess/GT TS pairs
# Creates train/val/test pickle files with guess and ground-truth TS geometries

set -e

RESULTS_FOLDER="results_musica/results_musica"
RXN_CSV="data/cyclo32_small_rand.csv"
OUT_DIR="/Users/leo/Documents/motsart_project/goflow/data/CYCLO/processed_data"
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
SEED=42

python -m motsart.learning.results_to_data_pkl \
    --results_folder "$RESULTS_FOLDER" \
    --rxn_csv "$RXN_CSV" \
    --out_dir "$OUT_DIR" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --test_ratio "$TEST_RATIO" \
    --seed "$SEED" \
    --check_irc_success \
    --group_by_rxn
