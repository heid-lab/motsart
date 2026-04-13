#!/bin/bash

# Compute validation statistics for TS methods
# This script compares cluster results with learning results

CLUSTER_FOLDER="results_musica/results_musica"
CLUSTER_TS_METHOD="racer_ts"

LEARNING_FOLDER="results_goflow/results_goflow"
AL_TS_METHOD="active_learning"

VALIDATOR="DFTValidator"
OUTPUT_CSV="results_goflow/stats_tot.csv"

MODE="both"  # Options: cluster, al, both

python -m motsart.validator.compute_stats \
    --cluster-folder "$CLUSTER_FOLDER" \
    --learning-folder "$LEARNING_FOLDER" \
    --validator "$VALIDATOR" \
    --output-csv "$OUTPUT_CSV" \
    --cluster-ts-method "$CLUSTER_TS_METHOD" \
    --al-ts-method "$AL_TS_METHOD" \
    --mode "$MODE"
