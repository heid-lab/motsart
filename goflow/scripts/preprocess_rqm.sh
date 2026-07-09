#!/bin/bash

H5_FILE="data/RQM/raw_data/B3LYPD3_TZVP.h5"
CSV_FILE="data/RQM/raw_data/B3LYPD3_TZVP_reaction_info.csv"
IRC_FILE="data/RQM/processed_data/irc_dict.pkl"
OUT_DIR="data/RQM_RC/processed_data/"

FEAT_DICT="data/RDB7/feat_dict_organic.pkl"

python -m goflow.preprocess_rqm \
    --h5_path "$H5_FILE" \
    --csv_path "$CSV_FILE" \
    --feat_dict_path "$FEAT_DICT" \
    --out_dir "$OUT_DIR" \
    --irc_dict "$IRC_FILE" \
    --train_ratio 0.9 \
    --val_ratio 0.05 \
    --test_ratio 0.05 \
    --include_r_pos \
    --n_confs_per_rxn=32 \
    --seed 42