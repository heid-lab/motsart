#!/bin/bash

python -m goflow.preprocess_rqm_multihead \
    --h5_path data/RQM/raw_data/B3LYPD3_TZVP.h5 \
    --csv_path data/RQM/raw_data/B3LYPD3_TZVP_reaction_info.csv \
    --irc_dict data/RQM/processed_data/irc_dict.pkl \
    --out_dir data/RQM_MH/processed_data/ \
    --n_confs_per_rxn 32 \
    --max_reactions 256