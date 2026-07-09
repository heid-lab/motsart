DATA_PATH="data/CYCLO"
FULL_PKL="$DATA_PATH/processed_data/data.pkl"

python -m goflow.split_preprocessed \
    --input_data_pkl "$FULL_PKL" \
    --output_dir "$DATA_PATH/processed_data" \
    --random_split_file