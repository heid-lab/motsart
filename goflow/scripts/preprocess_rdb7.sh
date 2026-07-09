DATA_PATH="data/RDB7/"
FULL_CSV="$DATA_PATH/raw_data/rdb7_full.csv"
FULL_XYZ="$DATA_PATH/raw_data/rdb7_ts.xyz"
FULL_PKL="$DATA_PATH/processed_data/data.pkl"

python -m goflow.preprocessing \
    --csv_file "$FULL_CSV" \
    --xyz_file "$FULL_XYZ" \
    --save_filepath "$DATA_PATH/processed_data/data.pkl"
