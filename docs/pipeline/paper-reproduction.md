# Paper Reproduction Workflow

## Goal

Compare TS methods from path guesser (for example `racer_ts`) against model-generated TS guesses (`learning`) on the same reaction set and validator settings.

## Inputs You Need

- Reaction CSV in moTSart format (no header, `rxn_id,rxn_smiles`)
- Path guesser results folder (output from Step 1 to Step 3)
- Model-generated TS samples (or a script that fetches them)
- Cluster/local run scripts adjusted to your environment paths

## Phase A: Run Baseline Pipeline

1. Choose your execution script. Local: `bash complex_and_ts_search_local.sh`. Cluster: `sbatch complex_and_ts_search_cpu.sh`.
2. Confirm baseline validation files exist under: `results*/R*/validation/{method}/validation_*.csv`

## Phase B: Build Learning Data Pickles

Use one of:

- `bash create_fine_tune_dft_data.sh`
- `bash create_preprocess_rtsp_pretrain_data.sh`

Before running, adjust script variables (`RESULTS_FOLDER`, `RXN_CSV`, `OUT_DIR`, split ratios) to your paths. Then on the GPU cluster, fine-tune the model on the data and generate samples for the test set.

## Phase C: Import Model-Generated TS Samples

Run:

```bash
bash fetch_and_push_data_pkl_to_results.sh
```

Adjust variables in that script so imported samples end up in your target `learning` results folder structure (for example `results_goflow/<project>/R*/ts/learning/ts_to_validate/`).

## Phase D: Validate Imported TS Guesses

Run validator on imported AL guesses:

```bash
sbatch validation_goflow.sh
```

Edit `RXN_FOLDER`, `CSV_FILE`, array settings, and resource flags for your cluster.

## Phase E: Compute Final Stats
Example:
```bash
python -m motsart.validator.compute_stats \
  --cluster-folder /path/to/baseline_results \
  --learning-folder /path/to/al_results \
  --validator DFTValidator \
  --output-csv /path/to/al_results/stats_al.csv \
  --cluster-ts-method racer_ts \
  --al-ts-method learning \
  --mode both
```

## Quick Sanity Checklist

- Path guesser and (learned) model inference runs use the same reaction IDs
- Validator choice is consistent across compared runs (`xtb` or `dft`)
- `path_guessers_to_validate` in `validator_cfg` includes the method you are evaluating
- Stats command points to folders containing `R*` subdirectories
