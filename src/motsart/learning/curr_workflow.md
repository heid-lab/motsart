# Current Maintainer Workflow (Environment-Specific)

This file is a short, environment-specific runbook used by maintainers.

For general users and reproducibility instructions, use:

- `docs/pipeline/paper-reproduction.md`

Current local sequence (edit paths and resources before running):

1. `bash push_to_cluster.sh`
2. `sbatch complex_and_ts_search_musica.sh`
3. `bash create_fine_tune_dft_data.sh` or `bash create_preprocess_rtsp_pretrain_data.sh`
4. `bash fetch_and_push_data_pkl_to_results.sh`
5. `sbatch validation_goflow.sh`
6. `python -m motsart.validator.compute_stats ...`
