# Cluster & HPC

We provide SLURM-oriented scripts for cluster workflows.

## Pipeline Scripts

| Script | Description |
|--------|-------------|
| `complex_and_ts_search_local.sh` | Local pipeline template |
| `complex_and_ts_search_cpu.sh` | SLURM CPU pipeline template |
| `complex_and_ts_search_gpu.sh` | GPU validator-oriented template |
| `validation_goflow.sh` | Validate model-generated TS guesses on cluster |

### Running on SLURM
As with all the other scripts, make sure that to adjust them according to your environment.
```bash
sbatch complex_and_ts_search_cpu.sh
```

## Parallel Execution with Hydra

```bash
python -m motsart.complex_finder.complex_finder -m \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=4 \
    env=cluster \
    "env.rxn_num=range(0,32)"
```

## Learning on Cluster

Typical workflow:

1. Push code and data: `bash push_to_musica.sh`
2. Run base pipeline: `sbatch complex_and_ts_search_musica.sh`
3. Prepare training/eval data: `bash create_fine_tune_dft_data.sh` or `bash create_preprocess_rtsp_pretrain_data.sh`
4. Import model-generated samples: `bash fetch_and_push_data_pkl_to_results.sh`
5. Validate model-generated guesses: `sbatch validation_goflow.sh`
6. Compute stats:

```bash
python -m motsart.validator.compute_stats \
  --cluster-folder /data/results_cluster \
  --learning-folder /data/results_goflow/finetune_noise_1_TS \
  --validator DFTValidator \
  --output-csv /data/results_goflow/finetune_noise_1_TS/stats_al.csv \
  --cluster-ts-method racer_ts \
  --al-ts-method learning \
  --mode both
```

For a general walkthrough, see [Paper Reproduction Workflow](../pipeline/paper-reproduction.md).
