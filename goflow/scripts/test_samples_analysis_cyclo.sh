#!/bin/bash

#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_samples_analysis_cyclo
#SBATCH --output=%x-%j.out

cd /home/leonard.galustian/projects/goflowv2 || exit

mamba activate gotennet

SAMPLE_PATH="/Users/leo/Documents/motsart_project/goflow/logs/test_save_all_samples_cyclo/multiruns/2025-12-09_07-48-51/0/test_samples/samples_all.pkl"

python -m goflow.test_samples_analysis $SAMPLE_PATH cyclo