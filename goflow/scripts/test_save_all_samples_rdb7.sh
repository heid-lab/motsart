#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_save_all_samples_rdb7
#SBATCH --output=slurm/%x-%j.out

cd /home/leonard.galustian/projects/goflowv2 || exit

mamba activate goflow

MODEL_PATH="logs/train_rdb7/multiruns/2025-12-17_15-26-43/0/checkpoints/epoch_308.ckpt"

python -m goflow.flow_train model.sample_method=gaussian model.num_samples=15 model.num_steps=25 model.representation.n_atom_rdkit_feats=36 model.use_energy_loss=False task_name=test_save_all_samples_rdb7 train=False data=rdb7 custom_model_weight_path=$MODEL_PATH