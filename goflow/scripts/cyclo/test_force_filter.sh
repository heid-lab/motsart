#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_cyclo_filters
#SBATCH --output=slurm/%x-%j.out
#SBATCH --time=1-23:59:59
#SBATCH --exclude=a-l40s-o-2

cd /home/leonard.galustian/projects/flowtorc/ || exit
mamba activate goflow

# Checkpoint path - update this to your trained model
CKPT_PATH=/home/leonard.galustian/projects/flowtorc/logs/pretrain_cyclo_from_rqm/runs/2026-02-03_09-00-07/checkpoints/epoch_121.ckpt

# Force filter (selects low-force structures using MACE)
python -m goflow.flow_train \
    model=multihead_flow \
    model.num_steps=25 \
    model.num_samples=75 \
    model.n_center_samples=25 \
    model.use_init_cond=true \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=true \
    model.active_heads='["TS"]' \
    model.test_filter_force=true \
    model.force_model_path="models/EGRET_1T.model" \
    project=test_cyclo_filters \
    task_name=test_cyclo_force \
    train=False \
    test=True \
    data=cyclo_pre \
    data.batch_size=400 \
    custom_model_weight_path=$CKPT_PATH
