#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_mh_ts_only
#SBATCH --output=slurm/%x-%j.out
#SBATCH --time=2-23:59:59
#SBATCH --exclude=a-l40s-o-2

cd /home/leonard.galustian/projects/flowtorc/ || exit
mamba activate goflow

python -m goflow.flow_train \
    seed=3 \
    model=multihead_flow \
    model.num_steps=25 \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=true \
    model.active_heads='["TS"]' \
    task_name=train_ts_test \
    data=rqm_multihead \
    train=False \
    data.batch_size=250 \
    custom_model_weight_path="/home/leonard.galustian/projects/flowtorc/logs/train_mh_ts_only/runs/2026-01-21_17-51-00/checkpoints/epoch_235.ckpt"
