#!/bin/bash

#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --job-name=pretrain_cyclo_from_rqm
#SBATCH --output=slurm/%x-%j.out
#SBATCH --time=3-23:59:59
#SBATCH --exclude=a-l40s-o-2

cd /home/leonard.galustian/projects/flowtorc/ || exit
mamba activate goflow

python -m goflow.flow_train \
    model=multihead_flow \
    model.num_steps=25 \
    model.use_init_cond=true \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=true \
    model.active_heads='["R", "TS", "P"]' \
    project=pretrain_cyclo \
    task_name=pretrain_cyclo_from_rqm \
    +ckpt_path=/home/leonard.galustian/projects/flowtorc/logs/train_rtsp_tag_IC_add/runs/2026-01-30_12-20-58/checkpoints/epoch_066.ckpt \
    data=cyclo_pre \
    data.batch_size=60
