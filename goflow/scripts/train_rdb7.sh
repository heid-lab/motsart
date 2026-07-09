#!/bin/bash

#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --job-name=train_rdb7_gauss
#SBATCH --output=slurm/%x-%j.out
#SBATCH --time=3-23:59:59
#SBATCH --exclude=a-l40s-o-2

cd /home/leonard.galustian/projects/flowtorc/ || exit
mamba activate goflow

python -m goflow.flow_train \
    seed=1 \
    model=multihead_flow \
    model.num_steps=25 \
    model.use_init_cond=true \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=false \
    model.representation.use_backbone_skip=true \
    model.active_heads='["TS"]' \
    model.prior_modes.TS='gaussian' \
    project=rdb7_mh_goflow \
    task_name=train_rdb7_gauss_skip \
    data=rdb7