#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=finetune_cyclo_from_rqm
#SBATCH --output=slurm/%x-%j.out
#SBATCH --time=3-23:59:59
#SBATCH --exclude=a-l40s-o-2

cd /home/leonard.galustian/projects/flowtorc/ || exit
mamba activate goflow

python -m goflow.flow_train \
    model=multihead_flow \
    model.num_steps=25 \
    model.use_init_cond=false \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=false \
    model.active_heads='["TS"]' \
    model.prior_modes.TS='pos_guess' \
    model.noise_levels.TS=0.0 \
    project=finetune_cyclo \
    task_name=finetune_noise_1_noIC_nonoise \
    +ckpt_path=/home/leonard.galustian/projects/flowtorc/logs/pretrain_cyclo_from_rqm/runs/2026-02-03_09-00-07/checkpoints/epoch_121.ckpt \
    data=cyclo \
    data.batch_size=140 