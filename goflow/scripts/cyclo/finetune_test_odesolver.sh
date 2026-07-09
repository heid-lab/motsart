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

CKPT_PATH=/home/leonard.galustian/projects/flowtorc/logs/finetune_noise_1_IC/runs/2026-02-06_15-47-17/checkpoints/epoch_059.ckpt

python -m goflow.flow_train \
    model=multihead_flow \
    model.num_steps=25 \
    model.use_init_cond=true \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=false \
    model.active_heads='["TS"]' \
    model.prior_modes.TS='pos_guess' \
    model.noise_levels.TS=0.1 \
    model.ode_solver=dopri8 \
    project=finetune_cyclo \
    task_name=finetune_test_odesolver_dopri8 \
    data=cyclo \
    data.batch_size=400 \
    train=False \
    test=True \
    custom_model_weight_path=$CKPT_PATH