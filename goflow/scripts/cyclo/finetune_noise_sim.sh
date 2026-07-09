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
    model.num_samples=3 \
    model.use_init_cond=true \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=false \
    model.active_heads='["TS"]' \
    model.prior_modes.TS='pos_guess' \
    model.noise_levels.TS=0.1 \
    model.sim_train=true \
    model.sim_num_steps=25 \
    model.sim_start_epoch=15 \
    model.sim_ramp_epochs=55 \
    model.sim_max_prob=0.25 \
    model.sim_max_t=0.3 \
    model.val_loss_mode='fm' \
    model.val_log_rollout=true \
    model.val_rollout_num_steps=25 \
    model.val_rollout_subset_batches=0 \
    project=finetune_cyclo \
    task_name=finetune_sim_IC_maxt03_v_fmlog_really \
    +ckpt_path=/home/leonard.galustian/projects/flowtorc/logs/pretrain_cyclo_from_rqm/runs/2026-02-03_09-00-07/checkpoints/epoch_121.ckpt \
    data=cyclo \
    data.batch_size=140 
