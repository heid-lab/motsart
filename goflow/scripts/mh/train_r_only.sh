#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_mh_r_only
#SBATCH --output=slurm/%x-%j.out
#SBATCH --time=2-12:00:00
#SBATCH --exclude=a-l40s-o-2

#cd /home/leonard.galustian/projects/flowtorc/ || exit
#mamba activate goflow

python -m goflow.flow_train \
    model=multihead_flow \
    model.num_steps=25 \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=true \
    model.active_heads='["R"]' \
    task_name=train_mh_r_only \
    data=rqm_multihead
