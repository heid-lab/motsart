#!/bin/bash

#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_rts_ch_ca
#SBATCH --output=slurm/%x-%j.out
#SBATCH --time=2-23:00:00
#SBATCH --exclude=a-l40s-o-2

cd /home/leonard.galustian/projects/flowtorc/ || exit
mamba activate goflow

python -m goflow.flow_train \
    model=multihead_flow \
    model.num_steps=25 \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=true \
    model.active_heads='["R", "TS"]' \
    model.use_cross_product=true \
    model.use_chiral_loss=true \
    model.chiral_loss_weight=0.1 \
    model.n_cross_product_layers=1 \
    task_name=train_rts_da_ch \
    data=rqm_multihead \
    data.batch_size=140

