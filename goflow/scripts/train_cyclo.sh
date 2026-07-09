#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_cyclo_rqm
#SBATCH --output=slurm/%x-%j.out

cd /home/leonard.galustian/projects/goflowv2/ || exit

mamba activate goflow

#python -m goflow.flow_train \
#    seed=1 \
#    model.lr=0.0001 \
#    model.num_steps=30 \
#    model.representation.n_interactions=6 \
#    model.representation.n_atom_basis=256 \
#    model.representation.n_atom_rdkit_feats=36 \
#    model.representation.numerical_size_scale=1.0 \
#    model.sample_method=pos_guess \
#    model.use_bond_loss=False \
#    task_name=train_cyclo_rqm_big \
#    data=cyclo \
#    ckpt_path=/home/leonard.galustian/projects/goflowv2/logs/train_rqm_big/runs/2026-01-07_07-17-01/epoch_400.ckpt
#                                           logs/train_rqm_big/runs/2026-01-07_07-17-01/checkpoints/epoch_400.ckpt
# /home/leonard.galustian/projects/goflowv2/logs/train_rqm_big/runs/2026-01-07_07-17-01/epoch_400.ckpt
# 
#python -m goflow.flow_train \
#    seed=1 \
#    model.lr=0.0001 \
#    model.num_steps=30 \
#    model.representation.cutoff_fn.cutoff=5 \
#    model.representation.n_interactions=3 \
#    model.representation.n_atom_basis=256 \
#    model.representation.n_atom_rdkit_feats=36 \
#    model.representation.numerical_size_scale=1.0 \
#    model.sample_method=pos_guess \
#    model.use_bond_loss=False \
#    task_name=train_cyclo_rqm \
#    data=cyclo \
#    ckpt_path=/home/leonard.galustian/projects/goflowv2/logs/train_rqm_lr05/runs/2025-12-24_08-47-29/checkpoints/epoch_283.ckpt

python -m goflow.flow_train \
    seed=1 \
    model.lr=0.0001 \
    model.num_steps=30 \
    model.representation.cutoff_fn.cutoff=5 \
    model.representation.n_interactions=3 \
    model.representation.n_atom_basis=256 \
    model.representation.n_atom_rdkit_feats=36 \
    model.representation.numerical_size_scale=0.35 \
    model.sample_method=pos_guess \
    model.use_bond_loss=False \
    task_name=train_cyclo_rdb7 \
    data=cyclo \
    ckpt_path=/home/leonard.galustian/projects/goflowv2/logs/train_rdb7/multiruns/2025-12-17_15-26-43/0/checkpoints/epoch_308.ckpt