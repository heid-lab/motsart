#!/bin/bash

#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --job-name=train_flowtorc
#SBATCH --output=slurm/%x-%j.out
#SBATCH --time=2-12:00:00
#SBATCH --exclude=a-l40s-o-2

cd /home/leonard.galustian/projects/flowtorc/ || exit
mamba activate goflow

python -m goflow.flow_train \
    seed=1 \
    model.lr=0.0001 \
    model.num_steps=35 \
    model.representation.n_interactions=4 \
    model.representation.n_atom_basis=256 \
    model.representation.numerical_size_scale=1.0 \
    model.sample_method=rdkit_reactant \
    task_name=train_torc_rdkit \
    data=rqm