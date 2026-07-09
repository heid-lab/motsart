#!/bin/bash

#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_cyclo_irc_true
#SBATCH --output=%x-%j.out

cd /home/leonard.galustian/projects/goflowv2 || exit

mamba activate goflow

#MODEL_PATH="logs/energy_pretrained_train_cyclo_energy_intense/runs/2025-12-17_10-34-51/checkpoints/epoch_140.ckpt" # energy based
#MODEL_PATH="logs/train_cyclo/runs/2025-12-17_10-23-30/checkpoints/epoch_248.ckpt"
MODEL_PATH="logs/train_cyclo/runs/2025-12-17_14-29-57/checkpoints/epoch_094.ckpt"

python -m goflow.flow_train model.sample_method=pos_guess model.num_samples=15 model.num_steps=25 model.representation.n_atom_rdkit_feats=36 model.use_energy_loss=False task_name=test_cyclo_irc_true train=False data=cyclo custom_model_weight_path=$MODEL_PATH