#!/bin/bash
#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --time=3-23:59:59
#SBATCH --exclude=a-l40s-o-2
#SBATCH --job-name=finetune_cyclo_dp
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --array=0-2

# Define configurations in an indexed array
# (Index 0=100, 1=300, 2=null)
CONFIGS=(50 500 1000)
NUM_DATAPOINTS=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

cd /home/leonard.galustian/projects/flowtorc/ || exit
mamba activate goflow

# 3. Handle the null logic for the Hydra argument
if [ "$NUM_DATAPOINTS" = "null" ]; then
    DATAPOINTS_ARG=""
else
    DATAPOINTS_ARG="data.num_datapoints=${NUM_DATAPOINTS}"
fi

echo "Running job array index $SLURM_ARRAY_TASK_ID with num_datapoints=${NUM_DATAPOINTS}"

# TODO!!: make batch_size smaller!
python -m goflow.flow_train \
    model=multihead_flow \
    model.num_steps=25 \
    model.use_init_cond=true \
    model.representation.numerical_size_scale=1.0 \
    model.representation.use_cross_attention=false \
    model.active_heads='["TS"]' \
    model.prior_modes.TS='pos_guess' \
    model.noise_levels.TS=0.1 \
    project=finetune_cyclo \
    task_name=finetune_dp_${NUM_DATAPOINTS} \
    +ckpt_path=/home/leonard.galustian/projects/flowtorc/logs/pretrain_cyclo_from_rqm/runs/2026-02-03_09-00-07/checkpoints/epoch_121.ckpt \
    data=cyclo \
    data.batch_size=140 \
    ${DATAPOINTS_ARG}