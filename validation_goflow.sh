#!/bin/bash
#SBATCH --job-name=validate_goflow
#SBATCH --output=slurm/slurm-%A.out
#SBATCH --error=slurm/slurm-%A.out
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
########### CHANGE ARRAY SIZE ###########
#SBATCH --array=1-64%2
#SBATCH --mem=4G


export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
ulimit -s unlimited
export OMP_STACKSIZE=64M
export OMPI_MCA_osc=^ucx

eval "$(mamba shell hook --shell bash)"
mamba activate motsart

########### CHANGE THIS ###########
RXN_FOLDER='results_goflow/finetune_noise_1_TS'
CSV_FILE='projects/motsart/data/cyclo32_small_rand.csv'

# Map all folder paths into a Bash array (sorted)
mapfile -t ALL_RXNS < <(find "$RXN_FOLDER" -maxdepth 1 -type d -name "R*" | sort -V)

# Use SLURM_ARRAY_TASK_ID to pick the specific folder
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_RXN_PATH="${ALL_RXNS[$INDEX]}"

# Extract names and IDs
RXN_NAME=$(basename "$CURRENT_RXN_PATH")
RXN_ID=${RXN_NAME#R}

echo "Task ID $SLURM_ARRAY_TASK_ID is processing: $RXN_NAME (ID: $RXN_ID)"

python -m motsart.validator.base_validator \
    env=musica \
    env.rxn_csv=$CSV_FILE \
    env.rxn_id=$RXN_ID \
    env.results_folder=$RXN_FOLDER \
    validator_cfg=cluster_goflow \
    validator_cfg.SP_nprocs=4 \
    validator=dft
