#!/bin/bash
#SBATCH --job-name=validate_engine
#SBATCH --output=slurm/slurm-%A_%a.out
#SBATCH --error=slurm/slurm-%A_%a.out
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --array=1-64%4
#SBATCH --mem=8G
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1

# SLURM array version of run_validation_engine_local.sh: validate one reaction
# per array task with a chosen PES engine (xtb | dft | mlip).
#
# For engine=mlip, request a GPU (the eSEN model runs on CUDA if available, else
# CPU) and make sure fairchem-core is installed in the `motsart` env (see
# experiments/install_mlip_deps.sh) and the model checkpoint is accessible.

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
ulimit -s unlimited
export OMP_STACKSIZE=64M
export OMPI_MCA_osc=^ucx

eval "$(mamba shell hook --shell bash)"
mamba activate motsart

# ------------------------------- CONFIG (edit me) -------------------------------
ENGINE="${ENGINE:-mlip}" # xtb | dft | mlip
ENV_NAME="${ENV_NAME:-cluster}"
VALIDATOR_CFG="${VALIDATOR_CFG:-cluster}"
RESULTS_FOLDER="${RESULTS_FOLDER:-results_goflow/results_goflow}"
CSV_FILE="${CSV_FILE:-data/cyclo32_small_rand.csv}"
TS_METHOD="${TS_METHOD:-racer_ts}" # racer_ts | rmsd_pp | learning
# export MOTSART_MLIP_MODEL=/path/to/esen_sm_conserving_all.pt   # optional override
# --------------------------------------------------------------------------------

# Map all reaction folders into a sorted array; pick this task's reaction.
mapfile -t ALL_RXNS < <(find "$RESULTS_FOLDER" -maxdepth 1 -type d -name "R*" | sort -V)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_RXN_PATH="${ALL_RXNS[$INDEX]}"
RXN_NAME=$(basename "$CURRENT_RXN_PATH")
RXN_ID=${RXN_NAME#R}

echo "Task $SLURM_ARRAY_TASK_ID -> $RXN_NAME (id $RXN_ID), engine=$ENGINE, ts_method=$TS_METHOD"

python -m motsart.validator.base_validator \
    env="$ENV_NAME" \
    env.rxn_csv="$CSV_FILE" \
    env.rxn_id="$RXN_ID" \
    env.results_folder="$RESULTS_FOLDER" \
    validator_cfg="$VALIDATOR_CFG" \
    "validator_cfg.path_guessers_to_validate=[$TS_METHOD]" \
    validator="$ENGINE"
