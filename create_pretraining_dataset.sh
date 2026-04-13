#!/bin/bash
#SBATCH --job-name=process_array
#SBATCH --output=slurm/slurm-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-23:59:59
#SBATCH --array=0-74

BATCH_SIZE=10
GLOBAL_START=256
GLOBAL_MAX=998

# Calculate start and end for this specific job
JOB_START=$(( GLOBAL_START + SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
JOB_END=$(( JOB_START + BATCH_SIZE - 1 ))

# Cap the end if it exceeds the global maximum
if [ "$JOB_END" -gt "$GLOBAL_MAX" ]; then
    JOB_END=$GLOBAL_MAX
fi

# Safety: If the start is past the max, exit
if [ "$JOB_START" -gt "$GLOBAL_MAX" ]; then
    echo "Job Start ($JOB_START) exceeds global max. Exiting."
    exit 0
fi

# Create the comma-separated list (e.g., "256,257,258")
# -s, tells seq to use a comma as the separator
RXN_LIST=$(seq -s, $JOB_START $JOB_END)

echo "Job Index: $SLURM_ARRAY_TASK_ID"
echo "Processing List: $RXN_LIST"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
ulimit -s unlimited
export OMP_STACKSIZE=64M
export OMPI_MCA_osc=^ucx

eval "$(mamba shell hook --shell bash)"
mamba activate motsart

# 4. Run commands with the list and -m flag
python -m motsart.complex_finder.complex_finder \
    env=musica \
    env.rxn_num=$RXN_LIST \
    afir_cfg=base \
    optim_cfg=cyclo32 \
    -m

python -m motsart.path_guessers.rmsd_pp.rmsd_pp_reaction_path_guesser \
    env=musica \
    env.rxn_num=$RXN_LIST \
    -m

python -m motsart.path_guessers.ts_conf_sampler \
    env=musica \
    env.rxn_num=$RXN_LIST \
    -m