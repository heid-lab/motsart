#!/bin/bash
#SBATCH --job-name=motsart_array
#SBATCH --output=slurm/slurm-%A.out
#SBATCH --error=slurm/slurm-%A.out

#SBATCH --array=0-2
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --time=3-23:00:00

module purge
module load gnu12/12.2.0
module load orca/6.1.0

RXN_NUM=$SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited
export OMP_STACKSIZE=64M
export OMPI_MCA_osc=^ucx

eval "$(mamba shell hook --shell bash)"
mamba activate motsart

python -m motsart.complex_finder.complex_finder env=tetrazine env.rxn_num=1 afir_cfg=leon optim_cfg=leon
python -m motsart.path_guessers.rmsd_pp.rmsd_pp_reaction_path_guesser env=tetrazine env.rxn_num=1
python -m motsart.path_guessers.ts_conf_sampler env=tetrazine env.rxn_num=1
python -m motsart.validator.base_validator \
    env=azide \
    validator_cfg=cluster \
    'validator_cfg.path_guessers_to_validate=[learning]' \
    validator=dft \
    env.rxn_num=$RXN_NUM