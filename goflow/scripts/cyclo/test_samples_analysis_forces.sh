#!/bin/bash

#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_samples_analysis_cyclo
#SBATCH --output=%x-%j.out

#50 100 300 500 1000 1200
for val in 50 100 300 500 1000 1200; do
    SAMPLE_PATH="/Users/leo/Documents/motsart_project/goflow/reaction_analysis/finetune_dp_${val}_TS/samples_all.pkl"
    OUT_PATH="reaction_analysis/cyclo/forces_analysis_${val}"

    EXTRA_FLAGS=""
    if [ -z "$FIRST_RUN" ]; then
        EXTRA_FLAGS="--evaluate-force-norm --evaluate-force-norm-guess"
        FIRST_RUN=1
    fi

    python -m goflow.test_samples_analysis \
        "$SAMPLE_PATH" \
        "$OUT_PATH" \
        $EXTRA_FLAGS \
        --force-backend xtb \
        --force-solvent water
done
