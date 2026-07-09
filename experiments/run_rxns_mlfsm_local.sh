#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate motsart

# ------------------------------- CONFIG (edit me) -------------------------------
RXN_NUM=0                         # index of the first reaction to run
N_RXNS=8                          # number of consecutive reactions to run
ENV=am                            # hydra env preset
CSV_FILE=data/reactions_sn2_e2.csv
RESULTS_FOLDER=results_sn2e2_mlfsm
OPTIM_CFG=am                      # complex-finder evolutionary search preset
AFIR_CFG=am
FSM_CFG=base                      # ML-FSM parameters (see FSMPathGuesserParams): base | test | local
VALIDATOR_CFG=ml_fsm              # path_guessers_to_validate=['ml_fsm']
VALIDATOR=mlip                    # mlip | dft | xtb
FULL_IRC=true
TS_METHOD=ml_fsm

RUN_COMPLEX_FINDER=false
RUN_MLFSM=false
RUN_VALIDATOR=true
# Optional: local OMol25 checkpoint (*.pt). Leave empty to use the gated FAIRChem
# registry model (esen-sm-conserving-all-omol) from the env=am preset.
MLIP_MODEL=""
# --------------------------------------------------------------------------------

case "$VALIDATOR" in
    mlip) VALIDATOR_CLASS=MLIPValidator ;;
    xtb) VALIDATOR_CLASS=GFN2XTBValidator ;;
    dft) VALIDATOR_CLASS=DFTValidator ;;
    *) VALIDATOR_CLASS="$VALIDATOR" ;;
esac

# FULL_IRC -> validator_cfg.skip_full_irc override (Hydra/Python boolean).
# full IRC enabled  => skip_full_irc=False;  disabled => skip_full_irc=True
case "$(printf '%s' "$FULL_IRC" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes) SKIP_FULL_IRC=False ;;
    0|false|no) SKIP_FULL_IRC=True ;;
    *) echo "Error: FULL_IRC must be true/false (got '$FULL_IRC')" >&2; exit 1 ;;
esac

COMMON=(
    env="$ENV"
    env.rxn_csv="$CSV_FILE"
    env.results_folder="$RESULTS_FOLDER"
)
if [ -n "$MLIP_MODEL" ]; then
    COMMON+=( env.mlip_model="$MLIP_MODEL" )
fi

for (( rxn_num=RXN_NUM; rxn_num<RXN_NUM+N_RXNS; rxn_num++ )); do
    echo "===== Running reaction $rxn_num (ml_fsm) ====="

    if [ "$RUN_COMPLEX_FINDER" = true ]; then
        python -m motsart.complex_finder.complex_finder "${COMMON[@]}" \
            env.rxn_num=$rxn_num afir_cfg="$AFIR_CFG" optim_cfg="$OPTIM_CFG" \
            || echo "[warn] complex_finder failed for rxn $rxn_num"
    fi

    if [ "$RUN_MLFSM" = true ]; then
        python -m motsart.path_guessers.ml_fsm.ml_fsm_reaction_path_guesser "${COMMON[@]}" \
            env.rxn_num=$rxn_num fsm_cfg="$FSM_CFG" \
            || echo "[warn] ml_fsm path guesser failed for rxn $rxn_num"
    fi

    if [ "$RUN_VALIDATOR" = true ]; then
        python -m motsart.validator.base_validator "${COMMON[@]}" \
            env.rxn_num=$rxn_num validator_cfg="$VALIDATOR_CFG" validator="$VALIDATOR" \
            validator_cfg.skip_full_irc=$SKIP_FULL_IRC \
            || echo "[warn] validator failed for rxn $rxn_num"
    fi
done

# ----- Summarize results -----
echo "===== Saddle-point optimization summary ====="
python experiments/summarize_cycles.py \
    --results-folder "$RESULTS_FOLDER" --validator "$VALIDATOR_CLASS" --ts-method "$TS_METHOD" \
    --out-csv "$RESULTS_FOLDER/summary_cycles_${TS_METHOD}_${VALIDATOR_CLASS}.csv" \
    || echo "[warn] summarize_cycles failed"

echo "===== Telemetry (timings & failure rates) ====="
python experiments/summarize_telemetry.py \
    --results-folder "$RESULTS_FOLDER" --out-prefix "$RESULTS_FOLDER/telemetry" \
    || echo "[warn] summarize_telemetry failed"
