#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate motsart

MODEL="${MOTSART_MLIP_MODEL:-esen-sm-conserving-all-omol}"
TASK="${MOTSART_MLIP_TASK:-omol}"
IDLE="${MLIP_IDLE_TIMEOUT:-7200}"
OET=src/motsart/validator/orca_validator/orca_external_tools

# Derive the same socket path the wrapper uses (model/task/device-keyed).
SOCKET=$(python - "$MODEL" "$TASK" <<'PY'
import sys
from motsart.validator.orca_validator.orca_external_tools.mlip_external import default_socket_path, resolve_device
model, task = sys.argv[1], (sys.argv[2] or None)
print(default_socket_path(model, task, resolve_device(None)))
PY
)
echo "Starting MLIP worker: model=$MODEL task=$TASK socket=$SOCKET (idle ${IDLE}s)"
exec python "$OET/mlip_server.py" --model "$MODEL" --task "$TASK" --socket "$SOCKET" --idle-timeout "$IDLE"
