#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
set -a
source "$root/.env"
set +a
export GENESIS_IPC_CLOTH_PARAMETERIZATION=native
export GENESIS_IPC_DETAILED_TIMERS=0
export GENESIS_IPC_CONVERGENCE_LOGS=1
export PYTHONUNBUFFERED=1
unset UIPC_BACKEND_PATH UIPC_BACKEND_DIR LD_PRELOAD
unset UIPC_NORMAL_LOSS_RATE_S UIPC_NORMAL_LOSS_SCOPE UIPC_NORMAL_LOSS_FEM_CONTACT_ID UIPC_NORMAL_LOSS_OTHER_CONTACT_ID
cuda_dirs=("$GENESIS_ENV"/lib/python3.12/site-packages/nvidia/*/lib)
cuda_path="$(IFS=:; echo "${cuda_dirs[*]}")"
export LD_LIBRARY_PATH="$cuda_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$GENESIS_PYTHON" "$root/tools/run_scene527_uipc028.py" "$@"
