#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/robodojo/train_robodojo.sh <stats|train|resume>

Modes:
  stats   Compute norm stats for the datasets in CONFIG_NAME.
  train   Start a new run; refuse to overwrite an existing run.
  resume  Resume the latest full checkpoint from the same run.

For a new experiment, edit only the CONFIGURATION block near the top.
EOF
}

mode="${1:-}"
case "${mode}" in
  stats|train|resume) ;;
  -h|--help|"") usage; exit 0 ;;
  *) echo "[ERROR] Unknown mode: ${mode}" >&2; usage >&2; exit 2 ;;
esac

# ------------------------- CONFIGURATION -------------------------
CONFIG_NAME="pi05_robodojo_stack_bowls_arx_x5_joint"
INIT_PARAMS="/mnt/resource/robodojo_checkpoints/pi05_robodojo_stack_bowls_arx_x5_joint/robodojo_sft_100ep/30000/params"
CHECKPOINT_ROOT="/mnt/resource/robodojo_checkpoints"
LEROBOT_HOME="/mnt/resource/robodojo_dataset/lerobot"
LOG_ROOT="/mnt/resource/robodojo_training_logs"

GPUS="0,1,2,3"
FSDP_DEVICES=2
# Number of new optimizer updates in this run; the initial checkpoint's step is not added.
TRAIN_STEPS=40000
BATCH_SIZE=256
NUM_WORKERS=8
SEED=0
SAVE_INTERVAL=5000
KEEP_PERIOD=10000
# ---------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] Missing environment: ${PYTHON}" >&2
  exit 1
fi

export HF_LEROBOT_HOME="${LEROBOT_HOME}"
export HF_DATASETS_CACHE="/tmp/openpi-cache-$(hostname)/hf/datasets"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONUNBUFFERED=1

mkdir -p "${HF_DATASETS_CACHE}"
cd "${REPO_ROOT}"

config_values="$(${PYTHON} - "${CONFIG_NAME}" <<'PY'
import sys
from openpi.training import config

cfg = config.get_config(sys.argv[1])
print(cfg.exp_name)
print(cfg.data.repo_id)
PY
)"
EXP_NAME="$(sed -n '1p' <<<"${config_values}")"
REPO_IDS="$(sed -n '2p' <<<"${config_values}")"

if [[ -z "${EXP_NAME}" || -z "${REPO_IDS}" ]]; then
  echo "[ERROR] CONFIG_NAME must define exp_name and repo_id: ${CONFIG_NAME}" >&2
  exit 1
fi

IFS=',' read -r -a repo_id_array <<<"${REPO_IDS}"
for repo_id in "${repo_id_array[@]}"; do
  if [[ ! -f "${LEROBOT_HOME}/${repo_id}/meta/info.json" ]]; then
    echo "[ERROR] Missing LeRobot dataset: ${LEROBOT_HOME}/${repo_id}" >&2
    exit 1
  fi
done

ASSET_ID="${REPO_IDS//,/_}"
NORM_STATS="${REPO_ROOT}/assets/${CONFIG_NAME}/${ASSET_ID}/norm_stats.json"
RUN_DIR="${CHECKPOINT_ROOT}/${CONFIG_NAME}/${EXP_NAME}"

echo "[RoboDojo train] mode=${mode}"
echo "[RoboDojo train] config=${CONFIG_NAME}"
echo "[RoboDojo train] datasets=${REPO_IDS}"
echo "[RoboDojo train] norm_stats=${NORM_STATS}"
echo "[RoboDojo train] run=${RUN_DIR}"

if [[ "${mode}" == "stats" ]]; then
  exec "${PYTHON}" scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
fi

if [[ ! -f "${NORM_STATS}" ]]; then
  echo "[ERROR] Missing norm stats. Run first:" >&2
  echo "        bash examples/robodojo/train_robodojo.sh stats" >&2
  exit 1
fi
if [[ ! -d "${INIT_PARAMS}" ]]; then
  echo "[ERROR] Missing initial params: ${INIT_PARAMS}" >&2
  exit 1
fi

mode_args=()
if [[ "${mode}" == "train" ]]; then
  if [[ -e "${RUN_DIR}" ]]; then
    echo "[ERROR] Refusing to overwrite existing run: ${RUN_DIR}" >&2
    echo "        Use resume if this is the same experiment." >&2
    exit 1
  fi
else
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[ERROR] Cannot resume missing run: ${RUN_DIR}" >&2
    exit 1
  fi
  mode_args+=(--resume)
fi

mkdir -p "${LOG_ROOT}"
log_file="${LOG_ROOT}/${EXP_NAME}_${mode}_$(date +%Y%m%d_%H%M%S).log"
echo "[RoboDojo train] log=${log_file}"

set -o pipefail
"${PYTHON}" scripts/train.py \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --weight-loader.params-path "${INIT_PARAMS}" \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --num-train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --fsdp-devices "${FSDP_DEVICES}" \
  --save-interval "${SAVE_INTERVAL}" \
  --keep-period "${KEEP_PERIOD}" \
  --save-full-state \
  "${mode_args[@]}" \
  2>&1 | tee -a "${log_file}"
