#!/usr/bin/env bash
set -euo pipefail

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/public/gaojiahuan/huggingface/lerobot}"
export HF_HOME="${HF_HOME:-/mnt/public/gaojiahuan/huggingface/cache/${HOSTNAME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/mnt/public/gaojiahuan/.cache/jax/openpi_demospeedup}"
export PATH="/usr/local/cuda/bin:$PATH"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=/usr/local/cuda}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$JAX_COMPILATION_CACHE_DIR"

readonly CONFIG_NAME="bagging_sku_0811_demospeedup_sm2sm"
readonly EXP_NAME="bagging_sku_0811_demospeedup_r2r4_dm10dh50df50po20_meta_ah20"
readonly CHECKPOINT_DIR="/mnt/public/gaojiahuan/openpi/checkpoints/$CONFIG_NAME/$EXP_NAME"
readonly PROXY_CHECKPOINT="/mnt/public/gaojiahuan/openpi/checkpoints/bagging_sku_0811_sm2sm/bagging_sku_0811_dm10dh50df50po20_meta/80000"
readonly LABEL_ROOT="${DEMOSPEEDUP_LABEL_ROOT:-/mnt/public/gaojiahuan/huggingface/demospeedup_labels}"
readonly STAGE="${1:-all}"
readonly SAMPLES="${DEMOSPEEDUP_SAMPLES:-10}"
readonly FRAME_BATCH_SIZE="${DEMOSPEEDUP_FRAME_BATCH_SIZE:-1}"
readonly NUM_STEPS="${DEMOSPEEDUP_NUM_STEPS:-10}"
readonly EPISODES="${DEMOSPEEDUP_EPISODES:-}"
readonly PARALLEL_LOG_ROOT="${DEMOSPEEDUP_LOG_ROOT:-/mnt/public/gaojiahuan/openpi/logs/demospeedup_label}"
readonly BUILD_LOG_ROOT="${DEMOSPEEDUP_BUILD_LOG_ROOT:-/mnt/public/gaojiahuan/openpi/logs/demospeedup_build}"
readonly PIPELINE_LOG_ROOT="${DEMOSPEEDUP_PIPELINE_LOG_ROOT:-/mnt/public/gaojiahuan/openpi/logs/demospeedup_pipeline}"
readonly NORM_NUM_WORKERS="${DEMOSPEEDUP_NORM_NUM_WORKERS:-8}"

DATASETS=(
  bagging_sku_fty_0702b_trimmed
  bagging_sku_fty_0703_trimmed
  bagging_sku_fty_0706+0707
  bagging_sku_fty_0708_trimmed
  bagging_sku_fty_0715_trimmed
  bagging_sku_fty_0713_dagger_filtered
  bagging_sku_fty_0714_trimmed
  bagging_sku_fty_0717_trimmed
  bagging_sku_zzc_0724_trimmed
  bagging_sku_zzc_0730_trimmed
  bagging_sku_zyx_0625
  bagging_sku_zzc_0805_dagger
  bagging_sku_zzc_0810
)

if [[ -n "${DEMOSPEEDUP_DATASET:-}" ]]; then
  DATASETS=("$DEMOSPEEDUP_DATASET")
fi

label_dataset() {
  local dataset="$1"
  local episode_spec="${2-$EPISODES}"
  local args=(
    --dataset "$dataset"
    --config-name bagging_sku_0811_sm2sm
    --checkpoint "$PROXY_CHECKPOINT"
    --label-root "$LABEL_ROOT"
    --samples "$SAMPLES"
    --frame-batch-size "$FRAME_BATCH_SIZE"
    --num-steps "$NUM_STEPS"
    --estimator kde
    --contamination 0.01
    --min-cluster-size 20
    --precision-quantile 0.4
  )
  if [[ -n "$episode_spec" ]]; then
    args+=(--episodes "$episode_spec")
  fi
  uv run examples/x2robot/demospeedup/label_entropy.py "${args[@]}"
}

label_worker() {
  local gpu="$1"
  shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  local job dataset episode_spec
  for job in "$@"; do
    IFS='|' read -r dataset episode_spec <<<"$job"
    echo "[$(date '+%F %T')] GPU $gpu starting $dataset episodes=${episode_spec:-all}"
    label_dataset "$dataset" "$episode_spec"
  done
}

label_all_parallel() {
  mkdir -p "$PARALLEL_LOG_ROOT"
  local pids=()

  # Split the dominant 0702b dataset across all GPUs, then balance the remaining
  # datasets by frame count. Per-episode NPZ outputs are independent. A final
  # metadata-only pass below consolidates the 0702b manifest after all shards finish.
  label_worker 0 \
    "bagging_sku_fty_0702b_trimmed|0:140" \
    "bagging_sku_zyx_0625|" \
    "bagging_sku_zzc_0724_trimmed|" \
    "bagging_sku_zzc_0805_dagger|" \
    "bagging_sku_zzc_0810|" \
    >>"$PARALLEL_LOG_ROOT/gpu0.log" 2>&1 &
  pids+=("$!")

  label_worker 1 \
    "bagging_sku_fty_0702b_trimmed|140:270" \
    "bagging_sku_fty_0706+0707|" \
    "bagging_sku_fty_0703_trimmed|" \
    >>"$PARALLEL_LOG_ROOT/gpu1.log" 2>&1 &
  pids+=("$!")

  label_worker 2 \
    "bagging_sku_fty_0702b_trimmed|270:400" \
    "bagging_sku_fty_0715_trimmed|" \
    "bagging_sku_fty_0713_dagger_filtered|" \
    "bagging_sku_zzc_0730_trimmed|" \
    >>"$PARALLEL_LOG_ROOT/gpu2.log" 2>&1 &
  pids+=("$!")

  label_worker 3 \
    "bagging_sku_fty_0702b_trimmed|400:529" \
    "bagging_sku_fty_0708_trimmed|" \
    "bagging_sku_fty_0714_trimmed|" \
    "bagging_sku_fty_0717_trimmed|" \
    >>"$PARALLEL_LOG_ROOT/gpu3.log" 2>&1 &
  pids+=("$!")

  printf '%s\n' "${pids[@]}" >"$PARALLEL_LOG_ROOT/worker_pids.txt"
  echo "Parallel label workers started: ${pids[*]}"
  echo "Logs: $PARALLEL_LOG_ROOT/gpu{0,1,2,3}.log"

  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "Label worker $pid failed; inspect logs in $PARALLEL_LOG_ROOT" >&2
      failed=1
    fi
  done
  if (( failed )); then
    return 1
  fi

  echo "Consolidating complete 0702b manifest"
  CUDA_VISIBLE_DEVICES=0 label_dataset bagging_sku_fty_0702b_trimmed ""
  touch "$PARALLEL_LOG_ROOT/COMPLETE"
}

build_dataset() {
  local dataset="$1"
  local output="$HF_LEROBOT_HOME/${dataset}_demospeedup_r2r4"
  if [[ -f "$output/meta/demospeedup.json" ]]; then
    echo "Skipping completed DemoSpeedup dataset: $output"
    validate_built_dataset "$dataset"
    return
  fi
  uv run examples/x2robot/demospeedup/build_dataset.py \
    --dataset "$dataset" \
    --label-root "$LABEL_ROOT" \
    --precision-stride 2 \
    --casual-stride 4 \
    --max-position-step 0.04 \
    --max-rotation-step 0.25 \
    --max-gripper-step 1.0
  validate_built_dataset "$dataset"
}

validate_built_dataset() {
  local dataset="$1"
  local output="$HF_LEROBOT_HOME/${dataset}_demospeedup_r2r4"
  local manifest="$output/meta/demospeedup.json"
  local info="$output/meta/info.json"
  local expected_episodes expected_frames expected_videos
  local parquet_count video_count episode_metadata_count

  jq -e \
    --arg source "$HF_LEROBOT_HOME/$dataset" \
    --arg output "$output" \
    '.schema_version == 1 and .method == "DemoSpeedup" and
     .source == $source and .output == $output and
     (.source_frames | type == "number") and
     (.output_frames | type == "number") and
     .output_frames > 0 and .output_frames <= .source_frames' \
    "$manifest" >/dev/null

  expected_episodes="$(jq -er '.total_episodes' "$info")"
  expected_frames="$(jq -er '.total_frames' "$info")"
  expected_videos="$(jq -er '.total_videos' "$info")"
  parquet_count="$(find "$output/data" -type f -name '*.parquet' | wc -l)"
  video_count="$(find "$output/videos" -type f -name '*.mp4' | wc -l)"
  episode_metadata_count="$(wc -l < "$output/meta/episodes.jsonl")"

  [[ "$(jq -er '.episodes | length' "$manifest")" -eq "$expected_episodes" ]]
  [[ "$(jq -er '.output_frames' "$manifest")" -eq "$expected_frames" ]]
  [[ "$parquet_count" -eq "$expected_episodes" ]]
  [[ "$episode_metadata_count" -eq "$expected_episodes" ]]
  [[ "$video_count" -eq "$expected_videos" ]]
  echo "Validated $dataset: episodes=$expected_episodes frames=$expected_frames videos=$expected_videos"
}

validate_all_built_datasets() {
  local dataset
  for dataset in "${DATASETS[@]}"; do
    validate_built_dataset "$dataset"
  done
}

build_worker() {
  local worker="$1"
  shift
  local dataset
  for dataset in "$@"; do
    echo "[$(date '+%F %T')] build worker $worker starting $dataset"
    build_dataset "$dataset"
  done
}

build_all_parallel() {
  mkdir -p "$BUILD_LOG_ROOT"
  local pids=()

  build_worker 0 \
    bagging_sku_fty_0702b_trimmed \
    >>"$BUILD_LOG_ROOT/worker0.log" 2>&1 &
  pids+=("$!")

  build_worker 1 \
    bagging_sku_zyx_0625 \
    bagging_sku_fty_0713_dagger_filtered \
    bagging_sku_zzc_0730_trimmed \
    bagging_sku_zzc_0805_dagger \
    bagging_sku_zzc_0810 \
    >>"$BUILD_LOG_ROOT/worker1.log" 2>&1 &
  pids+=("$!")

  build_worker 2 \
    bagging_sku_fty_0706+0707 \
    bagging_sku_fty_0715_trimmed \
    bagging_sku_fty_0703_trimmed \
    >>"$BUILD_LOG_ROOT/worker2.log" 2>&1 &
  pids+=("$!")

  build_worker 3 \
    bagging_sku_fty_0708_trimmed \
    bagging_sku_fty_0714_trimmed \
    bagging_sku_fty_0717_trimmed \
    bagging_sku_zzc_0724_trimmed \
    >>"$BUILD_LOG_ROOT/worker3.log" 2>&1 &
  pids+=("$!")

  printf '%s\n' "${pids[@]}" >"$BUILD_LOG_ROOT/worker_pids.txt"
  echo "Parallel build workers started: ${pids[*]}"
  echo "Logs: $BUILD_LOG_ROOT/worker{0,1,2,3}.log"

  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "Build worker $pid failed; inspect logs in $BUILD_LOG_ROOT" >&2
      failed=1
    fi
  done
  if (( failed )); then
    return 1
  fi
  touch "$BUILD_LOG_ROOT/COMPLETE"
}

train_policy() {
  export WANDB_PROJECT="${WANDB_PROJECT:-openpi}"
  export WANDB_ENTITY="${WANDB_ENTITY:-striding-algo}"

  local mode="--overwrite"
  local checkpoint
  if [[ -d "$CHECKPOINT_DIR" ]]; then
    for checkpoint in "$CHECKPOINT_DIR"/[1-9]*; do
      if [[ -d "$checkpoint" && "${checkpoint##*/}" =~ ^[0-9]+$ && -f "$checkpoint/_CHECKPOINT_METADATA" ]]; then
        mode="--resume"
        break
      fi
    done
  fi

  echo "Training $CONFIG_NAME with $mode (checkpoint_dir=$CHECKPOINT_DIR)"
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "$CONFIG_NAME" "$mode"
  mkdir -p "$PIPELINE_LOG_ROOT"
  touch "$PIPELINE_LOG_ROOT/TRAIN_COMPLETE"
}

compute_norm_stats() {
  validate_all_built_datasets
  uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME" --num-workers "$NORM_NUM_WORKERS"
  mkdir -p "$PIPELINE_LOG_ROOT"
  touch "$PIPELINE_LOG_ROOT/NORM_COMPLETE"
}

case "$STAGE" in
  label)
    for dataset in "${DATASETS[@]}"; do label_dataset "$dataset"; done
    ;;
  label-parallel)
    label_all_parallel
    ;;
  build)
    for dataset in "${DATASETS[@]}"; do build_dataset "$dataset"; done
    ;;
  build-parallel)
    build_all_parallel
    ;;
  norm)
    compute_norm_stats
    ;;
  train)
    train_policy
    ;;
  all)
    for dataset in "${DATASETS[@]}"; do label_dataset "$dataset"; done
    for dataset in "${DATASETS[@]}"; do build_dataset "$dataset"; done
    compute_norm_stats
    train_policy
    ;;
  *)
    echo "Usage: $0 {label|label-parallel|build|build-parallel|norm|train|all}" >&2
    exit 2
    ;;
esac
