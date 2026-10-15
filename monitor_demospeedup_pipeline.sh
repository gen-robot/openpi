#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/mnt/public/gaojiahuan/openpi"
readonly PIPELINE_SESSION="demospeedup_pipeline"
readonly MONITOR_LOG_ROOT="$ROOT/logs/demospeedup_pipeline"
readonly MONITOR_LOG="$MONITOR_LOG_ROOT/watchdog.log"
readonly PIPELINE_LOG="$MONITOR_LOG_ROOT/pipeline.log"
readonly BUILD_COMPLETE="$ROOT/logs/demospeedup_build/COMPLETE"
readonly NORM_COMPLETE="$MONITOR_LOG_ROOT/NORM_COMPLETE"
readonly TRAIN_COMPLETE="$MONITOR_LOG_ROOT/TRAIN_COMPLETE"
readonly CHECK_INTERVAL_SECONDS="${DEMOSPEEDUP_CHECK_INTERVAL_SECONDS:-7200}"
readonly STALE_LOG_SECONDS="${DEMOSPEEDUP_STALE_LOG_SECONDS:-3600}"
readonly STALE_RECHECK_SECONDS="${DEMOSPEEDUP_STALE_RECHECK_SECONDS:-60}"
readonly CONFIG_NAME="bagging_sku_0811_demospeedup_sm2sm"
readonly CHECKPOINT_DIR="$ROOT/checkpoints/$CONFIG_NAME/bagging_sku_0811_demospeedup_r2r4_dm10dh50df50po20_meta_ah20"

mkdir -p "$MONITOR_LOG_ROOT"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "$MONITOR_LOG"
}

training_pids() {
  pgrep -f "^$ROOT/.venv/bin/python3 scripts/train.py $CONFIG_NAME( |$)" || true
}

latest_checkpoint() {
  local checkpoint latest=-1
  if [[ -d "$CHECKPOINT_DIR" ]]; then
    for checkpoint in "$CHECKPOINT_DIR"/[0-9]*; do
      if [[ -d "$checkpoint" && "${checkpoint##*/}" =~ ^[0-9]+$ && -f "$checkpoint/_CHECKPOINT_METADATA" ]]; then
        if (( ${checkpoint##*/} > latest )); then
          latest="${checkpoint##*/}"
        fi
      fi
    done
  fi
  echo "$latest"
}

stop_pipeline() {
  local pid
  if tmux has-session -t "$PIPELINE_SESSION" 2>/dev/null; then
    tmux kill-session -t "$PIPELINE_SESSION"
  fi

  # tmux normally terminates the whole pipeline process group. Only target the
  # exact training command if a wedged child survives the session shutdown.
  for _ in {1..30}; do
    [[ -z "$(training_pids)" ]] && return 0
    sleep 1
  done
  while read -r pid; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done < <(training_pids)
  for _ in {1..30}; do
    [[ -z "$(training_pids)" ]] && return 0
    sleep 1
  done
  log "Training process did not stop after SIGTERM; refusing to start a duplicate process."
  return 1
}

start_pipeline() {
  local stages
  if [[ ! -f "$BUILD_COMPLETE" ]]; then
    stages="build-parallel && bash run_demospeedup_0811.sh norm && bash run_demospeedup_0811.sh train"
    stages="bash run_demospeedup_0811.sh $stages"
  elif [[ ! -f "$NORM_COMPLETE" ]]; then
    stages="bash run_demospeedup_0811.sh norm && bash run_demospeedup_0811.sh train"
  else
    stages="bash run_demospeedup_0811.sh train"
  fi

  log "Restarting pipeline from the earliest incomplete stage: $stages"
  tmux new-session -d -s "$PIPELINE_SESSION" \
    "cd '$ROOT' && set -o pipefail; { $stages; } 2>&1 | tee -a '$PIPELINE_LOG'"
}

restart_pipeline() {
  local reason="$1"
  log "$reason"
  stop_pipeline
  start_pipeline
}

check_training_health() {
  local pid log_age first_size second_size checkpoint
  pid="$(training_pids | head -n 1)"
  if [[ -z "$pid" ]]; then
    # Avoid mistaking a short stage transition for failure.
    sleep 30
    pid="$(training_pids | head -n 1)"
    if [[ -z "$pid" && ! -f "$TRAIN_COMPLETE" ]]; then
      restart_pipeline "Training process is absent before completion; recovering from the latest checkpoint."
      return
    fi
  fi

  [[ -f "$TRAIN_COMPLETE" ]] && return
  checkpoint="$(latest_checkpoint)"
  if [[ ! -f "$PIPELINE_LOG" ]]; then
    log "Training PID $pid is running; pipeline log does not exist yet; latest checkpoint=$checkpoint."
    return
  fi

  log_age=$(( $(date +%s) - $(stat -c %Y "$PIPELINE_LOG") ))
  if (( log_age <= STALE_LOG_SECONDS )); then
    log "Training healthy: pid=$pid log_age=${log_age}s latest_checkpoint=$checkpoint."
    return
  fi

  # A stale timestamp alone is not enough to restart. Re-poll the same live
  # process and log after one minute to rule out a transient save/compile pause.
  first_size="$(stat -c %s "$PIPELINE_LOG")"
  log "Training log is ${log_age}s old; rechecking pid=$pid and log progress in ${STALE_RECHECK_SECONDS}s."
  sleep "$STALE_RECHECK_SECONDS"
  second_size="$(stat -c %s "$PIPELINE_LOG" 2>/dev/null || echo -1)"
  if kill -0 "$pid" 2>/dev/null && [[ "$second_size" != "$first_size" ]]; then
    log "Training resumed log output during recheck; no restart needed."
    return
  fi
  restart_pipeline "Training is confirmed stalled (pid=$pid, unchanged log); recovering from checkpoint $checkpoint."
}

while true; do
  if [[ -f "$TRAIN_COMPLETE" ]]; then
    log "Training completion marker found; watchdog exiting."
    exit 0
  fi

  if tmux has-session -t "$PIPELINE_SESSION" 2>/dev/null; then
    if [[ -f "$NORM_COMPLETE" ]]; then
      check_training_health
    else
      log "Pre-training pipeline stage is running."
    fi
  else
    log "Pipeline session is not running before completion."
    start_pipeline
  fi

  sleep "$CHECK_INTERVAL_SECONDS"
done
