#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-cbs_full_$(date +%Y%m%d_%H%M%S)}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SEEDS="${SEEDS:-101 202 303}"
N_ENV="${N_ENV:-32}"
N_EVAL_ENV="${N_EVAL_ENV:-5}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/paper/rl_training_logs/$RUN_ID"
mkdir -p "$LOG_DIR"

SESSION_NAME="${RUN_ID//[^A-Za-z0-9_]/_}"
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION_NAME" \
    "cd '$ROOT_DIR' && env RUN_ID='$RUN_ID' MAX_PARALLEL='$MAX_PARALLEL' SEEDS='$SEEDS' N_ENV='$N_ENV' N_EVAL_ENV='$N_EVAL_ENV' bash '$ROOT_DIR/scripts/train_cbs_paper_experiments.sh' > '$LOG_DIR/master.log' 2>&1"
  echo "$SESSION_NAME" > "$LOG_DIR/master.tmux"
  pid="$(tmux display-message -p -t "$SESSION_NAME" '#{pid}')"
else
  setsid -f env \
    RUN_ID="$RUN_ID" \
    MAX_PARALLEL="$MAX_PARALLEL" \
    SEEDS="$SEEDS" \
    N_ENV="$N_ENV" \
    N_EVAL_ENV="$N_EVAL_ENV" \
    bash "$ROOT_DIR/scripts/train_cbs_paper_experiments.sh" \
    > "$LOG_DIR/master.log" 2>&1
  sleep 1
  pid="$(pgrep -af "RUN_ID=$RUN_ID|train_cbs_paper_experiments.sh" | head -1 | awk '{print $1}')"
fi

echo "$pid" > "$LOG_DIR/master.pid"
echo "Started CBS training queue"
echo "PID: $pid"
echo "Run ID: $RUN_ID"
echo "tmux session: ${SESSION_NAME:-none}"
echo "Master log: $LOG_DIR/master.log"
echo "PID file: $LOG_DIR/master.pid"
