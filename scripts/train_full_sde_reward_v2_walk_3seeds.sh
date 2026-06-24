#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS_DIR="$ROOT_DIR/myosuite/agents"
PYTHON_BIN="${PYTHON_BIN:-/home/fzh/anaconda3/envs/myosuite/bin/python}"
ENV_ID="${ENV_ID:-myoLegWalkT2Apca1RewardV2-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
N_ENV="${N_ENV:-32}"
N_EVAL_ENV="${N_EVAL_ENV:-5}"
SEEDS=(${SEEDS:-123 234 345})
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_${USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${USER}}"

mkdir -p "$ROOT_DIR/paper/training_logs/full_sde_reward_v2_walk"

for seed in "${SEEDS[@]}"; do
  log_file="$ROOT_DIR/paper/training_logs/full_sde_reward_v2_walk/seed_${seed}.log"
  session_name="full_sde_rv2_walk_${seed}"
  tmux kill-session -t "$session_name" >/dev/null 2>&1 || true
  tmux new-session -d -s "$session_name" "cd '$AGENTS_DIR' && exec '$PYTHON_BIN' hydra_sb3_launcher.py \
    --config-name hydra_myo_sb3_ppo_config_t2a \
    env='$ENV_ID' \
    seed='$seed' \
    n_env='$N_ENV' \
    n_eval_env='$N_EVAL_ENV' \
    total_timesteps='$TOTAL_TIMESTEPS' \
    job_name='full_sde_reward_v2_walk_seed_${seed}' \
    hydra.run.dir='./outputs/full_sde_reward_v2_walk/${RUN_TAG}/seed_${seed}' \
    hydra.sweep.dir='./outputs/full_sde_reward_v2_walk/${RUN_TAG}/seed_${seed}' >'$log_file' 2>&1"
  echo "seed=${seed} tmux_session=${session_name} run_tag=${RUN_TAG} log=${log_file}"
done
