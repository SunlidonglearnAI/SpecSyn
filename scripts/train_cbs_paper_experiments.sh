#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS_DIR="$ROOT_DIR/myosuite/agents"
RUN_ID="${RUN_ID:-cbs_rerun_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT_DIR/paper/rl_training_logs/$RUN_ID"
OUTPUT_ROOT="$AGENTS_DIR/outputs_cbs/$RUN_ID"
SEEDS="${SEEDS:-101 202 303}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
CONDA_ENV="${CONDA_ENV:-myosuite}"
N_ENV="${N_ENV:-32}"
N_EVAL_ENV="${N_EVAL_ENV:-5}"
FULL_TIMESTEPS="${FULL_TIMESTEPS:-100000000}"
FIXED_TIMESTEPS="${FIXED_TIMESTEPS:-100000000}"
T2A_TIMESTEPS="${T2A_TIMESTEPS:-100000000}"

mkdir -p "$LOG_DIR"

run_job() {
  local group="$1"
  local config="$2"
  local env_id="$3"
  local seed="$4"
  local timesteps="$5"
  local job_id="${group}__${env_id}__seed_${seed}"
  local log_file="$LOG_DIR/${job_id}.log"
  local output_dir="$OUTPUT_ROOT/$group/$env_id/seed_${seed}"

  (
    cd "$AGENTS_DIR" || exit 1
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    export MUJOCO_GL="${MUJOCO_GL:-glfw}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_${USER}}"
    export WANDB_MODE=disabled
    mkdir -p "$MPLCONFIGDIR"
    echo "[$(date --iso-8601=seconds)] START $job_id"
    echo "output_dir=$output_dir"
    conda run -n "$CONDA_ENV" python hydra_sb3_launcher.py \
      --config-path config \
      --config-name "$config" \
      "hydra.run.dir=$output_dir" \
      "env=$env_id" \
      "seed=$seed" \
      "n_env=$N_ENV" \
      "n_eval_env=$N_EVAL_ENV" \
      "total_timesteps=$timesteps" \
      "+use_wandb=false"
    status=$?
    echo "[$(date --iso-8601=seconds)] END $job_id status=$status"
    exit "$status"
  ) >"$log_file" 2>&1
}

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 30
  done
}

submit() {
  local group="$1"
  local config="$2"
  local env_id="$3"
  local timesteps="$4"
  local seed
  for seed in $SEEDS; do
    wait_for_slot
    run_job "$group" "$config" "$env_id" "$seed" "$timesteps" &
    echo "submitted ${group} ${env_id} seed=${seed}"
  done
}

echo "Run ID: $RUN_ID"
echo "Logs: $LOG_DIR"
echo "Outputs: $OUTPUT_ROOT"
echo "Seeds: $SEEDS"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo "N_ENV: $N_ENV"
echo "N_EVAL_ENV: $N_EVAL_ENV"

# Main paper comparison: fixed morphology, direct high-dimensional T2A, and Full SDE.
submit "fixed" "hydra_myo_sb3_ppo_config.yaml" "myoLegWalk-v0" "$FIXED_TIMESTEPS"
submit "fixed" "hydra_myo_sb3_ppo_config.yaml" "myoLegRoughTerrainWalk-v0" "$FIXED_TIMESTEPS"
submit "fixed" "hydra_myo_sb3_ppo_config.yaml" "myoLegStairTerrainWalk-v0" "$FIXED_TIMESTEPS"
submit "fixed" "hydra_myo_sb3_ppo_config.yaml" "myoLegHillyTerrainWalk-v0" "$FIXED_TIMESTEPS"

submit "direct_t2a" "hydra_myo_sb3_ppo_config_t2a1.yaml" "myoLegWalkT2A-v0" "$T2A_TIMESTEPS"
submit "direct_t2a" "hydra_myo_sb3_ppo_config_t2a1.yaml" "myoLegRoughTerrainT2A-v0" "$T2A_TIMESTEPS"
submit "direct_t2a" "hydra_myo_sb3_ppo_config_t2a1.yaml" "myoLegStairTerrainT2A-v0" "$T2A_TIMESTEPS"
submit "direct_t2a" "hydra_myo_sb3_ppo_config_t2a1.yaml" "myoLegHillyTerrainWalkT2A-v0" "$T2A_TIMESTEPS"

submit "full_sde" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegWalkT2Apca1-v0" "$FULL_TIMESTEPS"
submit "full_sde" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegRoughTerrainT2Apca1-v0" "$FULL_TIMESTEPS"
submit "full_sde" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegStairTerrainT2Apca1-v0" "$FULL_TIMESTEPS"
submit "full_sde" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegHillyTerrainWalkT2Apca1-v0" "$FULL_TIMESTEPS"

# Single-property ablations on flat walking.
submit "single_property" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegWalkT2AStrength-v0" "$FULL_TIMESTEPS"
submit "single_property" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegWalkT2AVelocity-v0" "$FULL_TIMESTEPS"
submit "single_property" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegWalkT2AStiffness-v0" "$FULL_TIMESTEPS"

# Symmetry ablation.
submit "symmetry" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegWalkT2Apca0-v0" "$FULL_TIMESTEPS"
submit "symmetry" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegRoughTerrainT2Apca0-v0" "$FULL_TIMESTEPS"

# Latent dimension ablation on flat walking.
submit "latent_k" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegWalkT2ApcaK3-v0" "$FULL_TIMESTEPS"
submit "latent_k" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegWalkT2ApcaK5-v0" "$FULL_TIMESTEPS"
submit "latent_k" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegWalkT2ApcaK7-v0" "$FULL_TIMESTEPS"
submit "latent_k" "hydra_myo_sb3_ppo_config_t2a.yaml" "myoLegWalkT2ApcaK9-v0" "$FULL_TIMESTEPS"

wait
echo "All submitted jobs completed at $(date --iso-8601=seconds)"
