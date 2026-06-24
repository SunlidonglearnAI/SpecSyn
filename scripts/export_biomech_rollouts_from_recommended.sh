#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/paper/biomech_rollouts_recommended}"
EPISODES="${EPISODES:-5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MAX_STEPS="${MAX_STEPS:-2000}"
DEVICE="${DEVICE:-cpu}"
CONDA_ENV="${CONDA_ENV:-myosuite}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_${USER:-myosuite}}"

cd "$ROOT_DIR"

run_validation() {
  local terrain="$1"
  shift
  conda run -n "$CONDA_ENV" python biomech_validate_success.py \
    "$@" \
    --episodes "$EPISODES" \
    --seeds $SEEDS \
    --max-steps "$MAX_STEPS" \
    --device "$DEVICE" \
    --terrain "$terrain" \
    --reset-type random \
    --filter-quality \
    --min-frames 80 \
    --min-distance 0.25 \
    --min-steps 2 \
    --min-speed 0.05 \
    --return-mad-threshold 3.0 \
    --output-dir "$OUT_DIR/$terrain"
}

# Walk-task biomechanical core: baseline, direct T2A, Full SDE, and single-property ablations.
run_validation "flat" \
  --condition "Fixed,myoLegWalk-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/fixed/myoLegWalk-v0/recommended" \
  --condition "T2A,myoLegWalkT2A-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/direct_t2a/myoLegWalkT2A-v0/recommended" \
  --condition "FullSDE,myoLegWalkT2Apca1-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/full_sde/myoLegWalkT2Apca1-v0/recommended" \
  --condition "StrengthOnly,myoLegWalkT2AStrength-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/single_property/myoLegWalkT2AStrength-v0/recommended" \
  --condition "VelocityOnly,myoLegWalkT2AVelocity-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/single_property/myoLegWalkT2AVelocity-v0/recommended" \
  --condition "StiffnessOnly,myoLegWalkT2AStiffness-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/single_property/myoLegWalkT2AStiffness-v0/recommended" \
  --condition "AsymSDE,myoLegWalkT2Apca0-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/symmetry/myoLegWalkT2Apca0-v0/recommended"

# Rough-task validation: enough to show terrain robustness without rerunning all 63 RL jobs.
run_validation "rough" \
  --condition "T2A,myoLegRoughTerrainT2A-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/direct_t2a/myoLegRoughTerrainT2A-v0/recommended" \
  --condition "FullSDE,myoLegRoughTerrainT2Apca1-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/full_sde/myoLegRoughTerrainT2Apca1-v0/recommended" \
  --condition "AsymSDE,myoLegRoughTerrainT2Apca0-v0,$ROOT_DIR/myosuite/agents/outputs_by_category/symmetry/myoLegRoughTerrainT2Apca0-v0/recommended"

python scripts/segment_gait_cycles.py "$OUT_DIR" --output-dir "$OUT_DIR/gait_cycles"

echo "Biomechanical rollout export complete: $OUT_DIR"
