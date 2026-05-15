#!/bin/bash
# Local LIBERO eval for Wan joint MoE (HF checkpoint). Single terminal — no policy server.
set -euo pipefail

STARVLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${STARVLA_DIR}"

export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

# === Paths (this cluster) ===
CKPT_DIR="${CKPT_DIR:-/SSD_DISK/users/wuruihan/sii_starvla_ckpt/ckpt_debug_libero}"
WAN_ROOT="${WAN_ROOT:-/SSD_DISK/users/wuruihan/sii_starvla_ckpt/hugg_model/Wan2.2-TI2V-5B-Diffusers}"
# Set STATS_JSON to your training run's dataset_statistics.json for real success rates.
STATS_JSON="${STATS_JSON:-}"

# === LIBERO env ===
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export PYTHONPATH="${PYTHONPATH}:${LIBERO_HOME}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
TASK_SUITE="${TASK_SUITE:-libero_goal}"
NUM_TRIALS="${NUM_TRIALS:-5}"
VIDEO_OUT="${VIDEO_OUT:-${CKPT_DIR}/libero_eval_local/${TASK_SUITE}}"

mkdir -p "${VIDEO_OUT}"

EXTRA_ARGS=()
if [[ -n "${STATS_JSON}" ]]; then
  EXTRA_ARGS+=(--stats-json "${STATS_JSON}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" examples/LIBERO/eval_files/eval_libero_wan_joint_local.py \
  --ckpt-dir "${CKPT_DIR}" \
  --wan-root "${WAN_ROOT}" \
  --task-suite "${TASK_SUITE}" \
  --num-trials "${NUM_TRIALS}" \
  --video-out "${VIDEO_OUT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
