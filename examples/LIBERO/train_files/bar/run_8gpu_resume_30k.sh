#!/usr/bin/env bash
# 8-GPU single-node WanDit4DiT resume from step 30k (run_id 20260623_023830).
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
STARVLA_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

export WAM_EXP_BENCHMARK=libero
export WAM_EXP_SUITE=all
export WAM_EXP_FRAMEWORK=wan_dit4dit
export WAM_RUN_ID=20260623_023830
export WAM_ROOT=/SSD_DISK_1/users/wuruihan/WAM
export NUM_PROCESSES=8
export PER_DEVICE_BS=1
export GRAD_ACCUM=1
export MAX_STEPS=80000
export SAVE_INTERVAL=10000

export WAM_WORK_ROOT="${WAM_ROOT}/work_dirs"
export run_root_dir="${WAM_WORK_ROOT}/libero/all/wan_dit4dit"
export run_id="${WAM_RUN_ID}"
output_dir="${run_root_dir}/${run_id}"
train_log="${output_dir}/train.log"

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${output_dir}/scripts"
cp "${SCRIPT_PATH}" "${output_dir}/scripts/"

ACCEL_CONFIG=$(python3 examples/Gemma4/_make_accelerate_config.py \
  --grad-accum "${GRAD_ACCUM}" \
  --num-processes "${NUM_PROCESSES}" \
  --zero-stage 2 \
  --out-dir "${WAM_WORK_ROOT}/cache/accelerate_configs")

echo "[8gpu-resume] run_id=${WAM_RUN_ID} suite=${WAM_EXP_SUITE}"
echo "[8gpu-resume] resume from ${output_dir}/checkpoints/steps_30000_pytorch_model.pt"
echo "[8gpu-resume] GPUs=${NUM_PROCESSES} grad_accum=${GRAD_ACCUM} log=${train_log}"
echo "[8gpu-resume] accelerate_config=${ACCEL_CONFIG}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate starVLA

set -o pipefail
accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_wan_dit4dit_libero.yaml \
  --framework.name WanDit4DiT \
  --framework.world_model.base_wm "${WAM_ROOT}/hugg_data/Wan2.2-TI2V-5B-Diffusers" \
  --framework.qwenvl.base_vlm "${WAM_ROOT}/hugg_data/Wan2.2-TI2V-5B-Diffusers" \
  --datasets.vla_data.data_root_dir "${WAM_ROOT}/hugg_data/LIBERO-Lerobot-IPEC" \
  --datasets.vla_data.data_mix libero_all_dit4dit \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BS}" \
  --trainer.freeze_modules backbone.vae,backbone.text_encoder \
  --trainer.learning_rate.base 1.0e-05 \
  --trainer.learning_rate.backbone 1.0e-05 \
  --trainer.learning_rate.action_model 1.0e-04 \
  --trainer.learning_rate.wm_projector 1.0e-04 \
  --trainer.loss_scale.video 0.1 \
  --framework.video_loss_weight 0.1 \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 500 \
  --trainer.is_resume true \
  --run_root_dir "${run_root_dir}" \
  --run_id "'${run_id}'" \
  2>&1 | tee -a "${train_log}"
