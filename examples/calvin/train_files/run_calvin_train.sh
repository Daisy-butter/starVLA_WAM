# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# # used for check save when communication
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
# export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI
freeze_module_list=''
base_vlm=/SSD_DISK/users/wuruihan/starVLA/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action
config_yaml=./examples/calvin/train_files/starvla_train_calvin.yaml
DIT_TYPE="DiT-B"
calvin_data_root=${CALVIN_BASE:-/SSD_DISK/users/wuruihan/sii_starvla/calvin_debug_small_test_dataset}
# Official ~1.3GB debug split (smoke / baseline). Full ABC→D: use calvin_abc_d + calvin_task_ABC_D.
data_mix=calvin_debug_smoke
run_root_dir=./results/Checkpoints
run_id=0118_starvla_qwenpi_calvin_debug_smoke
export action_input_dim=2048
# === End of environment variable configuration ===
###########################################################################################


# 不连 wandb、不上传：取消下一行注释即可（无需改 entity）
export WANDB_MODE=disabled

# ZeRO-2: shards optimizer/grads only — each GPU still holds a full model copy (often OOM on 24GB + 3B VLA).
# ZeRO-3: shards parameters across GPUs — use for RTX 3090 / 24GB Calvin+QwenPI smoke runs.
DEEPSPEED_ACCEL_CONFIG="${DEEPSPEED_ACCEL_CONFIG:-starVLA/config/deepseeds/deepspeed_zero3.yaml}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file "${DEEPSPEED_ACCEL_CONFIG}" \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.enable_gradient_checkpointing true \
  --framework.action_model.enable_dit_gradient_checkpointing true \
  --framework.action_model.repeated_diffusion_steps 1 \
  --datasets.vla_data.data_root_dir ${calvin_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Calvin \
  --wandb_entity "22307140084" \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
