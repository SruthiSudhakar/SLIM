#!/bin/bash
# Post-train Ctrl-World on RoboCasa OpenDrawer, resuming from the pretrained DROID checkpoint.
# 8-GPU accelerate run. Schedule (max 30k steps, ckpt 5k, val 1k, tag robocasa_opendrawer) is in config.py.
set -e
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World
source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh
conda activate ctrl-world

# Unique per-run output dir so restarts never overwrite previous runs' checkpoints/samples.
# Computed ONCE here and exported so all 8 accelerate processes share the same value.
# Override by exporting RUN_TAG yourself before calling this script.
export RUN_TAG="${RUN_TAG:-robocasa_opendrawer_$(date +%Y%m%d_%H%M%S)}"
echo ">>> RUN_TAG=$RUN_TAG   (outputs -> model_ckpt/$RUN_TAG)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=online SWANLAB_MODE=disabled accelerate launch \
  --multi_gpu --num_processes 8 --num_machines 1 --mixed_precision fp16 \
  --main_process_port 29501 \
  scripts/train_wm.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names robocasa_opendrawer_full \
  --svd_model_path checkpoints/svd \
  --clip_model_path checkpoints/clip \
  --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt
