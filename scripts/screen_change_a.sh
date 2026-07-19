#!/bin/bash
# Fast ablation screen for Change A (temporal action encoder), warm-started from an existing
# checkpoint. Short schedule + light validation so a variant reaches "does it help?" signal in
# well under an hour. Change A is wired as a residual + zero-init no-op, so at step 0 the model
# is identical to the warm-start checkpoint and only learns the delta.
#
# Usage:
#   # baseline (no temporal encoder) — for the probe's reference numbers:
#   USE_TEMPORAL_ACTION_ENCODER=0 bash scripts/screen_change_a.sh
#   # Change A on:
#   USE_TEMPORAL_ACTION_ENCODER=1 bash scripts/screen_change_a.sh
#
# Override any knob inline, e.g.  GPUS=0,1 MAX_TRAIN_STEPS=1500 bash scripts/screen_change_a.sh
set -e
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World
source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh
conda activate ctrl-world

# ---- screening knobs (all overridable from the environment) ----
export GPUS="${GPUS:-0,1}"                                   # 2 GPUs frees the rest
export USE_TEMPORAL_ACTION_ENCODER="${USE_TEMPORAL_ACTION_ENCODER:-1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1500}"            # warm-started delta converges fast
export VALIDATION_STEPS="${VALIDATION_STEPS:-500}"          # peek 3x during the run
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"    # ckpts to probe at 500/1000/1500
export VIDEO_NUM="${VIDEO_NUM:-3}"                          # 3 val videos, not 20 (big time saver)
# wandb: online by default (creds already in ~/.netrc). Set WANDB_MODE=offline for no sync,
# or WANDB_PROJECT to route screening runs to their own project.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-ctrl-world-screen}"

WARM_CKPT="${WARM_CKPT:-checkpoints/ctrl-world/checkpoint-10000.pt}"
DATASET="${DATASET:-robocasa_opendrawer_full}"

NPROC=$(echo "$GPUS" | awk -F',' '{print NF}')
VARIANT=$([ "$USE_TEMPORAL_ACTION_ENCODER" = "1" ] && echo "changeA" || echo "baseline")
export RUN_TAG="${RUN_TAG:-screen_${VARIANT}_$(date +%Y%m%d_%H%M%S)}"

echo ">>> RUN_TAG=$RUN_TAG  variant=$VARIANT  gpus=$GPUS (nproc=$NPROC)"
echo ">>> warm start: $WARM_CKPT   dataset: $DATASET"
echo ">>> steps=$MAX_TRAIN_STEPS val_every=$VALIDATION_STEPS ckpt_every=$CHECKPOINTING_STEPS video_num=$VIDEO_NUM"

echo ">>> wandb: mode=$WANDB_MODE project=$WANDB_PROJECT run=$RUN_TAG"
CUDA_VISIBLE_DEVICES="$GPUS" SWANLAB_MODE=disabled accelerate launch \
  --multi_gpu --num_processes "$NPROC" --num_machines 1 --mixed_precision fp16 \
  --main_process_port "${PORT:-29511}" \
  scripts/train_wm.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names "$DATASET" \
  --svd_model_path checkpoints/svd \
  --clip_model_path checkpoints/clip \
  --ckpt_path "$WARM_CKPT"

echo ">>> done. probe each checkpoint with:"
echo "    python scripts/action_sensitivity_probe.py --ckpt_path model_ckpt/$RUN_TAG/checkpoint-1500.pt --dataset_names $DATASET --out model_ckpt/$RUN_TAG/probe_1500.json"
