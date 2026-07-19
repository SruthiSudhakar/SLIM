#!/bin/bash
# Sequentially train LoRA on robocasa_opendrawer_target at rank 64, 128, 256 -- the rank sweep
# counterpart of the existing rank-16 run (model_ckpt/fewshot_opendrawer_lora). Matches that r16
# setup: 3000 steps, ckpt every 250, from the 9-task base, lr 1e-4, alpha==rank (scale 1.0).
# Effective batch 32 (batch 2 x accum 2 x 8 gpu) == the r16 run's batch-4 x 8, but lighter per-step
# memory so high ranks fit. Runs continue even if one fails.
# Outputs -> model_ckpt/fewshot_opendrawer_lora_r{RANK}/ ;  logs -> logs/sweep_opendrawer_lora_ranks/

set -u
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World
source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh
conda activate ctrl-world

LOG_DIR=logs/sweep_opendrawer_lora_ranks
mkdir -p "$LOG_DIR"

RANKS=(64 128 256)
SWEEP_START=$(date +%s)

for RANK in "${RANKS[@]}"; do
  RUN_TAG="fewshot_opendrawer_lora_r${RANK}"
  LOG_FILE="$LOG_DIR/r${RANK}.log"
  echo ""
  echo "=========================================================="
  echo ">>> [$(date '+%F %T')] START rank=$RANK  tag=$RUN_TAG"
  echo ">>> outputs -> model_ckpt/$RUN_TAG"
  echo ">>> log     -> $LOG_FILE"
  echo "=========================================================="

  START=$(date +%s)
  RUN_TAG="$RUN_TAG" \
  MAX_TRAIN_STEPS=3000 CHECKPOINTING_STEPS=250 VALIDATION_STEPS=1000 \
  TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=2 \
  USE_LORA=1 LORA_RANK="$RANK" LORA_ALPHA="$RANK" LEARNING_RATE=1e-4 \
  WANDB_MODE=online SWANLAB_MODE=disabled WANDB_PROJECT=fewshot_opendrawer \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --multi_gpu --num_processes 8 --num_machines 1 --mixed_precision fp16 \
    --main_process_port 29501 \
    scripts/train_wm.py \
    --dataset_root_path dataset_example \
    --dataset_meta_info_path dataset_meta_info \
    --dataset_names robocasa_opendrawer_target \
    --svd_model_path checkpoints/svd \
    --clip_model_path checkpoints/clip \
    --ckpt_path model_ckpt/robocasa_9task_base/checkpoint-30000.pt \
    >"$LOG_FILE" 2>&1
  STATUS=$?
  END=$(date +%s)
  ELAPSED=$((END - START))
  H=$((ELAPSED/3600)); M=$(((ELAPSED%3600)/60)); S=$((ELAPSED%60))

  if [ $STATUS -eq 0 ]; then
    echo ">>> [$(date '+%F %T')] DONE  rank=$RANK  status=OK   elapsed=${H}h${M}m${S}s"
  else
    echo ">>> [$(date '+%F %T')] DONE  rank=$RANK  status=FAIL(exit=$STATUS)  elapsed=${H}h${M}m${S}s"
    echo ">>> continuing to next rank; see $LOG_FILE for details"
  fi
done

SWEEP_END=$(date +%s)
TOTAL=$((SWEEP_END - SWEEP_START))
TH=$((TOTAL/3600)); TM=$(((TOTAL%3600)/60)); TS=$((TOTAL%60))
echo ""
echo ">>> [$(date '+%F %T')] SWEEP COMPLETE  total_elapsed=${TH}h${TM}m${TS}s"
