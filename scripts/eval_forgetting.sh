#!/bin/bash
# Experiment A (forgetting): eval ONE checkpoint on the 8 OTHER tasks the base was trained on
# (atomic 9-task set minus PickPlaceSinkToCounter, all success), so we can compare a ppsink-failure
# finetune against the base and see whether it degraded unrelated tasks.
#
# Fixed seed/gen_seed => every model sees the SAME start frames + diffusion noise (paired -> the
# base-vs-finetune delta is low-variance). atomic normalization; matches the ppsink eval protocol
# (15 steps, interact_num 6, random start).
#
# Usage:  bash scripts/eval_forgetting.sh <ckpt> <gpu> [n_success]
#   e.g.  bash scripts/eval_forgetting.sh model_ckpt/robocasa_9task_base/checkpoint-30000.pt 0
# Run the 4 models on 4 different GPUs in parallel; aggregate with scripts/aggregate_forgetting.py.
set -u
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World

CKPT="${1:?usage: eval_forgetting.sh <ckpt> <gpu> [n_success]}"
GPU="${2:?need gpu index}"
N="${3:-15}"                                  # success episodes per task (no failures in atomic)
STEPS="${NUM_INFERENCE_STEPS:-15}"
STAT=dataset_meta_info/robocasa_atomic_all/stat.json

TASKS=(CloseFridge CloseBlenderLid CoffeeSetupMug PickPlaceCounterToCabinet \
       PickPlaceToasterToCounter TurnOffStove TurnOnMicrowave TurnOnSinkFaucet)

echo ">>> forgetting eval of $CKPT on ${#TASKS[@]} other tasks (gpu $GPU, $N succ/task, $STEPS steps)"
for T in "${TASKS[@]}"; do
  echo "--- $T"
  NUM_INFERENCE_STEPS="$STEPS" bash scripts/eval_ckpt.sh "$CKPT" "$GPU" \
    --dataset robocasa_atomic_all \
    --select_by_success --task "$T" --n_success "$N" --n_fail 0 \
    --interact_num 6 --start_idx random --seed 42 --gen_seed 42 \
    --data_stat "$STAT"
done
echo ">>> done $CKPT. Aggregate with scripts/aggregate_forgetting.py"
