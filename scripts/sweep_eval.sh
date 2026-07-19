#!/bin/bash
# Evaluate every checkpoint-*.pt in a run dir on the held-out OpenDrawer val set, so we can plot
# quality vs. finetuning step. Fixed seeds -> paired, comparable rollouts across steps and arms.
#
# Usage:
#   bash scripts/sweep_eval.sh <run_dir> <gpu> [num_traj]
# e.g.
#   bash scripts/sweep_eval.sh model_ckpt/fewshot_opendrawer_fullft 0
#   bash scripts/sweep_eval.sh model_ckpt/fewshot_opendrawer_lora   1
# Include the step-0 zero-shot base as the shared origin of the curve:
#   bash scripts/sweep_eval.sh model_ckpt/robocasa_9task_base 2 10   # eval its checkpoint-30000 as "base"
#
# Each checkpoint's metrics land in <ckpt>_eval_<ts>/rollout_metrics.json (see eval_ckpt.sh).
set -e
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World

RUN_DIR="${1:?usage: sweep_eval.sh <run_dir> <gpu> [num_traj]}"
GPU="${2:?need gpu index}"
N="${3:-10}"
DS="${EVAL_DATASET:-robocasa_opendrawer_target}"
SEED="${SEED:-42}"
GEN_SEED="${GEN_SEED:-42}"

shopt -s nullglob
CKPTS=( "$RUN_DIR"/checkpoint-*.pt )
if [ ${#CKPTS[@]} -eq 0 ]; then echo "no checkpoint-*.pt in $RUN_DIR"; exit 1; fi

echo ">>> sweeping ${#CKPTS[@]} checkpoint(s) in $RUN_DIR on $DS (gpu $GPU, $N episodes, seed=$SEED)"
for CKPT in "${CKPTS[@]}"; do
  echo "--- eval $CKPT"
  bash scripts/eval_ckpt.sh "$CKPT" "$GPU" \
    --dataset "$DS" --num_traj "$N" --start_idx 0 --seed "$SEED" --gen_seed "$GEN_SEED"
done
echo ">>> done. Aggregate with: python scripts/plot_learning_curve.py"
