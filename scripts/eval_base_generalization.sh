#!/bin/bash
# Evaluate the 9-task base on ALL 18 tasks of robocasa_atomic_all AND robocasa_gr00t_rollouts
# (2 val episodes/task each = 36 per dataset), then aggregate per-task PSNR/MSE with a
# seen(9 trained) vs held-out(8) summary. Shows how the base generalizes to unseen tasks and to a
# different rollout dataset.
#
# Both evals use the base's TRAINING normalization (robocasa_atomic_all stat.json) so gr00t actions
# are fed on the same scale the model learned -> a fair generalization test, not a normalization shift.
#
# Usage:
#   bash scripts/eval_base_generalization.sh <gpu> [num_traj_per_dataset]
# e.g.  bash scripts/eval_base_generalization.sh 0          # all 36 each, sequential on gpu 0
set -e
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World

GPU="${1:?usage: eval_base_generalization.sh <gpu> [num_traj]}"
N="${2:-36}"                                    # 36 = every val episode (18 tasks x 2)
CKPT="${CKPT:-model_ckpt/robocasa_9task_base/checkpoint-30000.pt}"
STAT=dataset_meta_info/robocasa_atomic_all/stat.json   # base's training normalization
CDIR="$(dirname "$CKPT")"; CNAME="$(basename "$CKPT" .pt)"

run_one() {  # run_one <dataset>
  bash scripts/eval_ckpt.sh "$CKPT" "$GPU" \
    --dataset "$1" --num_traj "$N" --start_idx 0 --seed 42 --gen_seed 42 --data_stat "$STAT"
  ls -td "$CDIR/${CNAME}_eval_"*/ | head -1   # newest eval dir = the one just created
}

echo ">>> [1/2] base on robocasa_atomic_all"
ATOMIC=$(run_one robocasa_atomic_all)
echo ">>> [2/2] base on robocasa_gr00t_rollouts"
GROOT=$(run_one robocasa_gr00t_rollouts)

echo ">>> aggregating per-task ..."
python scripts/aggregate_pertask.py "$ATOMIC" "$GROOT" --csv "$CDIR/base_generalization.csv"
