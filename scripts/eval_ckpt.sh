#!/bin/bash
# Evaluate a Ctrl-World checkpoint with the replay rollout -- NO config.py edits.
# Output folder is auto-tagged by checkpoint, so you can run many at once on different GPUs.
#
# Usage:  bash scripts/eval_ckpt.sh <ckpt_path> <gpu> [eval_dataset] [num_traj] [start_idx] [seed]
#   <ckpt_path>    e.g. model_ckpt/robocasa_opendrawer_20260702_130917/checkpoint-30000.pt
#   <gpu>          GPU index (inference needs ~32 GB -> fits one A6000)
#   [eval_dataset] dataset name under dataset_example/ + dataset_meta_info/ (default robocasa_opendrawer_full)
#   [num_traj]     how many val episodes to roll out (default 6)
#   [start_idx]    start frame index: an int (e.g. 30), or 'random' for a random valid start per episode (default 0)
#   [seed]         RNG seed, only used when start_idx=random (default 0)
#
# Two checkpoints in parallel:
#   bash scripts/eval_ckpt.sh model_ckpt/RUN/checkpoint-30000.pt 0 &
#   bash scripts/eval_ckpt.sh model_ckpt/RUN/checkpoint-45000.pt 1 &
set -e
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World
source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh
conda activate ctrl-world

CKPT=$1
GPU=$2
DS=${3:-robocasa_opendrawer_full}
N=${4:-6}
START_IDX=${5:-0}
SEED=${6:-0}
# store eval videos right next to the checkpoint: <ckpt_dir>/<ckpt_name>_eval/
OUT="$(dirname "$CKPT")/$(basename "$CKPT" .pt)_eval"

echo ">>> eval $CKPT on $DS (gpu $GPU, $N episodes, start_idx=$START_IDX seed=$SEED) -> $OUT/"
CUDA_VISIBLE_DEVICES=$GPU python scripts/rollout_replay_traj.py \
  --task_type robocasa \
  --ckpt_path "$CKPT" \
  --svd_model_path checkpoints/svd --clip_model_path checkpoints/clip \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names "$DS" \
  --val_dataset_dir "dataset_example/$DS" \
  --data_stat_path "dataset_meta_info/$DS/stat.json" \
  --num_traj "$N" --out_dir "$OUT" \
  --start_idx "$START_IDX" --seed "$SEED"
