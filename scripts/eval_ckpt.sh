#!/bin/bash
# Evaluate a Ctrl-World checkpoint with the replay rollout -- NO config.py edits.
# Output folder is auto-tagged by checkpoint, so you can run many at once on different GPUs.
#
# Usage (ckpt + gpu are required positional; everything else is positional OR named):
#   bash scripts/eval_ckpt.sh <ckpt_path> <gpu> [eval_dataset] [num_traj] [start_idx] [seed] [interact_num]
#   bash scripts/eval_ckpt.sh <ckpt_path> <gpu> --dataset <name> --num_traj <N> \
#                             --start_idx <int|random> --seed <int> --interact_num <int> --gen_seed <int>
#
#   <ckpt_path>       e.g. model_ckpt/robocasa_opendrawer_20260702_130917/checkpoint-30000.pt
#   <gpu>             GPU index (inference needs ~32 GB -> fits one A6000)
#   --dataset         dataset name under dataset_example/ + dataset_meta_info/ (default robocasa_opendrawer_full)
#                     (aliases: --eval_dataset)
#   --num_traj        number of val episodes to roll out (default 6)  (alias: --num_trajs)
#   --start_idx       start frame index: int (e.g. 30), or 'random' for a random valid start (default 0)
#   --seed            RNG seed, only used when start_idx=random (default 0)
#   --interact_num    autoregressive rollout steps per episode; rollout length ~= (pred_step-1)*interact_num + 1
#                     (default: whatever config.py has -- 12 for replay)
#   --gen_seed        seed for diffusion sampling noise (default: same as --seed; env GEN_SEED also honored)
#
# Two checkpoints in parallel:
#   bash scripts/eval_ckpt.sh model_ckpt/RUN/checkpoint-30000.pt 0 &
#   bash scripts/eval_ckpt.sh model_ckpt/RUN/checkpoint-45000.pt 1 &
set -e
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World
source /proj/vondrick3/sruthi/miniconda3/etc/profile.d/conda.sh
conda activate ctrl-world

if [ $# -lt 2 ]; then
  echo "usage: bash scripts/eval_ckpt.sh <ckpt_path> <gpu> [flags or positional...]" >&2
  exit 1
fi

CKPT=$1
GPU=$2
shift 2

# defaults
DS=robocasa_opendrawer_full
N=1
START_IDX=0
SEED=0
INTERACT_NUM=""   # empty -> don't pass to python -> config.py default wins
GEN_SEED_ARG=""   # empty -> fall back to env GEN_SEED, then to SEED

# support mixed positional + named:
#   positional order (after ckpt/gpu): dataset, num_traj, start_idx, seed, interact_num
POS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset|--eval_dataset)     DS="$2";           shift 2;;
    --num_traj|--num_trajs)       N="$2";            shift 2;;
    --start_idx)                  START_IDX="$2";    shift 2;;
    --seed)                       SEED="$2";         shift 2;;
    --interact_num)               INTERACT_NUM="$2"; shift 2;;
    --gen_seed)                   GEN_SEED_ARG="$2"; shift 2;;
    --) shift; break;;
    --*)
      echo "Unknown flag: $1" >&2; exit 1;;
    *)
      case $POS in
        0) DS="$1";;
        1) N="$1";;
        2) START_IDX="$1";;
        3) SEED="$1";;
        4) INTERACT_NUM="$1";;
        *) echo "Too many positional args: $1" >&2; exit 1;;
      esac
      POS=$((POS+1))
      shift;;
  esac
done

# gen_seed precedence: explicit flag > env GEN_SEED > SEED
GEN_SEED="${GEN_SEED_ARG:-${GEN_SEED:-$SEED}}"

# store eval videos right next to the checkpoint, timestamped so runs don't clash:
#   <ckpt_dir>/<ckpt_name>_eval_<timestamp>/
OUT="$(dirname "$CKPT")/$(basename "$CKPT" .pt)_eval_$(date +%Y%m%d_%H%M%S)"

# write run metadata so evals can be told apart later
mkdir -p "$OUT"
cat > "$OUT/meta.json" <<EOF
{
  "ckpt_path": "$CKPT",
  "eval_dataset": "$DS",
  "num_traj": $N,
  "start_idx": "$START_IDX",
  "seed": $SEED,
  "gen_seed": $GEN_SEED,
  "interact_num": ${INTERACT_NUM:-null},
  "gpu": "$GPU",
  "run_time": "$(date +%Y-%m-%d_%H:%M:%S)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)",
  "command": "$0 $*"
}
EOF

echo ">>> eval $CKPT on $DS (gpu $GPU, $N episodes, start_idx=$START_IDX seed=$SEED interact_num=${INTERACT_NUM:-config}) -> $OUT/"
echo ">>> wrote $OUT/meta.json"

# build the python command; only pass --interact_num when set
INTERACT_ARGS=()
if [ -n "$INTERACT_NUM" ]; then
  INTERACT_ARGS=(--interact_num "$INTERACT_NUM")
fi

CUDA_VISIBLE_DEVICES=$GPU python scripts/rollout_replay_traj.py \
  --task_type robocasa \
  --ckpt_path "$CKPT" \
  --svd_model_path checkpoints/svd --clip_model_path checkpoints/clip \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names "$DS" \
  --val_dataset_dir "dataset_example/$DS" \
  --data_stat_path "dataset_meta_info/$DS/stat.json" \
  --num_traj "$N" --out_dir "$OUT" \
  --start_idx "$START_IDX" --seed "$SEED" --gen_seed "$GEN_SEED" \
  "${INTERACT_ARGS[@]}"
