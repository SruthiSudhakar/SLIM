#!/bin/bash
# Convert GR00T rollouts -> Ctrl-World dataset, sharded across 8 GPUs by task (~45 min),
# then one final meta pass. Resumable: re-running skips already-built episodes.
set -e
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World
PY=/proj/vondrick3/sruthi/miniconda3/envs/robocasa_dp/bin/python
REC=/proj/vondrick3/sruthi/Appaji/robot_critics_that_sweat_the_small_stuff/released_checkpoints_groot/gr00t_n1-5/multitask_learning/checkpoint-120000/evals/pretrain
NAME=robocasa_gr00t_rollouts

TASKS=(CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen \
       OpenCabinet OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove \
       PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack \
       TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet)

NGPU=8
# round-robin tasks -> GPUs
for g in $(seq 0 $((NGPU-1))); do
    grp=""
    for idx in $(seq $g $NGPU $((${#TASKS[@]}-1))); do grp="$grp,${TASKS[$idx]}"; done
    grp=${grp#,}
    echo "GPU $g -> $grp"
    CUDA_VISIBLE_DEVICES=$g $PY scripts/build_gr00t_rollout_dataset.py \
        --record_dir "$REC" --name "$NAME" --val_per_task 2 --no_train_video \
        --tasks "$grp" --no_meta > /tmp/convert_gr00t_$g.log 2>&1 &
done
wait
echo "=== all shards done; building meta ==="
$PY scripts/build_gr00t_rollout_dataset.py --name "$NAME" --meta_only
echo "=== DONE: dataset_example/$NAME + dataset_meta_info/$NAME ==="
