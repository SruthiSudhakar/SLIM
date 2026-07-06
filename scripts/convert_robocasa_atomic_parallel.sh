#!/bin/bash
# Convert ALL RoboCasa target/atomic demo tasks -> ONE pooled Ctrl-World dataset,
# sharded across 8 GPUs by task (~20-25 min for ~9k episodes), then one meta pass.
# Resumable: re-running skips already-built episodes.
set -e
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World
PY=/proj/vondrick3/sruthi/miniconda3/envs/robocasa_dp/bin/python
ATOMIC=/proj/vondrick3/sruthi/Appaji/robocasa_new/datasets/v1.0/target/atomic
NAME=robocasa_atomic_all

TASKS=(CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen \
       OpenCabinet OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove \
       PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack \
       TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet)

NGPU=8
for g in $(seq 0 $((NGPU-1))); do
    grp=""
    for idx in $(seq $g $NGPU $((${#TASKS[@]}-1))); do grp="$grp,${TASKS[$idx]}"; done
    grp=${grp#,}
    echo "GPU $g -> $grp"
    CUDA_VISIBLE_DEVICES=$g $PY scripts/build_robocasa_atomic_dataset.py \
        --atomic_root "$ATOMIC" --name "$NAME" --val_per_task 2 --no_train_video \
        --tasks "$grp" --no_meta > /tmp/convert_atomic_$g.log 2>&1 &
done
wait
echo "=== all shards done; building meta ==="
$PY scripts/build_robocasa_atomic_dataset.py --name "$NAME" --meta_only
echo "=== DONE: dataset_example/$NAME + dataset_meta_info/$NAME ==="
