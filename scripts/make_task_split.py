#!/usr/bin/env python3
"""Build task-subset "virtual datasets" by filtering a source dataset's sample lists.

Each sample in dataset_meta_info/<source>/{train,val}_sample.json is
    {'episode_id': 'TaskName__ep000173' | 'TaskName_14_50_..._seed100004', 'frame_ids': [...]}
The task name is the first '_'-delimited token (works for both robocasa_atomic_all and
robocasa_gr00t_rollouts ids). We write filtered sample JSONs + a stat.json into a new
dataset_meta_info/<name>/ dir and symlink dataset_example/<name> -> <source> so the dataloader
resolves the real latents/annotations without copying. Pass <name> via --dataset_names to
train_wm.py / eval_ckpt.sh.

Optional --success N keeps only episodes whose annotation 'success' == N (e.g. 0 = failures only);
this reads dataset_example/<source>/annotation/<mode>/<episode_id>.json.

Usage (from repo root):
    python scripts/make_task_split.py                       # builds the two default fewshot splits
    python scripts/make_task_split.py --name my_split --tasks TaskA,TaskB
    # gr00t PickPlaceSinkToCounter FAILURES, normalized with the base's (atomic) training stats:
    python scripts/make_task_split.py --name gr00t_ppsink_failures \
        --source robocasa_gr00t_rollouts --tasks PickPlaceSinkToCounter --success 0 \
        --stat-from robocasa_atomic_all
"""
import argparse
import glob
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = "robocasa_atomic_all"
META = os.path.join(REPO, "dataset_meta_info")
DATA = os.path.join(REPO, "dataset_example")

# --- default fewshot split (9 train / 8 holdout; OpenDrawer = target). Edit here to rebalance. ---
TRAIN_9 = [
    "CloseFridge", "CloseBlenderLid", "CoffeeSetupMug",
    "PickPlaceCounterToCabinet", "PickPlaceSinkToCounter", "PickPlaceToasterToCounter",
    "TurnOffStove", "TurnOnMicrowave", "TurnOnSinkFaucet",
]
TARGET_TASK = "OpenDrawer"


def task_of(episode_id):
    return episode_id.split("_")[0]


def success_ids(source, tasks, mode, success):
    """episode_ids under <source>/annotation/<mode> for the given tasks with annotation success==N."""
    keep = set()
    for task in tasks:
        for f in glob.glob(os.path.join(DATA, source, "annotation", mode, f"{task}_*.json")) + \
                 glob.glob(os.path.join(DATA, source, "annotation", mode, f"{task}__*.json")):
            try:
                d = json.load(open(f))
            except (OSError, json.JSONDecodeError):
                continue
            if int(d.get("success", -1)) == success:
                keep.add(d["episode_id"])
    return keep


def _filter(samples, tasks, allowed):
    return [s for s in samples
            if task_of(s["episode_id"]) in tasks
            and (allowed is None or s["episode_id"] in allowed)]


def _setup_data_dir(name, source, holdout_val_eps):
    """Point dataset_example/<name> at <source>'s latents/annotations.

    Normal case: a bare symlink. Holdout case: the val episodes physically live in <source>'s
    annotation/train/ (e.g. gr00t failures), but the dataloader reads annotation/<mode>/, so we build
    a real dir that re-exposes those episodes under annotation/val/ via per-episode symlinks. Latents
    resolve unchanged because each annotation's latent_video_path is 'latent_videos/train/...'.
    """
    link = os.path.join(DATA, name)
    if os.path.islink(link):
        os.remove(link)
    elif os.path.isdir(link):
        shutil.rmtree(link)
    elif os.path.exists(link):
        raise SystemExit(f"{link} exists and is not a symlink/dir; refusing to overwrite")

    if not holdout_val_eps:
        os.symlink(source, link)
        return "symlink -> " + source

    os.makedirs(os.path.join(link, "annotation", "val"))
    os.symlink(os.path.join("..", source, "latent_videos"), os.path.join(link, "latent_videos"))
    os.symlink(os.path.join("..", source, "videos"), os.path.join(link, "videos"))
    os.symlink(os.path.join("..", "..", source, "annotation", "train"),
               os.path.join(link, "annotation", "train"))
    for ep in holdout_val_eps:
        os.symlink(os.path.join("..", "..", "..", source, "annotation", "train", f"{ep}.json"),
                   os.path.join(link, "annotation", "val", f"{ep}.json"))
    return f"real dir (val/ = {len(holdout_val_eps)} held-out eps symlinked from {source}/annotation/train)"


def build_split(name, tasks, source=DEFAULT_SOURCE, success=None, stat_from=None, val_holdout=0):
    tasks = set(tasks)
    stat_from = stat_from or source
    out_meta = os.path.join(META, name)
    os.makedirs(out_meta, exist_ok=True)
    counts = {}

    if val_holdout:
        # Carve val out of the source TRAIN split by episode (use when the source has no matching val
        # episodes, e.g. gr00t failures live only in train). Deterministic: last N episode ids -> val.
        with open(os.path.join(META, source, "train_sample.json")) as f:
            samples = json.load(f)
        allowed = success_ids(source, tasks, "train", success) if success is not None else None
        kept = _filter(samples, tasks, allowed)
        eps = sorted({s["episode_id"] for s in kept})
        val_eps = set(eps[-val_holdout:]) if val_holdout < len(eps) else set()
        splits = {"train": [s for s in kept if s["episode_id"] not in val_eps],
                  "val":   [s for s in kept if s["episode_id"] in val_eps]}
        holdout_val_eps = val_eps
    else:
        holdout_val_eps = None
        splits = {}
        for mode in ("train", "val"):
            with open(os.path.join(META, source, f"{mode}_sample.json")) as f:
                samples = json.load(f)
            allowed = success_ids(source, tasks, mode, success) if success is not None else None
            splits[mode] = _filter(samples, tasks, allowed)

    for mode, kept in splits.items():
        with open(os.path.join(out_meta, f"{mode}_sample.json"), "w") as f:
            json.dump(kept, f)
        counts[mode] = len(kept)
        n_eps = len({s["episode_id"] for s in kept})
        print(f"  {mode}: {counts[mode]} samples across {n_eps} episode(s)")
    if success is not None and not splits["train"]:
        print(f"  [warn] no train samples matched success=={success} for {sorted(tasks)}")
    # normalization stats: copy from stat_from (use the model's TRAINING dataset for a cross-dataset
    # finetune so actions are normalized the same way the model learned).
    shutil.copyfile(os.path.join(META, stat_from, "stat.json"),
                    os.path.join(out_meta, "stat.json"))
    data_desc = _setup_data_dir(name, source, holdout_val_eps)
    tag = f"success=={success} " if success is not None else ""
    print(f"[{name}] {tag}tasks={sorted(tasks)} train={counts['train']} val={counts['val']} "
          f"stat<-{stat_from}  {data_desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="split name (omit to build the two default fewshot splits)")
    ap.add_argument("--tasks", help="comma-separated task names (with --name)")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="source dataset to filter")
    ap.add_argument("--success", type=int, default=None,
                    help="keep only episodes with annotation success==N (e.g. 0 = failures only)")
    ap.add_argument("--stat-from", dest="stat_from", default=None,
                    help="dataset whose stat.json to copy (default: --source)")
    ap.add_argument("--val-holdout", dest="val_holdout", type=int, default=0,
                    help="carve N episodes out of the source train split into val (for sources whose "
                         "val has no matching episodes, e.g. gr00t failures)")
    args = ap.parse_args()

    if args.name:
        build_split(args.name, [t.strip() for t in args.tasks.split(",")],
                    source=args.source, success=args.success, stat_from=args.stat_from,
                    val_holdout=args.val_holdout)
        return

    build_split("robocasa_9task_base", TRAIN_9)
    build_split("robocasa_opendrawer_target", [TARGET_TASK])
    print("\nDone (default fewshot splits).")


if __name__ == "__main__":
    main()
