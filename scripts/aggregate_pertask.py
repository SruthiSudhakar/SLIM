#!/usr/bin/env python3
"""Group a rollout eval's per-trajectory metrics by task and report seen-vs-held-out generalization.

eval_ckpt.sh writes <eval_dir>/rollout_metrics.json with a 'per_traj' list, each entry tagged with
'traj_id' = the episode id (e.g. 'OpenDrawer__ep000000' or 'OpenDrawer_14_50_..._seed100000'). The
task name is the first '_'-delimited token. This aggregates PSNR/MSE per task and marks whether each
task was in the base's 9 training tasks (seen) or held out.

Usage:
    python scripts/aggregate_pertask.py <eval_dir> [<eval_dir> ...] [--csv out.csv]
Each <eval_dir> is a '<ckpt>_eval_<ts>' folder (or pass several to combine, e.g. atomic + gr00t).
"""
import argparse
import collections
import csv
import json
import os

# the 9 tasks the base was trained on (keep in sync with scripts/make_task_split.py TRAIN_9)
SEEN = {
    "CloseFridge", "CloseBlenderLid", "CoffeeSetupMug",
    "PickPlaceCounterToCabinet", "PickPlaceSinkToCounter", "PickPlaceToasterToCounter",
    "TurnOffStove", "TurnOnMicrowave", "TurnOnSinkFaucet",
}


def task_of(traj_id):
    return traj_id.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dirs", nargs="+", help="one or more <ckpt>_eval_<ts> dirs")
    ap.add_argument("--csv", help="optional CSV output path")
    args = ap.parse_args()

    # task -> {source_label: [(psnr, mse), ...]}; source label = eval dataset from meta.json
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in args.eval_dirs:
        mpath = os.path.join(d, "rollout_metrics.json")
        if not os.path.isfile(mpath):
            print(f"[skip] no rollout_metrics.json in {d}")
            continue
        src = "?"
        meta_p = os.path.join(d, "meta.json")
        if os.path.isfile(meta_p):
            src = json.load(open(meta_p)).get("eval_dataset", "?")
        for t in json.load(open(mpath))["per_traj"]:
            rows[task_of(t["traj_id"])][src].append((t["mean_psnr"], t["mean_mse"]))

    sources = sorted({s for v in rows.values() for s in v})
    out = []
    for task in sorted(rows):
        for src in sources:
            vals = rows[task].get(src, [])
            if not vals:
                continue
            n = len(vals)
            psnr = sum(p for p, _ in vals) / n
            mse = sum(m for _, m in vals) / n
            out.append({"task": task, "dataset": src, "seen": task in SEEN,
                        "n": n, "mean_psnr": psnr, "mean_mse": mse})

    # ---- print table ----
    print(f"\n{'task':28s} {'dataset':26s} {'seen':5s} {'n':>3s} {'PSNR(dB)↑':>10s} {'MSE↓':>10s}")
    print("-" * 88)
    for r in out:
        print(f"{r['task']:28s} {r['dataset']:26s} {str(r['seen']):5s} {r['n']:3d} "
              f"{r['mean_psnr']:10.3f} {r['mean_mse']:10.2f}")

    # ---- seen vs held-out summary, per dataset ----
    print("\n=== seen (9 trained) vs held-out, mean PSNR over tasks ===")
    for src in sources:
        for grp, label in [(True, "seen   "), (False, "heldout")]:
            ps = [r["mean_psnr"] for r in out if r["dataset"] == src and r["seen"] == grp]
            if ps:
                print(f"  {src:26s} {label}: {sum(ps)/len(ps):7.3f} dB   ({len(ps)} tasks)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "dataset", "seen", "n", "mean_psnr", "mean_mse"])
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
