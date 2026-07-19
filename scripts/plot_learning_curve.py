#!/usr/bin/env python3
"""Aggregate the sweep_eval.sh outputs into a learning curve: quality vs. finetuning step,
one line per arm (full-FT, LoRA), sharing the step-0 zero-shot base as origin.

Reads <run_dir>/checkpoint-<step>_eval_*/rollout_metrics.json (the 'summary' block written by
rollout_replay_traj.py). If a checkpoint was evaluated more than once, the latest eval dir wins.

Usage (from repo root):
    python scripts/plot_learning_curve.py \
        --arm fullft=model_ckpt/fewshot_opendrawer_fullft \
        --arm lora=model_ckpt/fewshot_opendrawer_lora \
        --base model_ckpt/robocasa_9task_base \
        --out model_ckpt/opendrawer_learning_curve

Writes <out>.csv and <out>.png (PSNR + MSE vs. step).
"""
import argparse
import csv
import glob
import json
import os
import re


def collect(run_dir):
    """step -> summary dict, taking the newest eval dir per checkpoint step."""
    out = {}
    pat = re.compile(r"checkpoint-(\d+)_eval_")
    for mpath in sorted(glob.glob(os.path.join(run_dir, "checkpoint-*_eval_*", "rollout_metrics.json"))):
        m = pat.search(mpath)
        if not m:
            continue
        step = int(m.group(1))
        try:
            summary = json.load(open(mpath))["summary"]
        except (KeyError, json.JSONDecodeError):
            continue
        # sorted() is lexicographic on the timestamp suffix -> last wins = newest eval
        out[step] = {"mean_psnr": summary["mean_psnr"], "mean_mse": summary["mean_mse"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="name=run_dir (repeatable), e.g. fullft=model_ckpt/fewshot_opendrawer_fullft")
    ap.add_argument("--base", help="run dir whose checkpoint = zero-shot origin (plotted at step 0)")
    ap.add_argument("--out", default="opendrawer_learning_curve", help="output path prefix (.csv/.png)")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, run_dir = spec.split("=", 1)
        arms[name] = collect(run_dir)

    # zero-shot origin: best (usually only) checkpoint of the base run, plotted at step 0 for every arm
    base_point = None
    if args.base:
        base_pts = collect(args.base)
        if base_pts:
            base_point = base_pts[max(base_pts)]  # the base's final checkpoint
            for a in arms.values():
                a.setdefault(0, base_point)

    # ---- CSV ----
    csv_path = args.out + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "mean_psnr", "mean_mse"])
        for name, pts in arms.items():
            for step in sorted(pts):
                w.writerow([name, step, pts[step]["mean_psnr"], pts[step]["mean_mse"]])
    print(f"wrote {csv_path}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; CSV written, skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, pts in sorted(arms.items()):
        steps = sorted(pts)
        ax1.plot(steps, [pts[s]["mean_psnr"] for s in steps], marker="o", label=name)
        ax2.plot(steps, [pts[s]["mean_mse"] for s in steps], marker="o", label=name)
    if base_point is not None:
        ax1.axhline(base_point["mean_psnr"], ls="--", c="gray", lw=1, label="zero-shot base")
    ax1.set(xlabel="finetuning step", ylabel="mean PSNR (dB) ↑", title="OpenDrawer adaptation: PSNR")
    ax2.set(xlabel="finetuning step", ylabel="mean MSE ↓", title="OpenDrawer adaptation: MSE")
    ax1.legend(); ax2.legend(); ax1.grid(alpha=.3); ax2.grid(alpha=.3)
    fig.tight_layout()
    png_path = args.out + ".png"
    fig.savefig(png_path, dpi=130)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
