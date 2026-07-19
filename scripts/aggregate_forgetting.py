#!/usr/bin/env python3
"""Experiment A aggregator: per-task PSNR/MSE on the 8 other (un-finetuned) tasks for each model,
plus the delta vs the base -> how much each ppsink finetune forgot.

For each model it finds, per task, the NEWEST atomic eval dir under the checkpoint's folder whose
meta.json matches (eval_dataset=robocasa_atomic_all, task=T). Uses the success metrics (atomic has
no failures).

Usage (from repo root; first model is treated as the BASE reference):
    python scripts/aggregate_forgetting.py \
        base=model_ckpt/robocasa_9task_base/checkpoint-30000.pt \
        FullFT=model_ckpt/gr00t_ppsink_failures_fullft/checkpoint-2000.pt \
        LoRA_r16=model_ckpt/gr00t_ppsink_failures_lora/checkpoint-2000.pt \
        LoRA_r256=model_ckpt/gr00t_ppsink_failures_lora_r256/checkpoint-2000.pt
"""
import glob, json, os, sys

TASKS = ["CloseFridge", "CloseBlenderLid", "CoffeeSetupMug", "PickPlaceCounterToCabinet",
         "PickPlaceToasterToCounter", "TurnOffStove", "TurnOnMicrowave", "TurnOnSinkFaucet"]


def find_summary(ckpt, task):
    """newest atomic eval dir for (ckpt, task) -> summary dict, or None."""
    cdir = os.path.dirname(ckpt)
    cname = os.path.basename(ckpt)[:-3]           # strip .pt
    best = None
    for meta_p in glob.glob(os.path.join(cdir, f"{cname}_eval_*", "meta.json")):
        try:
            m = json.load(open(meta_p))
        except json.JSONDecodeError:
            continue
        if m.get("eval_dataset") != "robocasa_atomic_all" or m.get("task") != task:
            continue
        d = os.path.dirname(meta_p)
        if not os.path.exists(os.path.join(d, "rollout_metrics.json")):
            continue                              # skip failed/aborted evals (no metrics written)
        if best is None or d > best:              # dir name is timestamp-sorted -> newest wins
            best = d
    if best is None:
        return None
    return json.load(open(os.path.join(best, "rollout_metrics.json")))["summary"]


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    models = []
    for a in sys.argv[1:]:
        label, ckpt = a.split("=", 1)
        models.append((label, ckpt))

    # model -> task -> (psnr, mse)
    data = {}
    for label, ckpt in models:
        data[label] = {}
        for t in TASKS:
            s = find_summary(ckpt, t)
            if s is None:
                print(f"[warn] no eval for {label} / {t}")
                continue
            # atomic = all success -> use success metrics (fall back to overall)
            psnr = s.get("mean_psnr_success", s["mean_psnr"])
            mse = s.get("mean_mse_success", s["mean_mse"])
            data[label][t] = (psnr, mse)

    base_label = models[0][0]
    base = data[base_label]

    # ---- per-task PSNR table with deltas vs base ----
    print(f"\nPer-task PSNR (dB)  [Δ vs {base_label}]")
    hdr = f"{'task':26s}" + "".join(f"{lbl:>16s}" for lbl, _ in models)
    print(hdr); print("-" * len(hdr))
    for t in TASKS:
        row = f"{t:26s}"
        for lbl, _ in models:
            if t not in data[lbl]:
                row += f"{'--':>16s}"; continue
            p = data[lbl][t][0]
            if lbl == base_label:
                row += f"{p:>16.2f}"
            else:
                d = p - base[t][0] if t in base else float('nan')
                row += f"{p:>10.2f}[{d:+.2f}]"
        print(row)

    # ---- mean over the 8 tasks: PSNR and MSE, with delta vs base ----
    print("\nMean over 8 un-finetuned tasks:")
    print(f"  {'model':12s} {'PSNR':>8s} {'ΔPSNR':>8s}   {'MSE':>8s} {'ΔMSE':>8s}")
    for lbl, _ in models:
        ps = [data[lbl][t][0] for t in TASKS if t in data[lbl]]
        ms = [data[lbl][t][1] for t in TASKS if t in data[lbl]]
        if not ps:
            continue
        mp, mm = sum(ps) / len(ps), sum(ms) / len(ms)
        bp = sum(base[t][0] for t in TASKS if t in base) / max(len(base), 1)
        bm = sum(base[t][1] for t in TASKS if t in base) / max(len(base), 1)
        dp = "" if lbl == base_label else f"{mp - bp:+8.2f}"
        dm = "" if lbl == base_label else f"{mm - bm:+8.1f}"
        print(f"  {lbl:12s} {mp:8.2f} {dp:>8s}   {mm:8.1f} {dm:>8s}")
    print("\n(negative ΔPSNR or positive ΔMSE vs base = forgetting on tasks NOT finetuned on)")


if __name__ == "__main__":
    main()
