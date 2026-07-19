#!/usr/bin/env python3
"""Plot failure-PSNR and success-MSE vs LoRA rank for the gr00t ppsink-failures sweep, with
FullFT / base+gr00t / base as horizontal reference lines. Reads each run's rollout_metrics.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = "model_ckpt"
def summ(d): return json.load(open(os.path.join(M, d, "rollout_metrics.json")))["summary"]

lora = {  # rank -> eval dir
    16:  "gr00t_ppsink_failures_lora/checkpoint-2000_eval_20260714_150519",
    64:  "gr00t_ppsink_failures_lora_r64/checkpoint-2000_eval_20260715_092047",
    128: "gr00t_ppsink_failures_lora_r128/checkpoint-2000_eval_20260715_092053",
    256: "gr00t_ppsink_failures_lora_r256/checkpoint-2000_eval_20260715_092057",
}
refs = {
    "FullFT":     ("gr00t_ppsink_failures_fullft/checkpoint-2000_eval_20260714_150551", "tab:red"),
    "Base+gr00t": ("robocasa_9task_atomic_gr00t/checkpoint-30000_eval_20260715_093218", "tab:green"),
    "Base":       ("robocasa_9task_base/checkpoint-30000_eval_20260714_150406", "gray"),
}

ranks = sorted(lora)
fail_psnr = [summ(lora[r])["mean_psnr_failure"] for r in ranks]
succ_mse  = [summ(lora[r])["mean_mse_success"] for r in ranks]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(ranks, fail_psnr, "o-", color="tab:blue", lw=2, ms=8, label="LoRA", zorder=3)
ax2.plot(ranks, succ_mse,  "o-", color="tab:blue", lw=2, ms=8, label="LoRA", zorder=3)
for name, (d, c) in refs.items():
    s = summ(d)
    ax1.axhline(s["mean_psnr_failure"], ls="--", c=c, lw=1.6, label=name)
    ax2.axhline(s["mean_mse_success"], ls="--", c=c, lw=1.6, label=name)

for ax in (ax1, ax2):
    ax.set_xscale("log", base=2); ax.set_xticks(ranks); ax.set_xticklabels(ranks)
    ax.set_xlabel("LoRA rank"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax1.set_ylabel("Failure PSNR (dB)  ↑ better"); ax1.set_title("Target (failure) fidelity vs LoRA rank")
ax2.set_ylabel("Success MSE  ↓ better (forgetting)"); ax2.set_title("Forgetting on un-finetuned success rollouts")
fig.suptitle("gr00t PickPlaceSinkToCounter — LoRA rank sweep (2000 steps, ckpt-2000)", y=1.02)
fig.tight_layout()
out = os.path.join(M, "ppsink_lora_rank_sweep.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
for r, fp, sm in zip(ranks, fail_psnr, succ_mse):
    print(f"  r{r:<4d} failPSNR {fp:.2f}  succMSE {sm:.1f}")
