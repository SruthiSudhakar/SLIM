"""Critic-hacking pilot: how easily does gradient optimization fool the frozen critic?

Take a same-seed pair where the critic is CORRECT (failure A first, success B second, so the
margin m(A,B) > 0 => critic prefers the real success B). Then optimize ONLY the failure frame's
pixel_values (the same differentiable surface the world-model training loop would push on) to
drive m(A,B) NEGATIVE -- i.e. make the critic prefer the still-incomplete failure frame. That is
a hack: perceived progress rose with no real progress.

Runs three configs and compares how hard the hack is:
  baseline   - raw critic gradient
  smoothgrad - critic gradient averaged over K noisy copies (the de-spiking fix we validated)
  anchor     - an L2 penalty pulling the perturbation toward 0 (proxy for the KL/manifold leash)

Reuses all model/prompt/pairing machinery from critic_saliency.py; nothing is duplicated.

  EVAL=/proj/vondrick3/sruthi/Appaji/robot_critics_that_sweat_the_small_stuff/released_checkpoints_groot/gr00t_n1-5/multitask_learning/checkpoint-120000/evals/pretrain_for_robotcritics
  CUDA_VISIBLE_DEVICES=0 /proj/vondrick3/sruthi/miniconda3/envs/vlmoverlay/bin/python \
      scripts/critic_hack.py --eval_root "$EVAL" --task_name CloseFridge \
      --steps 150 --lr 0.02 --smooth 8 --anchor 0.5 --out_dir logs/critic_hack
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import critic_saliency as cs  # noqa: E402
from critic_saliency import (  # noqa: E402
    rv, resolve_margin_tokens, build_inputs, margin_from_inputs,
    build_pairs_same_seed, load_pair_frames, _patch_saliency_per_image, _diagnostics,
)

logger = cs.logger
DEFAULT_CHECKPOINT = cs.DEFAULT_CHECKPOINT
DEFAULT_EVAL_ROOT = cs.DEFAULT_EVAL_ROOT


def _margin_grad(model, inputs, base, delta, nA, tok_pos, tok_neg, smooth, noise_frac):
    """d m(A,B) / d delta, optionally SmoothGrad-averaged over `smooth` noisy copies.

    Also returns the clean (noise-free) margin for logging / the flip test.
    """
    def pv_from(d):
        return torch.cat([base[:nA] + d, base[nA:]], 0)

    if smooth:
        sigma = noise_frac * base[:nA].float().std().item()
        g = torch.zeros_like(delta, dtype=torch.float32)
        for _ in range(smooth):
            d2 = delta.detach().clone().requires_grad_(True)
            noise = torch.randn_like(d2.float()).to(d2.dtype) * sigma
            m = margin_from_inputs(model, {**inputs, "pixel_values": pv_from(d2 + noise)}, tok_pos, tok_neg)
            model.zero_grad(set_to_none=True)
            m.backward()
            g += d2.grad.float()
        g /= smooth
        with torch.no_grad():
            clean_m = float(margin_from_inputs(model, {**inputs, "pixel_values": pv_from(delta)}, tok_pos, tok_neg))
        return g, clean_m

    d = delta.detach().clone().requires_grad_(True)
    m = margin_from_inputs(model, {**inputs, "pixel_values": pv_from(d)}, tok_pos, tok_neg)
    model.zero_grad(set_to_none=True)
    m.backward()
    return d.grad.float(), float(m)


def hack(model, inputs, tok_pos, tok_neg, merge, steps, lr, eps, smooth, noise_frac, anchor):
    """Gradient-descend m(A,B) w.r.t. A's pixel_values to flip the critic's decision."""
    base = inputs["pixel_values"].detach()
    thw = inputs["image_grid_thw"]
    nA = int(thw[0].prod().item())  # number of patches for image A
    baseA_norm = base[:nA].float().norm().item()

    delta = torch.zeros_like(base[:nA])
    hist, flipped = [], None
    for step in range(steps):
        g, m = _margin_grad(model, inputs, base, delta, nA, tok_pos, tok_neg, smooth, noise_frac)
        g = g + 2.0 * anchor * delta.float()  # L2 anchor: penalty gradient pulls delta toward 0
        with torch.no_grad():
            delta -= lr * (g / (g.norm() + 1e-8)).to(delta.dtype)
            if eps:
                delta.clamp_(-eps, eps)
        hist.append(m)
        if flipped is None and m < 0.0:
            flipped = step + 1

    # high-frequency character of the perturbation (adversarial-junk signature)
    dgrid = _patch_saliency_per_image(
        torch.cat([delta, torch.zeros_like(base[nA:])], 0), thw, merge
    )[0]
    _, hf = _diagnostics(dgrid)
    return {
        "flipped": flipped, "m0": hist[0], "mfinal": hist[-1], "hist": hist,
        "rel_l2": delta.float().norm().item() / baseA_norm,
        "linf": delta.float().abs().max().item(), "hf": hf,
        "delta": delta, "base": base, "thw": thw, "nA": nA,
    }


def reconstruct_image(pixel_values, thw_row, off, processor, merge):
    """Un-patchify + un-normalize one image's pixel_values back to an HxWx3 uint8 array."""
    ip = processor.image_processor
    ps = ip.patch_size
    t, h, w = (int(x) for x in thw_row.tolist())
    seg = pixel_values[off:off + t * h * w].detach().float().cpu().numpy()
    seg = seg.reshape(t, h // merge, w // merge, merge, merge, 3, -1, ps, ps)[..., 0, :, :]
    #        axes: (t, hb, wb, mh, mw, ch, ph, pw) -> (t, ch, hb, mh, ph, wb, mw, pw)
    img = seg.transpose(0, 5, 1, 3, 6, 2, 4, 7).reshape(t, 3, h * ps, w * ps)[0]
    mean = np.array(ip.image_mean).reshape(3, 1, 1)
    std = np.array(ip.image_std).reshape(3, 1, 1)
    img = np.clip(img * std + mean, 0, 1).transpose(1, 2, 0)
    return (img * 255).astype(np.uint8)


def save_report(processor, merge, frameB, results, out_dir, tag):
    """One figure: original A / real B, then the hacked A + perturbation for EACH config,
    plus margin-vs-step curves and a stats table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    ref = results["baseline"]
    base, thw, nA = ref["base"], ref["thw"], ref["nA"]
    names = list(results.keys())  # baseline, smoothgrad, anchor
    orig_A = reconstruct_image(base, thw[0], 0, processor, merge)

    fig, ax = plt.subplots(3, 4, figsize=(19, 12))
    # row 0: original failure A, then the hacked A for each config
    ax[0, 0].imshow(orig_A); ax[0, 0].set_title(f"original failure A (m0={ref['m0']:+.2f})")
    for j, name in enumerate(names):
        r = results[name]
        hacked_pv = torch.cat([base[:nA] + r["delta"], base[nA:]], 0)
        ax[0, j + 1].imshow(reconstruct_image(hacked_pv, thw[0], 0, processor, merge))
        flip = f"FLIP@{r['flipped']}" if r["flipped"] else "no flip"
        ax[0, j + 1].set_title(f"hacked A [{name}]\nm={r['mfinal']:+.2f}  ({flip})")
    # row 1: real success B, then the |perturbation| heatmap for each config
    ax[1, 0].imshow(np.asarray(frameB)); ax[1, 0].set_title("real success B")
    for j, name in enumerate(names):
        r = results[name]
        dgrid = _patch_saliency_per_image(
            torch.cat([r["delta"], torch.zeros_like(base[nA:])], 0), thw, merge)[0]
        ax[1, j + 1].imshow(dgrid, cmap="jet"); ax[1, j + 1].set_title(f"|perturbation| [{name}]")
    # row 2: margin curves + stats table
    for name in names:
        ax[2, 0].plot(results[name]["hist"], label=name)
    ax[2, 0].axhline(0, ls="--", c="k", lw=1); ax[2, 0].set_xlabel("step")
    ax[2, 0].set_ylabel("m(A,B)  (<0 = hacked)"); ax[2, 0].legend(); ax[2, 0].set_title("margin vs step")
    rows = [f"{'config':<11}{'flip@':>6}{'m0':>7}{'mfin':>7}{'relL2':>8}{'Linf':>7}{'hf':>6}"]
    for name in names:
        r = results[name]
        rows.append(f"{name:<11}{str(r['flipped'] or '-'):>6}{r['m0']:>+7.2f}{r['mfinal']:>+7.2f}"
                    f"{r['rel_l2']:>8.3f}{r['linf']:>7.3f}{r['hf']:>6.2f}")
    ax[2, 1].axis("off"); ax[2, 1].text(0.0, 0.95, "\n".join(rows), family="monospace",
                                        va="top", fontsize=9, transform=ax[2, 1].transAxes)
    for a in [ax[0, 0], ax[0, 1], ax[0, 2], ax[0, 3], ax[1, 0], ax[1, 1], ax[1, 2], ax[1, 3]]:
        a.axis("off")
    ax[2, 2].axis("off"); ax[2, 3].axis("off")
    fig.suptitle(f"critic hacking: {tag}")
    fig.tight_layout()
    png = os.path.join(out_dir, f"{tag}_hack.png")
    fig.savefig(png, dpi=110); plt.close(fig)
    return png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT)
    p.add_argument("--task_name", default="CloseFridge")
    p.add_argument("--max_pixels", default="960x540")
    p.add_argument("--seed", default=None, help="Force a specific same-seed pair (e.g. 100000)")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.02, help="Step size (normalized-gradient step)")
    p.add_argument("--eps", type=float, default=0.0, help="Optional Linf cap on the perturbation (0=off)")
    p.add_argument("--smooth", type=int, default=8, help="SmoothGrad samples for the defense config")
    p.add_argument("--noise_frac", type=float, default=0.1)
    p.add_argument("--anchor", type=float, default=0.5, help="L2 penalty weight for the anchor config")
    p.add_argument("--n_pairs", type=int, default=1,
                   help="Sweep the first N same-seed pairs and print per-config aggregate stats "
                        "(no per-pair figure when N>1).")
    p.add_argument("--out_dir", default="logs/critic_hack")
    args = p.parse_args()

    processor, model = rv.load_ranker(args.checkpoint, args.max_pixels)
    for prm in model.parameters():
        prm.requires_grad_(False)
    tok_pos, tok_neg = resolve_margin_tokens(processor.tokenizer)
    merge = int(getattr(processor.image_processor, "merge_size", 2))

    configs = {
        "baseline": dict(smooth=0, anchor=0.0),
        "smoothgrad": dict(smooth=args.smooth, anchor=0.0),
        "anchor": dict(smooth=0, anchor=args.anchor),
    }

    def run_pair(spec):
        frameA, frameB = load_pair_frames(spec)
        inputs = build_inputs(processor, model, frameA, frameB, args.task_name)  # A first, B second
        out = {}
        for name, cfg in configs.items():
            out[name] = hack(model, inputs, tok_pos, tok_neg, merge, args.steps, args.lr, args.eps,
                             cfg["smooth"], args.noise_frac, cfg["anchor"])
        return frameB, out

    specs = build_pairs_same_seed(args.eval_root, args.task_name)
    if not specs:
        raise SystemExit(f"No same-seed pairs for {args.task_name}")

    # Single pair -> full figure. Multiple pairs -> aggregate stats, no per-pair figure.
    if args.n_pairs <= 1:
        spec = next((s for s in specs if s["tag"] == f"seed{args.seed}"), specs[0]) if args.seed else specs[0]
        logger.info(f"attacking {args.task_name} {spec['tag']} (drive m(A,B) below 0 -> prefer failure A)")
        frameB, results = run_pair(spec)
        for name, r in results.items():
            logger.info(f"[{name:<10}] flip@={r['flipped']}  m0={r['m0']:+.3f} -> mfinal={r['mfinal']:+.3f}  "
                        f"relL2={r['rel_l2']:.3f}  Linf={r['linf']:.3f}  hf={r['hf']:.3f}")
        png = save_report(processor, merge, frameB, results,
                          args.out_dir, f"{args.task_name}_{spec['tag']}")
        logger.info(f"wrote {png}")
        return

    chosen = specs[:args.n_pairs]
    logger.info(f"sweeping {len(chosen)} pairs for {args.task_name}")
    agg = {n: {"flip": [], "steps": [], "drop": [], "relL2": [], "Linf": [], "hf": []} for n in configs}
    for i, spec in enumerate(chosen):
        frameB, results = run_pair(spec)
        if i == 0:
            png = save_report(processor, merge, frameB, results,
                              args.out_dir, f"{args.task_name}_{spec['tag']}")
            logger.info(f"wrote {png}")
        for name, r in results.items():
            a = agg[name]
            a["flip"].append(r["flipped"] is not None)
            if r["flipped"] is not None:
                a["steps"].append(r["flipped"])
            a["drop"].append(r["m0"] - r["mfinal"])
            a["relL2"].append(r["rel_l2"]); a["Linf"].append(r["linf"]); a["hf"].append(r["hf"])
        b = results["baseline"]
        logger.info(f"  [{i+1}/{len(chosen)}] {spec['tag']} baseline flip@={b['flipped']} "
                    f"m0={b['m0']:+.2f}->{b['mfinal']:+.2f}")
    med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")
    logger.info(f"===== hacking summary [{args.task_name}, n={len(chosen)} pairs, steps={args.steps}, lr={args.lr}] =====")
    for name, a in agg.items():
        logger.info(f"[{name:<10}] flip_rate={np.mean(a['flip']):.2f}  med_steps_to_flip={med(a['steps']):.0f}  "
                    f"med_drop={med(a['drop']):+.2f}  med_relL2={med(a['relL2']):.3f}  "
                    f"med_Linf={med(a['Linf']):.3f}  med_hf={med(a['hf']):.2f}")


if __name__ == "__main__":
    main()
