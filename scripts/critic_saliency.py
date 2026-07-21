"""Experiment 1 (todo.MD): ∂(logit-margin)/∂pixels saliency for the Qwen2.5-VL critic.

Feed the frozen critic a real pair of frames, compute the gradient of its (differentiable)
logit-margin w.r.t. the input pixels, and inspect the saliency map. Verifies (a) sensitivity
concentrates on the gripper/object, not the background, and (b) it is not dominated by
high-frequency junk everywhere (which would predict fast adversarial hacking).

Pairs come from RoboCasa eval rollouts: the SAME episode seed reproduces the SAME initial
scene across a task's different rollout runs, so a seed that is a success (reward 1.0) in one
run and a failure (reward 0.0) in another gives an identical starting scene with a divergent
outcome. We take the last frame of each; the success frame shows more progress (label from the
reward), and any saliency on the (identical) background is unambiguously spurious.

The critic answers a signed number +/-ANSWER_MAGNITUDE: positive => the SECOND image shows more
progress. With ANSWER_MAGNITUDE=32, "32" tokenizes with leading token "3" and "-32" with
leading token "-", so at the first answer position the logit-margin is simply
    m(A_first, B_second) = logit(tok_pos="3") - logit(tok_neg="-")
Symmetrized to kill position bias:  m~ = (m(A,B) - m(B,A)) / 2.

Reuses the critic's own loading/prompt/frame/token machinery from the trl repo (no duplicated
model code). Run under the vlmoverlay env:

  EVAL=/proj/vondrick3/sruthi/Appaji/robot_critics_that_sweat_the_small_stuff/released_checkpoints_groot/gr00t_n1-5/multitask_learning/checkpoint-120000/evals/pretrain_for_robotcritics
  CUDA_VISIBLE_DEVICES=0 /proj/vondrick3/sruthi/miniconda3/envs/vlmoverlay/bin/python \
    scripts/critic_saliency.py \
    --checkpoint /proj/vondrick3/sruthi/Appaji/robot_critics_that_sweat_the_small_stuff/trl/outputs/checkpoint-4400 \
    --eval_root "$EVAL" --task_name CloseBlenderLid --out_dir logs/critic_saliency
  # --seed 100007 to force a seed; --videoA/--videoB for explicit mp4 overrides; --corr 40 for
  # the held-out margin-correlation signal-quality check.
"""

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import torch

# Reuse the critic's machinery from the trl repo (moved into robot_critics_that_sweat_the_small_stuff).
MYSCRIPTS = (
    "/proj/vondrick3/sruthi/Appaji/robot_critics_that_sweat_the_small_stuff/"
    "trl/examples/scripts/myscripts"
)
sys.path.insert(0, MYSCRIPTS)

import rank_videos as rv  # noqa: E402  load_ranker, get_frame_at_fraction, _build_prompt_text
from rank_serve_robocasa import task_token_for  # noqa: E402  curated-or-auto UPPER_SNAKE token
from sft_vlm_lego import ANSWER_MAGNITUDE, LEGO_USER_PROMPT_TEMPLATE  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr
)
logger = logging.getLogger("critic_saliency")

DEFAULT_CHECKPOINT = (
    "/proj/vondrick3/sruthi/Appaji/robot_critics_that_sweat_the_small_stuff/"
    "trl/outputs/checkpoint-4400"
)
DEFAULT_EVAL_ROOT = (
    "/proj/vondrick3/sruthi/Appaji/robot_critics_that_sweat_the_small_stuff/"
    "released_checkpoints_groot/gr00t_n1-5/multitask_learning/checkpoint-120000/"
    "evals/pretrain_for_robotcritics"
)


# --------------------------------------------------------------------------------------
# Same-seed success-vs-failure pairing from RoboCasa eval rollouts
# --------------------------------------------------------------------------------------
def _rewards_for_run(run_dir):
    """{seed_str -> reward} from a rollout dir's eval_log.json (test/sim_max_reward_<N>)."""
    with open(os.path.join(run_dir, "eval_log.json")) as f:
        log = json.load(f)
    out = {}
    for k, v in log.items():
        if k.startswith("test/sim_max_reward_"):
            out[k.rsplit("_", 1)[-1]] = float(v)
    return out


def find_same_seed_pairs(eval_root, task):
    """Seeds that are failure (0.0) in one run and success (1.0) in another, same task.

    Returns a list of dicts: {seed, failure_mp4, success_mp4}. The success mp4 is placed
    second by the caller, so the ground-truth symmetrized margin is > 0.
    """
    run_dirs = sorted(glob.glob(os.path.join(eval_root, f"{task}_14_50_*")))
    if not run_dirs:
        raise FileNotFoundError(f"No '{task}_14_50_*' rollout dirs under {eval_root}")

    succ, fail = {}, {}  # seed -> mp4 path (first run we find it in)
    for rd in run_dirs:
        media = os.path.join(rd, "media")
        for seed, r in _rewards_for_run(rd).items():
            mp4 = os.path.join(media, f"seed{seed}.mp4")
            if not os.path.isfile(mp4):
                continue
            if r >= 1.0:
                succ.setdefault(seed, mp4)
            elif r <= 0.0:
                fail.setdefault(seed, mp4)

    pairs = [
        {"seed": s, "failure_mp4": fail[s], "success_mp4": succ[s]}
        for s in sorted(set(succ) & set(fail))
    ]
    logger.info(
        f"[{task}] {len(run_dirs)} runs -> {len(pairs)} seeds with both a failure and a success"
    )
    return pairs


def _num_frames(path):
    return len(rv._get_video_reader(path))


def list_success_videos(eval_root, task):
    """All success (reward 1.0) rollout mp4s for the task's _14_50_ runs."""
    run_dirs = sorted(glob.glob(os.path.join(eval_root, f"{task}_14_50_*")))
    if not run_dirs:
        raise FileNotFoundError(f"No '{task}_14_50_*' rollout dirs under {eval_root}")
    vids = []
    for rd in run_dirs:
        media = os.path.join(rd, "media")
        for seed, r in _rewards_for_run(rd).items():
            if r >= 1.0:
                mp4 = os.path.join(media, f"seed{seed}.mp4")
                if os.path.isfile(mp4):
                    vids.append(mp4)
    return vids


# --------------------------------------------------------------------------------------
# Pair specs: {pathA, idxA, pathB, idxB, tag}. B is the ground-truth more-progressed frame
# (placed second), so the correct symmetrized margin is m~ > 0. Two ways to build them:
#   same_seed : last frame of a failure vs last frame of a same-seed success (cross-scene).
#   temporal  : frame i vs frame i+gap of ONE success rollout; later = more progress. Same
#               scene, only the robot/object moved -> cleanest isolation for saliency.
# --------------------------------------------------------------------------------------
def build_pairs_same_seed(eval_root, task):
    specs = []
    for pr in find_same_seed_pairs(eval_root, task):
        specs.append({
            "pathA": pr["failure_mp4"], "idxA": _num_frames(pr["failure_mp4"]) - 1,
            "pathB": pr["success_mp4"], "idxB": _num_frames(pr["success_mp4"]) - 1,
            "tag": f"seed{pr['seed']}", "labelA": "A(failure)", "labelB": "B(success)",
        })
    return specs


def build_pairs_temporal(eval_root, task, gap, stride):
    """Within-success-rollout pairs (i, i+gap), video-major (each video's pairs in time order)."""
    vids = list_success_videos(eval_root, task)
    if not vids:
        raise FileNotFoundError(f"No success rollouts for task {task}")
    specs = []
    for v in vids:
        n = _num_frames(v)
        stem = os.path.basename(v)[:-4]
        for i in range(0, n - gap, stride):
            specs.append({
                "pathA": v, "idxA": i, "pathB": v, "idxB": i + gap,
                "tag": f"{stem}_{i}-{i+gap}",
                "labelA": f"A(frame {i})", "labelB": f"B(frame {i+gap})",
            })
    logger.info(
        f"[{task}] {len(vids)} success rollouts -> {len(specs)} temporal pairs "
        f"(gap={gap}, stride={stride})"
    )
    return specs


def load_pair_frames(spec):
    """Extract the (frameA, frameB) PIL images for a pair spec (absolute frame indices)."""
    return rv.extract_frame(spec["pathA"], spec["idxA"]), rv.extract_frame(spec["pathB"], spec["idxB"])


# --------------------------------------------------------------------------------------
# Differentiable logit-margin
# --------------------------------------------------------------------------------------
def resolve_margin_tokens(tokenizer):
    """First-token ids of the positive vs negative answer strings ('3' vs '-' for +/-32)."""
    tok_pos = tokenizer.encode(str(ANSWER_MAGNITUDE), add_special_tokens=False)[0]
    tok_neg = tokenizer.encode(str(-ANSWER_MAGNITUDE), add_special_tokens=False)[0]
    logger.info(
        f"margin tokens: pos={tok_pos} ({tokenizer.decode([tok_pos])!r})  "
        f"neg={tok_neg} ({tokenizer.decode([tok_neg])!r})"
    )
    return tok_pos, tok_neg


def build_inputs(processor, model, frame_first, frame_second, task_name):
    """Processor inputs for the pair (frame_first, frame_second), on the model device."""
    user_text = LEGO_USER_PROMPT_TEMPLATE.format(task_token=task_token_for(task_name))
    text = rv._build_prompt_text(processor, [frame_first, frame_second], user_text)
    inputs = processor(
        text=[text], images=[[frame_first, frame_second]], return_tensors="pt", padding=True
    )
    return {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def margin_from_inputs(model, inputs, tok_pos, tok_neg):
    """logit(tok_pos) - logit(tok_neg) at the first answer position (grad-enabled forward)."""
    out = model(**{k: v for k, v in inputs.items() if k != "labels"})
    last_logits = out.logits[0, -1, :]  # left-padded, batch=1 -> generation position is last
    return last_logits[tok_pos] - last_logits[tok_neg]


# --------------------------------------------------------------------------------------
# Saliency: reshape flattened pixel-grad back onto the patch grid, per image
# --------------------------------------------------------------------------------------
def _patch_saliency_per_image(pixel_grad, grid_thw, merge):
    """Split flattened |grad| (summed over feature dim) into per-image (h_patch, w_patch) maps.

    Qwen2.x flattens patches in merge-block order: for each (h_block, w_block) the merge*merge
    sub-patches are contiguous. We invert that to recover the (grid_h, grid_w) layout.
    """
    s = pixel_grad.abs().float().sum(dim=1).cpu().numpy()  # [total_patches]
    maps, off = [], 0
    for (t, h, w) in grid_thw.tolist():
        n = t * h * w
        blk = s[off:off + n].reshape(t, h // merge, w // merge, merge, merge)
        blk = blk.transpose(0, 1, 3, 2, 4).reshape(t, h, w)  # -> (t, grid_h, grid_w)
        maps.append(blk[0])  # t == 1 for still frames
        off += n
    return maps


def _diagnostics(heat):
    """(center_fraction, high_freq_ratio) for checks (a) and (b)."""
    h, w = heat.shape
    total = heat.sum() + 1e-8
    y0, y1, x0, x1 = h // 4, 3 * h // 4, w // 4, 3 * w // 4  # central 50% box
    center_frac = float(heat[y0:y1, x0:x1].sum() / total)
    lap = np.abs(np.gradient(np.gradient(heat, axis=0), axis=0)) + np.abs(
        np.gradient(np.gradient(heat, axis=1), axis=1)
    )
    high_freq_ratio = float(lap.sum() / total)
    return center_frac, high_freq_ratio


def _save_overlays(frames, heats, margins, out_dir, tag, clip_pct=99.0, note="(expect m~>0)"):
    """Save a 2x2 figure (each image: original | saliency overlay) plus raw heatmaps.

    Raw input gradients are extremely heavy-tailed (the median patch is ~0.1-1% of the max),
    so normalizing by the max renders all but the top ~0.1% of patches as dark. We instead
    clip at the `clip_pct`-th percentile before normalizing, which reveals the real structure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ["image A (first)", "image B (second)"]
    for row, (frame, heat, title) in enumerate(zip(frames, heats, titles)):
        img = np.asarray(frame)
        # upsample patch-grid heat to the frame size for display
        from PIL import Image

        heat_img = np.asarray(
            Image.fromarray(heat.astype(np.float32)).resize(
                (img.shape[1], img.shape[0]), Image.BILINEAR
            )
        )
        # robust normalization: clip at a high percentile instead of the (outlier) max
        denom = np.percentile(heat_img, clip_pct) + 1e-8
        hn = np.clip(heat_img / denom, 0.0, 1.0)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(title)
        axes[row, 1].imshow(img)
        axes[row, 1].imshow(hn, cmap="jet", alpha=0.5)
        axes[row, 1].set_title(f"{title} — |d margin / d pixels|")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(
        f"{tag}   m(A,B)={margins['m_ab']:+.3f}  m(B,A)={margins['m_ba']:+.3f}  "
        f"m~={margins['m_tilde']:+.3f}   {note}"
    )
    fig.tight_layout()
    png = os.path.join(out_dir, f"{tag}_saliency.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    np.savez(os.path.join(out_dir, f"{tag}_heatmaps.npz"), heatA=heats[0], heatB=heats[1])
    return png


# --------------------------------------------------------------------------------------
# Main experiment paths
# --------------------------------------------------------------------------------------
def pixel_grad_for_margin(model, inputs, tok_pos, tok_neg, smoothgrad=0, noise_frac=0.1):
    """d margin / d pixel_values, optionally SmoothGrad-averaged over noisy input copies.

    Vanilla gradients are spiky; SmoothGrad (Smilkov 2017) averages the gradient over
    `smoothgrad` copies of the input perturbed with Gaussian noise (std = noise_frac * pixel
    std), which cancels random spikes while consistent, task-relevant signal survives.
    Returns (grad_tensor, clean_margin_float).
    """
    base = inputs["pixel_values"].detach()
    n = max(1, smoothgrad)
    sigma = noise_frac * base.float().std().item() if smoothgrad else 0.0
    grad_accum, clean_margin = None, None
    for k in range(n):
        pv = base if sigma == 0.0 else base + torch.randn_like(base.float()).to(base.dtype) * sigma
        pv = pv.detach().requires_grad_(True)
        m_t = margin_from_inputs(model, {**inputs, "pixel_values": pv}, tok_pos, tok_neg)
        model.zero_grad(set_to_none=True)
        m_t.backward()
        grad_accum = pv.grad.detach().float() if grad_accum is None else grad_accum + pv.grad.detach().float()
        if k == 0 and sigma == 0.0:
            clean_margin = float(m_t)
    if clean_margin is None:  # smoothgrad path: report the clean (noise-free) margin separately
        with torch.no_grad():
            clean_margin = float(margin_from_inputs(model, inputs, tok_pos, tok_neg))
    return grad_accum / n, clean_margin


def run_saliency(processor, model, tok_pos, tok_neg, merge, frameA, frameB, task, out_dir, tag,
                 clip_pct=99.0, smoothgrad=0, noise_frac=0.1,
                 labelA="A(first)", labelB="B(second)"):
    """Compute m(A,B), m(B,A), m~, and the saliency map of m(A,B) w.r.t. both images' pixels."""
    # symmetrized scalar margins (no grad needed for the reverse orientation)
    with torch.no_grad():
        inputs_ba = build_inputs(processor, model, frameB, frameA, task)
        m_ba = float(margin_from_inputs(model, inputs_ba, tok_pos, tok_neg))

    inputs_ab = build_inputs(processor, model, frameA, frameB, task)
    grad, m_ab = pixel_grad_for_margin(model, inputs_ab, tok_pos, tok_neg, smoothgrad, noise_frac)
    m_tilde = 0.5 * (m_ab - m_ba)

    heats = _patch_saliency_per_image(grad, inputs_ab["image_grid_thw"], merge)
    margins = {"m_ab": m_ab, "m_ba": m_ba, "m_tilde": m_tilde}
    note = f"({labelB} is more progress; expect m~>0)"
    png = _save_overlays([frameA, frameB], heats, margins, out_dir, tag, clip_pct=clip_pct, note=note)

    logger.info(f"m(A,B)={m_ab:+.4f}  m(B,A)={m_ba:+.4f}  m~={m_tilde:+.4f}")
    for name, heat in zip([labelA, labelB], heats):
        cf, hf = _diagnostics(heat)
        logger.info(f"  saliency[{name}]: center_frac={cf:.3f}  high_freq_ratio={hf:.4f}")
    logger.info(f"wrote {png}")
    return margins


def run_corr(processor, model, tok_pos, tok_neg, specs, task, n, label="", verbose=True):
    """Signal-quality check: m~ vs ground truth over pair specs (B placed 2nd => correct is m~>0).

    If more than n specs are available, take an evenly-spaced subsample so coverage stays
    spread across rollouts and (for temporal pairs) across time rather than clustered.
    """
    if not specs:
        logger.info(f"[corr]{label} n=0 (no pairs)")
        return np.asarray([])
    if len(specs) > n:
        idxs = np.unique(np.linspace(0, len(specs) - 1, n).round().astype(int))
        specs = [specs[i] for i in idxs]
    ms, correct = [], 0
    for i, s in enumerate(specs):
        fA, fB = load_pair_frames(s)
        with torch.no_grad():
            m_ab = float(margin_from_inputs(model, build_inputs(processor, model, fA, fB, task), tok_pos, tok_neg))
            m_ba = float(margin_from_inputs(model, build_inputs(processor, model, fB, fA, task), tok_pos, tok_neg))
        m_tilde = 0.5 * (m_ab - m_ba)
        ms.append(m_tilde)
        correct += int(m_tilde > 0)
        if verbose:
            logger.info(f"  [{i+1}/{len(specs)}] {s['tag']}  m~={m_tilde:+.4f}  {'OK' if m_tilde>0 else 'WRONG'}")
    ms = np.asarray(ms)
    logger.info(
        f"[corr]{label} n={len(ms)}  sign-agreement={correct/max(len(ms),1):.3f}  "
        f"mean(m~)={ms.mean():+.4f}  std={ms.std():.4f}  min={ms.min():+.4f}  max={ms.max():+.4f}"
    )
    return ms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT)
    p.add_argument("--task_name", default="CloseBlenderLid")
    p.add_argument("--max_pixels", default="960x540", help="WxH image budget")
    p.add_argument("--pairing", choices=["same_seed", "temporal"], default="same_seed",
                   help="same_seed: failure vs same-seed success (last frames). "
                        "temporal: frame i vs i+gap within one success rollout (later=more progress).")
    p.add_argument("--frame_gap", type=int, default=10, help="[temporal] frames between the two frames of a pair")
    p.add_argument("--frame_stride", type=int, default=5, help="[temporal] step between consecutive pair starts")
    p.add_argument("--gaps", default=None,
                   help="[temporal] comma-separated gap sweep (e.g. 5,10,20,40,80). corr-only, no saliency map.")
    p.add_argument("--seed", default=None, help="[same_seed] force a specific seed (e.g. 100007)")
    p.add_argument("--sal_index", type=int, default=None,
                   help="[temporal] start frame i for the saliency pair (default: middle of first rollout)")
    p.add_argument("--out_dir", default="logs/critic_saliency")
    p.add_argument("--corr", type=int, default=0, help="Also run margin-correlation over N same-seed pairs")
    p.add_argument("--clip_pct", type=float, default=99.0,
                   help="Percentile to clip the saliency at for display (raw grads are heavy-tailed; "
                        "max-normalization hides all but the top ~0.1%% of patches)")
    p.add_argument("--smoothgrad", type=int, default=0,
                   help="SmoothGrad samples: average |grad| over N noisy input copies to de-spike the map")
    p.add_argument("--noise_frac", type=float, default=0.1,
                   help="SmoothGrad noise std as a fraction of the pixel-value std")
    args = p.parse_args()

    processor, model = rv.load_ranker(args.checkpoint, args.max_pixels)
    for prm in model.parameters():
        prm.requires_grad_(False)
    tok_pos, tok_neg = resolve_margin_tokens(processor.tokenizer)
    merge = int(getattr(processor.image_processor, "merge_size", 2))

    # Gap sweep: corr-only across several temporal gaps (model loaded once). No saliency map.
    if args.gaps:
        for g in (int(x) for x in args.gaps.split(",")):
            specs = build_pairs_temporal(args.eval_root, args.task_name, g, args.frame_stride)
            run_corr(processor, model, tok_pos, tok_neg, specs, args.task_name, args.corr or 60,
                     label=f" gap={g}", verbose=False)
        return

    # Build the pool of pair specs for the chosen pairing scheme.
    if args.pairing == "temporal":
        specs = build_pairs_temporal(args.eval_root, args.task_name, args.frame_gap, args.frame_stride)
    else:
        specs = build_pairs_same_seed(args.eval_root, args.task_name)
    if not specs:
        raise SystemExit(f"No {args.pairing} pairs for task {args.task_name}")

    # Pick the pair to render the saliency map for.
    if args.pairing == "temporal":
        v0 = specs[0]["pathA"]
        i = args.sal_index if args.sal_index is not None else _num_frames(v0) // 2
        sal = {"pathA": v0, "idxA": i, "pathB": v0, "idxB": i + args.frame_gap,
               "tag": f"{os.path.basename(v0)[:-4]}_{i}-{i+args.frame_gap}",
               "labelA": f"A(frame {i})", "labelB": f"B(frame {i+args.frame_gap})"}
    elif args.seed:
        sal = next((s for s in specs if s["tag"] == f"seed{args.seed}"), specs[0])
    else:
        sal = specs[0]
    tag = f"{args.task_name}_{args.pairing}_{sal['tag']}"
    logger.info(f"saliency pair ({args.pairing}): {sal['tag']}  A={sal['pathA']}:{sal['idxA']}  "
                f"B={sal['pathB']}:{sal['idxB']}")

    frameA, frameB = load_pair_frames(sal)
    run_saliency(processor, model, tok_pos, tok_neg, merge, frameA, frameB, args.task_name,
                 args.out_dir, tag, clip_pct=args.clip_pct, smoothgrad=args.smoothgrad,
                 noise_frac=args.noise_frac, labelA=sal["labelA"], labelB=sal["labelB"])

    if args.corr:
        run_corr(processor, model, tok_pos, tok_neg, specs, args.task_name, args.corr)


if __name__ == "__main__":
    main()
