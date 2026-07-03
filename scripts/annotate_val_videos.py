"""
Post-hoc: add history/prediction markers to already-produced training val videos
(train_steps_*.mp4). Use this for runs launched before the marker was added to
train_wm.py (fresh runs get it natively). Non-destructive: writes to <dir>_annotated/.

Each val video is T = num_history + num_frames frames (default 6 + 5 = 11). This adds:
  * GREEN border on the first num_history frames (history: matches GT by construction)
  * RED border on the remaining frames (model-predicted future: judge bottom vs top here)
  * a white horizontal divider between the GT row (top) and history+prediction row (bottom)

Run (ctrl-world env):
  python scripts/annotate_val_videos.py /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World/model_ckpt/robocasa_opendrawer_20260702_130917/samples
  python scripts/annotate_val_videos.py model_ckpt/<RUN_TAG>/samples --num_history 6 --fps 2
"""
import os, glob, argparse
import numpy as np
from decord import VideoReader, cpu
import mediapy


def annotate_val_video(videos, num_history, border=5):
    GREEN = np.array([0, 200, 0], np.uint8)
    RED   = np.array([220, 0, 0], np.uint8)
    WHITE = np.array([255, 255, 255], np.uint8)
    out = videos.copy()
    T, H, W, _ = out.shape
    for t in range(T):
        c = GREEN if t < num_history else RED
        out[t, :border, :] = c; out[t, -border:, :] = c
        out[t, :, :border] = c; out[t, :, -border:] = c
    out[:, H // 2 - 1:H // 2 + 1, :] = WHITE
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_dir", help="dir with train_steps_*.mp4")
    ap.add_argument("--num_history", type=int, default=6)
    ap.add_argument("--fps", type=int, default=2)
    args = ap.parse_args()

    out_dir = args.samples_dir.rstrip("/") + "_annotated"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.samples_dir, "train_steps_*.mp4")))
    print(f"{len(files)} videos -> {out_dir}")
    for f in files:
        vr = VideoReader(f, ctx=cpu(0))
        vid = np.stack([vr[i].asnumpy() for i in range(len(vr))])
        ann = annotate_val_video(vid, args.num_history)
        mediapy.write_video(os.path.join(out_dir, os.path.basename(f)), ann, fps=args.fps)
    print("done")


if __name__ == "__main__":
    main()
