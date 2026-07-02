"""
Convert a RoboCasa v1.0 LeRobot dataset (e.g. .../target/atomic/OpenDrawer/<date>/lerobot)
into the annotation/video format that Ctrl-World's rollout_replay_traj.py expects.

Design choices (see CTRL_WORLD_IO_NOTES.md and the chat that produced this):
  * 3 views mapped to Ctrl-World's (2 exterior + 1 wrist) order:
        0 = robot0_agentview_left   (exterior)
        1 = robot0_agentview_right  (exterior)
        2 = robot0_eye_in_hand      (wrist)
  * 20 Hz -> 5 Hz by keeping every 4th frame (DROID pipeline is 15->5 by every 3rd).
  * Each view resized (stretched) to 192x320 (H x W) to match DROID's canvas so the
    hardcoded (4,72,40) latent asserts in the replay script hold.
  * 7-D EEF-only state = [eef_pos(3), eef_rpy(3 from quat), gripper(1)].
        eef_pos   = observation.state[7:10]   (end_effector_position_relative)
        eef_quat  = observation.state[10:14]  (assumed xyzw) -> euler 'xyz' radians
        gripper   = -(qpos[0]-qpos[1])         (finger separation, sign so closed is larger,
                                                matching DROID's 0=open/1=closed convention)
    Mobile base motion is intentionally dropped (atomic tasks, base ~static).
  * stat.json (state_01 / state_99 percentiles) is computed over the converted episodes.

This writes:
  <out_root>/<name>/annotation/val/<id>.json
  <out_root>/<name>/videos/val/<id>/{0,1,2}.mp4
  <meta_root>/<name>/stat.json

Run with an env that has pandas+pyarrow+decord+cv2+imageio (e.g. robocasa_dp):
  /proj/vondrick3/sruthi/miniconda3/envs/robocasa_dp/bin/python \
      scripts/convert_robocasa_to_ctrlworld.py --num_episodes 6
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from decord import VideoReader, cpu
import cv2
import imageio

VIEW_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]
DOWN = 4          # 20 Hz -> 5 Hz
OUT_H, OUT_W = 192, 320


def build_state_7d(state16):
    """state16: (N,16) -> (N,7) [x,y,z, r,p,y, gripper]."""
    eef_pos = state16[:, 7:10]
    eef_quat = state16[:, 10:14]                    # assumed xyzw
    # renormalize just in case
    eef_quat = eef_quat / (np.linalg.norm(eef_quat, axis=1, keepdims=True) + 1e-8)
    rpy = Rotation.from_quat(eef_quat).as_euler("xyz")   # (N,3) radians
    grip = -(state16[:, 14] - state16[:, 15])           # finger separation, closed -> larger
    return np.concatenate([eef_pos, rpy, grip[:, None]], axis=1).astype(np.float32)


def load_view_frames(video_path, keep_idx):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    n = len(vr)
    keep_idx = [min(i, n - 1) for i in keep_idx]
    try:
        frames = vr.get_batch(keep_idx).asnumpy()
    except Exception:
        frames = vr.get_batch(keep_idx).numpy()
    out = np.empty((len(frames), OUT_H, OUT_W, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = cv2.resize(f, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/proj/vondrick3/sruthi/Appaji/robocasa_new/"
                    "datasets/v1.0/target/atomic/OpenDrawer/20250816/lerobot")
    ap.add_argument("--out_root", default="dataset_example")
    ap.add_argument("--meta_root", default="dataset_meta_info")
    ap.add_argument("--name", default="robocasa_opendrawer")
    ap.add_argument("--num_episodes", type=int, default=6)
    ap.add_argument("--min_len_5hz", type=int, default=70,
                    help="skip episodes shorter than this many 5Hz frames "
                         "(rollout needs pred_step*interact_num+8 = 68)")
    args = ap.parse_args()

    src = args.src
    info = json.load(open(os.path.join(src, "meta", "info.json")))
    tasks = {}
    for line in open(os.path.join(src, "meta", "tasks.jsonl")):
        d = json.loads(line); tasks[d["task_index"]] = d["task"]
    print(f"fps={info['fps']}, total_episodes={info['total_episodes']}, tasks={tasks}")

    ann_dir = os.path.join(args.out_root, args.name, "annotation", "val")
    vid_root = os.path.join(args.out_root, args.name, "videos", "val")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(vid_root, exist_ok=True)

    parquets = sorted(glob.glob(os.path.join(src, "data", "chunk-000", "*.parquet")))
    all_states = []
    converted = []
    for pq in parquets:
        if len(converted) >= args.num_episodes:
            break
        ep = int(os.path.basename(pq).split("_")[1].split(".")[0])
        df = pd.read_parquet(pq)
        N = len(df)
        keep = list(range(0, N, DOWN))
        if len(keep) < args.min_len_5hz:
            print(f"  skip ep{ep}: {len(keep)} 5Hz frames < {args.min_len_5hz}")
            continue

        state16 = np.stack(df["observation.state"].values)
        states7 = build_state_7d(state16)[keep]           # (T,7)
        T = len(keep)

        # instruction: prefer the language task_description, else the task_name
        task_desc_idx = int(np.asarray(df["annotation.human.task_description"].iloc[0]).reshape(-1)[0])
        instruction = tasks.get(task_desc_idx, tasks.get(int(df["task_index"].iloc[0]), ""))

        # write the 3 view videos at 5Hz
        out_vid_dir = os.path.join(vid_root, str(ep))
        os.makedirs(out_vid_dir, exist_ok=True)
        videos_meta = []
        for vi, vk in enumerate(VIEW_KEYS):
            src_mp4 = os.path.join(src, "videos", "chunk-000", vk, f"episode_{ep:06d}.mp4")
            frames = load_view_frames(src_mp4, keep)      # (T,192,320,3)
            out_mp4 = os.path.join(out_vid_dir, f"{vi}.mp4")
            imageio.mimwrite(out_mp4, frames, fps=5, macro_block_size=1, quality=8)
            videos_meta.append({"video_path": f"videos/val/{ep}/{vi}.mp4"})

        anno = {
            "texts": [instruction],
            "episode_id": ep,
            "success": True,
            "video_length": T,
            "state_length": T,
            "raw_length": N,
            "videos": videos_meta,
            "states": states7.tolist(),          # (T,7)
            "joints": np.zeros((T, 8), np.float32).tolist(),  # unused in replay, shape must be (*,8)
        }
        json.dump(anno, open(os.path.join(ann_dir, f"{ep}.json"), "w"))
        all_states.append(states7)
        converted.append(ep)
        print(f"  ep{ep}: N={N} -> T={T} 5Hz frames | '{instruction}'")

    # stat.json (1st/99th percentile per dim over converted episodes)
    S = np.concatenate(all_states, axis=0)
    stat = {"state_01": np.percentile(S, 1, axis=0).tolist(),
            "state_99": np.percentile(S, 99, axis=0).tolist()}
    meta_dir = os.path.join(args.meta_root, args.name)
    os.makedirs(meta_dir, exist_ok=True)
    json.dump(stat, open(os.path.join(meta_dir, "stat.json"), "w"), indent=2)

    print(f"\nDone. Converted {len(converted)} episodes: {converted}")
    print(f"  annotations -> {ann_dir}")
    print(f"  videos      -> {vid_root}")
    print(f"  stat.json   -> {os.path.join(meta_dir, 'stat.json')}")
    print(f"  state_01 = {[round(x,3) for x in stat['state_01']]}")
    print(f"  state_99 = {[round(x,3) for x in stat['state_99']]}")
    print(f"\nEpisode ids for config.py robocasa block: {converted}")


if __name__ == "__main__":
    main()
