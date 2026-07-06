"""
Build ONE pooled Ctrl-World dataset from ALL RoboCasa v1.0 target/atomic demo tasks
(18 tasks, ~9k episodes). Multi-task version of build_robocasa_dataset.py, with the same
resumable + shardable design as build_gr00t_rollout_dataset.py.

Input: <atomic_root>/<Task>/<date>/lerobot/{data/chunk-000/episode_*.parquet,
       videos/chunk-000/<view>/episode_*.mp4, meta/tasks.jsonl}
Output (schema identical to build_robocasa_dataset.py -> train_wm.py needs ZERO changes):
  dataset_example/<name>/{annotation,latent_videos,videos}/{train,val}/<ep_id>/...
  dataset_meta_info/<name>/{stat.json,train_sample.json,val_sample.json}

Conventions reused verbatim: 7-D EEF pose [eef_pos(3), quat(xyzw)->euler(3), gripper=-(qpos0-qpos1)];
latents at 5Hz (every 4th of 20Hz); observation.state.* on a 3*T_lat grid so down_sample=3 holds.
Episode ids are DETERMINISTIC (<task>__ep<idx>), so re-runs skip built episodes and shards over
disjoint --tasks write to the same dir without colliding.

Run in robocasa_dp env (torch+cuda+diffusers+pyarrow+decord). Parallel: see scripts/convert_robocasa_atomic_parallel.sh
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq_meta
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from decord import VideoReader, cpu
import imageio

VIEW_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]
DOWN_LAT = 4
DS = 3
NAT_PER_STATE = DOWN_LAT / DS
H, W = 192, 320
SEQLEN = 8


def build_state_7d(state16):
    eef_pos = state16[:, 7:10]
    q = state16[:, 10:14]
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    rpy = Rotation.from_quat(q).as_euler("xyz")
    grip = -(state16[:, 14] - state16[:, 15])
    return np.concatenate([eef_pos, rpy, grip[:, None]], axis=1).astype(np.float64)


def read_view_frames(src_mp4, keep):
    vr = VideoReader(src_mp4, ctx=cpu(0), num_threads=2)
    n = len(vr)
    keep = [min(i, n - 1) for i in keep]
    try:
        return vr.get_batch(keep).asnumpy()
    except Exception:
        return vr.get_batch(keep).numpy()


@torch.no_grad()
def encode_latents(vae, frames, device):
    x = torch.from_numpy(frames).to(device).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    resized = (((x / 2 + 0.5).clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8))
    lat = []
    for i in range(0, len(x), 32):
        lat.append(vae.encode(x[i:i+32].to(vae.dtype)).latent_dist.sample().mul_(vae.config.scaling_factor).cpu())
    return torch.cat(lat, 0).to(torch.float16), resized


def find_lerobot(atomic_root, task):
    hits = sorted(glob.glob(os.path.join(atomic_root, task, "*", "lerobot")))
    return hits[0] if hits else None


def already_done(root, mode, ep_id):
    a = os.path.join(root, "annotation", mode, f"{ep_id}.json")
    lat = [os.path.join(root, "latent_videos", mode, ep_id, f"{i}.pt") for i in range(3)]
    return os.path.exists(a) and all(os.path.exists(p) for p in lat)


def ep_len(pq_path):
    return pq_meta.ParquetFile(pq_path).metadata.num_rows


def process(lerobot, idx, ep_id, mode, root, tasks_map, vae, device, write_video):
    df = pd.read_parquet(os.path.join(lerobot, "data", "chunk-000", f"episode_{idx:06d}.parquet"))
    N = len(df)
    s7 = build_state_7d(np.stack(df["observation.state"].values))
    L = list(range(0, N, DOWN_LAT)); T_lat = len(L)
    M = DS * T_lat
    grid = np.clip(np.round(np.arange(M) * NAT_PER_STATE).astype(int), 0, N - 1)
    s7g = s7[grid]
    tdi = int(np.asarray(df["annotation.human.task_description"].iloc[0]).reshape(-1)[0])
    lang = tasks_map.get(tdi, tasks_map.get(int(df["task_index"].iloc[0]), ""))

    lat_dir = os.path.join(root, "latent_videos", mode, ep_id); os.makedirs(lat_dir, exist_ok=True)
    if write_video:
        vid_dir = os.path.join(root, "videos", mode, ep_id); os.makedirs(vid_dir, exist_ok=True)
    for vi, vk in enumerate(VIEW_KEYS):
        src = os.path.join(lerobot, "videos", "chunk-000", vk, f"episode_{idx:06d}.mp4")
        lat, resized = encode_latents(vae, read_view_frames(src, L), device)
        torch.save(lat, os.path.join(lat_dir, f"{vi}.pt"))
        if write_video:
            imageio.mimwrite(os.path.join(vid_dir, f"{vi}.mp4"), resized, fps=5, macro_block_size=1, quality=8)

    anno = {
        "texts": [lang], "episode_id": ep_id, "success": 1,
        "video_length": T_lat, "state_length": T_lat, "raw_length": N,
        "videos": [{"video_path": f"videos/{mode}/{ep_id}/{i}.mp4"} for i in range(3)],
        "latent_videos": [{"latent_video_path": f"latent_videos/{mode}/{ep_id}/{i}.pt"} for i in range(3)],
        "states": s7[L].tolist(),
        "joints": np.zeros((T_lat, 8), np.float32).tolist(),
        "observation.state.cartesian_position": s7g[:, :6].tolist(),
        "observation.state.gripper_position": s7g[:, 6].tolist(),
        "observation.state.joint_position": np.zeros((M, 7), np.float32).tolist(),
    }
    os.makedirs(os.path.join(root, "annotation", mode), exist_ok=True)
    with open(os.path.join(root, "annotation", mode, f"{ep_id}.json"), "w") as f:
        json.dump(anno, f)
    return T_lat


def make_samples(anno_dir, seqlen=SEQLEN):
    out = []
    for jf in sorted(glob.glob(os.path.join(anno_dir, "*.json"))):
        a = json.load(open(jf)); n = a["video_length"]
        for s in range(0, max(1, n - int(seqlen * 0.5))):
            out.append({"episode_id": a["episode_id"], "frame_ids": [s]})
    return out


def build_meta(root, meta_dir):
    os.makedirs(meta_dir, exist_ok=True)
    train_states = [np.array(json.load(open(jf))["states"])
                    for jf in glob.glob(os.path.join(root, "annotation", "train", "*.json"))]
    if not train_states:
        print("build_meta: no train annotations -> nothing to do"); return
    S = np.concatenate(train_states, 0)
    json.dump({"state_01": np.percentile(S, 1, 0).tolist(), "state_99": np.percentile(S, 99, 0).tolist()},
              open(os.path.join(meta_dir, "stat.json"), "w"), indent=2)
    import random
    for mode in ["train", "val"]:
        samp = make_samples(os.path.join(root, "annotation", mode))
        random.Random(0).shuffle(samp)
        json.dump(samp, open(os.path.join(meta_dir, f"{mode}_sample.json"), "w"))
        print(f"{mode}: {len(glob.glob(os.path.join(root,'annotation',mode,'*.json')))} episodes, {len(samp)} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atomic_root", default="/proj/vondrick3/sruthi/Appaji/robocasa_new/datasets/v1.0/target/atomic")
    ap.add_argument("--out_root", default="dataset_example")
    ap.add_argument("--meta_root", default="dataset_meta_info")
    ap.add_argument("--name", default="robocasa_atomic_all")
    ap.add_argument("--svd_vae", default="checkpoints/svd")
    ap.add_argument("--val_per_task", type=int, default=2)
    ap.add_argument("--min_len_5hz", type=int, default=20)
    ap.add_argument("--no_train_video", action="store_true")
    ap.add_argument("--tasks", default="", help="comma-separated task filter (default: all). For sharding.")
    ap.add_argument("--no_meta", action="store_true")
    ap.add_argument("--meta_only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = os.path.join(args.out_root, args.name)
    meta_dir = os.path.join(args.meta_root, args.name)
    if args.meta_only:
        build_meta(root, meta_dir); print(f"meta -> {meta_dir}"); return

    from diffusers.models import AutoencoderKLTemporalDecoder
    device = "cuda"
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_vae, subfolder="vae").to(device).to(torch.float16).eval()

    all_tasks = sorted(d for d in os.listdir(args.atomic_root) if os.path.isdir(os.path.join(args.atomic_root, d)))
    task_filter = set(t for t in args.tasks.split(",") if t)
    tasks = [t for t in all_tasks if (not task_filter or t in task_filter)]
    print(f"tasks: {tasks}")

    plan = []  # (lerobot, idx, ep_id, mode, tasks_map)
    for task in tasks:
        lerobot = find_lerobot(args.atomic_root, task)
        if lerobot is None:
            print(f"  no lerobot dir for {task}, skip"); continue
        tasks_map = {json.loads(l)["task_index"]: json.loads(l)["task"]
                     for l in open(os.path.join(lerobot, "meta", "tasks.jsonl"))}
        parquets = sorted(glob.glob(os.path.join(lerobot, "data", "chunk-000", "episode_*.parquet")))
        nval = 0
        for p in parquets:
            idx = int(os.path.basename(p).split("_")[1].split(".")[0])
            ep_id = f"{task}__ep{idx:06d}"
            mode = "train"
            if nval < args.val_per_task and len(range(0, ep_len(p), DOWN_LAT)) >= args.min_len_5hz:
                mode = "val"; nval += 1
            plan.append((lerobot, idx, ep_id, mode, tasks_map))
    if args.limit:
        plan = plan[:args.limit]
    print(f"{len(plan)} episodes planned")

    done = skipped = errored = 0
    for lerobot, idx, ep_id, mode, tasks_map in plan:
        if already_done(root, mode, ep_id):
            skipped += 1; continue
        wv = not (mode == "train" and args.no_train_video)
        try:
            p = os.path.join(lerobot, "data", "chunk-000", f"episode_{idx:06d}.parquet")
            if len(range(0, ep_len(p), DOWN_LAT)) < args.min_len_5hz:
                continue
            T_lat = process(lerobot, idx, ep_id, mode, root, tasks_map, vae, device, wv)
            done += 1
            if done % 50 == 0:
                print(f"  built {done}, skipped {skipped} ... last {ep_id} T_lat={T_lat}")
        except Exception as e:
            errored += 1; print(f"  ERROR {ep_id}: {e}")
    print(f"built={done} skipped={skipped} errored={errored}")

    if not args.no_meta:
        build_meta(root, meta_dir)
        print(f"\nDONE -> {root}")


if __name__ == "__main__":
    main()
