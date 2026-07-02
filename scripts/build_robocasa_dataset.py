"""
Build a FULL train/val Ctrl-World training dataset from RoboCasa v1.0 LeRobot data
(OpenDrawer by default). Unlike convert_robocasa_to_ctrlworld.py (which made a small
val-only set for the replay demo), this produces everything scripts/train_wm.py needs:

  <out>/robocasa_opendrawer_full/
    annotation/{train,val}/<ep>.json
    latent_videos/{train,val}/<ep>/{0,1,2}.pt      # SVD-VAE latents (T_lat,4,24,40), fp16
    videos/{train,val}/<ep>/{0,1,2}.mp4            # 192x320, 5Hz (for replay-style eval)
  dataset_meta_info/robocasa_opendrawer_full/
    stat.json                                      # state_01[7], state_99[7] over TRAIN
    {train,val}_sample.json                        # [{episode_id, frame_ids:[start]}]

Design (see chat / CTRL_WORLD_IO_NOTES.md), all chosen to need ZERO changes to the repo:
  * Latents at 5 Hz = every 4th native (20 Hz) frame -> T_lat frames per view.
  * The loader (dataset_droid_exp33.py) hardcodes frame_len = joint_len/3 and reads
    state_id = rgb_id * down_sample (down_sample=3). It only ever reads state indices at
    multiples of 3. So we store observation.state.* on a length M = 3*T_lat grid, where
    index j maps to native frame round(j*4/3): at j=3k this is native frame 4k, i.e. exactly
    the moment of latent frame k. => perfect pose<->latent alignment with down_sample=3.
  * 7-D EEF pose = [eef_pos(3), eef_rpy(3 from quat xyzw), gripper(1) = -(qpos0-qpos1)].
    Base-relative frame kept on purpose (agentview cams are base-mounted). Base dropped.
  * Views order [agentview_left, agentview_right, eye_in_hand]; each stretched to 192x320.
  * Latent encoding matches extract_latent.py exactly:
    frames/255*2-1 -> bilinear 192x320 -> vae.encode().latent_dist.sample()*scaling_factor.

Run in an env with torch+cuda, diffusers, pyarrow, decord (robocasa_dp works):
  CUDA_VISIBLE_DEVICES=0 /proj/vondrick3/sruthi/miniconda3/envs/robocasa_dp/bin/python \
      scripts/build_robocasa_dataset.py            # full 514 episodes
  ... --limit 3                                    # quick smoke test
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from decord import VideoReader, cpu
import imageio
from diffusers.models import AutoencoderKLTemporalDecoder

VIEW_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]
DOWN_LAT = 4            # 20 Hz native -> 5 Hz latents (every 4th frame)
DS = 3                 # down_sample the loader uses (state grid = DS * T_lat)
NAT_PER_STATE = DOWN_LAT / DS   # native frames per state-grid step = 4/3
H, W = 192, 320
SEQLEN = 8             # matches create_meta_info.py sequence_length


def build_state_7d(state16):
    eef_pos = state16[:, 7:10]
    eef_quat = state16[:, 10:14]
    eef_quat = eef_quat / (np.linalg.norm(eef_quat, axis=1, keepdims=True) + 1e-8)
    rpy = Rotation.from_quat(eef_quat).as_euler("xyz")
    grip = -(state16[:, 14] - state16[:, 15])
    return np.concatenate([eef_pos, rpy, grip[:, None]], axis=1).astype(np.float64)  # (N,7)


def read_view_frames(src_mp4, keep_idx):
    vr = VideoReader(src_mp4, ctx=cpu(0), num_threads=2)
    n = len(vr)
    keep_idx = [min(i, n - 1) for i in keep_idx]
    try:
        f = vr.get_batch(keep_idx).asnumpy()
    except Exception:
        f = vr.get_batch(keep_idx).numpy()
    return f  # (T,256,256,3) uint8


@torch.no_grad()
def encode_latents(vae, frames_uint8, device):
    # frames: (T,H0,W0,3) uint8 -> (T,3,192,320) in [-1,1] -> VAE latent (T,4,24,40)
    x = torch.from_numpy(frames_uint8).to(device).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    resized = (((x / 2 + 0.5).clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8))
    lat = []
    for i in range(0, len(x), 32):
        b = x[i:i + 32].to(vae.dtype)
        lat.append(vae.encode(b).latent_dist.sample().mul_(vae.config.scaling_factor).cpu())
    return torch.cat(lat, 0).to(torch.float16), resized  # (T,4,24,40) fp16, resized frames


def process_episode(pq, ep, mode, out_ds, vae, device, write_video=True):
    df = pd.read_parquet(pq)
    N = len(df)
    state16 = np.stack(df["observation.state"].values)
    s7 = build_state_7d(state16)                       # (N,7) native 20Hz

    L = list(range(0, N, DOWN_LAT))                    # latent frames (native idx)
    T_lat = len(L)
    M = DS * T_lat                                     # state-grid length
    # native index for each state-grid position j; j=3k -> 4k exactly
    grid_native = np.clip(np.round(np.arange(M) * NAT_PER_STATE).astype(int), 0, N - 1)
    s7_grid = s7[grid_native]                          # (M,7)

    cart = s7_grid[:, :6].tolist()                     # (M,6) pos+rpy
    grip = s7_grid[:, 6].tolist()                      # (M,)
    states_5hz = s7[L].tolist()                        # (T_lat,7) for replay/stat

    # instruction
    tasks = out_ds["_tasks"]
    tdi = int(np.asarray(df["annotation.human.task_description"].iloc[0]).reshape(-1)[0])
    instruction = tasks.get(tdi, tasks.get(int(df["task_index"].iloc[0]), ""))

    ep_dir_lat = os.path.join(out_ds["root"], "latent_videos", mode, str(ep))
    ep_dir_vid = os.path.join(out_ds["root"], "videos", mode, str(ep))
    os.makedirs(ep_dir_lat, exist_ok=True)
    if write_video:
        os.makedirs(ep_dir_vid, exist_ok=True)

    for vi, vk in enumerate(VIEW_KEYS):
        src = os.path.join(out_ds["src"], "videos", "chunk-000", vk, f"episode_{ep:06d}.mp4")
        frames = read_view_frames(src, L)
        lat, resized = encode_latents(vae, frames, device)
        torch.save(lat, os.path.join(ep_dir_lat, f"{vi}.pt"))
        if write_video:
            imageio.mimwrite(os.path.join(ep_dir_vid, f"{vi}.mp4"), resized, fps=5,
                             macro_block_size=1, quality=8)

    anno = {
        "texts": [instruction],
        "episode_id": ep,
        "success": 1,
        "video_length": T_lat,
        "state_length": T_lat,
        "raw_length": N,
        "videos": [{"video_path": f"videos/{mode}/{ep}/{i}.mp4"} for i in range(3)],
        "latent_videos": [{"latent_video_path": f"latent_videos/{mode}/{ep}/{i}.pt"} for i in range(3)],
        "states": states_5hz,                                  # (T_lat,7) replay/create_meta
        "joints": np.zeros((T_lat, 8), np.float32).tolist(),   # (T_lat,8) replay assert
        "observation.state.cartesian_position": cart,          # (M,6) training
        "observation.state.gripper_position": grip,            # (M,) training
        "observation.state.joint_position": np.zeros((M, 7), np.float32).tolist(),  # length only
    }
    os.makedirs(os.path.join(out_ds["root"], "annotation", mode), exist_ok=True)
    with open(os.path.join(out_ds["root"], "annotation", mode, f"{ep}.json"), "w") as f:
        json.dump(anno, f)
    return T_lat, np.array(states_5hz)


def make_samples(anno_dir, seqlen=SEQLEN, start_interval=1):
    samples = []
    for jf in sorted(glob.glob(os.path.join(anno_dir, "*.json"))):
        a = json.load(open(jf))
        n = a["video_length"]
        end_idx = max(1, n - int(seqlen * 0.5))
        for s in range(0, end_idx, start_interval):
            samples.append({"episode_id": a["episode_id"], "frame_ids": [s]})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/proj/vondrick3/sruthi/Appaji/robocasa_new/"
                    "datasets/v1.0/target/atomic/OpenDrawer/20250816/lerobot")
    ap.add_argument("--out_root", default="dataset_example")
    ap.add_argument("--meta_root", default="dataset_meta_info")
    ap.add_argument("--name", default="robocasa_opendrawer_full")
    ap.add_argument("--svd_vae", default="checkpoints/svd")
    ap.add_argument("--val_count", type=int, default=20)
    ap.add_argument("--min_val_len", type=int, default=70)
    ap.add_argument("--limit", type=int, default=0, help="0 = all episodes")
    ap.add_argument("--no_train_video", action="store_true", help="skip writing train mp4s")
    args = ap.parse_args()

    device = "cuda"
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_vae, subfolder="vae").to(device).to(torch.float16).eval()

    tasks = {}
    for line in open(os.path.join(args.src, "meta", "tasks.jsonl")):
        d = json.loads(line); tasks[d["task_index"]] = d["task"]

    root = os.path.join(args.out_root, args.name)
    out_ds = {"root": root, "src": args.src, "_tasks": tasks}

    parquets = sorted(glob.glob(os.path.join(args.src, "data", "chunk-000", "*.parquet")))
    if args.limit:
        parquets = parquets[:args.limit]

    # split: first val_count episodes with enough length -> val, rest -> train
    val_eps, train_eps = [], []
    for pq in parquets:
        ep = int(os.path.basename(pq).split("_")[1].split(".")[0])
        if len(val_eps) < args.val_count:
            N = len(pd.read_parquet(pq, columns=["frame_index"]))
            if len(range(0, N, DOWN_LAT)) >= args.min_val_len:
                val_eps.append((pq, ep)); continue
        train_eps.append((pq, ep))
    print(f"train={len(train_eps)} episodes, val={len(val_eps)} episodes")

    train_states = []
    for mode, eps in [("val", val_eps), ("train", train_eps)]:
        wv = not (mode == "train" and args.no_train_video)
        for i, (pq, ep) in enumerate(eps):
            done = os.path.exists(os.path.join(root, "annotation", mode, f"{ep}.json"))
            if done:
                if mode == "train":
                    train_states.append(np.array(json.load(open(
                        os.path.join(root, "annotation", mode, f"{ep}.json")))["states"]))
                print(f"[{mode} {i+1}/{len(eps)}] ep{ep} exists, skip"); continue
            T_lat, st = process_episode(pq, ep, mode, out_ds, vae, device, write_video=wv)
            if mode == "train":
                train_states.append(st)
            print(f"[{mode} {i+1}/{len(eps)}] ep{ep}: T_lat={T_lat}")

    # meta: stat.json (over train) + sample jsons
    meta_dir = os.path.join(args.meta_root, args.name)
    os.makedirs(meta_dir, exist_ok=True)
    S = np.concatenate(train_states, 0)
    json.dump({"state_01": np.percentile(S, 1, 0).tolist(),
               "state_99": np.percentile(S, 99, 0).tolist()},
              open(os.path.join(meta_dir, "stat.json"), "w"), indent=2)
    for mode in ["train", "val"]:
        samp = make_samples(os.path.join(root, "annotation", mode))
        import random; random.Random(0).shuffle(samp)
        json.dump(samp, open(os.path.join(meta_dir, f"{mode}_sample.json"), "w"))
        print(f"{mode}: {samp and len(samp)} samples")

    print(f"\nDONE. dataset -> {root}\n  meta -> {meta_dir}")
    print("Train with:\n  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline accelerate launch --main_process_port 29501 \\")
    print(f"    scripts/train_wm.py --dataset_root_path {args.out_root} \\")
    print(f"    --dataset_meta_info_path {args.meta_root} --dataset_names {args.name} \\")
    print("    --svd_model_path checkpoints/svd --clip_model_path checkpoints/clip \\")
    print("    --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt")


if __name__ == "__main__":
    main()
