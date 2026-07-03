"""
Convert recorded GR00T policy rollouts (from run_eval.py --record_traj_dir, i.e. the
TrajectoryRecordingWrapper output) into a Ctrl-World world-model training dataset.

Input layout (one folder per seeded episode, possibly many tasks):
    <record_dir>/**/<Task>/seed<seed>/{0,1,2}.mp4     # agentview_left/right, eye_in_hand, 256x256, 20Hz
    <record_dir>/**/<Task>/seed<seed>/traj.npz         # eef_pos, eef_quat(xyzw), gripper_qpos,
                                                        # base_pos, base_quat, lang, task, seed, success

Output = identical schema to build_robocasa_dataset.py (train_wm.py needs ZERO changes).
Reuses the SAME conventions as the RoboCasa demo build: 7-D EEF pose
[eef_pos(3), quat(xyzw)->euler rpy(3), gripper = -(qpos0-qpos1)]; latents at 5Hz (every 4th of
20Hz); observation.state.* on a length-3*T_lat grid so down_sample=3 holds. Episodes pooled
across tasks; each episode's instruction is its recorded lang.

Resumable + shardable: episode ids are DETERMINISTIC (<run_dir>__seed<seed>), so re-runs skip
already-built episodes and multiple processes handling disjoint --tasks write to the SAME
dataset dir without colliding. Shards use --no_meta; run one final --meta_only pass at the end.

Run in robocasa_dp env (torch+cuda+diffusers+decord). Single GPU:
  CUDA_VISIBLE_DEVICES=0 <robocasa_dp python> scripts/build_gr00t_rollout_dataset.py --record_dir <path>
8-way shard (see run.txt) then:
  <robocasa_dp python> scripts/build_gr00t_rollout_dataset.py --name robocasa_gr00t_rollouts --meta_only
"""
import os, json, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from decord import VideoReader, cpu
import imageio

DOWN_LAT = 4
DS = 3
NAT_PER_STATE = DOWN_LAT / DS
H, W = 192, 320
SEQLEN = 8


def build_state_7d(eef_pos, eef_quat, grip_qpos):
    q = eef_quat / (np.linalg.norm(eef_quat, axis=1, keepdims=True) + 1e-8)
    rpy = Rotation.from_quat(q).as_euler("xyz")
    grip = -(grip_qpos[:, 0] - grip_qpos[:, 1])
    return np.concatenate([eef_pos, rpy, grip[:, None]], axis=1).astype(np.float64)


def read_frames(mp4, keep):
    vr = VideoReader(mp4, ctx=cpu(0), num_threads=2)
    n = len(vr)
    keep = [min(i, n - 1) for i in keep]
    try:
        return vr.get_batch(keep).asnumpy()
    except Exception:
        return vr.get_batch(keep).numpy()


@torch.no_grad()
def encode(vae, frames, device):
    x = torch.from_numpy(frames).to(device).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    resized = (((x / 2 + 0.5).clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8))
    lat = []
    for i in range(0, len(x), 32):
        lat.append(vae.encode(x[i:i+32].to(vae.dtype)).latent_dist.sample().mul_(vae.config.scaling_factor).cpu())
    return torch.cat(lat, 0).to(torch.float16), resized


def ep_id_for(ep_src):
    # deterministic, unique, filesystem-safe: <run_dir>__<seedNNNN>
    run = os.path.basename(os.path.dirname(os.path.dirname(ep_src)))
    return f"{run}__{os.path.basename(ep_src)}"


def already_done(root, mode, ep_id):
    a = os.path.join(root, "annotation", mode, f"{ep_id}.json")
    lat = [os.path.join(root, "latent_videos", mode, ep_id, f"{i}.pt") for i in range(3)]
    return os.path.exists(a) and all(os.path.exists(p) for p in lat)


def process(ep_src, ep_id, mode, root, vae, device, write_video):
    d = np.load(os.path.join(ep_src, "traj.npz"), allow_pickle=True)
    s7 = build_state_7d(d["eef_pos"], d["eef_quat"], d["gripper_qpos"])
    N = len(s7)
    L = list(range(0, N, DOWN_LAT)); T_lat = len(L)
    M = DS * T_lat
    grid = np.clip(np.round(np.arange(M) * NAT_PER_STATE).astype(int), 0, N - 1)
    s7g = s7[grid]
    lang = str(d["lang"]) if "lang" in d else ""

    lat_dir = os.path.join(root, "latent_videos", mode, ep_id); os.makedirs(lat_dir, exist_ok=True)
    if write_video:
        vid_dir = os.path.join(root, "videos", mode, ep_id); os.makedirs(vid_dir, exist_ok=True)
    for vi in range(3):
        frames = read_frames(os.path.join(ep_src, f"{vi}.mp4"), L)
        lat, resized = encode(vae, frames, device)
        torch.save(lat, os.path.join(lat_dir, f"{vi}.pt"))
        if write_video:
            imageio.mimwrite(os.path.join(vid_dir, f"{vi}.mp4"), resized, fps=5, macro_block_size=1, quality=8)

    anno = {
        "texts": [lang], "episode_id": ep_id, "success": int(bool(d["success"])) if "success" in d else 1,
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
    """Scan the whole dataset dir -> stat.json (over train 'states') + {train,val}_sample.json."""
    os.makedirs(meta_dir, exist_ok=True)
    train_states = []
    for jf in glob.glob(os.path.join(root, "annotation", "train", "*.json")):
        train_states.append(np.array(json.load(open(jf))["states"]))
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
    ap.add_argument("--record_dir", help="root of run_eval --record_traj_dir output")
    ap.add_argument("--out_root", default="dataset_example")
    ap.add_argument("--meta_root", default="dataset_meta_info")
    ap.add_argument("--name", default="robocasa_gr00t_rollouts")
    ap.add_argument("--svd_vae", default="checkpoints/svd")
    ap.add_argument("--val_per_task", type=int, default=2, help="held-out val episodes per task")
    ap.add_argument("--min_len_5hz", type=int, default=20)
    ap.add_argument("--no_train_video", action="store_true")
    ap.add_argument("--tasks", default="", help="comma-separated task filter (default: all). For sharding.")
    ap.add_argument("--no_meta", action="store_true", help="skip stat/sample writing (for shards)")
    ap.add_argument("--meta_only", action="store_true", help="only (re)build stat + sample jsons over the dataset")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = os.path.join(args.out_root, args.name)
    meta_dir = os.path.join(args.meta_root, args.name)

    if args.meta_only:
        build_meta(root, meta_dir)
        print(f"meta written -> {meta_dir}")
        return

    from diffusers.models import AutoencoderKLTemporalDecoder
    device = "cuda"
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_vae, subfolder="vae").to(device).to(torch.float16).eval()

    ep_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.record_dir, "**", "traj.npz"), recursive=True))
    if args.limit:
        ep_dirs = ep_dirs[:args.limit]
    task_filter = set(t for t in args.tasks.split(",") if t)
    by_task = {}
    for ed in ep_dirs:
        task = os.path.basename(os.path.dirname(ed))
        if task_filter and task not in task_filter:
            continue
        by_task.setdefault(task, []).append(ed)
    print(f"{sum(len(v) for v in by_task.values())} episodes across {len(by_task)} tasks"
          f"{' (filtered)' if task_filter else ''}: { {t: len(v) for t, v in by_task.items()} }")

    # per-task: first val_per_task -> val, rest -> train (keeps every task in val)
    plan = []
    for task, eds in by_task.items():
        for i, ed in enumerate(eds):
            plan.append((ed, "val" if i < args.val_per_task else "train"))

    done = skipped = errored = 0
    for ed, mode in plan:
        ep_id = ep_id_for(ed)
        if already_done(root, mode, ep_id):
            skipped += 1; continue
        wv = not (mode == "train" and args.no_train_video)
        try:
            d = np.load(os.path.join(ed, "traj.npz"), allow_pickle=True)
            if len(range(0, len(d["eef_pos"]), DOWN_LAT)) < args.min_len_5hz:
                print(f"  skip {ep_id} (too short)"); continue
            T_lat = process(ed, ep_id, mode, root, vae, device, wv)
            done += 1
            if done % 25 == 0:
                print(f"  [{mode}] {done} built, {skipped} skipped ... last {ep_id} T_lat={T_lat}")
        except Exception as e:
            errored += 1
            print(f"  ERROR on {ep_id}: {e}")
    print(f"built={done} skipped={skipped} errored={errored}")

    if not args.no_meta:
        build_meta(root, meta_dir)
        print(f"\nDONE -> {root}\nCombine with the demo set at train time:")
        print(f"  --dataset_names robocasa_opendrawer_full+{args.name}   (set config.prob=[0.5,0.5])")


if __name__ == "__main__":
    main()
