# Ctrl-World — I/O Format Notes (reading-only, for the RoboCasa plan)

Documentation of the exact inputs Ctrl-World ingests, extracted from the repo code
(`config.py`, `scripts/rollout_replay_traj.py`, `dataset/dataset_droid_exp33.py`,
`dataset_example/extract_latent.py`, `models/ctrl_world.py`, `scripts/train_wm.py`).
Nothing here was run/modified beyond the Phase-2 replay demo. All line refs are to the
committed repo.

---

## 1. Action format (what the model actually ingests)

**Dimensionality: 7-D per frame** = `[x, y, z, roll, pitch, yaw, gripper]`
(`action_dim = 7`, `config.py:65`). Built from two DROID fields concatenated:
`observation.state.cartesian_position` (6-D) + `observation.state.gripper_position` (1-D)
(`dataset_droid_exp33.py:190-192`). In the rollout the pre-concatenated `states` field
(shape `(N, 7)`) is used directly (`rollout_replay_traj.py:105`, the `car_action` array).

- **xyz** — end-effector Cartesian position, meters, in the robot base frame.
  Percentile bounds (from `dataset_meta_info/droid/stat.json`):
  x ∈ [0.268, 0.781], y ∈ [−0.442, 0.437], z ∈ [−0.043, 0.784].
- **roll/pitch/yaw** — 3-D orientation as **Euler angles in radians** (DROID convention).
  Bounds ≈ [−3.137, 3.137], [−1.214, 0.904], [−2.116, 1.992]. (`scipy ... Rotation`
  is imported for FK/adapter use, but the WM action vector itself is raw rpy, not a matrix/quat.)
- **gripper** — `gripper_position`, 1-D, range ≈ [0, 1] where **0 = open, 1 = closed**
  (`state_99[6] = 0.991`, `state_01[6] = 0.0`). For the replay task `gripper_max = 1.0`
  (`config.py:76`, used to rescale gripper only in the policy/keyboard/π0.5 paths).

**Normalization** (`normalize_bound`, `rollout_replay_traj.py:74-84`,
`dataset_droid_exp33.py:105-115`):
`x_norm = 2*(x − p01)/(p99 − p01 + 1e-8) − 1`, then clipped to **[−1, 1]**, per-dimension,
using the 1st/99th percentiles in `dataset_meta_info/<name>/stat.json`
(rollout hardcodes `data_stat_path = dataset_meta_info/droid/stat.json`, `config.py:86`).

**Temporal window & downsampling** (this is the "1-second / 15-step → 5-step" detail):
- Raw DROID runs at **15 Hz**. Preprocessing keeps every 3rd frame (`rgb_skip = 3` /
  `down_sample = 3`, `extract_latent.py:174`, `config.py:26`) → **5 Hz** latents/videos.
  So a **1-second chunk = 15 raw steps → 5 model steps** (`num_frames = 5`, `config.py:63`;
  `pred_step = 5`, `config.py:81`).
- Per world-model forward the action tensor is **(num_history + num_frames, 7) = (6 + 5, 7)
  = (11, 7)** (asserted `rollout_replay_traj.py:152, 306`): 6 sparse history poses followed
  by the 5-step (1 s) future action chunk.
- In training, the 15 Hz state arrays are indexed at the 5 Hz frame positions via
  `state_id = rgb_id * down_sample(3)` (`dataset_droid_exp33.py:167`). Future frame spacing
  uses `skip ∈ {1,2}` as data augmentation; at inference the replay uses 5 consecutive 5 Hz
  steps per interaction (`interact_num = 12`, `config.py:83`).
- The 7-D action is embedded per-frame by an MLP `7 → 1024` (`Action_encoder2`,
  `ctrl_world.py:71-107`, `frame_level_cond = True`), then the CLIP text embedding of the
  instruction is added to each frame token.

---

## 2. Camera / multi-view input format

**3 views per timestep** (`extract_latent.py:68-71`), in this fixed order:
| idx | DROID stream | type |
|-----|--------------|------|
| 0 | `observation.images.exterior_1_left` | third-person |
| 1 | `observation.images.exterior_2_left` | third-person |
| 2 | `observation.images.wrist_left`      | wrist |

- **Resolution:** each view resized to **192×320 (H×W)** (`size=(192,320)`,
  `extract_latent.py:173`), bilinear.
- **Latent encoding:** SVD VAE (`AutoencoderKLTemporalDecoder`, ÷8 spatial) →
  **(T, 4, 24, 40)** per view, saved as `.pt` (`extract_latent.py:106-117`).
- **View packing:** the 3 view latents are **stacked along the latent height** into a single
  tensor **(T, 4, 72, 40)** (72 = 3 × 24): view0 → `[:, :, 0:24]`, view1 → `[24:48]`,
  view2 → `[48:72]` (`dataset_droid_exp33.py:183-186`; rollout concatenates the 3 per-step
  view latents along the channel-height axis into `(1, 4, 72, 40)`, `rollout_replay_traj.py:266`,
  asserted throughout).
- **At the model** generation runs at `height = 192*3 = 576`, `width = 320`
  (`rollout_replay_traj.py:171`). The pipeline output is un-stacked back to per-view frames via
  `einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)`
  (`rollout_replay_traj.py:182`) → 3 views recovered as batch entries.
- **Batch layout:** condition-image latent per step is `(1, 4, 72, 40)`; the full conditioning
  stack across time is `(1, 11, 4, 72, 40)`.

---

## 3. History / memory input format (sparse history frames)

- **num_history = 6** sparse history frames are prepended to the 5 future frames → 11
  conditioning frames total (`config.py:64`).
- **History latents:** a rolling `his_cond` buffer of `(1, 4, 72, 40)` latents. Sparse
  retrieval indices used at inference (recent-weighted):
  **`history_idx = [0, 0, -8, -6, -4, -2]`** (`rollout_replay_traj.py:300`).
  *(Note: `config.py:88` carries a different default `[0,0,-12,-9,-6,-3]`; the replay script
  overrides it with the line-300 value — flag this if it matters for the next plan.)*
  → `his_cond_input` shape **(1, 6, 4, 72, 40)**; matching history poses `his_pose` (6, 7) are
  concatenated with the 5 future poses to form `action_cond` (11, 7)
  (`rollout_replay_traj.py:301-307`).
- **Autoregressive memory:** the buffer is seeded with `num_history*4 = 24` copies of the t=0
  frame (`rollout_replay_traj.py:268-271`); after each interaction step the model's **own
  predicted** last-frame latent is appended (`rollout_replay_traj.py:313-314`), so history is
  self-generated across the rollout (the memory-retrieval mechanism).
- **Training conditioning** (`ctrl_world.py:190-209`): history frames get Gaussian noise
  (σ up to 0.3) added as a robustness condition; the current frame is channel-stacked as the
  SVD image condition; diffusion noise is added only to the future frames and the **MSE loss is
  computed on the 5 future frames only** (EDM-style `predict_x0`). Data aug: history spacing
  `skip_his = skip*4` with `skip ∈ {1,2}`, and 15% of the time `skip_his = 0`
  (all history = current frame) (`dataset_droid_exp33.py:150-153`).

---

## 4. VLAW post-training entry point (do not run)

- **Script:** `scripts/train_wm.py`, launched with Accelerate (readme §Pre/Post-Training and
  §(3) "Post-train world model on down-stream tasks"):
  ```bash
  # smoke test on bundled subset
  WANDB_MODE=offline accelerate launch --main_process_port 29501 scripts/train_wm.py \
    --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
    --dataset_names droid_subset
  # full / down-stream dataset
  accelerate launch --main_process_port 29501 scripts/train_wm.py \
    --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
    --dataset_names <your_dataset>
  ```
  Pre-training and VLAW post-training use the **same** script; post-training just points at a
  down-stream dataset and (optionally) resumes from the released Ctrl-World checkpoint via
  `--ckpt_path` (`train_wm.py:41-43`, `load_state_dict(strict=True)`). Checkpoints are saved as
  `checkpoint-<step>.pt` every `checkpointing_steps = 20000` into `output_dir = model_ckpt/<tag>`
  (`train_wm.py:129-131`, `config.py:33,46`). Hardware target: 1–2 nodes × 8 A100/H100
  (readme §(0)).

- **Data format the trainer expects** (identical to `dataset_example/droid_subset`, produced by
  `extract_latent.py` then `dataset_meta_info/create_meta_info.py`):
  ```
  <dataset_root_path>/<dataset_name>/
    annotation/{train,val}/<episode_id>.json     # texts, states(7), observation.state.*, action.*, videos, latent_videos
    latent_videos/{train,val}/<episode_id>/{0,1,2}.pt   # per-view VAE latents, (T,4,24,40)
    videos/{train,val}/<episode_id>/{0,1,2}.mp4         # 192x320, 5 Hz, 3 views
  dataset_meta_info/<dataset_name>/
    stat.json                # {"state_01":[7], "state_99":[7]}  percentile norm bounds
    {train,val}_sample.json  # list of {"episode_id": id, "frame_ids": [start]}
  ```
  The loader is `dataset.dataset_droid_exp33.Dataset_mix` (`train_wm.py:68-70`); it reads latents
  from `latent_videos[cam_id]['latent_video_path']` and actions from
  `observation.state.cartesian_position` + `observation.state.gripper_position`.

- **Auxiliary — action adapter** (`models/action_adapter/train2.py`, weights
  `models/action_adapter/model2_15_9.pth`): a small MLP mapping joint position + joint velocity →
  future Cartesian pose. Used **only** in the π0.5 VLA-in-the-loop demo to convert a policy's
  joint-velocity actions into the 7-D Cartesian the WM ingests. **Not needed** for the DROID
  replay path (replay feeds recorded `states` directly).

---

## Quick reference (shapes at a glance)

| Tensor | Shape | Notes |
|--------|-------|-------|
| action per WM forward | `(11, 7)` | 6 history + 5 future; normalized [−1,1] |
| future action chunk   | `(5, 7)`  | 1 s @ 5 Hz (`num_frames = pred_step = 5`) |
| per-view pixel frame   | `192×320×3` | 3 views: 2 exterior + 1 wrist |
| per-view latent        | `(T, 4, 24, 40)` | SVD VAE ÷8 |
| packed multi-view latent | `(T, 4, 72, 40)` | 3 views stacked on height (72 = 3×24) |
| condition latent / step | `(1, 4, 72, 40)` | current frame |
| history condition      | `(1, 6, 4, 72, 40)` | sparse `history_idx=[0,0,-8,-6,-4,-2]` |
| generation canvas      | `H=576, W=320` | = 192×3 views stacked |
| norm stats             | `state_01[7], state_99[7]` | `dataset_meta_info/droid/stat.json` |
