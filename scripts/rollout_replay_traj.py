
# from openpi.training import config as config_pi
# from openpi.policies import policy_config
# from openpi_client import image_tools
import numpy as np


from accelerate import Accelerator
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import key_board_control, get_fk_solution

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import wandb
import json
from decord import VideoReader, cpu
import swanlab
import mediapy
import sys
from scipy.spatial.transform import Rotation as R



def _psnr_mse_per_frame(pred, gt, max_val=255.0):
    """Per-frame PSNR (dB) and MSE between predicted and GT rollout frames.
    pred, gt: (num_views, F, H, W, 3) uint8 arrays of the same shape. Returns two
    length-F lists: mean MSE and PSNR at each frame (averaged over views/pixels/channels)."""
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mses, psnrs = [], []
    for f in range(pred.shape[1]):
        mse = float(np.mean((pred[:, f] - gt[:, f]) ** 2))
        mses.append(mse)
        psnrs.append(99.0 if mse <= 1e-8 else float(10.0 * np.log10(max_val ** 2 / mse)))
    return mses, psnrs


class agent():
    def __init__(self,args):
          
        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # # load pi policy
        # if 'pi05' in args.policy_type:
        #     config = config_pi.get_config("pi05_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid' 
        # elif 'pi0fast' in args.policy_type:
        #     config = config_pi.get_config("pi0fast_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0fast_droid'
        # elif 'pi0' in args.policy_type:
        #     config = config_pi.get_config("pi0_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0_droid'
        # else:
        #     raise ValueError(f"Unknown policy type: {args.policy_type}")
        # self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # load ctrl-world model
        # Auto-detect architecture from the checkpoint so eval "just works" for both old
        # (baseline) and Change-A checkpoints without setting any flag: if the checkpoint
        # carries the temporal action-encoder weights, build the model with that module.
        _state_dict = torch.load(args.val_model_path, map_location='cpu')
        _has_temporal = any(k.startswith('action_encoder.temporal_encoder') for k in _state_dict)
        _has_action_mod = any(k.startswith('unet.action_mod') for k in _state_dict)  # Change B
        args.use_temporal_action_encoder = _has_temporal
        args.use_action_modulation = _has_action_mod
        print(f"[eval] checkpoint: temporal_action_encoder={_has_temporal} action_modulation={_has_action_mod} "
              f"-> building model to match")
        self.model = CrtlWorld(args)
        self.model.load_state_dict(_state_dict)  # strict: verifies arch matches exactly
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("load world model success")
        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None,:]
            self.state_p99 = np.array(data_stat['state_99'])[None,:]
        

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, id, start_idx=0, steps=8):
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps
        _amap = getattr(args, '_anno_path_map', None)
        annotation_path = _amap[id] if (_amap and id in _amap) else f"{val_dataset_dir}/annotation/val/{id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno['action'])
            except:
                length = anno["video_length"]
        frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)

        # get action and joint pos
        instruction = anno['texts'][0]
        car_action = np.array(anno['states'])
        car_action = car_action[frames_ids]
        joint_pos = np.array(anno['joints'])
        joint_pos = joint_pos[frames_ids]

        # get videos (per camera view)
        device = self.device
        video_dict =[]
        video_latent = []
        for id in range(len(anno['videos'])):
            video_path = f"{val_dataset_dir}/{anno['videos'][id]['video_path']}"
            if os.path.exists(video_path):
                # ---- raw-video path (unchanged): read mp4, then VAE-encode ----
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
                try:
                    true_video = vr.get_batch(range(length)).asnumpy()
                except:
                    true_video = vr.get_batch(range(length)).numpy()
                true_video = true_video[frames_ids]
                video_dict.append(true_video)

                # encode video
                true_video = torch.from_numpy(true_video).to(self.dtype).to(device)
                x = true_video.permute(0,3,1,2).to(device) / 255.0*2-1
                vae = self.model.pipeline.vae
                with torch.no_grad():
                    batch_size = 32
                    latents = []
                    for i in range(0, len(x), batch_size):
                        batch = x[i:i+batch_size]
                        latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                        latents.append(latent)
                    x = torch.cat(latents, dim=0)
                video_latent.append(x)
            else:
                # ---- fallback: raw video missing (e.g. gr00t train), load the PRECOMPUTED latent ----
                # (stored .pt is already scaled the same way as the encode path above; same format
                # the model trained on). video_dict is only used for a dead var, so use a placeholder.
                lat_rel = anno['latent_videos'][id]['latent_video_path']
                lat = torch.load(f"{val_dataset_dir}/{lat_rel}", map_location='cpu')
                lat.requires_grad = False
                fids = [min(int(fi), lat.shape[0]-1) for fi in frames_ids]
                x = lat[fids].to(self.dtype).to(device)   # [T, 4, 24, 40]
                video_latent.append(x)
                video_dict.append(x)  # placeholder (only feeds the unused `video_first`)

        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None, gen_seed=None):
        args = self.args
        image_cond = video_latent_cond

        # action should be normed
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames+args.num_history, args.action_dim)


        # predict future frames
        with torch.no_grad():
            bsz = action_cond.shape[0]
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)           
            pipeline = self.model.pipeline
            
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height*3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
                # Seed the diffusion sampling so baseline vs Change A are generated under IDENTICAL
                # noise (paired), removing run-to-run sampling variance from the comparison.
                generator=(torch.Generator(device=self.device).manual_seed(int(gen_seed))
                           if gen_seed is not None else None),
            )
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)


        # decode ground truth video
        true_video = torch.stack(video_latent_true, dim=0) # (bsz, 8,32,32)
        decoded_video = []
        bsz,frame_num = true_video.shape[:2]
        true_video = true_video.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,true_video.shape[0],args.decode_chunk_size):
            chunk = true_video[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video = torch.cat(decoded_video,dim=0)
        true_video = true_video.reshape(bsz,frame_num,*true_video.shape[1:])
        true_video = ((true_video / 2.0 + 0.5).clamp(0, 1)*255)
        true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)

        # decode predicted video
        decoded_video = []
        bsz,frame_num = latents.shape[:2]
        x = latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,x.shape[0],args.decode_chunk_size):
            chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)

        # concatenate true videos and video
        videos_cat = np.concatenate([true_video,videos],axis=-3) # (3, 8, 256, 256, 3)
        videos_cat = np.concatenate([video for video in videos_cat],axis=-2).astype(np.uint8) 

        return videos_cat, true_video, videos, latents  # np.uint8:(3, 8, 128, 256, 3) or (3, 8, 192, 320, 3)

        
if __name__ == "__main__":
    from config import wm_args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--task_type', type=str, default='replay')
    # --- flexible eval overrides (avoid editing config.py; run many checkpoints in parallel) ---
    parser.add_argument('--data_stat_path', type=str, default=None,
                        help="normalization stats; MUST match what the checkpoint trained on")
    parser.add_argument('--val_dataset_dir', type=str, default=None,
                        help="eval-set dir; val episode ids auto-discovered from its annotation/val/")
    parser.add_argument('--num_traj', type=int, default=None, help="limit number of val episodes")
    parser.add_argument('--interact_num', type=int, default=None,
                        help="autoregressive rollout steps per episode (overrides config). "
                             "Rollout length in frames ~= (pred_step-1)*interact_num + 1.")
    parser.add_argument('--save_tag', type=str, default=None,
                        help="suffix on the output folder so parallel / other-checkpoint runs don't mix")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="write videos directly here (e.g. next to the checkpoint); overrides save_dir/task_name")
    parser.add_argument('--start_idx', type=str, default=None,
                        help="start frame index per episode: an int (e.g. 30), or 'random' for a random "
                             "valid start per episode. Default keeps the config value (0 for a fresh eval set).")
    parser.add_argument('--seed', type=int, default=0,
                        help="RNG seed used only when --start_idx random (for reproducible random starts)")
    parser.add_argument('--gen_seed', type=int, default=None,
                        help="seed for the diffusion SAMPLING noise, so baseline vs Change A are generated "
                             "under identical noise (paired). None -> unseeded. Use the SAME value for both models.")
    parser.add_argument('--select_by_success', action='store_true',
                        help="select eval episodes per task = n_success success + n_fail failure rollouts, "
                             "drawn from BOTH annotation/train and annotation/val (uses the 'success' field "
                             "in each annotation). Overrides --num_traj.")
    parser.add_argument('--n_success', type=int, default=5, help="success episodes per task (with --select_by_success)")
    parser.add_argument('--n_fail', type=int, default=5, help="failure episodes per task (with --select_by_success)")
    parser.add_argument('--task', type=str, default=None,
                        help="restrict --select_by_success to a single task (e.g. PickPlaceSinkToCounter)")
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args

    args = merge_args(args, args_new)
    args.gen_seed = args_new.gen_seed  # may be None; merge_args skips None so set explicitly

    # --- apply eval overrides (run after merge so they win over the config branch) ---
    import glob as _glob
    import random as _random
    _random.seed(args_new.seed)

    # frames spanned by one episode rollout: get_traj_info samples `steps` frames stride `skip`
    _span = int(args.pred_step * args.interact_num + 8) * args.skip_step

    def _episode_length(vid):
        _amap = getattr(args, '_anno_path_map', None)
        _p = _amap[vid] if (_amap and vid in _amap) else f"{args.val_dataset_dir}/annotation/val/{vid}.json"
        with open(_p) as f:
            anno = json.load(f)
        try:
            return len(anno['action'])
        except Exception:
            return anno['video_length']

    def _start_for(vid):
        # None -> keep default (0); an int string -> fixed; 'random' -> random valid start
        if args_new.start_idx is None:
            return 0
        if args_new.start_idx == 'random':
            hi = max(0, _episode_length(vid) - _span)
            return _random.randint(0, hi)
        return int(args_new.start_idx)

    if args_new.select_by_success and args.val_dataset_dir is not None:
        # Per task: n_success success + n_fail failure rollouts, drawn from train+val combined.
        # Task = filename prefix before the first _<digit>; success = the annotation's 'success' field
        # (read from the file head only, so we don't parse the big state arrays for every candidate).
        import re as _re
        root = args.val_dataset_dir
        def _task_success(p):
            # task = first '_'-delimited token; robocasa task names are CamelCase with no '_' or
            # digits, so this handles BOTH gr00t ids ('PickPlaceSinkToCounter_14_50_...') and atomic
            # ids ('CloseFridge__ep000000'). (The old '^(.+?)_\d' regex only matched gr00t ids.)
            task = os.path.basename(p).split('_')[0]
            with open(p) as f:
                head = f.read(4096)
            sm = _re.search(r'"success"\s*:\s*(\d+)', head)
            return task, (int(sm.group(1)) if sm else None)
        paths = (_glob.glob(os.path.join(root, "annotation", "train", "*.json")) +
                 _glob.glob(os.path.join(root, "annotation", "val", "*.json")))
        by_task = {}
        for p in paths:
            task, succ = _task_success(p)
            if succ is None:
                continue
            if args_new.task and task != args_new.task:  # --task: restrict to one task
                continue
            eid = os.path.basename(p)[:-5]
            by_task.setdefault(task, {1: [], 0: []})[1 if succ == 1 else 0].append((eid, p))
        if args_new.task and not by_task:
            raise SystemExit(f"--task {args_new.task}: no annotations found under {root}/annotation/*/")
        val_id, anno_map, success_map = [], {}, {}
        for task in sorted(by_task):
            for label, n in [(1, args_new.n_success), (0, args_new.n_fail)]:
                pool = sorted(by_task[task][label])
                _random.shuffle(pool)
                picked = pool[:n]
                if len(picked) < n:
                    print(f"  WARN task {task} {'success' if label else 'fail'}: only {len(picked)}/{n} available")
                for eid, p in picked:
                    val_id.append(eid); anno_map[eid] = p; success_map[eid] = label
        args.val_id = val_id
        args._anno_path_map = anno_map      # get_traj_info / _episode_length read annotations via this
        args._success_map = success_map
        args.start_idx = [_start_for(vid) for vid in val_id]
        args.instruction = [""] * len(val_id)
        print(f"select_by_success: {len(val_id)} episodes ({args_new.n_success} succ + {args_new.n_fail} fail "
              f"per task, {len(by_task)} tasks) drawn from train+val")
    elif args_new.val_dataset_dir is not None:
        ids = sorted(os.path.basename(p)[:-5]
                     for p in _glob.glob(os.path.join(args.val_dataset_dir, "annotation", "val", "*.json")))
        if args_new.num_traj is not None:
            ids = ids[:args_new.num_traj]
        args.val_id = ids
        args.start_idx = [_start_for(vid) for vid in ids]
        args.instruction = [""] * len(ids)
        print(f"eval set {args.val_dataset_dir}: {len(ids)} val episodes")
        print(f"start_idx ({args_new.start_idx if args_new.start_idx is not None else 0}): {args.start_idx}")
    elif args_new.num_traj is not None:
        args.val_id = args.val_id[:args_new.num_traj]
        args.start_idx = args.start_idx[:args_new.num_traj]
        args.instruction = args.instruction[:args_new.num_traj]
        if args_new.start_idx is not None:
            args.start_idx = [_start_for(vid) for vid in args.val_id]
    if args_new.save_tag is not None:
        args.task_name = f"{args.task_name}_{args_new.save_tag}"
    _outdir = getattr(args, 'out_dir', None) or f"{args.save_dir}/{args.task_name}/video"
    print(f"ckpt={args.ckpt_path}\n  stat={args.data_stat_path}\n  output -> {_outdir}/")

    # create rollout agent
    Agent = agent(args)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    print(f'rollout with {args.task_type}')

    rollout_metrics = []  # per-trajectory drift metrics (pred vs GT); dumped to rollout_metrics.json

    for traj_idx, (val_id_i, text_i, start_idx_i) in enumerate(zip(args.val_id, args.instruction, args.start_idx)):
        # read ground truth trajectory informations
        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8))
        text_i = instruction
        print("text_i:",instruction, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])

        # create buffers and push first frames to history buffer
        predicted_latents = None
        video_to_save = []
        traj_mse, traj_psnr = [], []  # per-frame drift for this trajectory
        info_to_save = []
        his_cond = []
        his_joint = []
        his_eef = []
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
        for i in range(Agent.args.num_history*4):
            his_cond.append(first_latent)  # (1, 4, 72, 40)
            his_joint.append(joint_pos_gt[0:1])  # (1, 7)
            his_eef.append(eef_gt[0:1])  # (1, 7)

        # interact loop
        for i in range(interact_num):
            # ground truth video
            start_id = int(i*(pred_step-1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]
            
            # prepare input for policy
            joint_first = his_joint[-1][0]
            state_first = his_eef[-1][0]
            if i==0:
                video_first = [v[0] for v in video_dict]
            else:
                video_first = [v[-1] for v in video_dict_pred]
            assert joint_first.shape == (8,), f"Expected joint_first shape (8,), got {joint_first.shape}"
            assert state_first.shape == (7,), f"Expected state_first shape (7,), got {state_first.shape}"
            
            # forward policy
            print("################ policy forward ####################")
            # in the trajectory replay model, we use action recorded in trajetcory
            cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, 7)
            print("cartesian space action", cartesian_pose[0]) # output xyz and gripper for debug
            print("cartesian space action", cartesian_pose[-1]) # output xyz and gripper for debug
            
            print("################ world model forward ################")
            print(f'traj_id:{val_id_i}, interact step: {i}/{interact_num}')
            # retrive history cond and action cond
            history_idx = [0,0,-8,-6,-4,-2]
            his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)  # (4, 7)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)
            assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
            assert action_cond.shape == (int(num_history+num_frames), 7), f"Expected action_cond shape ({int(num_history+num_frames)}, 7), got {action_cond.shape}"
            assert his_cond_input.shape == (1, int(num_history), 4, 72, 40), f"Expected his_cond_input shape (1, {int(num_history)}, 72, 40), got {his_cond_input.shape}"
            # forward world model. Deterministic per-(trajectory, interaction) seed derived from
            # args.gen_seed: identical across baseline/Change A runs (model-independent) but distinct
            # per step -> paired, reproducible generation. None -> unseeded (old behaviour).
            gen_seed = None if getattr(args, 'gen_seed', None) is None else int(args.gen_seed)*1_000_003 + traj_idx*101 + i
            videos_cat, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm(action_cond, video_latent_true, current_latent, his_cond=his_cond_input,text=text_i if Agent.args.text_cond else None, gen_seed=gen_seed)

            print("################ record information ################")
            # push current step to history buffer
            his_eef.append(cartesian_pose[pred_step-1:pred_step]) #(1,7)
            his_cond.append(torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
            if i == interact_num - 1:
                video_to_save.append(videos_cat)  # save all frames for the last interaction step
            else:
                video_to_save.append(videos_cat[:pred_step-1]) # last frame is the first frame of next step, so we remove it here

            # accumulate PSNR/MSE drift on the SAME frames we keep in the saved rollout
            n_keep = pred_step if i == interact_num - 1 else pred_step - 1
            mse_f, psnr_f = _psnr_mse_per_frame(video_dict_pred[:, :n_keep], true_videos[:, :n_keep])
            traj_mse.extend(mse_f); traj_psnr.extend(psnr_f)
                
        
        # save rollout video and info with parameters
        video = np.concatenate(video_to_save, axis=0)
        task_name = args.task_name
        text_id = text_i.replace(' ', '_').replace(',', '').replace('.', '').replace('\'', '').replace('\"', '')[:30]
        # --out_dir (if set) puts videos directly there (e.g. next to the checkpoint);
        # otherwise the default synthetic_traj/<task_name>/video/ layout.
        # No per-video timestamp: the eval run's out_dir is already timestamped (see eval_ckpt.sh).
        out_dir = getattr(args, 'out_dir', None) or f"{args.save_dir}/{task_name}/video"
        filename_video = f"{out_dir}/traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}.mp4"
        os.makedirs(os.path.dirname(filename_video), exist_ok=True)
        mediapy.write_video(filename_video, video, fps=4)
        print(f"Saving video to {filename_video}")
        print("##########################################################################")

        rollout_metrics.append({
            'traj_id': str(val_id_i), 'start_idx': int(start_idx_i),
            'success': getattr(args, '_success_map', {}).get(val_id_i),  # 1/0 (None if not selected by success)
            'num_frames': len(traj_psnr),
            'mean_psnr': float(np.mean(traj_psnr)) if traj_psnr else 0.0,
            'mean_mse': float(np.mean(traj_mse)) if traj_mse else 0.0,
            'psnr_per_frame': traj_psnr, 'mse_per_frame': traj_mse,
        })

    # ---- aggregate + write drift metrics (pred vs GT over the autoregressive rollout) ----
    if rollout_metrics:
        # all trajectories share the same rollout length; average per frame index for the drift curve
        min_len = min(len(m['psnr_per_frame']) for m in rollout_metrics)
        psnr_curve = np.mean([m['psnr_per_frame'][:min_len] for m in rollout_metrics], axis=0).tolist()
        mse_curve = np.mean([m['mse_per_frame'][:min_len] for m in rollout_metrics], axis=0).tolist()
        summary = {
            'ckpt': args.ckpt_path,
            'use_temporal_action_encoder': bool(getattr(args, 'use_temporal_action_encoder', False)),
            'dataset': getattr(args, 'dataset_names', None),
            'num_traj': len(rollout_metrics),
            'rollout_frames': min_len,
            'mean_psnr': float(np.mean([m['mean_psnr'] for m in rollout_metrics])),
            'mean_mse': float(np.mean([m['mean_mse'] for m in rollout_metrics])),
            'psnr_per_frame': psnr_curve,
            'mse_per_frame': mse_curve,
        }
        # breakdown by success/failure (when episodes were selected with --select_by_success)
        if any(m.get('success') is not None for m in rollout_metrics):
            for lab, name in [(1, 'success'), (0, 'failure')]:
                sub = [m for m in rollout_metrics if m.get('success') == lab]
                if sub:
                    summary[f'n_{name}'] = len(sub)
                    summary[f'mean_psnr_{name}'] = float(np.mean([m['mean_psnr'] for m in sub]))
                    summary[f'mean_mse_{name}'] = float(np.mean([m['mean_mse'] for m in sub]))
        metrics_path = os.path.join(_outdir, 'rollout_metrics.json')
        os.makedirs(_outdir, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump({'summary': summary, 'per_traj': rollout_metrics}, f, indent=2)
        print("\n===== rollout drift metrics =====")
        print(f"  ckpt              : {summary['ckpt']}")
        print(f"  temporal_encoder  : {summary['use_temporal_action_encoder']}")
        print(f"  trajectories      : {summary['num_traj']}  ({summary['rollout_frames']} frames each)")
        print(f"  mean PSNR (dB, higher=better) : {summary['mean_psnr']:.3f}")
        print(f"  mean MSE  (lower=better)      : {summary['mean_mse']:.2f}")
        print(f"  PSNR first->last frame        : {psnr_curve[0]:.2f} -> {psnr_curve[-1]:.2f}  (drift over rollout)")
        if 'mean_psnr_success' in summary or 'mean_psnr_failure' in summary:
            print("  --- by outcome ---")
            if 'mean_psnr_success' in summary:
                print(f"  SUCCESS (n={summary.get('n_success')}): PSNR {summary['mean_psnr_success']:.3f}  MSE {summary['mean_mse_success']:.1f}")
            if 'mean_psnr_failure' in summary:
                print(f"  FAILURE (n={summary.get('n_failure')}): PSNR {summary['mean_psnr_failure']:.3f}  MSE {summary['mean_mse_failure']:.1f}")
        print(f"  wrote {metrics_path}")
        print("=================================")


# CUDA_VISIBLE_DEVICES=0 python rollout_replay_traj.py
        
        
