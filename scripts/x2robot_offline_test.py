"""Offline evaluation for X2Robot: inference and plot pred vs GT."""

import os
import dataclasses
import glob
import json
import logging
from pathlib import Path

import tyro
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    dataset_dir: str = "datasets/x2robot/throw_0113"  # Support multiple datasets separated by comma
    policy_config: str = "throw_sm2m"
    policy_dir: str = "checkpoints/throw_sm2m/throw_0113_sm2m_h5f3/29999"
    policy_mode: str | None = None  # Auto-detect from policy_dir if None (s2s, s2m, sm2m, sm2sm)
    state_history_size: int | None = None  # Auto-load from config if None
    state_future_size: int | None = None
    state_step: int | None = None  # Auto-load from config if None; step size for sampling future states
    move_steps: int = 15
    num_episodes: int | None = None
    split: str | None = None  # "train", "val", or None
    val_ratio: float = 0.1
    split_seed: int = 42
    output_dir: str = "offline_test_results"


def build_state(frame_data: dict, policy_mode: str) -> np.ndarray:
    """Build state from frame data."""
    slave = np.concatenate([
        frame_data['follow_left_position'], frame_data['follow_left_rotation'],
        [frame_data['follow_left_gripper']], frame_data['follow_right_position'],
        frame_data['follow_right_rotation'], [frame_data['follow_right_gripper']]
    ]).astype(np.float32)
    
    if policy_mode in ["s2s", "s2m"]:
        return slave
    
    master = np.concatenate([
        frame_data['master_left_position'], frame_data['master_left_rotation'],
        [frame_data['master_left_gripper']], frame_data['master_right_position'],
        frame_data['master_right_rotation'], [frame_data['master_right_gripper']]
    ]).astype(np.float32)
    
    return np.concatenate([slave, master])


def run_inference(policy, episode_path: str, args: Args):
    """Run inference on episode and return predictions and GT."""
    episode_name = os.path.basename(episode_path)
    with open(os.path.join(episode_path, f"{episode_name}.json"), 'r') as f:
        episode_data = json.load(f)
    
    frames = episode_data['data']
    total_frames = len(frames)
    videos = {k: cv2.VideoCapture(os.path.join(episode_path, f"{k}Img.mp4")) 
              for k in ['left', 'face', 'right']}
    
    all_preds, all_gts = [], []
    current_idx = 0
    
    while current_idx < total_frames:
        # Read current frame images
        images = {}
        for k, cap in videos.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            ret, frame = cap.read()
            images[f"{k}_wrist_view" if k != "face" else "face_view"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Build state sequence (sampled at intervals of state_step)
        indices = [max(0, min(current_idx + i * args.state_step, total_frames - 1)) 
                   for i in range(-args.state_history_size, args.state_future_size + 1)]
        state_seq = np.array([build_state(frames[i], args.policy_mode) for i in indices], dtype=np.float32)
        
        # Inference
        obs = {'images': images, 'prompt': '', 'state': state_seq}
        action_pred = policy.infer(obs)['actions']
        
        if args.policy_mode == "sm2sm":
            action_pred = action_pred[:, 14:28]  # Extract master part
        
        action_pred = action_pred[:args.move_steps]
        
        # Extract GT (from next frame)
        gt_length = min(args.move_steps, total_frames - current_idx - 1)
        if gt_length > 0:
            gt_actions = []
            for i in range(gt_length):
                idx = min(current_idx + 1 + i, total_frames - 1)
                frame = frames[idx]
                gt = np.concatenate([
                    frame['master_left_position'], frame['master_left_rotation'],
                    [frame['master_left_gripper']], frame['master_right_position'],
                    frame['master_right_rotation'], [frame['master_right_gripper']]
                ]).astype(np.float32)
                gt_actions.append(gt)
            
            all_preds.append(action_pred[:gt_length])
            all_gts.append(np.array(gt_actions))
        
        current_idx += args.move_steps
    
    for cap in videos.values():
        cap.release()
    
    return (np.concatenate(all_preds) if all_preds else np.zeros((0, 14)),
            np.concatenate(all_gts) if all_gts else np.zeros((0, 14)))


def plot_results(pred: np.ndarray, gt: np.ndarray, name: str, output_path: str):
    """Plot 6 subplots: left/right xyz, rpy, gripper."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Episode: {name}', fontsize=16)
    
    configs = [
        (0, 0, [0, 1, 2], 'Left Arm XYZ', 'Position'),
        (0, 1, [3, 4, 5], 'Left Arm RPY', 'Rotation'),
        (0, 2, [6], 'Left Gripper', 'Gripper'),
        (1, 0, [7, 8, 9], 'Right Arm XYZ', 'Position'),
        (1, 1, [10, 11, 12], 'Right Arm RPY', 'Rotation'),
        (1, 2, [13], 'Right Gripper', 'Gripper')
    ]
    
    labels = {3: ['R', 'P', 'Y'], 1: [''], 'default': ['X', 'Y', 'Z']}
    
    for row, col, indices, title, ylabel in configs:
        ax = axes[row, col]
        lbls = labels.get(len(indices), labels['default'])
        for i, idx in enumerate(indices):
            ax.plot(gt[:, idx], f'--', alpha=0.7, label=f'GT {lbls[i]}')
            ax.plot(pred[:, idx], alpha=0.7, label=f'Pred {lbls[i]}')
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(args: Args):
    # Auto-detect policy_mode from policy_dir if not specified
    if args.policy_mode is None:
        for mode in ['sm2sm', 'sm2m', 's2m', 's2s']:
            if mode in args.policy_dir.lower():
                args.policy_mode = mode
                logging.info(f"Auto-detected policy_mode from path: {args.policy_mode}")
                break
        if args.policy_mode is None:
            raise ValueError(f"Could not detect policy_mode from path: {args.policy_dir}. Please specify --policy-mode")
    
    # Load config params if not specified
    cfg = _config.get_config(args.policy_config)
    if args.state_history_size is None:
        args.state_history_size = getattr(cfg.data, 'state_history_size', 0)
        logging.info(f"Using state_history_size from config: {args.state_history_size}")
    if args.state_future_size is None:
        args.state_future_size = getattr(cfg.data, 'state_future_size', 0)
        logging.info(f"Using state_future_size from config: {args.state_future_size}")
    if args.state_step is None:
        args.state_step = getattr(cfg.data, 'state_step', 1)
        logging.info(f"Using state_step from config: {args.state_step}")
    
    # Load policy
    logging.info(f"Loading policy from {args.policy_dir}")
    policy = _policy_config.create_trained_policy(cfg, args.policy_dir)
    
    # Find and filter episodes from multiple datasets
    dataset_dirs = [d.strip() for d in args.dataset_dir.split(',')]
    all_episodes = []
    
    for dataset_dir in dataset_dirs:
        # Find episodes in this dataset
        dataset_episodes = sorted([p for p in glob.glob(f'{dataset_dir}/*') 
                                   if os.path.isdir(p) and glob.glob(f'{p}/*.mp4')])
        
        # Apply split to each dataset separately
        if args.split:
            rng = np.random.RandomState(args.split_seed)
            indices = np.arange(len(dataset_episodes))
            rng.shuffle(indices)
            val_size = int(len(dataset_episodes) * args.val_ratio)
            indices = indices[:val_size] if args.split == "val" else indices[val_size:]
            dataset_episodes = [dataset_episodes[i] for i in sorted(indices)]
            logging.info(f"Dataset {dataset_dir}: {len(dataset_episodes)} episodes in {args.split} split")
        else:
            logging.info(f"Dataset {dataset_dir}: {len(dataset_episodes)} episodes")
        
        all_episodes.extend(dataset_episodes)
    
    episodes = all_episodes
    logging.info(f"Total: {len(episodes)} episodes across {len(dataset_dirs)} dataset(s)")
    
    if args.num_episodes:
        episodes = episodes[:args.num_episodes]
    
    # Process episodes
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ep_path in tqdm(episodes, desc="Processing"):
        ep_name = os.path.basename(ep_path)
        pred, gt = run_inference(policy, ep_path, args)
        if len(pred) > 0:
            plot_results(pred, gt, ep_name, str(output_dir / f"{ep_name}.jpg"))
            logging.info(f"Saved {ep_name}")
    
    logging.info(f"Done! Results in {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
