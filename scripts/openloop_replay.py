"""
uv run scripts/openloop_replay.py \
    --num-episodes 10 \
    --dataset-path ./datasets/x2robot/microwave/ \
    --policy.config microwave_1218_sm2sm \
    --policy.dir ./checkpoints/microwave_1218_sm2sm/microwave_1218_sm2sm/29999/
"""

import os
import glob
import json
import time
import dataclasses
from pathlib import Path
from collections import defaultdict

import tqdm
import tyro
import numpy as np
import torch
import torchvision
import einops
import matplotlib.pyplot as plt

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define the order of keys for state/action concatenation
# Modify this list to change the order or add new dimensions
STATE_ACTION_KEYS = [
    "follow_left_position",
    "follow_left_rotation",
    "follow_left_gripper",
    "follow_right_position",
    "follow_right_rotation",
    "follow_right_gripper",
    'master_left_position',
    'master_left_rotation', 
    'master_left_gripper',
    'master_right_position',
    'master_right_rotation',
    "master_right_gripper",
    
]

FILE_CAME_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4"
}


def decode_video_torchvision(file_name, keyframes_only=True, backend='pyav'):
    """Decode video using torchvision.io.VideoReader"""
    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(file_name, "video")
    reader.seek(0, keyframes_only=keyframes_only)

    loaded_frames = []
    for frame in reader:
        loaded_frames.append(frame["data"])

    reader.container.close()
    reader = None
    loaded_frames = torch.stack(loaded_frames).numpy()

    return loaded_frames


def process_action(file_path):
    """Process action data from JSON file"""
    file_name = os.path.basename(file_path)
    action_path = os.path.join(file_path, f"{file_name}.json")
    
    trajectories = defaultdict(list)
    
    with open(action_path, 'r') as file:
        actions = json.load(file)
        data = actions['data']
        
        for action in data:
            for key, val in action.items():
                trajectories[key].append(val)
    
    trajectories = {k: np.array(v, dtype=np.float32) for k, v in trajectories.items()}
    
    return trajectories



@dataclasses.dataclass
class Checkpoint:
    """Policy checkpoint configuration"""
    config: str  # Training config name (e.g., "microwave_1218_lora")
    dir: str  # Checkpoint directory path


@dataclasses.dataclass
class Args:
    """Arguments for the openloop replay script"""
    policy: Checkpoint
    dataset_path: str  # Path to the dataset directory
    num_episodes: int = -1  # Number of episodes to process (-1 for all)
    action_horizon: int = 40  # Action horizon for sliding window
    output_dir: str = 'openloop_results'


def create_policy(checkpoint: Checkpoint):
    """Create a policy from checkpoint"""
    return _policy_config.create_trained_policy(
        _config.get_config(checkpoint.config), checkpoint.dir
    )


def visualize_and_save_results(ground_truths, model_predicts, episode_path, output_dir, key_dims):
    """Visualize comparison between ground truth and model predictions"""
    # Create subplots - one for each key in STATE_ACTION_KEYS
    num_groups = len(STATE_ACTION_KEYS)
    fig, axes = plt.subplots(num_groups, 1, figsize=(14, 4 * num_groups))
    if num_groups == 1:
        axes = [axes]
    
    # Colors for different dimensions within each group
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Track current dimension index in the full array
    current_idx = 0
    
    for group_idx, key in enumerate(STATE_ACTION_KEYS):
        ax = axes[group_idx]
        num_dims = key_dims[key]
        
        # Plot each dimension in this group
        for i in range(num_dims):
            color = colors[i % len(colors)]
            # Ground truth: solid line
            ax.plot(ground_truths[:, current_idx + i], 
                   label=f'Dim {i} (GT)', 
                   color=color, 
                   linestyle='-', 
                   linewidth=1.5)
            # Model prediction: dashed line
            ax.plot(model_predicts[:, current_idx + i], 
                   label=f'Dim {i} (Pred)', 
                   color=color, 
                   linestyle='--', 
                   linewidth=1.5)
        
        ax.set_title(f'{key}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        current_idx += num_dims
    
    plt.tight_layout()
    
    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_name = Path(episode_path).name
    ts = time.strftime('%Y%m%d_%H%M%S')
    png_name = out_dir / f'openloop_{episode_name}_{ts}.png'
    
    plt.savefig(str(png_name), dpi=100)
    plt.close()
    print(f"Saved comparison to {png_name}")


def process_episode(episode_path, policy, output_dir, action_horizon):
    """Process a single episode for openloop evaluation"""
    print(f"Processing {episode_path}")
    
    # Load action data
    pose_dicts = process_action(episode_path)
    
    # Get dimension info for each key from the loaded data
    key_dims = {}
    for key in STATE_ACTION_KEYS:
        if key in pose_dicts:
            data = pose_dicts[key]
            # If it's 1D, make it 2D with shape (n, 1)
            if data.ndim == 1:
                pose_dicts[key] = data.reshape(-1, 1)
                key_dims[key] = 1
            else:
                key_dims[key] = data.shape[1]
    
    # Concatenate states/actions according to STATE_ACTION_KEYS
    real_poses = np.concatenate([pose_dicts[key] for key in STATE_ACTION_KEYS], axis=1)
    
    # Load video frames
    video_frames = {}
    for key in FILE_CAME_MAPPING:
        video_path = os.path.join(episode_path, FILE_CAME_MAPPING[key])
        frames = decode_video_torchvision(video_path)
        frames = einops.rearrange(frames, 't c h w -> t h w c')
        video_frames[key] = frames
    
    # Perform openloop evaluation with sliding window
    ground_truths = []
    model_predicts = []
    
    current_chunk = None
    chunk_step_idx = 0
    
    for i in range(len(real_poses) - action_horizon - 10):
        # Run inference when chunk is exhausted or not initialized
        if current_chunk is None or chunk_step_idx >= action_horizon:
            obs = {
                'images': {
                    key: video_frames[key][i] for key in FILE_CAME_MAPPING
                },
                'prompt': '',
                'state': real_poses[i]
            }
            
            action_pred_result = policy.infer(obs)
            current_chunk = action_pred_result['actions']
            chunk_step_idx = 0
        
        # Skip if insufficient future steps
        if i + action_horizon >= len(real_poses):
            break
        
        predicted_action = current_chunk[chunk_step_idx]
        ground_truth_action = real_poses[i]
        
        model_predicts.append(predicted_action)
        ground_truths.append(ground_truth_action)
        
        chunk_step_idx += 1
    
    # Convert to arrays and visualize
    ground_truths = np.array(ground_truths)
    model_predicts = np.array(model_predicts)
    
    visualize_and_save_results(ground_truths, model_predicts, episode_path, output_dir, key_dims)


def main(args: Args) -> None:
    """Main function for openloop replay"""
    print(f"Creating policy from {args.policy.dir}")
    policy = create_policy(args.policy)
    
    # Collect all episode paths
    all_episode_paths = []
    for episode_path in sorted(glob.glob(f'{args.dataset_path}/*')):
        if os.path.isdir(episode_path):
            has_mp4_files = len(glob.glob(f'{episode_path}/*.mp4')) > 0
            if has_mp4_files:
                all_episode_paths.append(episode_path)
    
    # Limit number of episodes if specified
    if args.num_episodes > 0:
        all_episode_paths = all_episode_paths[:args.num_episodes]
    
    print(f"Found {len(all_episode_paths)} episodes to process")
    
    # Process each episode
    for episode_path in tqdm.tqdm(all_episode_paths, desc="Processing episodes"):
        process_episode(episode_path, policy, args.output_dir, args.action_horizon)
    
    print(f"\nOpenloop evaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))