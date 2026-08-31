import os
import shutil
import glob
import json
import numpy as np
import einops
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
import tyro
import time


# Configuration
REPO_NAME = "plugin_0107_test"  # TODO: Change to your dataset name
RAW_DATASET_PATHS = [
    './datasets/x2robot/plugin_0107/',  # Add more paths as needed
    # Add more paths as needed
]

FILE_CAMERA_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4"
}

# Keys as they appear in the original JSON file
ACTION_KEYS = [
    'follow_left_position',
    'follow_left_rotation', 
    'follow_left_gripper',
    'follow_right_position',
    'follow_right_rotation',
    'follow_right_gripper',
    'master_left_position',
    'master_left_rotation',
    'master_left_gripper', 
    'master_right_position',
    'master_right_rotation',
    'master_right_gripper',
]


def decode_video(video_path: str) -> np.ndarray:
    torchvision.set_video_backend('pyav')
    reader = torchvision.io.VideoReader(video_path, "video")
    reader.seek(0, keyframes_only=True)
    
    frames = []
    for frame in reader:
        frames.append(frame["data"])
    
    reader.container.close()
    return torch.stack(frames).numpy()


def process_episode(episode_path: str, low_resolution: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Load actions from JSON
    t0 = time.time()
    episode_name = os.path.basename(episode_path)
    json_path = os.path.join(episode_path, f"{episode_name}.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)['data']
    
    # Collect trajectories
    trajectories = {key: [] for key in ACTION_KEYS}
    
    for frame_data in data:
        for key in ACTION_KEYS:
            trajectories[key].append(frame_data[key])
    
    # Convert to numpy arrays and reshape grippers
    actions = {}
    for key, vals in trajectories.items():
        arr = np.array(vals, dtype=np.float32)
        # Reshape gripper values from (T,) to (T, 1)
        if 'gripper' in key:
            arr = arr.reshape(-1, 1)
        actions[key] = arr
    
    # Concatenate into state and action arrays (T, 14)
    state_action = np.concatenate([actions[key] for key in ACTION_KEYS], axis=1)
    
    t1 = time.time()
    print(f"  JSON loading & parsing: {t1-t0:.3f}s")
    
    # Load videos (resize to low resolution if requested)
    videos = {}
    for view_name, filename in FILE_CAMERA_MAPPING.items():
        video_path = os.path.join(episode_path, filename)
        t_vid_start = time.time()
        frames = decode_video(video_path)  # (T, C, H, W), uint8
        t_vid_decode = time.time()

        if low_resolution:
            # Resize to 240x320 using bilinear interpolation
            t = torch.as_tensor(frames, dtype=torch.float32)
            t = F.interpolate(t, size=(240, 320), mode="bilinear", align_corners=False)
            frames = t.clamp(0, 255).byte().numpy()

        frames = einops.rearrange(frames, 't c h w -> t h w c')  # (T, H, W, C)
        videos[view_name] = frames
        t_vid_end = time.time()
        print(f"  Video {view_name}: decode={t_vid_decode-t_vid_start:.3f}s, resize={t_vid_end-t_vid_decode:.3f}s")
    
    return state_action, videos


def find_episodes(raw_paths: list[str]) -> list[str]:
    """
    Find all episode directories containing MP4 files.
    
    Args:
        raw_paths: List of raw dataset root paths
        
    Returns:
        List of episode directory paths
    """
    episode_paths = []
    
    for raw_path in raw_paths:
        for dir_path in glob.glob(f'{raw_path}/*'):
            if os.path.isdir(dir_path):
                mp4_files = glob.glob(f'{dir_path}/*.mp4')
                if len(mp4_files) > 0:
                    episode_paths.append(dir_path)
    
    return sorted(episode_paths)


def main(
        push_to_hub: bool = False,
        debug: bool = False,
        debug_episodes: int = 3,
        low_resolution: bool = True,
):
    """
    Convert X2Robot data to LeRobot format.
    
    Args:
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        debug: Run in debug mode (process limited episodes)
        debug_episodes: Number of episodes to process in debug mode
    """
    print(f"HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
    
    # Clean up existing dataset
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset
    shape = (240, 320, 3) if low_resolution else (480, 640, 3)
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="ARX",
        fps=20,
        features={
            "face_view": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            },
            "left_wrist_view": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            },
            "right_wrist_view": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (28,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (28,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )
    
    # Find all episodes
    episode_paths = find_episodes(RAW_DATASET_PATHS)
    print(f"Found {len(episode_paths)} episodes")
    if debug:
        episode_paths = episode_paths[:debug_episodes]
        print(f"Debug mode: only processing first {debug_episodes} episodes")
    
    # Process each episode
    for episode_path in tqdm.tqdm(episode_paths, desc="Processing episodes"):
        try:
            t_episode_start = time.time()
            print(f"\nProcessing {os.path.basename(episode_path)}:")
            state_action, videos = process_episode(episode_path, low_resolution)
            t_process_done = time.time()
            print(f"  Episode processing: {t_process_done-t_episode_start:.3f}s")
            
            # Add frames to dataset
            t_add_start = time.time()
            num_frames = len(state_action)
            for i in range(num_frames - 1):
                dataset.add_frame({
                    "face_view": videos["face_view"][i],
                    "left_wrist_view": videos["left_wrist_view"][i],
                    "right_wrist_view": videos["right_wrist_view"][i],
                    "state": state_action[i],      # Current state
                    "actions": state_action[i + 1],  # Next state as action
                    "task": '',  # Empty task description
                })
            t_add_done = time.time()
            print(f"  add_frame ({num_frames} frames): {t_add_done-t_add_start:.3f}s")
            
            t_save_start = time.time()
            dataset.save_episode()
            t_save_done = time.time()
            print(f"  save_episode: {t_save_done-t_save_start:.3f}s")
            print(f"  Total episode time: {t_save_done-t_episode_start:.3f}s")
            
        except Exception as e:
            print(f"Error processing {episode_path}: {e}")
            continue
    
    print(f"Dataset saved at {HF_LEROBOT_HOME}/{REPO_NAME}")
    
    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(main)
