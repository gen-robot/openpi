"""
V4: Highly optimized X2Robot data conversion with episode-level parallelism.

Optimization strategy:
1. Parallel DECODING: N episodes × 3 cameras = 3N ffmpeg processes
2. Sequential dataset building (fast, images already on disk)
3. Parallel ENCODING: N episodes × 3 cameras = 3N encoding threads
4. Final cleanup

Key insight: Completely bypass lerobot's save_episode encoding by:
- Temporarily setting video_keys to empty during save_episode
- Manually encoding all videos after all episodes are saved
"""
import os
import shutil
import glob
import json
import subprocess
import numpy as np
import tqdm
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from lerobot.common.datasets.video_utils import encode_video_frames
import tyro
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

    
# Suppress SVT-AV1 encoder output using environment variable
os.environ['SVT_LOG'] = '0'


# Configuration
REPO_NAME = "throw_0113_sm2m"
RAW_DATASET_PATHS = [
    './datasets/x2robot/throw_0113/',
]

FILE_CAMERA_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4"
}

STATE_KEYS = [
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

ACTION_KEYS = [
    # 'follow_left_position',
    # 'follow_left_rotation', 
    # 'follow_left_gripper',
    # 'follow_right_position',
    # 'follow_right_rotation',
    # 'follow_right_gripper',
    'master_left_position',
    'master_left_rotation',
    'master_left_gripper', 
    'master_right_position',
    'master_right_rotation',
    'master_right_gripper',
]


def get_dim_from_keys(keys: list[str]) -> int:
    """Calculate the total dimension from a list of keys.
    
    Position keys have 3 dims, rotation keys have 3 dims, gripper keys have 1 dim.
    """
    dim = 0
    for key in keys:
        if 'gripper' in key:
            dim += 1
        elif 'position' in key:
            dim += 3
        elif 'rotation' in key:
            dim += 3
        else:
            raise ValueError(f"Unknown key type: {key}")
    return dim


@dataclass
class EpisodeInfo:
    """Information about a single episode."""
    episode_path: str
    episode_index: int
    num_frames: int = 0


def find_episodes(raw_paths: list[str]) -> list[str]:
    """Find all episode directories containing MP4 files."""
    episode_paths = []
    for raw_path in raw_paths:
        for dir_path in glob.glob(f'{raw_path}/*'):
            if os.path.isdir(dir_path):
                mp4_files = glob.glob(f'{dir_path}/*.mp4')
                if len(mp4_files) > 0:
                    episode_paths.append(dir_path)
    return sorted(episode_paths)


def load_json_data(episode_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and parse JSON data for an episode."""
    episode_name = os.path.basename(episode_path)
    json_path = os.path.join(episode_path, f"{episode_name}.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)['data']
    
    # Collect all keys needed
    all_keys = set(STATE_KEYS) | set(ACTION_KEYS)
    trajectories = {key: [] for key in all_keys}
    for frame_data in data:
        for key in all_keys:
            trajectories[key].append(frame_data[key])
    
    # Convert to arrays
    arrays = {}
    for key, vals in trajectories.items():
        arr = np.array(vals, dtype=np.float32)
        if 'gripper' in key:
            arr = arr.reshape(-1, 1)
        arrays[key] = arr
    
    state_array = np.concatenate([arrays[key] for key in STATE_KEYS], axis=1)
    action_array = np.concatenate([arrays[key] for key in ACTION_KEYS], axis=1)
    return state_array, action_array


def decode_video_ffmpeg(
    input_path: str, 
    output_dir: Path, 
    target_size: tuple[int, int] = (320, 240)
) -> int:
    """
    Decode video to PNG frames using ffmpeg.
    
    Args:
        input_path: Path to input video
        output_dir: Directory to save PNG frames
        target_size: (width, height) for output frames
        
    Returns:
        Number of frames decoded
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_pattern = str(output_dir / "frame_%06d.png")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale={target_size[0]}:{target_size[1]}",
        "-start_number", "0",
        output_pattern
    ]
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {input_path}: {result.stderr}")
    
    # Count output frames
    frames = list(output_dir.glob("frame_*.png"))
    return len(frames)


def decode_single_video(
    episode_path: str,
    episode_index: int,
    camera_name: str,
    video_filename: str,
    output_root: Path,
    target_size: tuple[int, int]
) -> tuple[int, str, int]:
    """Decode a single video file. Returns (episode_idx, camera_name, num_frames)."""
    video_path = os.path.join(episode_path, video_filename)
    output_dir = output_root / "images" / camera_name / f"episode_{episode_index:06d}"
    num_frames = decode_video_ffmpeg(video_path, output_dir, target_size)
    return episode_index, camera_name, num_frames


def encode_single_video(
    episode_index: int,
    camera_name: str,
    output_root: Path,
    fps: int = 20
) -> tuple[int, str]:
    """Encode a single video file. Returns (episode_idx, camera_name)."""
    img_dir = output_root / "images" / camera_name / f"episode_{episode_index:06d}"
    video_path = output_root / "videos" / "chunk-000" / camera_name / f"episode_{episode_index:06d}.mp4"
    
    video_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video_frames(img_dir, video_path, fps, overwrite=True)
    
    return episode_index, camera_name


class NoEncodingLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that skips video encoding and image deletion in save_episode."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_video_encoding = False
        self._skip_image_write = False
    
    def _save_image(self, image, fpath: Path) -> None:
        """Skip image saving if _skip_image_write is True (images already on disk)."""
        if self._skip_image_write:
            return
        super()._save_image(image, fpath)
    
    def save_episode(self, episode_data: dict | None = None) -> None:
        """Override save_episode to optionally skip video encoding and image deletion."""
        if not self._skip_video_encoding:
            super().save_episode(episode_data)
            return
            
        # Modified save_episode that skips video encoding and image deletion
        if not episode_data:
            episode_buffer = self.episode_buffer
        
        from lerobot.common.datasets.lerobot_dataset import (
            validate_episode_buffer,
            compute_episode_stats,
            get_episode_data_index,
            check_timestamps_sync,
        )

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        # SKIP video encoding - we'll do it in parallel later
        # if len(self.meta.video_keys) > 0:
        #     video_paths = self.encode_episode_videos(episode_index)
        #     for key in self.meta.video_keys:
        #         episode_buffer[key] = video_paths[key]

        # Custom save_episode that skips update_video_info
        self._save_episode_meta_no_video_check(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        # SKIP video file count check for now - videos will be created later
        # video_files = list(self.root.rglob("*.mp4"))
        # assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # SKIP image deletion - we need them for encoding
        # img_dir = self.root / "images"
        # if img_dir.is_dir():
        #     shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()
    
    def _save_episode_meta_no_video_check(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        """Save episode metadata without checking video files."""
        from lerobot.common.datasets.lerobot_dataset import (
            write_info,
            write_episode,
            write_episode_stats,
            aggregate_stats,
        )
        
        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += episode_length

        chunk = self.meta.get_episode_chunk(episode_index)
        if chunk >= self.meta.total_chunks:
            self.meta.info["total_chunks"] += 1

        self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}
        self.meta.info["total_videos"] += len(self.meta.video_keys)
        
        # SKIP update_video_info - videos don't exist yet
        # if len(self.meta.video_keys) > 0:
        #     self.meta.update_video_info()

        write_info(self.meta.info, self.meta.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.meta.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.meta.root)

        self.meta.episodes_stats[episode_index] = episode_stats
        self.meta.stats = aggregate_stats([self.meta.stats, episode_stats]) if self.meta.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.meta.root)
    
    def finalize_video_info(self) -> None:
        """Update video info after all videos are encoded."""
        self.meta.update_video_info()
        from lerobot.common.datasets.lerobot_dataset import write_info
        write_info(self.meta.info, self.meta.root)
    
    @classmethod
    def create(cls, **kwargs) -> "NoEncodingLeRobotDataset":
        """Create a NoEncodingLeRobotDataset with the same API as LeRobotDataset."""
        parent_obj = LeRobotDataset.create(**kwargs)
        obj = cls.__new__(cls)
        obj.__dict__.update(parent_obj.__dict__)
        obj._skip_video_encoding = False
        obj._skip_image_write = False
        return obj


def main(
    push_to_hub: bool = False,
    debug: bool = False,
    debug_episodes: int = 3,
    low_resolution: bool = True,
    num_workers: int = 10,
):
    """
    V4: Highly optimized conversion with episode-level parallelism.
    
    Args:
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        debug: Run in debug mode (process limited episodes)
        debug_episodes: Number of episodes to process in debug mode
        low_resolution: Use low resolution (240x320) instead of full (480x640)
        num_workers: Number of parallel workers for decoding and encoding
    """
    print(f"HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
    print(f"V4: Episode-level parallelism (num_workers={num_workers})")
    
    # Clean up existing dataset
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Find all episodes
    episode_paths = find_episodes(RAW_DATASET_PATHS)
    print(f"Found {len(episode_paths)} episodes")
    if debug:
        episode_paths = episode_paths[:debug_episodes]
        print(f"Debug mode: only processing first {debug_episodes} episodes")
    
    target_size = (320, 240) if low_resolution else (640, 480)
    shape = (target_size[1], target_size[0], 3)  # (H, W, C)
    
    total_start = time.time()
    
    # ========================================
    # Create Dataset (before any directory creation)
    # ========================================
    print("\nCreating LeRobotDataset...")
    dataset = NoEncodingLeRobotDataset.create(
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
                "shape": (get_dim_from_keys(STATE_KEYS),),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (get_dim_from_keys(ACTION_KEYS),),
                "names": ["actions"],
            },
        },
        image_writer_threads=0,  # Disable image writer - images are already on disk
        image_writer_processes=0,
    )
    
    # Enable skip encoding mode
    dataset._skip_video_encoding = True
    
    # ========================================
    # PHASE 1: Parallel Decoding
    # ========================================
    print(f"\n{'='*60}")
    print("PHASE 1: Parallel video decoding (ffmpeg)")
    print(f"{'='*60}")
    
    t_decode_start = time.time()
    
    # Create all decode tasks: each episode × each camera
    decode_tasks = []
    for ep_idx, ep_path in enumerate(episode_paths):
        for camera_name, video_filename in FILE_CAMERA_MAPPING.items():
            decode_tasks.append((ep_path, ep_idx, camera_name, video_filename, output_path, target_size))
    
    episode_frame_counts = {}  # {ep_idx: {camera: num_frames}}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(decode_single_video, *task): task[:3]  # (ep_path, ep_idx, camera)
            for task in decode_tasks
        }
        
        with tqdm.tqdm(total=len(decode_tasks), desc="Decoding videos") as pbar:
            for future in as_completed(futures):
                try:
                    ep_idx, camera_name, num_frames = future.result()
                    if ep_idx not in episode_frame_counts:
                        episode_frame_counts[ep_idx] = {}
                    episode_frame_counts[ep_idx][camera_name] = num_frames
                except Exception as e:
                    ep_path, ep_idx, camera = futures[future]
                    print(f"Error decoding episode {ep_idx} camera {camera}: {e}")
                    raise
                pbar.update(1)
    
    t_decode_end = time.time()
    print(f"Decoding completed in {t_decode_end - t_decode_start:.2f}s")
    
    # ========================================
    # PHASE 2: Build Dataset (sequential, fast)
    # ========================================
    print(f"\n{'='*60}")
    print("PHASE 2: Building dataset (state/action only, skip image I/O)")
    print(f"{'='*60}")
    
    t_build_start = time.time()
    
    # Create a dummy image - won't be saved since _skip_image_write is True
    dummy_image = np.zeros((shape[0], shape[1], shape[2]), dtype=np.uint8)
    
    # Also enable skip image write
    dataset._skip_image_write = True
    
    # Suppress lerobot's internal progress bars during save_episode
    import datasets
    datasets.disable_progress_bars()
    
    for ep_idx, ep_path in tqdm.tqdm(enumerate(episode_paths), total=len(episode_paths), desc="Building dataset"):
        state_array, action_array = load_json_data(ep_path)
        num_frames = len(state_array)
        
        for i in range(num_frames - 1):
            # Use dummy images - they're already on disk from Phase 1
            frame_data = {
                "face_view": dummy_image,
                "left_wrist_view": dummy_image,
                "right_wrist_view": dummy_image,
                "state": state_array[i],
                "actions": action_array[i + 1],
                "task": '',
            }
            
            dataset.add_frame(frame_data)
        
        # Save episode (no encoding, no image deletion)
        dataset.save_episode()
    
    # Re-enable progress bars
    # datasets.enable_progress_bars()
    
    t_build_end = time.time()
    print(f"Dataset building completed in {t_build_end - t_build_start:.2f}s")
    
    # ========================================
    # PHASE 3: Parallel Encoding
    # ========================================
    print(f"\n{'='*60}")
    print("PHASE 3: Parallel video encoding (AV1)")
    print(f"{'='*60}")
    
    t_encode_start = time.time()
    
    # Create all encode tasks
    encode_tasks = []
    for ep_idx in range(len(episode_paths)):
        for camera_name in FILE_CAMERA_MAPPING.keys():
            encode_tasks.append((ep_idx, camera_name, output_path, 20))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(encode_single_video, *task): task[:2]
            for task in encode_tasks
        }
        
        with tqdm.tqdm(total=len(encode_tasks), desc="Encoding videos") as pbar:
            for future in as_completed(futures):
                try:
                    ep_idx, camera_name = future.result()
                except Exception as e:
                    ep_idx, camera_name = futures[future]
                    print(f"Error encoding episode {ep_idx} camera {camera_name}: {e}")
                    raise
                pbar.update(1)
    
    t_encode_end = time.time()
    print(f"Encoding completed in {t_encode_end - t_encode_start:.2f}s")
    
    # ========================================
    # PHASE 4: Finalize and Cleanup
    # ========================================
    print("\nFinalizing video info...")
    dataset.finalize_video_info()
    
    print("Cleaning up images directory...")
    img_dir = output_path / "images"
    if img_dir.is_dir():
        shutil.rmtree(img_dir)
    
    # ========================================
    # Summary
    # ========================================
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Episodes processed: {len(episode_paths)}")
    print(f"  Phase 1 (decode): {t_decode_end - t_decode_start:.2f}s")
    print(f"  Phase 2 (build):  {t_build_end - t_build_start:.2f}s")
    print(f"  Phase 3 (encode): {t_encode_end - t_encode_start:.2f}s")
    print(f"  Total:            {total_time:.2f}s")
    print(f"  Average per episode: {total_time/len(episode_paths):.2f}s")
    print(f"{'='*60}")
    print(f"Dataset saved at {output_path}")
    
    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(main)
