"""Convert X2Robot raw data to unified LeRobot format.

Stores all slave+master state data in a single 28-D vector per frame,
enabling mode selection (s2s/s2m/sm2m/sm2sm) at training time without re-conversion.

State/Action dimension layout (28-D):
  [0:3]   follow_left_position
  [3:6]   follow_left_rotation
  [6:7]   follow_left_gripper
  [7:10]  follow_right_position
  [10:13] follow_right_rotation
  [13:14] follow_right_gripper
  [14:17] master_left_position
  [17:20] master_left_rotation
  [20:21] master_left_gripper
  [21:24] master_right_position
  [24:27] master_right_rotation
  [27:28] master_right_gripper

Usage:
  uv run examples/x2robot/convert_x2robot_unified.py \\
    --raw-paths ./datasets/x2robot/place_goods/ \\
    --repo-name place_goods_unified \\
    --task "Put the goods on your left into the bag on your right."
"""

import glob
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

os.environ["SVT_LOG"] = "0"

# Fixed key ordering that defines the 28-D layout.
# Each entry: (json_key, dimension)
STATE_KEYS: list[tuple[str, int]] = [
    ("follow_left_position", 3),
    ("follow_left_rotation", 3),
    ("follow_left_gripper", 1),
    ("follow_right_position", 3),
    ("follow_right_rotation", 3),
    ("follow_right_gripper", 1),
    ("master_left_position", 3),
    ("master_left_rotation", 3),
    ("master_left_gripper", 1),
    ("master_right_position", 3),
    ("master_right_rotation", 3),
    ("master_right_gripper", 1),
]

STATE_DIM = sum(d for _, d in STATE_KEYS)  # 28

CAMERA_MAPPING: dict[str, str] = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4",
}


def find_episodes(raw_paths: list[str]) -> list[str]:
    """Find all episode directories containing MP4 files."""
    episode_paths = []
    for raw_path in raw_paths:
        for dir_path in sorted(glob.glob(f"{raw_path}/*")):
            if os.path.isdir(dir_path) and glob.glob(f"{dir_path}/*.mp4"):
                episode_paths.append(dir_path)
    return sorted(episode_paths)


def load_episode_data(episode_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load JSON and return (state_array, action_array) with the unified 28-D layout.

    state_array[i] = values at frame i.
    action_array[i] = values at frame i+1 (next-frame-as-action convention).
    Both have shape (num_frames-1, 28).
    """
    episode_name = os.path.basename(episode_path)
    json_path = os.path.join(episode_path, f"{episode_name}.json")

    with open(json_path) as f:
        frames = json.load(f)["data"]

    num_frames = len(frames)
    all_values = np.empty((num_frames, STATE_DIM), dtype=np.float32)

    for i, frame in enumerate(frames):
        offset = 0
        for key, dim in STATE_KEYS:
            val = frame[key]
            if dim == 1:
                all_values[i, offset] = float(val)
            else:
                all_values[i, offset : offset + dim] = val
            offset += dim

    # state[i], actions[i] = values[i], values[i+1]
    return all_values[:-1], all_values[1:]


def transcode_video(
    src: str,
    dst: Path,
    target_size: tuple[int, int],
    fps: int = 20,
) -> int:
    """Transcode a video to AV1 and return frame count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-vf", f"scale={target_size[0]}:{target_size[1]}",
        "-c:v", "libsvtav1",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-g", "2",
        "-crf", "30",
        str(dst),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env={**os.environ, "SVT_LOG": "0"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}: {result.stderr}")

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            str(dst),
        ],
        capture_output=True, text=True,
    )
    return int(probe.stdout.strip())


def _transcode_one(
    episode_path: str,
    episode_index: int,
    camera_name: str,
    video_filename: str,
    output_root: Path,
    target_size: tuple[int, int],
) -> tuple[int, str, int]:
    src = os.path.join(episode_path, video_filename)
    dst = output_root / "videos" / "chunk-000" / camera_name / f"episode_{episode_index:06d}.mp4"
    n = transcode_video(src, dst, target_size)
    return episode_index, camera_name, n


def _dummy_video_stats(num_frames: int) -> dict:
    return {
        "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),
        "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
        "mean": np.array([[[0.4]], [[0.4]], [[0.4]]]),
        "std": np.array([[[0.25]], [[0.25]], [[0.25]]]),
        "count": np.array([num_frames]),
    }


class _NoVideoIODataset(LeRobotDataset):
    """LeRobotDataset subclass that skips video I/O in save_episode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_media = False
        self._video_frame_count = 0

    def _save_image(self, image, fpath: Path) -> None:
        if self._skip_media:
            return
        super()._save_image(image, fpath)

    def save_episode(self, episode_data: dict | None = None) -> None:
        if not self._skip_media:
            super().save_episode(episode_data)
            return

        episode_buffer = self.episode_buffer if not episode_data else episode_data

        from lerobot.common.datasets.compute_stats import get_feature_stats
        from lerobot.common.datasets.lerobot_dataset import (
            aggregate_stats,
            validate_episode_buffer,
            write_episode,
            write_episode_stats,
            write_info,
        )

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(t) for t in tasks])

        for key, ft in self.features.items():
            if key in ("index", "episode_index", "task_index") or ft["dtype"] in ("image", "video"):
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        ep_stats = {}
        for key, data in episode_buffer.items():
            if key not in self.features:
                continue
            ft = self.features[key]
            if ft["dtype"] == "string":
                continue
            if ft["dtype"] in ("image", "video"):
                ep_stats[key] = _dummy_video_stats(self._video_frame_count)
            else:
                ep_stats[key] = get_feature_stats(data, axis=0, keepdims=data.ndim == 1)

        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += episode_length
        chunk = self.meta.get_episode_chunk(episode_index)
        if chunk >= self.meta.total_chunks:
            self.meta.info["total_chunks"] += 1
        self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}
        self.meta.info["total_videos"] += len(self.meta.video_keys)

        write_info(self.meta.info, self.meta.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.meta.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.meta.root)

        self.meta.episodes_stats[episode_index] = ep_stats
        self.meta.stats = aggregate_stats([self.meta.stats, ep_stats]) if self.meta.stats else ep_stats
        write_episode_stats(episode_index, ep_stats, self.meta.root)

        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()

    @classmethod
    def create(cls, **kwargs) -> "_NoVideoIODataset":
        parent = LeRobotDataset.create(**kwargs)
        obj = cls.__new__(cls)
        obj.__dict__.update(parent.__dict__)
        obj._skip_media = False
        obj._video_frame_count = 0
        return obj


@dataclass
class ConvertArgs:
    raw_paths: list[str]
    repo_name: str
    task: str = ""
    low_resolution: bool = True
    num_workers: int = 10
    fps: int = 20
    debug: bool = False
    debug_episodes: int = 3
    push_to_hub: bool = False


def main(args: ConvertArgs):
    print(f"HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
    print(f"State/Action dim: {STATE_DIM}")

    output_path = HF_LEROBOT_HOME / args.repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    episode_paths = find_episodes(args.raw_paths)
    print(f"Found {len(episode_paths)} episodes")
    if args.debug:
        episode_paths = episode_paths[: args.debug_episodes]
        print(f"Debug mode: processing {len(episode_paths)} episodes")

    target_size = (320, 240) if args.low_resolution else (640, 480)
    shape = (target_size[1], target_size[0], 3)

    total_start = time.time()

    # -- Create dataset --
    dataset = _NoVideoIODataset.create(
        repo_id=args.repo_name,
        robot_type="ARX",
        fps=args.fps,
        features={
            cam: {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"]}
            for cam in CAMERA_MAPPING
        }
        | {
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["actions"]},
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )
    dataset._skip_media = True

    # -- Phase 1: parallel video transcoding --
    print(f"\n{'=' * 60}")
    print("Phase 1: Parallel video transcoding")
    print(f"{'=' * 60}")
    t1 = time.time()

    tasks = [
        (ep, idx, cam, vid, output_path, target_size)
        for idx, ep in enumerate(episode_paths)
        for cam, vid in CAMERA_MAPPING.items()
    ]

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_transcode_one, *t): t[:3] for t in tasks}
        with tqdm.tqdm(total=len(futures), desc="Transcoding") as pbar:
            for fut in as_completed(futures):
                fut.result()
                pbar.update(1)

    t1_end = time.time()
    print(f"Transcoding: {t1_end - t1:.1f}s")

    # -- Phase 2: build dataset metadata --
    print(f"\n{'=' * 60}")
    print("Phase 2: Building dataset")
    print(f"{'=' * 60}")
    t2 = time.time()

    import datasets as _hf_datasets
    _hf_datasets.disable_progress_bars()

    dummy_img = np.zeros(shape, dtype=np.uint8)

    for ep_idx, ep_path in tqdm.tqdm(
        enumerate(episode_paths), total=len(episode_paths), desc="Building"
    ):
        state_arr, action_arr = load_episode_data(ep_path)
        n = len(state_arr)
        dataset._video_frame_count = n

        for i in range(n):
            dataset.add_frame({
                "face_view": dummy_img,
                "left_wrist_view": dummy_img,
                "right_wrist_view": dummy_img,
                "state": state_arr[i],
                "actions": action_arr[i],
                "task": args.task,
            })
        dataset.save_episode()

    t2_end = time.time()
    print(f"Building: {t2_end - t2:.1f}s")

    # -- Finalize --
    dataset.meta.update_video_info()
    from lerobot.common.datasets.lerobot_dataset import write_info
    write_info(dataset.meta.info, dataset.meta.root)

    img_dir = output_path / "images"
    if img_dir.is_dir():
        shutil.rmtree(img_dir)

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done: {len(episode_paths)} episodes in {total:.1f}s ({total / len(episode_paths):.1f}s/ep)")
    print(f"Dataset saved at {output_path}")
    print(f"{'=' * 60}")

    if args.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(main)
