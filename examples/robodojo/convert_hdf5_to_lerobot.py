#!/usr/bin/env python3
"""Convert official RoboDojo ARX X5 HDF5 demonstrations to LeRobot.

Each source file is one episode. RoboDojo stores state and action as
``left arm (6), left gripper (1), right arm (6), right gripper (1)``.

The official data satisfies ``action[t] == state[t + 1]``. We validate that
contract, write the action stored by RoboDojo, and omit the final source row so
every emitted row represents a transition with a following state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm

CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
MOTOR_NAMES = [
    *[f"left_{index}" for index in range(6)],
    "left_ee_0",
    *[f"right_{index}" for index in range(6)],
    "right_ee_0",
]


class ConversionError(RuntimeError):
    """The source data does not satisfy the expected RoboDojo contract."""


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    source_frames: int
    fps: int
    instruction: str

    @property
    def output_frames(self) -> int:
        return self.source_frames - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Exact output dataset name.")
    parser.add_argument(
        "--output-home",
        type=Path,
        default=HF_LEROBOT_HOME,
        help=f"Dataset parent directory (default: {HF_LEROBOT_HOME}).",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--mode", choices=("video", "image"), default="video")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _scalar(dataset: h5py.Dataset) -> Any:
    value = dataset[()]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _instruction(file: h5py.File) -> str:
    if "instruction" not in file:
        raise ConversionError(f"Missing instruction in {file.filename}")
    value = _scalar(file["instruction"])
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(f"Invalid instruction in {file.filename}: {value!r}")
    return value.strip()


def _fps(file: h5py.File) -> int:
    key = "additional_info/frequency"
    if key not in file:
        raise ConversionError(f"Missing {key} in {file.filename}")
    fps = int(_scalar(file[key]))
    if fps <= 0:
        raise ConversionError(f"Invalid frequency in {file.filename}: {fps}")
    return fps


def _column(group: h5py.Group, key: str, width: int) -> np.ndarray:
    if key not in group:
        raise ConversionError(f"Missing {group.name}/{key}")
    value = np.asarray(group[key][:], dtype=np.float32)
    if value.ndim == 1 and width == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[1] != width:
        raise ConversionError(
            f"Unexpected {group.name}/{key} shape {value.shape}; expected (T, {width})"
        )
    return value


def robot_vector(file: h5py.File, group_name: str) -> np.ndarray:
    if group_name not in file:
        raise ConversionError(f"Missing group {group_name} in {file.filename}")
    group = file[group_name]
    vector = np.concatenate(
        [
            _column(group, "left_arm_joint_states", 6),
            _column(group, "left_ee_joint_states", 1),
            _column(group, "right_arm_joint_states", 6),
            _column(group, "right_ee_joint_states", 1),
        ],
        axis=1,
    )
    if not np.isfinite(vector).all():
        raise ConversionError(f"Non-finite values in {group_name} of {file.filename}")
    return vector


def decode_rgb(payload: Any) -> np.ndarray:
    if isinstance(payload, np.ndarray) and payload.dtype == np.uint8 and payload.ndim == 1:
        encoded = payload
    elif isinstance(payload, bytes | bytearray | np.bytes_ | np.void):
        encoded = np.frombuffer(bytes(payload).rstrip(b"\0"), dtype=np.uint8)
    else:
        raise ConversionError(f"Unsupported encoded image type: {type(payload)!r}")

    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ConversionError("OpenCV failed to decode an HDF5 image")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if image_rgb.shape != (480, 640, 3) or image_rgb.dtype != np.uint8:
        raise ConversionError(
            f"Unexpected image shape/dtype: {image_rgb.shape}, {image_rgb.dtype}"
        )
    return np.ascontiguousarray(image_rgb)


def inspect_episode(path: Path) -> EpisodeInfo:
    with h5py.File(path, "r") as file:
        state = robot_vector(file, "state")
        action = robot_vector(file, "action")
        if state.shape != action.shape or len(state) < 2:
            raise ConversionError(
                f"Invalid state/action horizons in {path}: {state.shape}, {action.shape}"
            )
        if not np.allclose(action[:-1], state[1:], rtol=0.0, atol=1e-6):
            error = float(np.max(np.abs(action[:-1] - state[1:])))
            raise ConversionError(
                f"action[t] != state[t+1] in {path}; max_abs_error={error}"
            )

        for source_name in CAMERA_ALIASES:
            key = f"vision/{source_name}/colors"
            if key not in file or len(file[key]) != len(state):
                raise ConversionError(f"Missing or misaligned camera {key} in {path}")
            decode_rgb(file[key][0])
            decode_rgb(file[key][-1])

        return EpisodeInfo(
            path=path,
            source_frames=len(state),
            fps=_fps(file),
            instruction=_instruction(file),
        )


def discover_episodes(raw_dir: Path, max_episodes: int | None) -> list[EpisodeInfo]:
    if max_episodes is not None and max_episodes <= 0:
        raise ConversionError("--max-episodes must be positive")
    paths = sorted(raw_dir.glob("episode_*.hdf5"))
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise ConversionError(f"No episode_*.hdf5 files found in {raw_dir}")
    episodes = [inspect_episode(path) for path in tqdm.tqdm(paths, desc="Inspecting")]
    frequencies = {episode.fps for episode in episodes}
    if len(frequencies) != 1:
        raise ConversionError(f"Mixed source frequencies are not supported: {frequencies}")
    return episodes


def create_dataset(
    *,
    root: Path,
    repo_id: str,
    fps: int,
    mode: Literal["video", "image"],
) -> LeRobotDataset:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [MOTOR_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [MOTOR_NAMES],
        },
    }
    for output_name in CAMERA_ALIASES.values():
        features[f"observation.images.{output_name}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type="arx_x5",
        features=features,
        use_videos=mode == "video",
        image_writer_processes=0,
        image_writer_threads=1,
    )


def convert_episode(dataset: LeRobotDataset, episode: EpisodeInfo) -> None:
    with h5py.File(episode.path, "r") as file:
        state = robot_vector(file, "state")
        action = robot_vector(file, "action")
        for index in range(episode.output_frames):
            frame: dict[str, Any] = {
                "observation.state": state[index],
                "action": action[index],
                "task": episode.instruction,
            }
            for source_name, output_name in CAMERA_ALIASES.items():
                frame[f"observation.images.{output_name}"] = decode_rgb(
                    file[f"vision/{source_name}/colors"][index]
                )
            dataset.add_frame(frame)
    dataset.save_episode()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_home = args.output_home.resolve()
    target = output_home / args.repo_id

    episodes = discover_episodes(raw_dir, args.max_episodes)
    expected_frames = sum(episode.output_frames for episode in episodes)
    fps = episodes[0].fps
    print(f"source={raw_dir}")
    print(f"episodes={len(episodes)} expected_frames={expected_frames} fps={fps}")
    print(f"target={target}")

    if target.exists():
        if not args.overwrite:
            raise ConversionError(f"Target already exists: {target}; pass --overwrite to replace it")
        shutil.rmtree(target)
    output_home.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(root=target, repo_id=args.repo_id, fps=fps, mode=args.mode)
    for episode in tqdm.tqdm(episodes, desc="Converting", unit="episode"):
        convert_episode(dataset, episode)

    if dataset.meta.total_episodes != len(episodes) or dataset.meta.total_frames != expected_frames:
        raise ConversionError(
            "Output count mismatch: "
            f"episodes={dataset.meta.total_episodes}/{len(episodes)}, "
            f"frames={dataset.meta.total_frames}/{expected_frames}"
        )
    print(
        f"conversion=PASS episodes={dataset.meta.total_episodes} "
        f"frames={dataset.meta.total_frames} target={target}"
    )


if __name__ == "__main__":
    main()
