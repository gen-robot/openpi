#!/usr/bin/env python3
"""Convert accepted RoboDojo PICO DAgger episodes to LeRobot.

The source is the append-only ``robodojo-pico-dagger-v2`` format. Only
successful, system-valid, accepted episodes with an intervention are selected.
The training row at t uses state[t] as the observation and state[t + 1] as the
action, so the final source step is not emitted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import hashlib
from itertools import pairwise
import json
from pathlib import Path
import shutil
from typing import Any, Literal

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm

FORMAT_VERSION = "robodojo-pico-dagger-v2"
CONVERTER_VERSION = "dev-openpi-robodojo-dagger-v1"
FPS = 25
STATE_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)
MOTOR_NAMES = [
    *[f"left_{index}" for index in range(6)],
    "left_ee_0",
    *[f"right_{index}" for index in range(6)],
    "right_ee_0",
]
CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


class ConversionError(RuntimeError):
    """The source data does not satisfy the expected DAgger contract."""


@dataclass(frozen=True)
class EpisodeSource:
    manifest_path: Path
    episode_dir: Path
    manifest: dict[str, Any]
    steps: list[dict[str, Any]]

    @property
    def output_frames(self) -> int:
        return len(self.steps) - 1


class SequentialVideoReader:
    """Read increasing frame indices from one MP4 without repeated seeking."""

    def __init__(self, path: Path):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ConversionError(f"Cannot open video: {path}")
        self.next_index = 0

    def read_rgb(self, target_index: int) -> np.ndarray:
        if target_index < self.next_index:
            raise ConversionError(
                f"Non-monotonic video frame index for {self.path}: "
                f"requested={target_index}, next={self.next_index}"
            )
        selected = None
        while self.next_index <= target_index:
            ok, frame_bgr = self.capture.read()
            if not ok or frame_bgr is None:
                raise ConversionError(f"Video ended before frame {target_index}: {self.path}")
            if self.next_index == target_index:
                selected = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.next_index += 1
        if selected is None or selected.shape != (480, 640, 3):
            raise ConversionError(f"Unexpected decoded image from {self.path}")
        return np.ascontiguousarray(selected)

    def close(self) -> None:
        self.capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Exact output dataset name.")
    parser.add_argument(
        "--output-home",
        type=Path,
        default=HF_LEROBOT_HOME,
        help=f"Dataset parent directory (default: {HF_LEROBOT_HOME}).",
    )
    parser.add_argument("--task-name")
    parser.add_argument("--policy-name")
    parser.add_argument("--config-name")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--mode", choices=("video", "image"), default="video")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"Expected JSON object: {path}")
    return value


def read_steps(path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    header_count = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record_type") == "episode_header":
                    header_count += 1
                elif record.get("record_type") == "step":
                    steps.append(record)
                else:
                    raise ConversionError(
                        f"Unknown record_type at {path}:{line_number}: "
                        f"{record.get('record_type')!r}"
                    )
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Cannot read JSONL {path}: {exc}") from exc
    if header_count != 1:
        raise ConversionError(f"Expected one episode header in {path}, got {header_count}")
    return steps


def state_vector(step: dict[str, Any]) -> np.ndarray:
    try:
        source = step["observation"]["data_without_vision"]["state"]
        values: list[float] = []
        for key in STATE_KEYS:
            part = source[key]
            if not isinstance(part, list):
                raise TypeError(f"{key} is not a list")
            values.extend(part)
        vector = np.asarray(values, dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"Invalid state at step {step.get('step_index')}: {exc}") from exc
    if vector.shape != (14,) or not np.isfinite(vector).all():
        raise ConversionError(
            f"Invalid state vector at step {step.get('step_index')}: {vector.shape}"
        )
    return vector


def instruction(step: dict[str, Any]) -> str:
    try:
        value = step["observation"]["data_without_vision"]["instruction"]
    except KeyError as exc:
        raise ConversionError(f"Missing instruction at step {step.get('step_index')}") from exc
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(f"Invalid instruction at step {step.get('step_index')}: {value!r}")
    return value.strip()


def frequency(step: dict[str, Any]) -> int:
    try:
        return int(step["observation"]["data_without_vision"]["additional_info"]["frequency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"Missing frequency at step {step.get('step_index')}") from exc


def video_frame_index(step: dict[str, Any]) -> int:
    try:
        value = int(step["observation"]["video_frame_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"Invalid video frame index at step {step.get('step_index')}") from exc
    if value < 0:
        raise ConversionError(f"Negative video frame index: {value}")
    return value


def _matches_optional(value: Any, expected: str | None) -> bool:
    return expected is None or value == expected


def is_selected(manifest: dict[str, Any], args: argparse.Namespace) -> bool:
    outcome = manifest.get("outcome", {})
    metadata = manifest.get("metadata", {})
    return (
        outcome.get("accepted_for_training") is True
        and outcome.get("success") is True
        and outcome.get("system_valid") is True
        and outcome.get("has_intervention") is True
        and _matches_optional(metadata.get("task_name"), args.task_name)
        and _matches_optional(metadata.get("policy_name"), args.policy_name)
        and _matches_optional(metadata.get("config_name"), args.config_name)
    )


def validate_episode(manifest_path: Path, manifest: dict[str, Any]) -> EpisodeSource:
    episode_dir = manifest_path.parent
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ConversionError(
            f"Unsupported format in {manifest_path}: {manifest.get('format_version')!r}"
        )
    step_path = episode_dir / manifest.get("step_file", "steps.jsonl")
    steps = read_steps(step_path)
    if manifest.get("step_count") != len(steps) or len(steps) < 2:
        raise ConversionError(
            f"Invalid step count in {manifest_path}: "
            f"declared={manifest.get('step_count')}, actual={len(steps)}"
        )
    if [step.get("step_index") for step in steps] != list(range(len(steps))):
        raise ConversionError(f"Non-contiguous step_index in {step_path}")
    frequencies = {frequency(step) for step in steps}
    if frequencies != {FPS}:
        raise ConversionError(f"Expected frequency {FPS}, got {frequencies}: {step_path}")

    frame_indices = [video_frame_index(step) for step in steps]
    if any(right <= left for left, right in pairwise(frame_indices)):
        raise ConversionError(f"video_frame_index is not strictly increasing: {step_path}")

    camera_files = manifest.get("camera_files", {})
    camera_counts = manifest.get("camera_frame_counts", {})
    max_frame = max(frame_indices)
    for source_name in CAMERA_ALIASES:
        relative_path = camera_files.get(source_name)
        video_path = episode_dir / relative_path if relative_path else None
        if video_path is None or not video_path.is_file():
            raise ConversionError(f"Missing camera video {source_name}: {manifest_path}")
        count = camera_counts.get(source_name)
        if not isinstance(count, int) or count <= max_frame:
            raise ConversionError(
                f"Camera {source_name} does not cover frame {max_frame}: count={count}"
            )

    prompts = {instruction(step) for step in steps}
    if len(prompts) != 1:
        raise ConversionError(f"Multiple instructions in {step_path}: {prompts}")
    for step in steps:
        state_vector(step)

    return EpisodeSource(
        manifest_path=manifest_path,
        episode_dir=episode_dir,
        manifest=manifest,
        steps=steps,
    )


def discover_episodes(args: argparse.Namespace) -> list[EpisodeSource]:
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise ConversionError(f"Source root is not a directory: {source_root}")
    episodes = []
    for manifest_path in source_root.glob("**/episodes/*/manifest.json"):
        manifest = read_json(manifest_path)
        if is_selected(manifest, args):
            episodes.append(validate_episode(manifest_path, manifest))
    episodes.sort(
        key=lambda episode: (
            episode.manifest.get("time", {}).get("start_time", ""),
            str(episode.manifest_path),
        )
    )
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ConversionError("--max-episodes must be positive")
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise ConversionError("No accepted intervention episodes matched the filters")
    return episodes


def create_dataset(
    *, root: Path, repo_id: str, mode: Literal["video", "image"]
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
        fps=FPS,
        robot_type="arx_x5",
        features=features,
        use_videos=mode == "video",
        image_writer_processes=0,
        image_writer_threads=1,
    )


def convert_episode(dataset: LeRobotDataset, episode: EpisodeSource) -> None:
    readers = {
        source_name: SequentialVideoReader(
            episode.episode_dir / episode.manifest["camera_files"][source_name]
        )
        for source_name in CAMERA_ALIASES
    }
    try:
        for index in range(episode.output_frames):
            step = episode.steps[index]
            next_step = episode.steps[index + 1]
            frame: dict[str, Any] = {
                "observation.state": state_vector(step),
                "action": state_vector(next_step),
                "task": instruction(step),
            }
            source_frame_index = video_frame_index(step)
            for source_name, output_name in CAMERA_ALIASES.items():
                frame[f"observation.images.{output_name}"] = readers[source_name].read_rgb(
                    source_frame_index
                )
            dataset.add_frame(frame)
        dataset.save_episode()
    finally:
        for reader in readers.values():
            reader.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(episode_index: int, episode: EpisodeSource) -> dict[str, Any]:
    metadata = episode.manifest["metadata"]
    outcome = episode.manifest["outcome"]
    step_path = episode.episode_dir / episode.manifest.get("step_file", "steps.jsonl")
    return {
        "lerobot_episode_index": episode_index,
        "source_episode_id": episode.manifest.get("episode_id"),
        "source_manifest": str(episode.manifest_path.resolve()),
        "source_manifest_sha256": sha256(episode.manifest_path),
        "source_steps": str(step_path.resolve()),
        "source_steps_sha256": sha256(step_path),
        "source_step_count": len(episode.steps),
        "output_frame_count": episode.output_frames,
        "run_id": metadata.get("run_id"),
        "layout_id": metadata.get("layout_id"),
        "layout_seed": metadata.get("layout_seed"),
        "attempt_id": metadata.get("attempt_id"),
        "eval_seed": metadata.get("eval_seed"),
        "task_name": metadata.get("task_name"),
        "instruction": instruction(episode.steps[0]),
        "policy_name": metadata.get("policy_name"),
        "policy_checkpoint": metadata.get("policy_checkpoint"),
        "outcome_summary": outcome.get("summary"),
        "first_source_video_frame_index": video_frame_index(episode.steps[0]),
        "last_emitted_source_video_frame_index": video_frame_index(episode.steps[-2]),
        "dropped_final_source_video_frame_index": video_frame_index(episode.steps[-1]),
    }


def safe_target(output_home: Path, repo_id: str) -> Path:
    output_home = output_home.resolve()
    target = (output_home / repo_id).resolve()
    if target == output_home or not target.is_relative_to(output_home):
        raise ConversionError(f"Unsafe output target: {target}")
    return target


def print_inventory(episodes: list[EpisodeSource], target: Path) -> None:
    layouts = {episode.manifest["metadata"].get("layout_id") for episode in episodes}
    print(f"selected_episodes={len(episodes)}")
    print(f"unique_layouts={len(layouts)}")
    print(f"source_steps={sum(len(episode.steps) for episode in episodes)}")
    print(f"output_frames={sum(episode.output_frames for episode in episodes)}")
    print(f"fps={FPS}")
    print(f"target={target}")


def main() -> None:
    args = parse_args()
    episodes = discover_episodes(args)
    target = safe_target(args.output_home, args.repo_id)
    print_inventory(episodes, target)
    if args.inspect_only:
        print("inspect_only=PASS")
        return

    if target.exists():
        if not args.overwrite:
            raise ConversionError(f"Target already exists: {target}; pass --overwrite")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    expected_frames = sum(episode.output_frames for episode in episodes)
    dataset = create_dataset(root=target, repo_id=args.repo_id, mode=args.mode)
    entries = []
    for episode_index, episode in enumerate(
        tqdm.tqdm(episodes, desc="Converting", unit="episode")
    ):
        convert_episode(dataset, episode)
        entries.append(provenance(episode_index, episode))

    if dataset.meta.total_episodes != len(episodes) or dataset.meta.total_frames != expected_frames:
        raise ConversionError(
            "Output count mismatch: "
            f"episodes={dataset.meta.total_episodes}/{len(episodes)}, "
            f"frames={dataset.meta.total_frames}/{expected_frames}"
        )

    conversion_manifest = {
        "converter_format": CONVERTER_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_root": str(args.source_root.resolve()),
        "selection": {
            "accepted_for_training": True,
            "success": True,
            "system_valid": True,
            "has_intervention": True,
            "task_name": args.task_name,
            "policy_name": args.policy_name,
            "config_name": args.config_name,
            "keep_repeated_layouts": True,
        },
        "mapping": {
            "fps": FPS,
            "state": "source observation state at t",
            "action": "source observation state at t+1",
            "final_source_step": "dropped",
            "image_index": "source observation.video_frame_index at t",
            "camera_aliases": CAMERA_ALIASES,
            "motor_names": MOTOR_NAMES,
        },
        "output": {
            "repo_id": args.repo_id,
            "root": str(target),
            "codebase_version": dataset.meta.info.get("codebase_version"),
            "robot_type": "arx_x5",
            "fps": FPS,
            "episodes": len(episodes),
            "frames": expected_frames,
        },
        "episodes": entries,
    }
    manifest_path = target / "conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(conversion_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"conversion_manifest={manifest_path}")
    print(
        f"conversion=PASS episodes={dataset.meta.total_episodes} "
        f"frames={dataset.meta.total_frames} target={target}"
    )


if __name__ == "__main__":
    main()
