"""Build an entropy-guided accelerated LeRobot dataset from DemoSpeedup labels.

Precision (low-entropy) regions use the conservative stride (default 2), while
Casualness regions use the aggressive stride (default 4). Episode boundaries,
precision transitions, gripper transitions, and configured motion limits can add
frames back. Source datasets and entropy sidecars remain unchanged.

Example:
  uv run examples/x2robot/demospeedup/build_dataset.py \
    --dataset bagging_sku_zzc_0810 --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile

from lerobot.common.datasets.utils import load_episodes
from lerobot.common.datasets.utils import load_episodes_stats
from lerobot.common.datasets.utils import write_episode
from lerobot.common.datasets.utils import write_episode_stats
from lerobot.common.datasets.utils import write_info
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

try:
    from examples.x2robot.filter_lerobot_dagger import compute_episode_stats
    from examples.x2robot.filter_lerobot_dagger import copy_extra_metadata
    from examples.x2robot.filter_lerobot_dagger import filter_video
    from examples.x2robot.filter_lerobot_dagger import format_episode_path
    from examples.x2robot.filter_lerobot_dagger import reset_indices
    from examples.x2robot.filter_lerobot_dagger import resolve_dataset
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from filter_lerobot_dagger import compute_episode_stats
    from filter_lerobot_dagger import copy_extra_metadata
    from filter_lerobot_dagger import filter_video
    from filter_lerobot_dagger import format_episode_path
    from filter_lerobot_dagger import reset_indices
    from filter_lerobot_dagger import resolve_dataset


DEFAULT_LABEL_ROOT = Path("/mnt/public/gaojiahuan/huggingface/demospeedup_labels")


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    dataset: Path
    output: Path | None = None
    label_root: Path = DEFAULT_LABEL_ROOT
    precision_stride: int = 2
    casual_stride: int = 4
    transition_boundary: int = 1
    gripper_threshold: float = 0.05
    max_position_step: float | None = 0.04
    max_rotation_step: float | None = 0.25
    max_gripper_step: float | None = 1.0
    action_dims: int = 14
    min_episode_frames: int = 2
    dry_run: bool = False


def binary_transition_mask(values: np.ndarray, boundary: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    mask = np.zeros(len(values), dtype=bool)
    if not len(values):
        return mask
    transitions = np.flatnonzero(values[1:] != values[:-1]) + 1
    for transition in transitions:
        mask[max(0, transition - boundary) : min(len(values), transition + boundary + 1)] = True
    return mask


def stride_keep_mask(
    precision: np.ndarray,
    *,
    precision_stride: int = 2,
    casual_stride: int = 4,
    transition_boundary: int = 1,
) -> np.ndarray:
    """Downsample without jumping across a Precision/Casualness transition."""
    precision = np.asarray(precision, dtype=bool)
    if not len(precision):
        return np.zeros(0, dtype=bool)
    if precision_stride < 1 or casual_stride < 1:
        raise ValueError("Sampling strides must be positive")
    keep = np.zeros(len(precision), dtype=bool)
    keep[0] = True
    cursor = 0
    while cursor < len(precision) - 1:
        stride = precision_stride if precision[cursor] else casual_stride
        candidate = min(cursor + stride, len(precision) - 1)
        transitions = np.flatnonzero(precision[cursor + 1 : candidate + 1] != precision[cursor])
        if len(transitions):
            candidate = cursor + 1 + int(transitions[0])
        keep[candidate] = True
        cursor = candidate
    keep |= binary_transition_mask(precision, transition_boundary)
    keep[-1] = True
    return keep


def gripper_keyframes(actions: np.ndarray, *, action_dims: int, threshold: float, boundary: int) -> np.ndarray:
    actions = np.asarray(actions)
    if action_dims < 14 or actions.shape[1] < action_dims:
        raise ValueError(f"Expected at least 14 action dimensions, got {actions.shape}")
    gripper_dims = np.asarray([6, 13], dtype=np.int64)
    changed = np.any(np.abs(np.diff(actions[:, gripper_dims], axis=0)) > threshold, axis=1)
    transition_frames = np.concatenate(([False], changed))
    keyframes = np.zeros(len(actions), dtype=bool)
    for transition in np.flatnonzero(transition_frames):
        keyframes[max(0, transition - boundary) : min(len(actions), transition + boundary + 1)] = True
    return keyframes


def _wrapped_rotation_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = right - left
    return np.abs((delta + np.pi) % (2 * np.pi) - np.pi)


def motion_limit_violated(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_position_step: float | None,
    max_rotation_step: float | None,
    max_gripper_step: float | None,
) -> bool:
    position_dims = np.asarray([0, 1, 2, 7, 8, 9])
    rotation_dims = np.asarray([3, 4, 5, 10, 11, 12])
    gripper_dims = np.asarray([6, 13])
    if max_position_step is not None and np.max(np.abs(right[position_dims] - left[position_dims])) > max_position_step:
        return True
    if (
        max_rotation_step is not None
        and np.max(_wrapped_rotation_delta(left[rotation_dims], right[rotation_dims])) > max_rotation_step
    ):
        return True
    return bool(
        max_gripper_step is not None and np.max(np.abs(right[gripper_dims] - left[gripper_dims])) > max_gripper_step
    )


def enforce_motion_limits(
    keep: np.ndarray,
    actions: np.ndarray,
    *,
    max_position_step: float | None,
    max_rotation_step: float | None,
    max_gripper_step: float | None,
) -> tuple[np.ndarray, int]:
    """Recursively add source frames until retained endpoint jumps satisfy limits."""
    keep = np.asarray(keep, dtype=bool).copy()
    actions = np.asarray(actions)
    while True:
        retained = np.flatnonzero(keep)
        additions: list[int] = []
        for left_index, right_index in itertools.pairwise(retained):
            if right_index - left_index <= 1:
                continue
            if motion_limit_violated(
                actions[left_index],
                actions[right_index],
                max_position_step=max_position_step,
                max_rotation_step=max_rotation_step,
                max_gripper_step=max_gripper_step,
            ):
                additions.append((int(left_index) + int(right_index)) // 2)
        if not additions:
            break
        keep[additions] = True

    # Adjacent source frames cannot be repaired by downsampling; report them so
    # they are visible in the manifest and safety review.
    unresolved = 0
    retained = np.flatnonzero(keep)
    for left_index, right_index in itertools.pairwise(retained):
        if motion_limit_violated(
            actions[left_index],
            actions[right_index],
            max_position_step=max_position_step,
            max_rotation_step=max_rotation_step,
            max_gripper_step=max_gripper_step,
        ):
            unresolved += 1
    return keep, unresolved


def build_keep_mask(table: pa.Table, label_path: Path, config: BuildConfig) -> tuple[np.ndarray, dict]:
    with np.load(label_path) as labels:
        precision = np.asarray(labels["precision"], dtype=bool)
        normalized_entropy = np.asarray(labels["normalized_entropy"], dtype=np.float32)
    if len(precision) != len(table):
        raise ValueError(f"Label/table length mismatch for {label_path}: {len(precision)} != {len(table)}")
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float64)[:, : config.action_dims]
    keep = stride_keep_mask(
        precision,
        precision_stride=config.precision_stride,
        casual_stride=config.casual_stride,
        transition_boundary=config.transition_boundary,
    )
    keep |= gripper_keyframes(
        actions,
        action_dims=config.action_dims,
        threshold=config.gripper_threshold,
        boundary=config.transition_boundary,
    )
    before_limits = int(keep.sum())
    keep, unresolved = enforce_motion_limits(
        keep,
        actions,
        max_position_step=config.max_position_step,
        max_rotation_step=config.max_rotation_step,
        max_gripper_step=config.max_gripper_step,
    )
    summary = {
        "source_frames": len(table),
        "output_frames": int(keep.sum()),
        "precision_frames": int(precision.sum()),
        "casual_frames": int((~precision).sum()),
        "frames_added_by_motion_limits": int(keep.sum()) - before_limits,
        "unresolved_source_motion_limit_edges": unresolved,
        "mean_normalized_entropy": float(np.mean(normalized_entropy)),
    }
    return keep, summary


def build_dataset(config: BuildConfig) -> None:
    source = resolve_dataset(config.dataset)
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset not found: {source}")
    output = (
        resolve_dataset(config.output)
        if config.output
        else source.with_name(f"{source.name}_demospeedup_r{config.precision_stride}r{config.casual_stride}")
    )
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    label_dir = config.label_root / source.name
    manifest_path = label_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Entropy manifest not found: {manifest_path}")
    label_manifest = json.loads(manifest_path.read_text())

    with (source / "meta" / "info.json").open() as file:
        info = json.load(file)
    episodes = load_episodes(source)
    episodes_stats = load_episodes_stats(source)
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    plans = []
    summaries = []
    for episode_index in sorted(episodes):
        data_path = source / format_episode_path(info["data_path"], episode_index, info["chunks_size"])
        table = pq.read_table(data_path)
        label_path = label_dir / f"episode_{episode_index:06d}.npz"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing entropy labels: {label_path}")
        keep, summary = build_keep_mask(table, label_path, config)
        summary["source_episode_index"] = episode_index
        summary["dropped"] = int(keep.sum()) < config.min_episode_frames
        summaries.append(summary)
        plans.append((episode_index, table, keep, summary["dropped"]))
        print(
            f"episode {episode_index:06d}: {len(table):5d} -> {int(keep.sum()):5d}, "
            f"precision {summary['precision_frames']:5d}, safety +{summary['frames_added_by_motion_limits']:4d}, "
            f"unresolved {summary['unresolved_source_motion_limit_edges']:3d}"
            + (" [drop]" if summary["dropped"] else "")
        )

    kept_plans = [plan for plan in plans if not plan[3]]
    total_source_frames = sum(len(plan[1]) for plan in plans)
    total_output_frames = sum(int(plan[2].sum()) for plan in kept_plans)
    print(
        f"summary: episodes {len(plans)} -> {len(kept_plans)}, "
        f"frames {total_source_frames} -> {total_output_frames}, "
        f"speedup ratio {total_source_frames / max(total_output_frames, 1):.3f}x"
    )
    if config.dry_run:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary_dir:
        temporary_output = Path(temporary_dir) / output.name
        copy_extra_metadata(source, temporary_output)
        output_episode_index = 0
        global_index = 0
        progress = tqdm.tqdm(
            total=len(kept_plans) * (len(video_keys) + 1),
            desc="Writing DemoSpeedup dataset",
            unit="file",
            dynamic_ncols=True,
        )
        for source_episode_index, table, keep, _ in kept_plans:
            filtered = table.filter(pa.array(keep))
            filtered = reset_indices(filtered, output_episode_index, global_index, info["fps"])
            output_data_path = temporary_output / format_episode_path(
                info["data_path"], output_episode_index, info["chunks_size"]
            )
            output_data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(filtered, output_data_path, compression="snappy")
            progress.update()

            for video_key in video_keys:
                source_video = source / format_episode_path(
                    info["video_path"], source_episode_index, info["chunks_size"], video_key
                )
                output_video = temporary_output / format_episode_path(
                    info["video_path"], output_episode_index, info["chunks_size"], video_key
                )
                filter_video(
                    source_video,
                    output_video,
                    keep,
                    info["fps"],
                    info["features"][video_key]["info"]["video.codec"],
                )
                progress.update()

            episode = dict(episodes[source_episode_index])
            episode["episode_index"] = output_episode_index
            episode["length"] = len(filtered)
            write_episode(episode, temporary_output)
            write_episode_stats(
                output_episode_index,
                compute_episode_stats(filtered, info["features"], episodes_stats[source_episode_index]),
                temporary_output,
            )
            global_index += len(filtered)
            output_episode_index += 1
        progress.close()

        output_info = dict(info)
        output_info["total_episodes"] = output_episode_index
        output_info["total_frames"] = global_index
        output_info["total_videos"] = output_episode_index * len(video_keys)
        output_info["total_chunks"] = math.ceil(output_episode_index / info["chunks_size"])
        output_info["splits"] = {"train": f"0:{output_episode_index}"}
        write_info(output_info, temporary_output)
        build_manifest = {
            "schema_version": 1,
            "method": "DemoSpeedup",
            "source": str(source.resolve()),
            "output": str(output.resolve()),
            "label_manifest": label_manifest,
            "build_parameters": dataclasses.asdict(config)
            | {
                "dataset": str(config.dataset),
                "output": str(config.output) if config.output else None,
                "label_root": str(config.label_root),
            },
            "source_frames": total_source_frames,
            "output_frames": total_output_frames,
            "nominal_execution_speedup": total_source_frames / max(total_output_frames, 1),
            "episodes": summaries,
        }
        manifest_output = temporary_output / "meta" / "demospeedup.json"
        manifest_output.write_text(json.dumps(build_manifest, indent=2, ensure_ascii=False) + "\n")
        temporary_output.rename(output)
    print(f"Wrote DemoSpeedup dataset to {output}")


def optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "off", "disabled"}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive number or 'none'")
    return parsed


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--precision-stride", type=int, default=2)
    parser.add_argument("--casual-stride", type=int, default=4)
    parser.add_argument("--transition-boundary", type=int, default=1)
    parser.add_argument("--gripper-threshold", type=float, default=0.05)
    parser.add_argument("--max-position-step", type=optional_positive_float, default=0.04)
    parser.add_argument("--max-rotation-step", type=optional_positive_float, default=0.25)
    parser.add_argument("--max-gripper-step", type=optional_positive_float, default=1.0)
    parser.add_argument("--action-dims", type=int, default=14)
    parser.add_argument("--min-episode-frames", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.precision_stride < 1 or args.casual_stride < 1:
        parser.error("sampling strides must be positive")
    if args.transition_boundary < 0:
        parser.error("transition-boundary must be non-negative")
    if args.gripper_threshold < 0:
        parser.error("gripper-threshold must be non-negative")
    return BuildConfig(**vars(args))


if __name__ == "__main__":
    build_dataset(parse_args())
