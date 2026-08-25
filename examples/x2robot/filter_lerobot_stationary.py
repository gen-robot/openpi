"""Compress stationary spans in an X2Robot LeRobot dataset.

A window is stationary only when position, rotation, and gripper ranges all stay
below their respective thresholds. Long stationary spans are reduced to boundary
frames, while parquet rows, videos, episode metadata, indices, timestamps, and
statistics are rebuilt together.

Examples:
  uv run examples/x2robot/filter_lerobot_stationary.py bagging_sku_fty_0703 --dry-run

  uv run examples/x2robot/filter_lerobot_stationary.py /data/source \
    --output /data/source_motion_only --window-seconds 1.0
"""

import argparse
import json
import math
from pathlib import Path
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
    # Direct script execution places examples/x2robot, rather than the repository root, on sys.path.
    from filter_lerobot_dagger import compute_episode_stats
    from filter_lerobot_dagger import copy_extra_metadata
    from filter_lerobot_dagger import filter_video
    from filter_lerobot_dagger import format_episode_path
    from filter_lerobot_dagger import reset_indices
    from filter_lerobot_dagger import resolve_dataset


def parse_dims(spec: str, state_dim: int, name: str) -> np.ndarray:
    """Parse comma-separated indices and inclusive-exclusive ranges."""
    indices: list[int] = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, end_text = token.split(":", maxsplit=1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else state_dim
            indices.extend(range(start, end))
        else:
            indices.append(int(token))
    if not indices:
        raise ValueError(f"{name} must select at least one dimension")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate dimensions: {spec!r}")
    if min(indices) < 0 or max(indices) >= state_dim:
        raise ValueError(f"{name} {spec!r} is outside state dimension {state_dim}")
    return np.asarray(indices, dtype=np.int64)


def default_motion_dims(state_dim: int) -> tuple[str, str, str]:
    if state_dim == 14:
        return "0:3,7:10", "3:6,10:13", "6,13"
    if state_dim >= 28:
        return "0:3,7:10,14:17,21:24", "3:6,10:13,17:20,24:27", "6,13,20,27"
    raise ValueError(
        f"Cannot infer X2Robot pose/gripper dimensions from a {state_dim}-D state; "
        "pass --position-dims, --rotation-dims, and --gripper-dims explicitly"
    )


def stationary_window_mask(
    state: np.ndarray,
    window_frames: int,
    position_dims: np.ndarray,
    rotation_dims: np.ndarray,
    gripper_dims: np.ndarray,
    position_threshold: float,
    rotation_threshold: float,
    gripper_threshold: float,
) -> np.ndarray:
    """Return frames covered by a qualifying stationary window."""
    if len(state) < window_frames:
        return np.zeros(len(state), dtype=bool)

    position = state[:, position_dims]
    # Euler angles can cross +/-pi without physical motion; unwrap before measuring ranges.
    rotation = np.unwrap(state[:, rotation_dims], axis=0)
    gripper = state[:, gripper_dims]

    def within_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(values, window_frames, axis=0)
        ranges = np.ptp(windows, axis=-1)
        return np.max(ranges, axis=1) <= threshold

    stationary_windows = (
        within_threshold(position, position_threshold)
        & within_threshold(rotation, rotation_threshold)
        & within_threshold(gripper, gripper_threshold)
    )
    coverage_delta = np.zeros(len(state) + 1, dtype=np.int64)
    starts = np.flatnonzero(stationary_windows)
    np.add.at(coverage_delta, starts, 1)
    np.add.at(coverage_delta, starts + window_frames, -1)
    return np.cumsum(coverage_delta[:-1]) > 0


def stationary_keep_mask(stationary: np.ndarray, boundary_frames: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    keep = np.ones(len(stationary), dtype=bool)
    padded = np.concatenate(([False], stationary, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    ranges = [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]
    for start, end in ranges:
        drop_start = min(start + boundary_frames, end)
        drop_end = max(end - boundary_frames, drop_start)
        keep[drop_start:drop_end] = False
    return keep, ranges


def filter_dataset(args: argparse.Namespace) -> None:
    source = resolve_dataset(args.dataset)
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset not found: {source}")
    output = resolve_dataset(args.output) if args.output else source.with_name(f"{source.name}_motion_only")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    with (source / "meta" / "info.json").open() as file:
        info = json.load(file)
    episodes = load_episodes(source)
    episodes_stats = load_episodes_stats(source)
    if args.state_key not in info["features"]:
        raise KeyError(f"State key {args.state_key!r} is not present in dataset features")
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    window_frames = max(2, math.ceil(args.window_seconds * info["fps"]))

    plans = []
    total_source_frames = 0
    total_removed_frames = 0
    for episode_index in sorted(episodes):
        data_path = source / format_episode_path(info["data_path"], episode_index, info["chunks_size"])
        table = pq.read_table(data_path)
        state = np.asarray(table[args.state_key].to_pylist(), dtype=np.float64)
        default_position, default_rotation, default_gripper = default_motion_dims(state.shape[1])
        position_dims = parse_dims(args.position_dims or default_position, state.shape[1], "--position-dims")
        rotation_dims = parse_dims(args.rotation_dims or default_rotation, state.shape[1], "--rotation-dims")
        gripper_dims = parse_dims(args.gripper_dims or default_gripper, state.shape[1], "--gripper-dims")
        stationary = stationary_window_mask(
            state,
            window_frames,
            position_dims,
            rotation_dims,
            gripper_dims,
            args.position_threshold,
            args.rotation_threshold,
            args.gripper_threshold,
        )
        keep, ranges = stationary_keep_mask(stationary, args.boundary_frames)
        kept_frames = int(keep.sum())
        dropped = kept_frames < args.min_episode_frames
        if dropped:
            kept_frames = 0
        removed = len(table) - int(keep.sum())
        total_source_frames += len(table)
        total_removed_frames += len(table) if dropped else removed
        plans.append((episode_index, table, keep, kept_frames, dropped))
        durations = [f"{(end - start) / info['fps']:.2f}s" for start, end in ranges]
        duration_summary = ", ".join(durations[:5]) + (", ..." if len(durations) > 5 else "")
        print(
            f"episode {episode_index:06d}: {len(table):5d} -> {kept_frames:5d}, "
            f"stationary spans {len(ranges):3d}, removed {removed:5d}"
            + (f" ({duration_summary})" if duration_summary else "")
            + (" [drop episode]" if dropped else "")
        )

    kept_plans = [plan for plan in plans if not plan[4]]
    total_output_frames = sum(plan[3] for plan in kept_plans)
    print(
        f"summary: episodes {len(episodes)} -> {len(kept_plans)}, "
        f"frames {total_source_frames} -> {total_output_frames}, removed {total_removed_frames}"
    )
    if args.dry_run:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary_dir:
        temporary_output = Path(temporary_dir) / output.name
        copy_extra_metadata(source, temporary_output)
        output_episode_index = 0
        global_index = 0
        progress = tqdm.tqdm(
            total=len(kept_plans) * (len(video_keys) + 1),
            desc="Writing dataset",
            unit="file",
            dynamic_ncols=True,
        )
        for source_episode_index, table, keep, kept_frames, _ in kept_plans:
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
                codec = info["features"][video_key]["info"]["video.codec"]
                progress.set_postfix(episode=f"{source_episode_index:06d}", output=video_key)
                filter_video(source_video, output_video, keep, info["fps"], codec)
                progress.update()

            episode = dict(episodes[source_episode_index])
            episode["episode_index"] = output_episode_index
            episode["length"] = kept_frames
            write_episode(episode, temporary_output)
            write_episode_stats(
                output_episode_index,
                compute_episode_stats(filtered, info["features"], episodes_stats[source_episode_index]),
                temporary_output,
            )
            global_index += kept_frames
            output_episode_index += 1
        progress.close()

        output_info = dict(info)
        output_info["total_episodes"] = output_episode_index
        output_info["total_frames"] = global_index
        output_info["total_videos"] = output_episode_index * len(video_keys)
        output_info["total_chunks"] = math.ceil(output_episode_index / info["chunks_size"])
        output_info["splits"] = {"train": f"0:{output_episode_index}"}
        write_info(output_info, temporary_output)
        temporary_output.rename(output)
    print(f"Wrote filtered dataset to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="LeRobot dataset path or name under the default LeRobot root")
    parser.add_argument("--output", type=Path, help="Output path (default: INPUT_motion_only)")
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--position-threshold", type=float, default=0.002, help="Position range in a window")
    parser.add_argument("--rotation-threshold", type=float, default=0.01, help="Rotation range in radians")
    parser.add_argument("--gripper-threshold", type=float, default=0.002, help="Gripper range in a window")
    parser.add_argument("--position-dims", help="Comma-separated indices/ranges; inferred for 14-D/28-D X2Robot")
    parser.add_argument("--rotation-dims", help="Comma-separated indices/ranges; inferred for 14-D/28-D X2Robot")
    parser.add_argument("--gripper-dims", help="Comma-separated indices/ranges; inferred for 14-D/28-D X2Robot")
    parser.add_argument("--boundary-frames", type=int, default=1, help="Frames retained at each stationary boundary")
    parser.add_argument("--min-episode-frames", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Report filtering without writing output")
    args = parser.parse_args()
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    for name in ("position_threshold", "rotation_threshold", "gripper_threshold"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.boundary_frames < 1:
        parser.error("--boundary-frames must be positive")
    if args.min_episode_frames < 2:
        parser.error("--min-episode-frames must be at least 2")
    return args


if __name__ == "__main__":
    filter_dataset(parse_args())
