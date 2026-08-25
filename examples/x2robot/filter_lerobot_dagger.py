"""Remove pre-takeover rollout and stationary spans from an X2Robot LeRobot dataset.

The converted dataset does not contain an intervention flag, so takeover boundaries
must come from either a JSON mapping or a previously rollout-trimmed dataset.

Examples:
  python examples/x2robot/filter_lerobot_dagger.py bagging_4sku_ny_0501 \
    --takeover-dataset bagging_4sku_ny_0501_modified --dry-run

  python examples/x2robot/filter_lerobot_dagger.py bagging_4sku_ny_0501 \
    --takeover-frames takeover_frames.json --output bagging_4sku_ny_0501_filtered

The takeover JSON must map episode indices to the first retained source frame, for
example: {"1": 5747, "2": 5722}. Missing episodes start at frame 0.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from lerobot.common.datasets.compute_stats import get_feature_stats
from lerobot.common.datasets.utils import load_episodes
from lerobot.common.datasets.utils import load_episodes_stats
from lerobot.common.datasets.utils import write_episode
from lerobot.common.datasets.utils import write_episode_stats
from lerobot.common.datasets.utils import write_info
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

DEFAULT_DATA_ROOT = Path("/mnt/public/gaojiahuan/huggingface/lerobot")


def resolve_dataset(path: Path) -> Path:
    return path if path.is_absolute() else DEFAULT_DATA_ROOT / path


def format_episode_path(pattern: str, episode_index: int, chunks_size: int, video_key: str | None = None) -> Path:
    return Path(
        pattern.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            video_key=video_key,
        )
    )


def parse_motion_dims(spec: str, state_dim: int) -> slice:
    try:
        start_text, end_text = spec.split(":", maxsplit=1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else state_dim
    except ValueError as error:
        raise ValueError(f"Invalid --motion-dims {spec!r}; expected START:END") from error
    if not 0 <= start < end <= state_dim:
        raise ValueError(f"--motion-dims {spec!r} is outside state dimension {state_dim}")
    return slice(start, end)


def load_takeover_frames(path: Path | None) -> dict[int, int]:
    if path is None:
        return {}
    with path.open() as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("--takeover-frames must contain a JSON object")
    frames = {int(episode_index): int(frame_index) for episode_index, frame_index in data.items()}
    if any(episode_index < 0 or frame_index < 0 for episode_index, frame_index in frames.items()):
        raise ValueError("Takeover episode and frame indices must be non-negative")
    return frames


def table_vectors(table: pa.Table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist())


def infer_takeover_frame(source: pa.Table, reference: pa.Table, episode_index: int) -> int:
    source_state = table_vectors(source, "state")
    reference_state = table_vectors(reference, "state")
    if not len(reference_state):
        raise ValueError(f"Reference episode {episode_index} is empty")

    candidates = np.flatnonzero(np.all(source_state == reference_state[0], axis=1))
    match_length = min(16, len(reference_state))
    while True:
        matches = [
            int(candidate)
            for candidate in candidates
            if candidate + match_length <= len(source_state)
            and np.array_equal(source_state[candidate : candidate + match_length], reference_state[:match_length])
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches or match_length == len(reference_state):
            raise ValueError(
                f"Could not uniquely match takeover reference episode {episode_index}: "
                f"found {len(matches)} matches using {match_length} frames"
            )
        candidates = np.asarray(matches)
        match_length = min(match_length * 2, len(reference_state))


def stationary_keep_mask(
    state: np.ndarray,
    motion_dims: slice,
    threshold: float,
    min_stationary_frames: int,
) -> np.ndarray:
    keep = np.ones(len(state), dtype=bool)
    if len(state) < min_stationary_frames:
        return keep

    stationary_edges = np.max(np.abs(np.diff(state[:, motion_dims], axis=0)), axis=1) <= threshold
    padded = np.concatenate(([False], stationary_edges, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    for start, end in zip(starts, ends, strict=True):
        frame_end = end + 1
        if frame_end - start >= min_stationary_frames:
            keep[start:frame_end] = False
    return keep


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index == -1:
        return table
    return table.set_column(column_index, name, pa.array(values, type=table.schema.field(name).type))


def reset_indices(table: pa.Table, episode_index: int, global_index: int, fps: int) -> pa.Table:
    length = len(table)
    table = replace_column(table, "frame_index", np.arange(length, dtype=np.int64))
    table = replace_column(table, "episode_index", np.full(length, episode_index, dtype=np.int64))
    table = replace_column(table, "index", np.arange(global_index, global_index + length, dtype=np.int64))
    return replace_column(table, "timestamp", np.arange(length, dtype=np.float32) / fps)


def contiguous_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def video_encoder(codec: str) -> str:
    encoders = {
        "av1": "libsvtav1",
        "h264": "libx264",
        "hevc": "libx265",
        "vp9": "libvpx-vp9",
    }
    try:
        return encoders[codec]
    except KeyError as error:
        raise ValueError(f"Unsupported source video codec {codec!r}") from error


def filter_video(source: Path, destination: Path, mask: np.ndarray, fps: int, codec: str) -> None:
    ranges = contiguous_ranges(mask)
    if not ranges:
        raise ValueError(f"No frames selected from {source}")
    select = "+".join(f"between(n\\,{start}\\,{end - 1})" for start, end in ranges)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"select={select},setpts=N/{fps}/TB",
            "-an",
            "-c:v",
            video_encoder(codec),
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-g",
            "2",
            "-crf",
            "23",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed for {source}: {result.stderr.strip()}")

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_frames = int(probe.stdout.strip())
    expected_frames = int(mask.sum())
    if actual_frames != expected_frames:
        raise RuntimeError(f"Video frame count mismatch for {destination}: {actual_frames} != {expected_frames}")


def feature_array(table: pa.Table, key: str) -> np.ndarray:
    column_type = table.schema.field(key).type
    if pa.types.is_fixed_size_list(column_type) or pa.types.is_list(column_type):
        return np.asarray(table[key].to_pylist())
    return table[key].to_numpy()


def compute_episode_stats(
    table: pa.Table,
    features: dict,
    source_stats: dict,
) -> dict:
    stats = {}
    for key, feature in features.items():
        if feature["dtype"] == "string":
            continue
        if feature["dtype"] in ("image", "video"):
            stats[key] = {name: value.copy() for name, value in source_stats[key].items()}
            stats[key]["count"] = np.array([len(table)])
            continue
        if key not in table.column_names:
            continue
        values = feature_array(table, key)
        stats[key] = get_feature_stats(values, axis=0, keepdims=values.ndim == 1)
    return stats


def copy_extra_metadata(source: Path, destination: Path) -> None:
    skipped = {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}
    for item in (source / "meta").iterdir():
        if item.name in skipped:
            continue
        target = destination / "meta" / item.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def load_reference_table(root: Path, info: dict, episode_index: int) -> pa.Table:
    path = root / format_episode_path(info["data_path"], episode_index, info["chunks_size"])
    return pq.read_table(path)


def filter_dataset(args: argparse.Namespace) -> None:
    source = resolve_dataset(args.dataset)
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset not found: {source}")
    output = resolve_dataset(args.output) if args.output else source.with_name(f"{source.name}_filtered")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    with (source / "meta" / "info.json").open() as file:
        info = json.load(file)
    episodes = load_episodes(source)
    episodes_stats = load_episodes_stats(source)
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    takeover_frames = load_takeover_frames(args.takeover_frames)

    reference = resolve_dataset(args.takeover_dataset) if args.takeover_dataset else None
    reference_info = None
    if reference:
        with (reference / "meta" / "info.json").open() as file:
            reference_info = json.load(file)

    plans = []
    total_source_frames = 0
    total_rollout_frames = 0
    total_stationary_frames = 0
    for episode_index in sorted(episodes):
        data_path = source / format_episode_path(info["data_path"], episode_index, info["chunks_size"])
        table = pq.read_table(data_path)
        total_source_frames += len(table)
        if reference:
            takeover_frame = infer_takeover_frame(
                table,
                load_reference_table(reference, reference_info, episode_index),
                episode_index,
            )
        else:
            takeover_frame = takeover_frames.get(episode_index, 0)
        if takeover_frame >= len(table):
            raise ValueError(f"Episode {episode_index} takeover frame {takeover_frame} is outside length {len(table)}")

        state = table_vectors(table, args.state_key)
        motion_dims = parse_motion_dims(args.motion_dims, state.shape[1])
        suffix_mask = stationary_keep_mask(
            state[takeover_frame:],
            motion_dims,
            args.stationary_threshold,
            args.min_stationary_frames,
        )
        mask = np.zeros(len(table), dtype=bool)
        mask[takeover_frame:] = suffix_mask
        kept_frames = int(mask.sum())
        dropped = kept_frames < args.min_episode_frames
        if dropped:
            kept_frames = 0
        total_rollout_frames += takeover_frame
        total_stationary_frames += len(table) - takeover_frame - int(suffix_mask.sum())
        plans.append((episode_index, table, mask, kept_frames, dropped))
        print(
            f"episode {episode_index:06d}: {len(table):5d} -> {kept_frames:5d} "
            f"(rollout {takeover_frame:5d}, stationary {len(table) - takeover_frame - int(suffix_mask.sum()):4d})"
            + (" [drop]" if dropped else "")
        )

    kept_plans = [plan for plan in plans if not plan[4]]
    total_output_frames = sum(plan[3] for plan in kept_plans)
    print(
        f"summary: episodes {len(episodes)} -> {len(kept_plans)}, frames {total_source_frames} -> {total_output_frames}, "
        f"rollout removed {total_rollout_frames}, stationary removed {total_stationary_frames}"
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
        for source_episode_index, table, mask, kept_frames, _ in kept_plans:
            filtered = table.filter(pa.array(mask))
            filtered = reset_indices(filtered, output_episode_index, global_index, info["fps"])
            output_data_path = temporary_output / format_episode_path(
                info["data_path"], output_episode_index, info["chunks_size"]
            )
            output_data_path.parent.mkdir(parents=True, exist_ok=True)
            progress.set_postfix(episode=f"{source_episode_index:06d}", output="parquet")
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
                filter_video(source_video, output_video, mask, info["fps"], codec)
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
    parser.add_argument("dataset", type=Path, help=f"Dataset path or name under {DEFAULT_DATA_ROOT}")
    parser.add_argument("--output", type=Path, help="Output path or name (default: INPUT_filtered)")
    takeover = parser.add_mutually_exclusive_group()
    takeover.add_argument("--takeover-frames", type=Path, help="JSON mapping episode index to takeover frame")
    takeover.add_argument(
        "--takeover-dataset",
        type=Path,
        help="Previously rollout-trimmed dataset used to infer each takeover frame",
    )
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--motion-dims", default="0:14", help="State dimensions used for motion detection")
    parser.add_argument("--stationary-threshold", type=float, default=1e-3)
    parser.add_argument("--min-stationary-frames", type=int, default=20)
    parser.add_argument("--min-episode-frames", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Report filtering without writing output")
    args = parser.parse_args()
    if args.stationary_threshold < 0:
        parser.error("--stationary-threshold must be non-negative")
    if args.min_stationary_frames < 2:
        parser.error("--min-stationary-frames must be at least 2")
    if args.min_episode_frames < 1:
        parser.error("--min-episode-frames must be positive")
    return args


if __name__ == "__main__":
    filter_dataset(parse_args())
