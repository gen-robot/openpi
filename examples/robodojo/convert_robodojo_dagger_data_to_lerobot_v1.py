#!/usr/bin/env python3
"""Merge RoboDojo PICO DAgger sessions into one LeRobot dataset.

Two things this adds over ``examples/robodojo/convert_dagger_to_lerobot.py`` in
``dev-openpi-xjy-robodojo``, whose reading and validation logic it follows:

1. **Several source sessions become one dataset.** DAgger for a task is
   collected over multiple sittings (seed0/1/2 for stack_bowls); training wants
   them as a single repo_id, and provenance has to survive the merge.

2. **Takeover metadata is written into ``meta/episodes.jsonl``.** The collector
   records, per step and per arm, whether control sits with the policy or the
   human. That is the supervision signal STEAM 2.0 is built on, and the stock
   converter drops it. Here each episode carries ``takeover_frames`` (onsets,
   matching the real-robot convention), plus ``takeover_intervals`` and
   ``takeover_arms``, which the simulator gives us and the real rig does not.

The state contract is RoboDojo's own 14-dim joint vector -- left arm joints (6),
left gripper (1), right arm joints (6), right gripper (1) -- because RoboDojo
evaluates policies in joint space. The source also carries end-effector poses
(xyz + quaternion per arm), so those ride along as ``observation.ee_pose``: the
real-robot pipeline works in end-effector space, and keeping both here lets the
same analysis run on either. The field is emitted only when present, so a
source without it still converts.

The training row at t uses state[t] as observation and state[t+1] as action, so
the final source step is not emitted -- same convention as the SFT converter,
which keeps the two mergeable under one repo_id.

Usage
-----
    export HF_LEROBOT_HOME=/mnt/resource/robodojo_dataset/lerobot
    python examples/robodojo/convert_robodojo_dagger_data_to_lerobot_v1.py \\
        --sources /mnt/resource/robodojo_dataset/raw_mp4_json/stack_bowls_seed0_xjy_20260814_dagger \\
                  /mnt/resource/robodojo_dataset/raw_mp4_json/stack_bowls_seed1_xjy_20260830_dagger \\
                  /mnt/resource/robodojo_dataset/raw_mp4_json/stack_bowls_seed2_xjy_20260831_dagger \\
        --repo-id robodojo_stack_bowls_dagger_ep175
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import numpy as np
import tqdm

FORMAT_VERSION = "robodojo-pico-dagger-v2"
CONVERTER_VERSION = "convert_robodojo_dagger_data_to_lerobot_v1"
FPS = 25
STATE_DIM = 14

STATE_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)
# Present in robodojo-pico-dagger-v2 under state; emitted when found.
EE_POSE_KEYS = ("left_ee_pose", "right_ee_pose")
EE_POSE_DIM = 14

MOTOR_NAMES = [
    *[f"left_{i}" for i in range(6)], "left_ee_0",
    *[f"right_{i}" for i in range(6)], "right_ee_0",
]
EE_POSE_NAMES = [
    *[f"left_ee_{c}" for c in ("x", "y", "z", "qx", "qy", "qz", "qw")],
    *[f"right_ee_{c}" for c in ("x", "y", "z", "qx", "qy", "qz", "qw")],
]
CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
HUMAN_SOURCES = frozenset({"human"})


class ConversionError(RuntimeError):
    """The source data does not satisfy the expected DAgger contract."""


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    session: Path
    episode_dir: Path
    manifest: dict[str, Any]
    steps: list[dict[str, Any]]

    @property
    def output_frames(self) -> int:
        return len(self.steps) - 1


# --------------------------------------------------------------------------
# reading and validation
# --------------------------------------------------------------------------
def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ConversionError(f"Expected a JSON object in {path}")
    return value


def read_steps(path: Path) -> list[dict[str, Any]]:
    steps, headers = [], 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("record_type")
            if kind == "episode_header":
                headers += 1
            elif kind == "step":
                steps.append(record)
            else:
                raise ConversionError(f"Unknown record_type {kind!r} in {path}")
    if headers != 1:
        raise ConversionError(f"Expected one episode header in {path}, got {headers}")
    return steps


def _payload(step: dict[str, Any]) -> dict[str, Any]:
    try:
        return step["observation"]["data_without_vision"]
    except KeyError as exc:
        raise ConversionError(f"Missing payload at step {step.get('step_index')}") from exc


def state_vector(step: dict[str, Any]) -> np.ndarray:
    source = _payload(step).get("state", {})
    values: list[float] = []
    for key in STATE_KEYS:
        part = source.get(key)
        if not isinstance(part, list):
            raise ConversionError(f"{key} missing or not a list at step {step.get('step_index')}")
        values.extend(part)
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (STATE_DIM,) or not np.isfinite(vector).all():
        raise ConversionError(f"Bad state at step {step.get('step_index')}: {vector.shape}")
    return vector


def ee_pose_vector(step: dict[str, Any]) -> np.ndarray | None:
    """Return the 14-dim end-effector pose, or None when the source lacks it."""
    source = _payload(step).get("state", {})
    if not all(k in source for k in EE_POSE_KEYS):
        return None
    values: list[float] = []
    for key in EE_POSE_KEYS:
        part = source[key]
        if not isinstance(part, list) or len(part) != 7:
            raise ConversionError(f"{key} must be a 7-vector at step {step.get('step_index')}")
        values.extend(part)
    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ConversionError(f"Non-finite ee pose at step {step.get('step_index')}")
    return vector


def instruction(step: dict[str, Any]) -> str:
    value = _payload(step).get("instruction")
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(f"Invalid instruction at step {step.get('step_index')}: {value!r}")
    return value.strip()


def frequency(step: dict[str, Any]) -> int:
    try:
        return int(_payload(step)["additional_info"]["frequency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"Missing frequency at step {step.get('step_index')}") from exc


def video_frame_index(step: dict[str, Any]) -> int:
    try:
        value = int(step["observation"]["video_frame_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"Bad video_frame_index at step {step.get('step_index')}") from exc
    if value < 0:
        raise ConversionError(f"Negative video_frame_index: {value}")
    return value


def validate_episode(session: Path, manifest_path: Path) -> EpisodeSource:
    manifest = read_json(manifest_path)
    episode_dir = manifest_path.parent
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ConversionError(
            f"Unsupported format in {manifest_path}: {manifest.get('format_version')!r}"
        )
    steps = read_steps(episode_dir / manifest.get("step_file", "steps.jsonl"))
    declared = manifest.get("step_count")
    if declared != len(steps) or len(steps) < 2:
        raise ConversionError(
            f"Step count mismatch in {manifest_path}: declared={declared}, actual={len(steps)}"
        )
    if [s.get("step_index") for s in steps] != list(range(len(steps))):
        raise ConversionError(f"Non-contiguous step_index in {episode_dir}")
    freqs = {frequency(s) for s in steps}
    if freqs != {FPS}:
        raise ConversionError(f"Expected frequency {FPS}, got {freqs}: {episode_dir}")

    idx = [video_frame_index(s) for s in steps]
    if any(b <= a for a, b in zip(idx, idx[1:])):
        raise ConversionError(f"video_frame_index not strictly increasing: {episode_dir}")

    counts = manifest.get("camera_frame_counts", {})
    files = manifest.get("camera_files", {})
    for src in CAMERA_ALIASES:
        rel = files.get(src)
        if not rel or not (episode_dir / rel).is_file():
            raise ConversionError(f"Missing camera video {src}: {manifest_path}")
        n = counts.get(src)
        if not isinstance(n, int) or n <= max(idx):
            raise ConversionError(f"Camera {src} does not cover frame {max(idx)}: count={n}")

    if len({instruction(s) for s in steps}) != 1:
        raise ConversionError(f"Multiple instructions in {episode_dir}")
    for s in steps:
        state_vector(s)
    return EpisodeSource(session=session, episode_dir=episode_dir,
                         manifest=manifest, steps=steps)


def is_selected(manifest: dict[str, Any], *, require_intervention: bool,
                include_failures: bool = False) -> bool:
    outcome = manifest.get("outcome", {})
    ok = (outcome.get("accepted_for_training") is True
          and outcome.get("system_valid") is True)
    if not include_failures:
        # DAgger corpora keep only successful takes; pure-rollout corpora for
        # the difficulty axis need the failures too (that is their point).
        ok = ok and outcome.get("success") is True
    if require_intervention:
        ok = ok and outcome.get("has_intervention") is True
    return ok


def discover(sources: Iterable[Path], *, require_intervention: bool,
             include_failures: bool = False,
             max_episodes: int | None) -> list[EpisodeSource]:
    found: list[EpisodeSource] = []
    for root in sources:
        root = root.resolve()
        if not root.is_dir():
            raise ConversionError(f"Source is not a directory: {root}")
        n_before = len(found)
        for manifest_path in sorted(root.glob("episodes/*/manifest.json")):
            if is_selected(read_json(manifest_path),
                           require_intervention=require_intervention,
                           include_failures=include_failures):
                found.append(validate_episode(root, manifest_path))
        print(f"  {root.name}: {len(found) - n_before} episodes selected")
    # Deterministic across runs: session order as given, then start time.
    order = {r.resolve(): i for i, r in enumerate(sources)}
    found.sort(key=lambda e: (order[e.session],
                              e.manifest.get("time", {}).get("start_time", ""),
                              e.episode_dir.name))
    if max_episodes is not None:
        found = found[:max_episodes]
    if not found:
        raise ConversionError("No episodes matched the selection rules")
    return found


# --------------------------------------------------------------------------
# takeover extraction -- the reason this converter exists
# --------------------------------------------------------------------------
def takeover_spans(steps: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Human-controlled spans, as [start, end) over emitted frames.

    A takeover shows up as ``policy -> handoff_hold -> human -> policy`` on one
    arm's ``*_arm_source``. The onset recorded here is the first ``human``
    frame, which is the moment the operator's motion starts being executed and
    the analogue of the real rig's takeover timestamp.
    """
    left, right = [], []
    for s in steps[:limit]:
        iv = s.get("intervention") or {}
        left.append(iv.get("left_arm_source") in HUMAN_SOURCES)
        right.append(iv.get("right_arm_source") in HUMAN_SOURCES)

    spans: list[dict[str, Any]] = []
    combined = [a or b for a, b in zip(left, right)]
    start = None
    for i, flag in enumerate(combined + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            arms = set()
            if any(left[start:i]):
                arms.add("left")
            if any(right[start:i]):
                arms.add("right")
            spans.append({
                "start": start,
                "end": i,
                "arm": "both" if len(arms) == 2 else next(iter(arms), "unknown"),
            })
            start = None
    return spans


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------
VIDEO_SIZE = (640, 480)          # native; no rescale


def transcode_selected(src: Path, dst: Path, keep: list[int], *, fps: int,
                       vcodec: str, crf: int, threads: int) -> int:
    """Decode once, keep the wanted frames, encode once.

    ``video_frame_index`` skips the occasional frame, so this is a selection
    rather than a trim; doing it inside one ffmpeg decode/encode pair avoids
    both per-frame Python decoding and a second re-encode by LeRobot.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    W, H = VIDEO_SIZE
    fsz = W * H * 3
    keep_set = set(keep)
    last = max(keep) if keep else -1

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-threads", str(threads), "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", vcodec, "-pix_fmt", "yuv420p", "-r", str(fps),
         "-g", "2", "-crf", str(crf), "-threads", str(threads), str(dst)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env={**os.environ, "SVT_LOG": "0"})

    written = idx = 0
    try:
        while idx <= last:
            buf = dec.stdout.read(fsz)
            if len(buf) < fsz:
                break
            if idx in keep_set:
                enc.stdin.write(buf)
                written += 1
            idx += 1
    finally:
        for stream in (dec.stdout, enc.stdin):
            try:
                stream.close()
            except OSError:
                pass
        dec.terminate()
        dec.wait()
        rc = enc.wait()
    if rc != 0:
        raise ConversionError(f"encode failed ({rc}) for {src}")
    if written != len(keep):
        raise ConversionError(f"{src}: wrote {written} frames, planned {len(keep)}")
    return written


def dummy_video_stats(n: int) -> dict:
    """Stand-in per-channel stats, so metadata never reads the pixels back."""
    return {
        "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),
        "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
        "mean": np.array([[[0.4]], [[0.4]], [[0.4]]]),
        "std": np.array([[[0.25]], [[0.25]], [[0.25]]]),
        "count": np.array([n]),
    }


def episode_stats_without_video(buffer: dict, features: dict, n: int) -> dict:
    from lerobot.common.datasets.compute_stats import get_feature_stats

    stats = {}
    for key, data in buffer.items():
        if key not in features:
            continue
        ft = features[key]["dtype"]
        if ft == "string":
            continue
        if ft in ("image", "video"):
            stats[key] = dummy_video_stats(n)
        else:
            stats[key] = get_feature_stats(data, axis=0, keepdims=data.ndim == 1)
    return stats


class NoVideoIOLeRobotDataset(LeRobotDataset):
    """LeRobotDataset whose save_episode writes no images and no video."""

    def _save_image(self, image, fpath: Path) -> None:  # noqa: D102
        return

    def save_episode(self, episode_data: dict | None = None) -> None:  # noqa: D102
        from lerobot.common.datasets.lerobot_dataset import (
            aggregate_stats, validate_episode_buffer, write_episode,
            write_episode_stats, write_info,
        )

        buffer = episode_data or self.episode_buffer
        validate_episode_buffer(buffer, self.meta.total_episodes, self.features)
        length = buffer.pop("size")
        tasks = buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = buffer["episode_index"]

        buffer["index"] = np.arange(self.meta.total_frames,
                                    self.meta.total_frames + length)
        buffer["episode_index"] = np.full((length,), episode_index)
        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)
        buffer["task_index"] = np.array([self.meta.get_task_index(t) for t in tasks])

        for key, ft in self.features.items():
            if key in ("index", "episode_index", "task_index"):
                continue
            if ft["dtype"] in ("image", "video"):
                continue
            buffer[key] = np.stack(buffer[key])

        self._wait_image_writer()
        self._save_episode_table(buffer, episode_index)
        stats = episode_stats_without_video(buffer, self.features, length)

        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += length
        if self.meta.get_episode_chunk(episode_index) >= self.meta.total_chunks:
            self.meta.info["total_chunks"] += 1
        self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}
        self.meta.info["total_videos"] += len(self.meta.video_keys)
        write_info(self.meta.info, self.meta.root)

        record = {"episode_index": episode_index, "tasks": episode_tasks,
                  "length": length}
        self.meta.episodes[episode_index] = record
        write_episode(record, self.meta.root)
        self.meta.episodes_stats[episode_index] = stats
        self.meta.stats = (aggregate_stats([self.meta.stats, stats])
                           if self.meta.stats else stats)
        write_episode_stats(episode_index, stats, self.meta.root)

        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()

    def finalize_video_info(self) -> None:
        """Read the transcoded files back once, to fill in real video info."""
        from lerobot.common.datasets.lerobot_dataset import write_info

        self.meta.update_video_info()
        write_info(self.meta.info, self.meta.root)

    @classmethod
    def create(cls, **kwargs) -> "NoVideoIOLeRobotDataset":  # noqa: D102
        parent = LeRobotDataset.create(**kwargs)
        obj = cls.__new__(cls)
        obj.__dict__.update(parent.__dict__)
        return obj


def create_dataset(root: Path, repo_id: str, with_ee: bool) -> NoVideoIOLeRobotDataset:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,),
                              "names": [MOTOR_NAMES]},
        "action": {"dtype": "float32", "shape": (STATE_DIM,),
                   "names": [MOTOR_NAMES]},
    }
    if with_ee:
        features["observation.ee_pose"] = {
            "dtype": "float32", "shape": (EE_POSE_DIM,), "names": [EE_POSE_NAMES],
        }
    for out in CAMERA_ALIASES.values():
        features[f"observation.images.{out}"] = {
            "dtype": "video", "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    return NoVideoIOLeRobotDataset.create(
        repo_id=repo_id, root=root, fps=FPS, robot_type="arx_x5",
        features=features, use_videos=True,
        image_writer_processes=0, image_writer_threads=1,
    )


def transcode_all(episodes: list[EpisodeSource], root: Path, *, workers: int,
                  vcodec: str, crf: int, threads: int) -> None:
    """Phase 1: every episode x camera video, in parallel."""
    jobs = []
    for i, ep in enumerate(episodes):
        keep = [video_frame_index(s) for s in ep.steps[:ep.output_frames]]
        for src_name, out_name in CAMERA_ALIASES.items():
            jobs.append((
                ep.episode_dir / ep.manifest["camera_files"][src_name],
                root / "videos" / "chunk-000" / f"observation.images.{out_name}"
                / f"episode_{i:06d}.mp4",
                keep,
            ))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(transcode_selected, src, dst, keep, fps=FPS,
                        vcodec=vcodec, crf=crf, threads=threads): src
            for src, dst, keep in jobs
        }
        with tqdm.tqdm(total=len(jobs), desc="transcoding") as bar:
            for fut in as_completed(futures):
                fut.result()
                bar.update(1)
    print(f"  transcoded {len(jobs)} videos in {time.time() - t0:.0f}s")


def write_frames(dataset: NoVideoIOLeRobotDataset, ep: EpisodeSource,
                 with_ee: bool, blank: np.ndarray) -> None:
    """Phase 2: rows and metadata only; the pixels are already on disk."""
    for i in range(ep.output_frames):
        step, nxt = ep.steps[i], ep.steps[i + 1]
        frame: dict[str, Any] = {
            "observation.state": state_vector(step),
            "action": state_vector(nxt),
            "task": instruction(step),
        }
        if with_ee:
            pose = ee_pose_vector(step)
            if pose is None:
                raise ConversionError(f"ee pose vanished mid-episode: {ep.episode_dir}")
            frame["observation.ee_pose"] = pose
        for out in CAMERA_ALIASES.values():
            frame[f"observation.images.{out}"] = blank
        dataset.add_frame(frame)
    dataset.save_episode()


def inject_metadata(root: Path, episodes: list[EpisodeSource],
                    extras: list[dict[str, Any]]) -> None:
    """Merge takeover and provenance keys into ``meta/episodes.jsonl``.

    LeRobot writes that file itself, so the custom keys go in afterwards. The
    write is atomic (tmp + replace) because training jobs may be reading it.
    """
    meta = root / "meta" / "episodes.jsonl"
    records = [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]
    if len(records) != len(extras):
        raise ConversionError(
            f"meta has {len(records)} episodes but {len(extras)} were converted"
        )
    tmp = meta.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for record, extra in zip(records, extras):
            record.update(extra)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, meta)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", type=Path, nargs="+", required=True,
                   help="One or more DAgger session roots; merged in this order.")
    p.add_argument("--repo-id", required=True,
                   help="Output dataset name, e.g. robodojo_stack_bowls_dagger_ep175")
    p.add_argument("--output-home", type=Path, default=None,
                   help="Defaults to $HF_LEROBOT_HOME.")
    p.add_argument("--num-workers", type=int, default=12,
                   help="Parallel ffmpeg transcodes.")
    p.add_argument("--ffmpeg-threads", type=int, default=4)
    p.add_argument("--vcodec", default="libx264")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--include-failures", action="store_true",
                   help="Also keep episodes whose outcome.success is false "
                        "(pure-rollout corpora for the difficulty axis).")
    p.add_argument("--allow-no-intervention", action="store_true",
                   help="Keep accepted episodes that contain no takeover.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--inspect-only", action="store_true",
                   help="Report the inventory and takeover statistics, write nothing.")
    args = p.parse_args()

    print(f"discovering episodes in {len(args.sources)} session(s):")
    episodes = discover(args.sources,
                        require_intervention=not args.allow_no_intervention,
                        include_failures=args.include_failures,
                        max_episodes=args.max_episodes)

    with_ee = ee_pose_vector(episodes[0].steps[0]) is not None
    spans = [takeover_spans(e.steps, e.output_frames) for e in episodes]
    n_tk = sum(len(s) for s in spans)
    human = sum(sp["end"] - sp["start"] for s in spans for sp in s)
    frames = sum(e.output_frames for e in episodes)
    print(f"\n{len(episodes)} episodes, {frames} frames "
          f"({frames / FPS / 60:.1f} min at {FPS} fps)")
    print(f"takeovers: {n_tk} total, {n_tk / len(episodes):.2f} per episode, "
          f"{sum(1 for s in spans if not s)} episodes with none")
    print(f"human-controlled frames: {human} ({100 * human / frames:.1f}%)")
    print(f"end-effector poses in source: {'yes' if with_ee else 'no (joint only)'}")
    if args.inspect_only:
        return

    home = args.output_home or HF_LEROBOT_HOME
    target = Path(home).resolve() / args.repo_id
    if target.exists():
        if not args.overwrite:
            raise ConversionError(f"{target} exists; pass --overwrite to replace it")
        shutil.rmtree(target)
    print(f"\nwriting {target}")

    # The dataset root has to be created by LeRobot itself (it insists the
    # directory does not already exist), so it comes before the transcode that
    # writes into its videos/ tree.
    dataset = create_dataset(target, args.repo_id, with_ee)

    print("\nphase 1/2 — transcoding videos")
    transcode_all(episodes, target, workers=args.num_workers,
                  vcodec=args.vcodec, crf=args.crf, threads=args.ffmpeg_threads)

    print("phase 2/2 — writing rows and metadata")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    extras: list[dict[str, Any]] = []
    for ep, sp in zip(tqdm.tqdm(episodes, desc="rows"), spans):
        write_frames(dataset, ep, with_ee, blank)
        extras.append({
            "takeover_frames": [s["start"] for s in sp],
            "takeover_seconds": [round(s["start"] / FPS, 3) for s in sp],
            "takeover_intervals": [[s["start"], s["end"]] for s in sp],
            "takeover_arms": [s["arm"] for s in sp],
            "human_frame_count": sum(s["end"] - s["start"] for s in sp),
            "source_session": ep.session.name,
            "source_episode": ep.episode_dir.name,
            # Outcome ride-along: a mixed success/failure rollout corpus is
            # unusable without per-episode labels, and the stage-difficulty
            # axis needs exactly this bit.
            "success": bool(ep.manifest.get("outcome", {}).get("success")),
            "score": ep.manifest.get("outcome", {}).get("score"),
            "termination_reason": ep.manifest.get("outcome", {}).get("termination_reason"),
        })
    dataset.finalize_video_info()
    inject_metadata(target, episodes, extras)

    (target / "meta" / "conversion.json").write_text(json.dumps({
        "converter": CONVERTER_VERSION,
        "converted_at": datetime.now(UTC).isoformat(),
        "source_format": FORMAT_VERSION,
        "sources": [str(s.resolve()) for s in args.sources],
        "episodes": len(episodes),
        "frames": frames,
        "takeovers": n_tk,
        "human_frames": human,
        "ee_poses_present": with_ee,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"done: {len(episodes)} episodes, {n_tk} takeovers -> {target}")


if __name__ == "__main__":
    main()
