"""
Raw mp4_json -> LeRobot, with static frames dropped in the same pass.

Replaces the two-step pipeline

    filter_x2robot_data_v2.py   raw -> raw_filter_static   (decode + encode)
    convert_..._v5.py           raw_filter_static -> LeRobot (decode + encode)

with one pass that decodes and encodes each video exactly once. Measured on a
640x480 / 8338-frame take: 23.5s -> 10.9s per video, and no intermediate copy of
the dataset on disk.

Why not an ffmpeg select expression: the keep set is a decimation, not a few
cuts -- about 1150 runs of ~3.5 frames for a 6600-frame episode. Expressed as
between() terms that is a 26 KB filter string evaluated per frame. Piping raw
frames and picking them in Python is far cheaper, and ffmpeg still does the
decode and the encode with all its threads. Scaling happens in the decoder, so
only 320x240 crosses the pipe.

The frame conventions match the old pipeline exactly: the video gets len(keep)
frames, the parquet gets len(keep) - 1 rows, and action[i] is the state of the
next KEPT frame (the old filter rewrote the json before the converter shifted).

Usage:
    python3 convert_x2robot_data_to_lerobot_qiuyi_v1.py \
        --args.repo_name make_whiskey_sour_xpc_0811_sft_perfect \
        --args.raw_paths /mnt/resource/.../beijing_xpc_20260811_sft_perfect

    # keep every frame (plain conversion)
    ... --args.no-filter-static
"""
import dataclasses
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm
import tyro

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "mp4_json_edit"))
from filter_x2robot_data_v2 import (  # noqa: E402
    detect_intervention_start,
    filter_stationary_frames,
    get_state_arrays,
)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME  # noqa: E402

os.environ["SVT_LOG"] = "0"

FILE_CAMERA_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4",
}
STATE_KEYS = [
    "follow_left_position", "follow_left_rotation", "follow_left_gripper",
    "follow_right_position", "follow_right_rotation", "follow_right_gripper",
    "master_left_position", "master_left_rotation", "master_left_gripper",
    "master_right_position", "master_right_rotation", "master_right_gripper",
]
ACTION_KEYS = list(STATE_KEYS)


@dataclass
class ConvertArgs:
    raw_paths: list[str] = dataclasses.field(default_factory=list)
    repo_name: str = "x2robot_dataset"
    task: str = ("Pick up the goods on your left hand and place them into the "
                 "bag on your right hand.")
    push_to_hub: bool = False
    debug: bool = False
    debug_episodes: int = 3
    low_resolution: bool = True
    num_workers: int = 10
    ffmpeg_threads: int = 4
    """Threads per ffmpeg process. Left at 0 (auto) each one grabs the whole box,
    which thrashes once num_workers videos run at once. 320x240 does not scale
    past a handful anyway."""
    filter_static: bool = True
    """Drop frames where neither arm moved -- the old filter_x2robot_data_v2 step."""
    takeover_detect: bool = False
    """Trim everything before the first intervention (the old script's default ON;
    every call in this project passed --no_takeover_detect, so this defaults OFF)."""
    crop_before_grasp: bool = False
    """Drop everything before the gripper first closes on an object.

    For takes recorded at a base position that has since been corrected, the
    reach up to that first grasp teaches coordinates the robot can no longer
    reproduce, while everything after it happens wherever the previous step left
    things and is still valid. Cropping keeps the second half."""
    crop_arm: str = "right"
    """Which gripper marks the cut: 'right' or 'left'."""


def get_dim_from_keys(keys: list[str]) -> int:
    dim = 0
    for key in keys:
        if "gripper" in key:
            dim += 1
        elif "position" in key or "rotation" in key:
            dim += 3
        elif "joint" in key:
            dim += 7
        else:
            raise ValueError(f"Unknown key type: {key}")
    return dim


def find_episodes(raw_paths: list[str]) -> list[str]:
    out = []
    for raw_path in raw_paths:
        for d in glob.glob(f"{raw_path}/*"):
            if os.path.isdir(d) and glob.glob(f"{d}/*.mp4"):
                out.append(d)
    return sorted(out)


def _json_path(episode_path: str) -> str:
    return os.path.join(episode_path, f"{os.path.basename(episode_path)}.json")


def load_raw_frames(episode_path: str) -> list:
    with open(_json_path(episode_path)) as f:
        return json.load(f)["data"]


def arrays_from_frames(frames: list) -> tuple[np.ndarray, np.ndarray]:
    all_keys = set(STATE_KEYS) | set(ACTION_KEYS)
    cols = {k: [] for k in all_keys}
    for fr in frames:
        for k in all_keys:
            cols[k].append(fr[k])
    arrs = {}
    for k, v in cols.items():
        a = np.array(v, dtype=np.float32)
        arrs[k] = a.reshape(-1, 1) if "gripper" in k else a
    state = np.concatenate([arrs[k] for k in STATE_KEYS], axis=1)
    action = np.concatenate([arrs[k] for k in ACTION_KEYS], axis=1)
    return state, action


def probe_frames(path: str) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


def find_first_grasp(frames: list, arm: str = "right") -> int | None:
    """Raw index of the frame where the gripper finishes closing on an object.

    The channel rests near 0, opens to ~4.4, holds, then falls and settles near
    1.0-1.3 -- it never returns to 0 because the object is between the fingers,
    and that settle value differs per machine. So the falling EDGE is detected,
    not a fixed low threshold: first frame above 3.5, then the first frame that
    has dropped more than 1.0 below the open plateau, then the first frame where
    the channel goes flat.
    """
    key = f"follow_{arm}_gripper"
    g = np.array([fr[key] for fr in frames], dtype=np.float64)
    o = np.where(g > 3.5)[0]
    if len(o) == 0:
        return None
    t0 = int(o[0])
    plateau = float(np.median(g[t0:t0 + 15]))
    f = np.where(g[t0 + 5:] < plateau - 1.0)[0]
    if len(f) == 0:
        return None
    t_edge = int(t0 + 5 + f[0])
    d = np.diff(g[t_edge:t_edge + 60])
    s = next((i for i in range(max(len(d) - 5, 0))
              if np.all(np.abs(d[i:i + 5]) < 0.03)), None)
    t_grasp = int(t_edge + s) if s is not None else int(
        t_edge + np.argmin(g[t_edge:t_edge + 30]))
    return min(t_grasp, len(g) - 1)


def plan_episode(episode_path: str, filter_static: bool, takeover_detect: bool,
                 crop_before_grasp: bool = False, crop_arm: str = "right"):
    """Decide which frames survive, clamped to what the videos actually hold.

    The json and the mp4s can disagree by a frame or two. The old pipeline let
    that slide, which silently produced a parquet longer than its video; here
    the keep list is clamped to the shortest camera and the shortfall reported.
    """
    frames = load_raw_frames(episode_path)
    n_json = len(frames)
    if filter_static:
        arrays = get_state_arrays(frames)
        start = detect_intervention_start(arrays) if takeover_detect else 0
        keep = filter_stationary_frames(arrays, start)
    else:
        keep = list(range(n_json))
    vid_counts = {}
    for cam, fn in FILE_CAMERA_MAPPING.items():
        p = os.path.join(episode_path, fn)
        vid_counts[cam] = probe_frames(p) if os.path.exists(p) else -1
    usable = min([c for c in vid_counts.values() if c > 0], default=0)
    clamped = [i for i in keep if i < usable]

    grasp = None
    cropped_off = 0
    if crop_before_grasp:
        grasp = find_first_grasp(frames, crop_arm)
        if grasp is None:
            # no grasp means no defensible cut point -- caller drops the episode
            return {"path": episode_path, "n_json": n_json, "keep": [],
                    "dropped_by_clamp": 0, "video_frames": vid_counts,
                    "usable": usable, "grasp": None, "cropped_off": 0,
                    "no_grasp": True}
        before = len(clamped)
        clamped = [i for i in clamped if i >= grasp]
        cropped_off = before - len(clamped)

    return {
        "path": episode_path, "n_json": n_json, "keep": clamped,
        "dropped_by_clamp": len(keep) - len(clamped) - cropped_off,
        "video_frames": vid_counts, "usable": usable,
        "grasp": grasp, "cropped_off": cropped_off, "no_grasp": False,
    }


def transcode_selected(video_path: str, output_path: Path, keep: list[int],
                       target_size: tuple[int, int], fps: int,
                       vcodec: str = "libx264", crf: int = 23,
                       threads: int = 4) -> int:
    """Decode once, keep the wanted frames, encode once. Returns frames written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    W, H = target_size
    fsz = W * H * 3
    keep_set = set(keep)
    last = max(keep) if keep else -1

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-threads", str(threads), "-i", video_path,
         "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", vcodec, "-pix_fmt", "yuv420p", "-r", str(fps),
         "-g", "2", "-crf", str(crf), "-threads", str(threads), str(output_path)],
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
        try:
            dec.stdout.close()
        except OSError:
            pass
        dec.terminate()
        dec.wait()
        try:
            enc.stdin.close()
        except OSError:
            pass
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError(f"encode failed ({rc}) for {video_path}")
    if written != len(keep):
        raise RuntimeError(
            f"{video_path}: wrote {written} frames but planned {len(keep)}")
    return written


def transcode_one(task):
    ep_path, ep_idx, cam, filename, out_root, keep, target_size, fps, threads = task
    n = transcode_selected(os.path.join(ep_path, filename),
                           out_root / "videos" / "chunk-000" / cam /
                           f"episode_{ep_idx:06d}.mp4",
                           keep, target_size, fps, threads=threads)
    return ep_idx, cam, n


def get_dummy_video_stats(num_frames: int) -> dict:
    """生成视频特征的伪统计值（与真实值非常接近）"""
    return {
        "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),  # RGB channels
        "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
        "mean": np.array([[[0.4]], [[0.4]], [[0.4]]]),  # typical mean
        "std": np.array([[[0.25]], [[0.25]], [[0.25]]]),  # typical std
        "count": np.array([num_frames])
    }

def compute_episode_stats_with_dummy_video(
    episode_buffer: dict, 
    features: dict,
    video_frame_count: int
) -> dict:
    """计算episode统计，对视频特征使用伪值"""
    from lerobot.common.datasets.compute_stats import get_feature_stats
    
    ep_stats = {}
    for key, data in episode_buffer.items():
        if key not in features:
            continue
            
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            # 使用伪统计值，不读取图片
            ep_stats[key] = get_dummy_video_stats(video_frame_count)
        else:
            # 对于其他特征，正常计算统计
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1
            ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
    
    return ep_stats

class NoVideoIOLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that skips all video/image I/O in save_episode."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_all_media = False
        self._video_frame_count = 0  # 设置每个episode的视频帧数
    
    def _save_image(self, image, fpath: Path) -> None:
        """Skip all image saving."""
        if self._skip_all_media:
            return
        super()._save_image(image, fpath)
    
    def save_episode(self, episode_data: dict | None = None) -> None:
        """Override save_episode to skip all video/image I/O and use dummy stats."""
        if not self._skip_all_media:
            super().save_episode(episode_data)
            return
            
        if not episode_data:
            episode_buffer = self.episode_buffer
        
        from lerobot.common.datasets.lerobot_dataset import (
            validate_episode_buffer,
            get_episode_data_index,
            check_timestamps_sync,
            write_info,
            write_episode,
            write_episode_stats,
            aggregate_stats,
        )

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        
        # 使用自定义的统计计算函数（对视频使用伪值）
        ep_stats = compute_episode_stats_with_dummy_video(
            episode_buffer, self.features, self._video_frame_count
        )

        # Save episode metadata
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
    
    def finalize_video_info(self) -> None:
        """Update video info after all videos are transcoded."""
        self.meta.update_video_info()
        from lerobot.common.datasets.lerobot_dataset import write_info
        write_info(self.meta.info, self.meta.root)
    
    @classmethod
    def create(cls, **kwargs) -> "NoVideoIOLeRobotDataset":
        """Create a NoVideoIOLeRobotDataset."""
        parent_obj = LeRobotDataset.create(**kwargs)
        obj = cls.__new__(cls)
        obj.__dict__.update(parent_obj.__dict__)
        obj._skip_all_media = False
        obj._video_frame_count = 0
        return obj


def main(args: ConvertArgs):
    t_start = time.time()
    if not args.raw_paths:
        raise SystemExit("--args.raw_paths is required")

    episode_paths = find_episodes(args.raw_paths)
    print(f"Found {len(episode_paths)} episodes")
    if args.debug:
        episode_paths = episode_paths[:args.debug_episodes]

    target_size = (320, 240) if args.low_resolution else (640, 480)
    shape = (target_size[1], target_size[0], 3)
    fps = 20

    # ---- phase 0: decide the keep set before touching a single video --------
    print(f"\nPhase 0: planning frames "
          f"(filter_static={args.filter_static}, takeover_detect={args.takeover_detect}, "
          f"crop_before_grasp={args.crop_before_grasp}"
          f"{' arm=' + args.crop_arm if args.crop_before_grasp else ''})")
    plans = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(plan_episode, p, args.filter_static,
                          args.takeover_detect, args.crop_before_grasp,
                          args.crop_arm): p
                for p in episode_paths}
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc="Planning"):
            try:
                plans.append(fut.result())
            except Exception as e:
                print(f"  [skip] {futs[fut]}: {e}")
    no_grasp = [p for p in plans if p.get("no_grasp")]
    plans = [p for p in plans if len(p["keep"]) >= 2]
    if args.crop_before_grasp:
        cut = sum(p.get("cropped_off", 0) for p in plans)
        gr = [p["grasp"] for p in plans if p.get("grasp") is not None]
        print(f"  crop: dropped {cut} frames before the first {args.crop_arm} grasp "
              f"(grasp at raw frame {np.mean(gr):.0f} on average)" if gr else
              "  crop: no grasp found in any episode")
        if no_grasp:
            print(f"  {len(no_grasp)} episode(s) had no detectable grasp and were "
                  f"dropped entirely:")
            for p in no_grasp[:5]:
                print(f"    {os.path.basename(p['path'])}")
    plans.sort(key=lambda p: p["path"])
    if not plans:
        raise SystemExit("no usable episodes")

    tot_json = sum(p["n_json"] for p in plans)
    tot_keep = sum(len(p["keep"]) for p in plans)
    clamped = [p for p in plans if p["dropped_by_clamp"] > 0]
    print(f"  {len(plans)} episodes, {tot_json} raw frames -> {tot_keep} kept "
          f"({100 * tot_keep / max(tot_json, 1):.1f}%)")
    if clamped:
        print(f"  {len(clamped)} episode(s) had fewer video frames than json rows; "
              f"keep list clamped:")
        for p in clamped[:5]:
            print(f"    {os.path.basename(p['path'])}: json {p['n_json']}, "
                  f"video {p['usable']}, dropped {p['dropped_by_clamp']}")

    output_path = HF_LEROBOT_HOME / args.repo_name
    if output_path.exists():
        import shutil
        print(f"\nRemoving existing {output_path}")
        shutil.rmtree(output_path)

    dataset = NoVideoIOLeRobotDataset.create(
        repo_id=args.repo_name,
        robot_type="ARX",
        fps=fps,
        features={
            "face_view": {"dtype": "video", "shape": shape,
                          "names": ["height", "width", "channel"]},
            "left_wrist_view": {"dtype": "video", "shape": shape,
                                "names": ["height", "width", "channel"]},
            "right_wrist_view": {"dtype": "video", "shape": shape,
                                 "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (get_dim_from_keys(STATE_KEYS),),
                      "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (get_dim_from_keys(ACTION_KEYS),),
                        "names": ["actions"]},
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )
    # the videos are written by phase 1, so LeRobot must not try to encode or
    # count them itself (its save_episode asserts one video per episode so far).
    dataset._skip_all_media = True

    # ---- phase 1: one decode + one encode per video ------------------------
    print(f"\nPhase 1: transcoding {len(plans) * 3} videos "
          f"(num_workers={args.num_workers}, ffmpeg_threads={args.ffmpeg_threads}, "
          f"{os.cpu_count()} cores)")
    t0 = time.time()
    tasks = []
    for ep_idx, plan in enumerate(plans):
        for cam, fn in FILE_CAMERA_MAPPING.items():
            tasks.append((plan["path"], ep_idx, cam, fn, output_path,
                          plan["keep"], target_size, fps, args.ffmpeg_threads))
    failures = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(transcode_one, t): t[:4] for t in tasks}
        with tqdm.tqdm(total=len(tasks), desc="Transcoding") as bar:
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    ep_path, ep_idx, cam, _ = futs[fut]
                    failures.append((ep_idx, cam, str(e)))
                bar.update(1)
    t_transcode = time.time() - t0
    if failures:
        print(f"\n{len(failures)} video(s) failed:")
        for ep_idx, cam, msg in failures[:10]:
            print(f"  ep {ep_idx} {cam}: {msg}")
        raise SystemExit("aborting: the dataset would be inconsistent")
    print(f"Transcoding completed in {t_transcode:.1f}s")

    # ---- phase 2: parquet rows, on the same keep list ----------------------
    print("\nPhase 2: building dataset")
    t0 = time.time()
    # the videos are already written; LeRobot still type-checks the frame dict,
    # so hand it one throwaway array and let _save_image drop it on the floor.
    dummy_image = np.zeros(shape, dtype=np.uint8)
    for plan in tqdm.tqdm(plans, desc="Building"):
        frames = load_raw_frames(plan["path"])
        keep = plan["keep"]
        kept = [frames[i] for i in keep]
        state_array, action_array = arrays_from_frames(kept)
        n = len(state_array)
        dataset._video_frame_count = n - 1
        for i in range(n - 1):
            dataset.add_frame({
                "face_view": dummy_image,
                "left_wrist_view": dummy_image,
                "right_wrist_view": dummy_image,
                "state": state_array[i],
                "actions": action_array[i + 1],
                "task": args.task,
            })
        dataset.save_episode()
    t_build = time.time() - t0

    dataset.finalize_video_info()
    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done: {args.repo_name}")
    print(f"  episodes            {len(plans)}")
    print(f"  frames  raw -> kept {tot_json} -> {tot_keep} "
          f"({100 * tot_keep / max(tot_json, 1):.1f}%)")
    print(f"  phase 1 transcode   {t_transcode:.1f}s")
    print(f"  phase 2 build       {t_build:.1f}s")
    print(f"  total               {total:.1f}s "
          f"({total / max(len(plans), 1):.2f}s per episode)")
    print(f"  output              {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
