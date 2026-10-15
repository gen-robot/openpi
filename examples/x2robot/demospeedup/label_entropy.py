"""Label X2Robot demonstrations with π0 action entropy and binary precision.

The 0811 checkpoint is used only as the frozen proxy/entropy estimator. Output is
written as episode-level NPZ sidecars; the source LeRobot dataset is never changed.

Example:
  HF_LEROBOT_HOME=/mnt/public/gaojiahuan/huggingface/lerobot \
  CUDA_VISIBLE_DEVICES=0 uv run examples/x2robot/demospeedup/label_entropy.py \
    --dataset bagging_sku_zzc_0810 --episodes 0:5
"""
# The labeling tool intentionally samples normalized actions through the frozen
# model so it can share each observation prefix and avoid result unnormalization.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training.data_loader import FilteredLeRobotDataset

try:
    from examples.x2robot.demospeedup.entropy import segment_precision
    from examples.x2robot.demospeedup.entropy import temporal_action_entropy
except ModuleNotFoundError:
    # Direct execution places this directory, rather than the repository root, on sys.path.
    from entropy import segment_precision
    from entropy import temporal_action_entropy


DEFAULT_CHECKPOINT = Path(
    "/mnt/public/gaojiahuan/openpi/checkpoints/bagging_sku_0811_sm2sm/bagging_sku_0811_dm10dh50df50po20_meta/80000"
)
DEFAULT_DATA_ROOT = Path("/mnt/public/gaojiahuan/huggingface/lerobot")
DEFAULT_LABEL_ROOT = Path("/mnt/public/gaojiahuan/huggingface/demospeedup_labels")


@dataclasses.dataclass(frozen=True)
class LabelConfig:
    dataset: str
    config_name: str = "bagging_sku_0811_sm2sm"
    checkpoint: Path = DEFAULT_CHECKPOINT
    data_root: Path = DEFAULT_DATA_ROOT
    label_root: Path = DEFAULT_LABEL_ROOT
    episodes: str | None = None
    samples: int = 10
    frame_batch_size: int = 1
    num_steps: int = 10
    action_dims: int = 14
    estimator: str = "kde"
    kde_bandwidth: float = 1.0
    contamination: float = 0.01
    min_cluster_size: int = 20
    min_samples: int | None = None
    time_weight: float = 1.0
    precision_quantile: float = 0.4
    min_precision_run: int = 3
    seed: int = 0
    append_meta: bool = True
    overwrite: bool = False
    max_frames: int | None = None


def parse_episode_selection(spec: str | None, total_episodes: int) -> list[int]:
    if spec is None:
        return list(range(total_episodes))
    selected: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, end_text = token.split(":", maxsplit=1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else total_episodes
            selected.update(range(start, end))
        else:
            selected.add(int(token))
    invalid = [index for index in selected if not 0 <= index < total_episodes]
    if invalid:
        raise ValueError(f"Episode indices outside [0, {total_episodes}): {invalid}")
    return sorted(selected)


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def item_to_observation(item: dict, *, append_meta: bool) -> dict:
    prompt = str(item["task"])
    meta = item.get("meta")
    if append_meta and meta is not None and str(meta):
        prompt = f"{prompt} {str(meta).lstrip()}"
    return {
        "images": {
            "face_view": _numpy(item["face_view"]),
            "left_wrist_view": _numpy(item["left_wrist_view"]),
            "right_wrist_view": _numpy(item["right_wrist_view"]),
        },
        "state": _numpy(item["state"]),
        "prompt": prompt,
    }


def sample_normalized_chunks(
    policy, shared_sampler, observations: list[dict], *, samples: int, num_steps: int, rng
) -> tuple[np.ndarray, object]:
    """Sample K normalized chunks while encoding each observation prefix once."""
    transformed = [policy._input_transform(jax.tree.map(lambda x: x, observation)) for observation in observations]
    stacked = jax.tree.map(lambda *values: np.stack(values), *transformed)
    stacked = jax.tree.map(jnp.asarray, stacked)
    model_observation = _model.Observation.from_dict(stacked)

    rng, noise_rng, sample_rng = jax.random.split(rng, 3)
    model = policy._model
    noise = jax.random.normal(
        noise_rng,
        (len(observations), samples, model.action_horizon, model.action_dim),
    )
    chunks = shared_sampler(sample_rng, model_observation, num_steps=num_steps, noise=noise)
    chunks = np.asarray(jax.block_until_ready(chunks))
    return chunks, rng


def _atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _selected_dataset(config: LabelConfig, episode_indices: list[int], *, fps: int, state_offsets: list[int]):
    return FilteredLeRobotDataset(
        config.dataset,
        root=config.data_root / config.dataset,
        episodes=episode_indices,
        delta_timestamps={"state": [offset / fps for offset in state_offsets]},
    )


def label_episode(
    policy,
    shared_sampler,
    dataset,
    config: LabelConfig,
    *,
    episode_index: int,
    dataset_start: int,
    dataset_stop: int,
    rng,
) -> tuple[dict, object]:
    full_episode_len = dataset_stop - dataset_start
    episode_len = full_episode_len if config.max_frames is None else min(full_episode_len, config.max_frames)
    chunks = np.empty(
        (episode_len, config.samples, policy._model.action_horizon, policy._model.action_dim),
        dtype=np.float16,
    )
    started = time.monotonic()
    with tqdm.tqdm(
        total=episode_len,
        desc=f"Labeling episode {episode_index:06d}",
        unit="frame",
        dynamic_ncols=True,
        smoothing=0.1,
        postfix={"samples": config.samples, "steps": config.num_steps},
    ) as progress:
        for start in range(0, episode_len, config.frame_batch_size):
            stop = min(start + config.frame_batch_size, episode_len)
            observations = [
                item_to_observation(dataset[dataset_start + index], append_meta=config.append_meta)
                for index in range(start, stop)
            ]
            batch_chunks, rng = sample_normalized_chunks(
                policy,
                shared_sampler,
                observations,
                samples=config.samples,
                num_steps=config.num_steps,
                rng=rng,
            )
            chunks[start:stop] = batch_chunks.astype(np.float16)
            progress.update(stop - start)

    raw_entropy = temporal_action_entropy(
        chunks,
        action_dims=config.action_dims,
        estimator=config.estimator,
        kde_bandwidth=config.kde_bandwidth,
    )
    result = segment_precision(
        raw_entropy,
        contamination=config.contamination,
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        time_weight=config.time_weight,
        precision_quantile=config.precision_quantile,
        min_precision_run=config.min_precision_run,
        random_state=config.seed + episode_index,
    )
    summary = {
        "episode_index": episode_index,
        "frames": episode_len,
        "precision_frames": int(result.precision.sum()),
        "casual_frames": int(result.casualness.sum()),
        "outlier_frames": int(result.outlier_mask.sum()),
        "elapsed_seconds": time.monotonic() - started,
    }
    arrays = {
        "raw_entropy": result.raw_entropy.astype(np.float32),
        "cleaned_entropy": result.cleaned_entropy.astype(np.float32),
        "normalized_entropy": result.normalized_entropy.astype(np.float32),
        "normalized_time": result.normalized_time.astype(np.float32),
        "outlier_mask": result.outlier_mask,
        "cluster_labels": result.cluster_labels.astype(np.int32),
        "precision": result.precision,
        "casualness": result.casualness,
    }
    return {"summary": summary, "arrays": arrays}, rng


def run(config: LabelConfig) -> None:
    if config.samples < 2:
        raise ValueError("At least two stochastic samples are required")
    if config.frame_batch_size < 1:
        raise ValueError("frame_batch_size must be positive")
    if not config.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")
    source = config.data_root / config.dataset
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset not found: {source}")

    train_config = _config.get_config(config.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        config.checkpoint,
        sample_kwargs={"num_steps": config.num_steps},
    )
    shared_sampler = nnx_utils.module_jit(policy._model.sample_action_groups)
    with (source / "meta" / "info.json").open() as file:
        info = json.load(file)
    total_episodes = int(info["total_episodes"])
    selected_episodes = parse_episode_selection(config.episodes, total_episodes)
    if not selected_episodes:
        raise ValueError("Episode selection is empty")
    state_offsets = list(range(-train_config.data.state_history_size, train_config.data.state_future_size + 1))
    output = config.label_root / config.dataset
    output.mkdir(parents=True, exist_ok=True)
    rng = jax.random.key(config.seed)
    summaries = []
    logging.info(
        "Loading %d selected episode(s) once; the first sampling batch includes JAX compilation",
        len(selected_episodes),
    )
    dataset = _selected_dataset(config, selected_episodes, fps=int(info["fps"]), state_offsets=state_offsets)

    for local_episode_index, episode_index in enumerate(selected_episodes):
        output_path = output / f"episode_{episode_index:06d}.npz"
        if output_path.exists() and not config.overwrite:
            logging.info("Skipping existing label: %s", output_path)
            with np.load(output_path) as saved:
                summaries.append(
                    {
                        "episode_index": episode_index,
                        "frames": len(saved["precision"]),
                        "precision_frames": int(saved["precision"].sum()),
                        "casual_frames": int(saved["casualness"].sum()),
                        "outlier_frames": int(saved["outlier_mask"].sum()),
                        "elapsed_seconds": None,
                    }
                )
            continue
        dataset_start = int(dataset.episode_data_index["from"][local_episode_index])
        dataset_stop = int(dataset.episode_data_index["to"][local_episode_index])
        labeled, rng = label_episode(
            policy,
            shared_sampler,
            dataset,
            config,
            episode_index=episode_index,
            dataset_start=dataset_start,
            dataset_stop=dataset_stop,
            rng=rng,
        )
        _atomic_save_npz(output_path, **labeled["arrays"])
        summaries.append(labeled["summary"])
        logging.info("episode %06d complete: %s", episode_index, labeled["summary"])

    manifest = {
        "schema_version": 1,
        "dataset": config.dataset,
        "source": str(source),
        "config_name": config.config_name,
        "checkpoint": str(config.checkpoint.resolve()),
        "pipeline": [
            "per_frame_action_entropy",
            "isolation_forest_interpolate_outliers",
            "concatenate_entropy_and_time",
            "episode_minmax_normalization",
            "hdbscan_contiguous_low_entropy_clusters",
            "precision_casualness_binary_labels",
        ],
        "parameters": dataclasses.asdict(config)
        | {
            "checkpoint": str(config.checkpoint),
            "data_root": str(config.data_root),
            "label_root": str(config.label_root),
        },
        "episodes": summaries,
    }
    temporary_manifest = output / "manifest.tmp.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary_manifest.replace(output / "manifest.json")
    logging.info("Wrote labels and manifest to %s", output)


def parse_args() -> LabelConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config-name", default="bagging_sku_0811_sm2sm")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--episodes", help="Comma-separated indices/ranges, e.g. 0:10,15")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--action-dims", type=int, default=14)
    parser.add_argument("--estimator", choices=("kde", "std"), default="kde")
    parser.add_argument("--kde-bandwidth", type=float, default=1.0)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--time-weight", type=float, default=1.0)
    parser.add_argument("--precision-quantile", type=float, default=0.4)
    parser.add_argument("--min-precision-run", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-append-meta", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Debug only: truncate every selected episode")
    args = parser.parse_args()
    return LabelConfig(
        dataset=args.dataset,
        config_name=args.config_name,
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        label_root=args.label_root,
        episodes=args.episodes,
        samples=args.samples,
        frame_batch_size=args.frame_batch_size,
        num_steps=args.num_steps,
        action_dims=args.action_dims,
        estimator=args.estimator,
        kde_bandwidth=args.kde_bandwidth,
        contamination=args.contamination,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        time_weight=args.time_weight,
        precision_quantile=args.precision_quantile,
        min_precision_run=args.min_precision_run,
        seed=args.seed,
        append_meta=not args.no_append_meta,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
