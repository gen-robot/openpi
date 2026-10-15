"""Action-entropy estimation and DemoSpeedup precision segmentation.

The binary-label pipeline is intentionally episode-local:

1. Estimate one entropy value per frame from stochastic action chunks.
2. Detect entropy outliers with Isolation Forest and interpolate over them.
3. Concatenate cleaned entropy and the frame's normalized time coordinate.
4. Min-max normalize both features within the episode.
5. Use HDBSCAN to discover temporally contiguous low-entropy regions.
6. Map low-entropy clusters to precision=1 and all other frames to casualness=1.

Outlier frames are never removed: doing so would break alignment with video, state,
and action rows. Only their entropy value is replaced before clustering.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SegmentationResult:
    raw_entropy: np.ndarray
    cleaned_entropy: np.ndarray
    normalized_entropy: np.ndarray
    normalized_time: np.ndarray
    outlier_mask: np.ndarray
    cluster_labels: np.ndarray
    precision: np.ndarray

    @property
    def casualness(self) -> np.ndarray:
        return ~self.precision


def _validate_entropy(entropy: np.ndarray) -> np.ndarray:
    values = np.asarray(entropy, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected 1-D entropy, got shape {values.shape}")
    if not len(values):
        raise ValueError("Entropy trajectory must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("Entropy trajectory contains NaN or infinity")
    return values


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    value_min = float(np.min(values))
    value_range = float(np.max(values) - value_min)
    if value_range <= np.finfo(np.float64).eps:
        return np.zeros_like(values)
    return (values - value_min) / value_range


def interpolate_outliers(values: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
    """Replace outliers by linear interpolation without removing time steps."""
    values = _validate_entropy(values)
    outlier_mask = np.asarray(outlier_mask, dtype=bool)
    if outlier_mask.shape != values.shape:
        raise ValueError(f"Outlier mask shape {outlier_mask.shape} != entropy shape {values.shape}")
    if not np.any(outlier_mask):
        return values.copy()
    valid = ~outlier_mask
    if not np.any(valid):
        return np.full_like(values, np.median(values))
    indices = np.arange(len(values))
    cleaned = values.copy()
    cleaned[outlier_mask] = np.interp(indices[outlier_mask], indices[valid], values[valid])
    return cleaned


def isolation_forest_clean(
    entropy: np.ndarray,
    *,
    contamination: float = 0.01,
    random_state: int = 0,
    n_estimators: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect scalar entropy outliers and interpolate over detected points."""
    values = _validate_entropy(entropy)
    if not 0.0 < contamination <= 0.5:
        raise ValueError(f"contamination must be in (0, 0.5], got {contamination}")
    # Isolation Forest is unstable on very short or constant sequences and adds no
    # useful information in those cases.
    if len(values) < 8 or np.ptp(values) <= np.finfo(np.float64).eps:
        return values.copy(), np.zeros(len(values), dtype=bool)
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as error:
        raise ImportError("DemoSpeedup entropy cleaning requires scikit-learn; run `uv sync`") from error

    detector = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    outlier_mask = detector.fit_predict(values[:, None]) == -1
    return interpolate_outliers(values, outlier_mask), outlier_mask


def kde_entropy(samples: np.ndarray, *, bandwidth: float = 1.0) -> float:
    """KDE entropy used by DemoSpeedup for a set of normalized action samples."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"Expected samples with shape (N, D), got {samples.shape}")
    if not len(samples):
        raise ValueError("At least one action sample is required")
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    squared_distance = np.sum((samples[:, None, :] - samples[None, :, :]) ** 2, axis=-1)
    density = np.exp(-squared_distance / (2.0 * bandwidth**2)).mean(axis=1)
    return float(-np.log(density + 1e-8).mean())


def temporal_action_entropy(
    action_chunks: np.ndarray,
    *,
    action_dims: int = 14,
    estimator: str = "kde",
    kde_bandwidth: float = 1.0,
) -> np.ndarray:
    """Align overlapping stochastic action chunks and estimate per-frame entropy.

    Args:
        action_chunks: Normalized predictions shaped (T, K, H, D), where T is
            episode length, K stochastic samples, and H action horizon.
        action_dims: Leading executed-action dimensions used for entropy. For the
            0811 sm2sm policy this defaults to the 14 follower/slave dimensions.
        estimator: ``kde`` (faithful to DemoSpeedup temporal aggregation) or ``std``.
    """
    chunks = np.asarray(action_chunks)
    if chunks.ndim != 4:
        raise ValueError(f"Expected action chunks with shape (T, K, H, D), got {chunks.shape}")
    episode_len, _, horizon, action_dim = chunks.shape
    if not 0 < action_dims <= action_dim:
        raise ValueError(f"action_dims must be in [1, {action_dim}], got {action_dims}")
    if estimator not in {"kde", "std"}:
        raise ValueError(f"Unknown entropy estimator {estimator!r}")

    entropy = np.empty(episode_len, dtype=np.float64)
    for frame_index in range(episode_len):
        first_query = max(0, frame_index - horizon + 1)
        aligned = np.concatenate(
            [chunks[query, :, frame_index - query, :action_dims] for query in range(first_query, frame_index + 1)],
            axis=0,
        )
        if estimator == "kde":
            entropy[frame_index] = kde_entropy(aligned, bandwidth=kde_bandwidth)
        else:
            entropy[frame_index] = float(np.std(aligned, axis=0).mean())
    return entropy


def _remove_short_true_runs(mask: np.ndarray, min_run_size: int) -> np.ndarray:
    if min_run_size <= 1:
        return mask
    result = mask.copy()
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    for start, end in zip(starts, ends, strict=True):
        if end - start < min_run_size:
            result[start:end] = False
    return result


def segment_precision(
    entropy: np.ndarray,
    *,
    contamination: float = 0.01,
    min_cluster_size: int = 20,
    min_samples: int | None = None,
    time_weight: float = 1.0,
    precision_quantile: float = 0.4,
    min_precision_run: int = 3,
    random_state: int = 0,
) -> SegmentationResult:
    """Convert per-frame entropy into Precision/Casualness labels with HDBSCAN.

    ``precision=True`` means low entropy and therefore the conservative sampling
    stride. HDBSCAN noise and high-entropy clusters are Casualness.
    """
    raw_entropy = _validate_entropy(entropy)
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if time_weight <= 0:
        raise ValueError("time_weight must be positive")
    if not 0.0 < precision_quantile < 1.0:
        raise ValueError("precision_quantile must be in (0, 1)")

    cleaned_entropy, outlier_mask = isolation_forest_clean(
        raw_entropy,
        contamination=contamination,
        random_state=random_state,
    )
    # Concatenate first, then normalize each feature within this trajectory. The
    # explicit order mirrors the method description and keeps all statistics local.
    entropy_time = np.column_stack([cleaned_entropy, np.arange(len(raw_entropy), dtype=np.float64)])
    features = np.column_stack([minmax_normalize(entropy_time[:, 0]), minmax_normalize(entropy_time[:, 1])])
    normalized_entropy = features[:, 0]
    normalized_time = features[:, 1]
    features[:, 1] *= time_weight

    # HDBSCAN requires min_cluster_size <= number of samples. Short episodes use
    # the deterministic entropy-quantile fallback below.
    cluster_labels = np.full(len(raw_entropy), -1, dtype=np.int64)
    effective_min_cluster = min(min_cluster_size, len(raw_entropy))
    if effective_min_cluster >= 2 and len(raw_entropy) >= 4:
        try:
            import hdbscan
        except ImportError as error:
            raise ImportError("DemoSpeedup precision segmentation requires hdbscan; run `uv sync`") from error
        cluster_labels = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_cluster,
            min_samples=min_samples,
        ).fit_predict(features)

    entropy_threshold = float(np.quantile(normalized_entropy, precision_quantile))
    precision = np.zeros(len(raw_entropy), dtype=bool)
    for cluster in np.unique(cluster_labels[cluster_labels >= 0]):
        members = cluster_labels == cluster
        if float(np.mean(normalized_entropy[members])) <= entropy_threshold:
            precision[members] = True

    # HDBSCAN can return a single mixed cluster or all noise on short/simple
    # trajectories. Retain a deterministic, episode-local low-entropy fallback.
    if not np.any(precision):
        precision = normalized_entropy <= entropy_threshold
    precision = _remove_short_true_runs(precision, min_precision_run)

    return SegmentationResult(
        raw_entropy=raw_entropy,
        cleaned_entropy=cleaned_entropy,
        normalized_entropy=normalized_entropy,
        normalized_time=normalized_time,
        outlier_mask=outlier_mask,
        cluster_labels=cluster_labels,
        precision=precision,
    )
