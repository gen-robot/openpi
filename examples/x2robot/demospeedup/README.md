# DemoSpeedup for π0 / X2Robot

This integration uses the frozen `bagging_sku_0811_sm2sm` step-80000
checkpoint as the proxy policy, estimates conditional action entropy on the
original 0811 datasets, rebuilds accelerated LeRobot datasets, and fine-tunes a
new π0 policy from the same checkpoint.

DemoSpeedup changes learned robot execution speed. It does not reduce π0 model
latency or the number of flow-matching steps at deployment.

## Binary entropy labels

The implementation adopts the requested pipeline, episode by episode:

1. Generate `K=10` action chunks from independent π0 noise samples. For frame
   `t`, align predictions made at earlier frames whose chunks cover `t`, then
   estimate KDE entropy over the executed follower/slave 14 action dimensions.
2. Apply Isolation Forest to the scalar per-frame entropy. Detected outliers are
   replaced by linear interpolation; frames are not deleted because that would
   break observation/action/video alignment.
3. Concatenate cleaned entropy with frame time.
4. Min-max normalize both columns inside that episode.
5. Run HDBSCAN in `(normalized_entropy, normalized_time)` space. Time makes
   clusters local along the trajectory rather than grouping every low-entropy
   point globally.
6. Clusters whose mean entropy is below the episode's configured entropy
   quantile are `Precision`; high-entropy clusters and HDBSCAN noise are
   `Casualness`. If HDBSCAN finds no usable low-entropy cluster, the same episode
   quantile is used as a deterministic fallback.

Defaults are Isolation Forest contamination `0.01`, HDBSCAN
`min_cluster_size=20`, and precision quantile `0.4`. These are calibration
parameters, not universal constants. Inspect entropy videos/curves and tune them
on a held-out set before full training.

## Files and artifacts

- `entropy.py`: entropy estimation, outlier interpolation, HDBSCAN segmentation.
- `label_entropy.py`: loads the 0811 checkpoint and emits one NPZ sidecar per
  episode under `demospeedup_labels/<dataset>`.
- `build_dataset.py`: applies Precision stride 2 and Casualness stride 4, then
  retains segmentation boundaries, gripper transitions, and frames needed by
  motion limits. It rebuilds parquet, video, indices, timestamps, episode stats,
  and metadata into a new `_demospeedup_r2r4` dataset.
- `meta/demospeedup.json` in each output dataset records checkpoint, parameters,
  per-episode counts, and the nominal compression/execution-speedup ratio.

## Pipeline

Install locked dependencies first:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Before processing the full collection, label and inspect a small slice:

```bash
export CUDA_VISIBLE_DEVICES=0
export DEMOSPEEDUP_DATASET=bagging_sku_zzc_0810
export DEMOSPEEDUP_EPISODES=0:1
export DEMOSPEEDUP_FRAME_BATCH_SIZE=1
bash run_demospeedup_0811.sh label
```

For the complete 13-dataset label pass on four GPUs, use the balanced parallel
launcher. It safely resumes existing per-episode labels and writes one log per
GPU under `logs/demospeedup_label/`:

```bash
DEMOSPEEDUP_FRAME_BATCH_SIZE=2 bash run_demospeedup_0811.sh label-parallel
tail -f logs/demospeedup_label/gpu0.log
```

`DEMOSPEEDUP_DATASET` is important for a true smoke test. Without it, the
episode range is applied independently to all 13 source datasets. The first
sampling batch also includes JAX compilation and is expected to be much slower
than later batches. The launcher enables a persistent JAX compilation cache, so
later runs with the same batch/sample shapes can reuse compiled executables.

Labeling shares each frame's image/text prefix across all stochastic samples.
For a faster approximate entropy pass, reduce only the proxy sampler workload
(this does not change final-policy inference):

```bash
export DEMOSPEEDUP_SAMPLES=6
export DEMOSPEEDUP_NUM_STEPS=5
bash run_demospeedup_0811.sh label
```

Clear `DEMOSPEEDUP_EPISODES` before building the complete datasets; the builder
intentionally refuses partial label sets.

```bash
unset DEMOSPEEDUP_DATASET DEMOSPEEDUP_EPISODES
unset DEMOSPEEDUP_SAMPLES DEMOSPEEDUP_NUM_STEPS
bash run_demospeedup_0811.sh label
bash run_demospeedup_0811.sh build
bash run_demospeedup_0811.sh norm
bash run_demospeedup_0811.sh train
```

Or run all stages sequentially:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_demospeedup_0811.sh all
```

The training config is `bagging_sku_0811_demospeedup_sm2sm`. It keeps the 0811
state history/future (`3/2`), action horizon (`20`), prompt metadata behavior,
batch size, and training steps. Its weight loader points to the supplied 0811
step-80000 checkpoint; normalization statistics are recomputed from accelerated
datasets before training.

## Validation gates

Run a dry build first and inspect `unresolved` motion-limit counts:

```bash
uv run examples/x2robot/demospeedup/build_dataset.py \
  --dataset bagging_sku_zzc_0810 --dry-run
```

Do not deploy a dataset with unresolved adjacent-source limit violations without
review. Compare baseline, uniform stride-2, and DemoSpeedup with identical task
starts. Report task success, successful-run completion time, safety stops,
endpoint error, maximum action delta, and inference p50/p95 separately.
