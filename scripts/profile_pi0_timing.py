"""Profile Pi0 model timing: prefix vs denoise breakdown using JAX profiler."""

import dataclasses
import logging
import time
import json
from pathlib import Path

import tyro
import jax
import jax.numpy as jnp
import cv2
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.shared import nnx_utils


@dataclasses.dataclass
class Args:
    dataset_dir: str = "datasets/x2robot/plugusb_0119"
    policy_config: str = "plugusb_sm2sm"
    policy_dir: str = "checkpoints/plugusb_sm2sm/plugusb_0119+0120_sm2sm_h9f3oro_a30_dm10dh50df75po30/29999/"
    num_tests: int = 10


def build_state(frame_data: dict, policy_mode: str) -> np.ndarray:
    """Build state from frame data."""
    slave = np.concatenate([
        frame_data['follow_left_position'], frame_data['follow_left_rotation'],
        [frame_data['follow_left_gripper']], frame_data['follow_right_position'],
        frame_data['follow_right_rotation'], [frame_data['follow_right_gripper']]
    ]).astype(np.float32)
    
    if policy_mode in ["s2s", "s2m"]:
        return slave
    
    master = np.concatenate([
        frame_data['master_left_position'], frame_data['master_left_rotation'],
        [frame_data['master_left_gripper']], frame_data['master_right_position'],
        frame_data['master_right_rotation'], [frame_data['master_right_gripper']]
    ]).astype(np.float32)
    
    return np.concatenate([slave, master])


def main(args: Args):
    # Load config
    cfg = _config.get_config(args.policy_config)
    
    logging.info(f"Loading policy from {args.policy_dir}")
    policy = _policy_config.create_trained_policy(cfg, args.policy_dir)
    model = policy._model
    
    # Auto-detect policy_mode
    policy_mode = None
    for mode in ['sm2sm', 'sm2m', 's2m', 's2s']:
        if mode in args.policy_dir.lower():
            policy_mode = mode
            break
    
    # Load an episode
    episodes = sorted([p for p in Path(args.dataset_dir).glob('*') if p.is_dir()])
    ep_path = episodes[0]
    episode_name = ep_path.name
    logging.info(f"Using episode: {episode_name}")
    
    with open(ep_path / f"{episode_name}.json", 'r') as f:
        episode_data = json.load(f)
    
    frames = episode_data['data']
    total_frames = len(frames)
    
    # Load videos
    videos = {
        k: cv2.VideoCapture(str(ep_path / f"{k}Img.mp4")) 
        for k in ['left', 'face', 'right']
    }
    
    # Get state sequence length from config
    state_history_size = getattr(cfg.data, 'state_history_size', 0)
    state_future_size = getattr(cfg.data, 'state_future_size', 0)
    state_step = getattr(cfg.data, 'state_step', 1)
    
    # Prepare test observations
    test_obs_list = []
    for frame_idx in range(0, min(args.num_tests, total_frames)):
        # Read images
        images = {}
        for k, cap in videos.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                images[f"{k}_wrist_view" if k != "face" else "face_view"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Build state sequence
        indices = [max(0, min(frame_idx + i * state_step, total_frames - 1)) 
                   for i in range(-state_history_size, state_future_size + 1)]
        state_seq = np.array([build_state(frames[i], policy_mode) for i in indices], dtype=np.float32)
        
        obs = {'images': images, 'prompt': '', 'state': state_seq}
        test_obs_list.append(obs)
    
    for cap in videos.values():
        cap.release()
    
    logging.info(f"Prepared {len(test_obs_list)} test observations")
    
    # Prepare first observation for warmup
    obs0 = test_obs_list[0]
    obs0_transformed = policy._input_transform(jax.tree.map(lambda x: x, obs0))
    obs0_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], obs0_transformed)
    observation0 = _model.Observation.from_dict(obs0_jax)
    rng = jax.random.key(0)
    
    # Test different number of steps
    steps_to_test = [1, 5, 10, 15, 20]
    
    # Warmup all paths
    logging.info("Warming up all paths...")
    for _ in range(3):
        _ = policy.infer(test_obs_list[0])  # 10 steps via policy.infer
    for steps in steps_to_test:
        for _ in range(2):
            _ = policy._sample_actions(rng, observation0, num_steps=steps)
    
    # Test full inference timing (using policy.infer like the real usage)
    logging.info("Testing full inference (policy.infer)...")
    full_times = []
    for i, obs in enumerate(test_obs_list):
        start = time.perf_counter()
        _ = policy.infer(obs)
        elapsed = time.perf_counter() - start
        full_times.append(elapsed * 1000)
    
    # Test different steps
    step_results = {}
    for steps in steps_to_test:
        logging.info(f"Testing with {steps} denoise step(s)...")
        times = []
        for i, obs in enumerate(test_obs_list):
            obs_transformed = policy._input_transform(jax.tree.map(lambda x: x, obs))
            obs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], obs_transformed)
            observation = _model.Observation.from_dict(obs_jax)
            
            start = time.perf_counter()
            _ = policy._sample_actions(rng, observation, num_steps=steps)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        step_results[steps] = np.mean(times)
    
    # Test input transform time
    logging.info("Testing input transform time...")
    input_transform_times = []
    for i, obs in enumerate(test_obs_list):
        start = time.perf_counter()
        obs_transformed = policy._input_transform(jax.tree.map(lambda x: x, obs))
        obs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], obs_transformed)
        observation = _model.Observation.from_dict(obs_jax)
        elapsed = time.perf_counter() - start
        input_transform_times.append(elapsed * 1000)
    
    # Calculate: 
    # N steps total = prefix + N * step_time
    # Use linear regression to find prefix and step_time
    steps_arr = np.array(list(step_results.keys()))
    times_arr = np.array(list(step_results.values()))
    
    # Linear regression: time = prefix + step_time * steps
    A = np.vstack([steps_arr, np.ones(len(steps_arr))]).T
    step_time_per, prefix_time = np.linalg.lstsq(A, times_arr, rcond=None)[0]
    
    # Print results
    print("\n" + "="*80)
    print("PI0 TIMING BREAKDOWN")
    print("="*80)
    
    print(f"\nRaw measurements by steps:")
    for steps, t in sorted(step_results.items()):
        predicted = prefix_time + step_time_per * steps
        print(f"  {steps:2d} steps: {t:.2f} ms (predicted: {predicted:.2f} ms, diff: {t - predicted:+.2f} ms)")
    
    print(f"\nLinear regression: time = {prefix_time:.2f} + {step_time_per:.2f} * steps")
    print(f"  => Prefix (intercept): {prefix_time:.2f} ms")
    print(f"  => Per step (slope):   {step_time_per:.2f} ms")
    
    denoise_10_steps = step_time_per * 10
    model_time_10 = prefix_time + denoise_10_steps
    transform_time = np.mean(full_times) - step_results[10]
    
    print(f"\nFinal breakdown (for 10 steps):")
    print(f"  Prefix (Image encoding + LLM + KV cache): {prefix_time:.2f} ms ({prefix_time/np.mean(full_times)*100:.1f}%)")
    print(f"  Denoise loop (10 steps, action expert):   {denoise_10_steps:.2f} ms ({denoise_10_steps/np.mean(full_times)*100:.1f}%)")
    print(f"  Data transforms (input + output):         {transform_time:.2f} ms ({transform_time/np.mean(full_times)*100:.1f}%)")
    print(f"    - Input transform only:                 {np.mean(input_transform_times):.2f} ms")
    print(f"    - Output transform + other:             {transform_time - np.mean(input_transform_times):.2f} ms")
    
    print(f"\nVerification:")
    print(f"  policy.infer (10 steps): {np.mean(full_times):.2f} ms")
    print(f"  _sample_actions (10 steps): {step_results[10]:.2f} ms")
    print(f"  Prefix + Denoise(10) = {model_time_10:.2f} ms (should ≈ {step_results[10]:.2f} ms)")
    print("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))


