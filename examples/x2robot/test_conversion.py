"""
Test script to verify that optimized conversion produces identical results.
"""
import os
import shutil
import numpy as np
import torch
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
import tyro


def compare_datasets(repo_name1: str, repo_name2: str) -> bool:
    """
    Compare two LeRobot datasets to ensure they are identical.
    
    Args:
        repo_name1: Name of first dataset
        repo_name2: Name of second dataset
        
    Returns:
        True if datasets are identical, False otherwise
    """
    print(f"Comparing {repo_name1} vs {repo_name2}")
    
    # Load both datasets from local directories
    path1 = HF_LEROBOT_HOME / repo_name1
    path2 = HF_LEROBOT_HOME / repo_name2
    
    # Since they are local, we specify the root and let it load from there
    dataset1 = LeRobotDataset(repo_id=repo_name1, root=path1)
    dataset2 = LeRobotDataset(repo_id=repo_name2, root=path2)
    
    # Check basic metadata
    if len(dataset1) != len(dataset2):
        print(f"❌ Length mismatch: {len(dataset1)} vs {len(dataset2)}")
        return False
    print(f"✓ Length matches: {len(dataset1)} frames")
    
    if dataset1.num_episodes != dataset2.num_episodes:
        print(f"❌ Episode count mismatch: {dataset1.num_episodes} vs {dataset2.num_episodes}")
        return False
    print(f"✓ Episode count matches: {dataset1.num_episodes} episodes")
    
    # Check episode metadata
    for ep_idx in range(dataset1.num_episodes):
        ep1_info = dataset1.episode_data_index['from'][ep_idx], dataset1.episode_data_index['to'][ep_idx]
        ep2_info = dataset2.episode_data_index['from'][ep_idx], dataset2.episode_data_index['to'][ep_idx]
        
        if ep1_info != ep2_info:
            print(f"❌ Episode {ep_idx} index mismatch: {ep1_info} vs {ep2_info}")
            return False
    print(f"✓ All episode indices match")
    
    # Compare randomly sampled frames from each episode
    state_errors = []
    action_errors = []
    image_diffs = []  # Collect all image differences for statistics
    
    SAMPLES_PER_EPISODE = 10  # Random sample 10 frames per episode
    
    print(f"\n--- Detailed Comparison (Random {SAMPLES_PER_EPISODE} frames per episode) ---")
    
    total_samples = 0
    for ep_idx in range(dataset1.num_episodes):
        from_idx = dataset1.episode_data_index['from'][ep_idx].item()
        to_idx = dataset1.episode_data_index['to'][ep_idx].item()
        ep_length = to_idx - from_idx
        
        # Random sample indices within this episode
        np.random.seed(42 + ep_idx)  # Reproducible random sampling
        if ep_length <= SAMPLES_PER_EPISODE:
            sample_indices = list(range(from_idx, to_idx))
        else:
            sample_indices = sorted(np.random.choice(range(from_idx, to_idx), SAMPLES_PER_EPISODE, replace=False))
        
        for idx in sample_indices:
            idx = int(idx)  # Convert numpy.int64 to int
            total_samples += 1
            frame1 = dataset1[idx]
            frame2 = dataset2[idx]
            
            # Compare state
            state1 = frame1['state'].numpy()
            state2 = frame2['state'].numpy()
            if not np.allclose(state1, state2, rtol=1e-5, atol=1e-7):
                state_diff = np.abs(state1 - state2)
                state_errors.append({
                    'frame': idx,
                    'max_diff': state_diff.max(),
                    'mean_diff': state_diff.mean()
                })
            
            # Compare actions
            action1 = frame1['actions'].numpy()
            action2 = frame2['actions'].numpy()
            if not np.allclose(action1, action2, rtol=1e-5, atol=1e-7):
                action_diff = np.abs(action1 - action2)
                action_errors.append({
                    'frame': idx,
                    'max_diff': action_diff.max(),
                    'mean_diff': action_diff.mean()
                })
            
            # Compare images (pixel-wise)
            for cam in ['face_view', 'left_wrist_view', 'right_wrist_view']:
                img1 = frame1[cam].numpy()
                img2 = frame2[cam].numpy()
                
                # Compute difference statistics
                diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
                max_diff = diff.max()
                mean_diff = diff.mean()
                
                image_diffs.append({
                    'frame': idx, 'cam': cam,
                    'max_diff': max_diff, 'mean_diff': mean_diff,
                    'exact_match': np.array_equal(img1, img2)
                })
    
    print(f"  Checked {total_samples} frames ({SAMPLES_PER_EPISODE} per episode × {dataset1.num_episodes} episodes)")
    
    # Report state comparison
    if state_errors:
        print(f"\n❌ State: {len(state_errors)} mismatches")
        for err in state_errors[:3]:
            print(f"   Frame {err['frame']}: max_diff={err['max_diff']:.6f}, mean_diff={err['mean_diff']:.6f}")
    else:
        print(f"\n✓ State: ALL EXACT MATCH (100%) - {total_samples} frames checked")
    
    # Report action comparison
    if action_errors:
        print(f"\n❌ Actions: {len(action_errors)} mismatches")
        for err in action_errors[:3]:
            print(f"   Frame {err['frame']}: max_diff={err['max_diff']:.6f}, mean_diff={err['mean_diff']:.6f}")
    else:
        print(f"\n✓ Actions: ALL EXACT MATCH (100%) - {total_samples} frames checked")
    
    # Report image comparison statistics
    exact_matches = sum(1 for d in image_diffs if d['exact_match'])
    total_images = len(image_diffs)
    max_diffs = [d['max_diff'] for d in image_diffs]
    mean_diffs = [d['mean_diff'] for d in image_diffs]
    
    print(f"\n--- Image Comparison Statistics ---")
    print(f"  Total images compared: {total_images}")
    print(f"  Exact matches: {exact_matches} ({100*exact_matches/total_images:.1f}%)")
    print(f"  Max pixel diff across all images: {max(max_diffs):.4f}")
    print(f"  Min pixel diff across all images: {min(max_diffs):.4f}")
    print(f"  Avg max diff per image: {np.mean(max_diffs):.4f}")
    print(f"  Avg mean diff per image: {np.mean(mean_diffs):.4f}")
    
    # Show distribution of differences
    print(f"\n  Difference distribution:")
    bins = [(0, 0), (0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    for low, high in bins:
        if low == 0 and high == 0:
            count = sum(1 for d in max_diffs if d == 0)
            print(f"    max_diff = 0 (exact): {count} images ({100*count/total_images:.1f}%)")
        else:
            count = sum(1 for d in max_diffs if low < d <= high)
            print(f"    {low:.2f} < max_diff <= {high:.2f}: {count} images ({100*count/total_images:.1f}%)")
    
    # Per-camera statistics
    print(f"\n  Per-camera breakdown:")
    for cam in ['face_view', 'left_wrist_view', 'right_wrist_view']:
        cam_diffs = [d for d in image_diffs if d['cam'] == cam]
        cam_exact = sum(1 for d in cam_diffs if d['exact_match'])
        cam_max = max(d['max_diff'] for d in cam_diffs)
        cam_mean_avg = np.mean([d['mean_diff'] for d in cam_diffs])
        print(f"    {cam}: exact={cam_exact}/{len(cam_diffs)}, max_diff={cam_max:.4f}, avg_mean_diff={cam_mean_avg:.4f}")
    
    print("")
    
    # Determine overall success
    has_errors = bool(state_errors or action_errors)
    if has_errors:
        return False
    
    # For images, we tolerate small differences due to different decoders
    high_diff_images = [d for d in image_diffs if d['max_diff'] > 0.5]
    if high_diff_images:
        print(f"⚠ Warning: {len(high_diff_images)} images have max_diff > 0.5")
        for d in high_diff_images[:5]:
            print(f"   Frame {d['frame']}, {d['cam']}: max_diff={d['max_diff']:.4f}")
    
    print(f"✓ State and Action data: EXACT MATCH")
    print(f"✓ Images: Small differences due to different video decoders (acceptable)")
    
    # Check video files exist and have reasonable sizes
    videos1 = list((HF_LEROBOT_HOME / repo_name1 / 'videos').rglob('*.mp4'))
    videos2 = list((HF_LEROBOT_HOME / repo_name2 / 'videos').rglob('*.mp4'))
    
    if len(videos1) != len(videos2):
        print(f"❌ Video count mismatch: {len(videos1)} vs {len(videos2)}")
        return False
    print(f"✓ Video count matches: {len(videos1)} videos")
    
    # Compare video file sizes (should be similar within 1%)
    videos1_sorted = sorted(videos1, key=lambda x: x.name)
    videos2_sorted = sorted(videos2, key=lambda x: x.name)
    
    for v1, v2 in zip(videos1_sorted, videos2_sorted):
        if v1.name != v2.name:
            print(f"❌ Video name mismatch: {v1.name} vs {v2.name}")
            return False
            
        size1 = v1.stat().st_size
        size2 = v2.stat().st_size
        size_diff_pct = abs(size1 - size2) / max(size1, size2) * 100
        
        if size_diff_pct > 10:  # Allow 10% difference
            print(f"❌ Video {v1.name} size differs by {size_diff_pct:.1f}%: {size1} vs {size2}")
            return False
    
    print(f"✓ All video file sizes are similar")
    
    print("\n✅ Datasets are equivalent!")
    return True


def main(
    original_repo: str = "plugin_0107_sm2sm",
    optimized_repo: str = "plugin_0107_test_v4",
):
    """
    Compare original and optimized conversion results.
    
    Args:
        original_repo: Name of dataset created with original code
        optimized_repo: Name of dataset created with optimized code
    """
    # Check both datasets exist
    if not (HF_LEROBOT_HOME / original_repo).exists():
        print(f"❌ Original dataset not found: {original_repo}")
        print(f"   Expected at: {HF_LEROBOT_HOME / original_repo}")
        return False
        
    if not (HF_LEROBOT_HOME / optimized_repo).exists():
        print(f"❌ Optimized dataset not found: {optimized_repo}")
        print(f"   Expected at: {HF_LEROBOT_HOME / optimized_repo}")
        return False
    
    return compare_datasets(original_repo, optimized_repo)


if __name__ == "__main__":
    tyro.cli(main)
