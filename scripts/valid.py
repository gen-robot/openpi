"""Validation script for evaluating a fine-tuned model on a dataset.

This script evaluates a trained model on a validation dataset and computes MSE metrics
for action predictions. It supports custom grouping of action dimensions (e.g., position,
rotation, gripper) to provide detailed performance analysis.

Example usage:
    # Basic validation with overall MSE
    uv run scripts/valid.py pi05_libero --checkpoint-dir checkpoints/pi05_libero/my_experiment/20000

    # With custom action grouping for a 14-dim action (ALOHA: 2 arms x 7 dims)
    uv run scripts/valid.py pi0_aloha_sim --checkpoint-dir checkpoints/pi0_aloha_sim/exp/20000 \
        --action-groups '{"left_pos": [0,1,2], "left_rot": [3,4,5], "left_gripper": [6], \
                         "right_pos": [7,8,9], "right_rot": [10,11,12], "right_gripper": [13]}'
    
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0, uv run scripts/valid.py \
        --config-name microwave_1218 \
        --checkpoint-dir checkpoints/microwave_1218/microwave_1218/29999/ \
        --batch-size 32 \
        --max-samples 5000 \
        --action-groups '{"left_pos": [0,1,2], "left_rot": [3,4,5], "left_gripper": [6], "right_pos": [7,8,9], "right_rot": [10,11,12], "right_gripper": [13]}'

    # Limit number of validation samples
    uv run scripts/valid.py pi05_droid --checkpoint-dir checkpoints/pi05_droid/exp/10000 \
        --max-samples 1000
"""

import dataclasses
import json
import logging
import platform
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm_loggable.auto as tqdm
import tyro

import openpi.models.model as _model
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


@dataclasses.dataclass
class ValidArgs:
    """Arguments for the validation script."""

    config_name: str
    """Name of the training config to use for validation."""

    checkpoint_dir: str
    """Path to the checkpoint directory containing the trained model."""

    max_samples: int | None = None
    """Maximum number of samples to evaluate. If None, evaluate entire dataset."""

    batch_size: int = 1
    """Batch size for evaluation (currently only supports 1)."""

    action_groups: str | None = None
    """JSON string defining action dimension groups for detailed metrics.
    
    Example: '{"position": [0,1,2], "rotation": [3,4,5], "gripper": [6]}'
    Each group will have its own MSE computed separately in addition to overall MSE.
    """

    dataset_split: str = "validation"
    """Dataset split to use for validation ('train' or 'validation')."""


def compute_mse(pred_actions: np.ndarray, true_actions: np.ndarray) -> dict[str, float]:
    """Compute MSE between predicted and true actions.
    
    Args:
        pred_actions: Predicted actions of shape (num_samples, action_horizon, action_dim)
        true_actions: True actions of shape (num_samples, action_horizon, action_dim)
    
    Returns:
        Dictionary containing MSE value
    """
    mse = np.mean((pred_actions - true_actions) ** 2)
    return {"mse": float(mse)}


def compute_grouped_mse(
    pred_actions: np.ndarray,
    true_actions: np.ndarray,
    action_groups: dict[str, list[int]] | None = None,
) -> dict[str, float]:
    """Compute MSE for overall actions and for each action group.
    
    Args:
        pred_actions: Predicted actions of shape (num_samples, action_horizon, action_dim)
        true_actions: True actions of shape (num_samples, action_horizon, action_dim)
        action_groups: Dictionary mapping group names to lists of dimension indices.
                      Example: {"position": [0,1,2], "rotation": [3,4,5], "gripper": [6]}
    
    Returns:
        Dictionary containing overall MSE and per-group MSE values
    """
    metrics = {}
    
    # Overall MSE
    overall_mse = np.mean((pred_actions - true_actions) ** 2)
    metrics["overall_mse"] = float(overall_mse)
    
    # Per-group MSE
    if action_groups is not None:
        for group_name, indices in action_groups.items():
            pred_group = pred_actions[:, :, indices]
            true_group = true_actions[:, :, indices]
            group_mse = np.mean((pred_group - true_group) ** 2)
            metrics[f"{group_name}_mse"] = float(group_mse)
    
    return metrics


def validate_policy(
    policy: Any,
    data_loader: _data_loader.DataLoader,
    max_samples: int | None = None,
    action_groups: dict[str, list[int]] | None = None,
) -> dict[str, float]:
    """Run validation on the policy using the data loader.
    
    Args:
        policy: The policy to evaluate
        data_loader: Data loader providing validation samples
        max_samples: Maximum number of samples to evaluate
        action_groups: Optional grouping of action dimensions for detailed metrics
    
    Returns:
        Dictionary containing validation metrics
    """
    all_pred_actions = []
    all_true_actions = []
    
    total_inference_time = 0.0
    num_samples = 0
    
    logging.info("Starting validation...")
    
    # Try to determine total number of samples
    total_samples = None
    if max_samples is not None:
        total_samples = max_samples
    else:
        # Try to get length from the underlying data loader
        try:
            if hasattr(data_loader, "_data_loader"):  # noqa: SLF001
                raw_loader = data_loader._data_loader  # noqa: SLF001
                if hasattr(raw_loader, "torch_loader"):
                    # TorchDataLoader case - multiply by batch_size to get total samples
                    num_batches = len(raw_loader.torch_loader)
                    batch_size = raw_loader.torch_loader.batch_size
                    total_samples = num_batches * batch_size
                elif hasattr(raw_loader, "_dataset"):  # noqa: SLF001
                    # Try to get from dataset
                    total_samples = len(raw_loader._dataset)  # noqa: SLF001
        except (AttributeError, TypeError):
            # If we can't get the length, that's okay
            pass
    
    # Create progress bar - if total is None, tqdm will show a counter without percentage
    pbar = tqdm.tqdm(total=total_samples, desc="Validating")
    
    try:
        # Iterate through the data loader (which returns Observation and actions)
        for observation, true_action in data_loader:
            if max_samples is not None and num_samples >= max_samples:
                break
            
            # Convert to numpy arrays
            obs_dict = _model.Observation.to_dict(observation)
            obs_dict = jax.tree.map(lambda x: np.asarray(x) if hasattr(x, "__array__") else x, obs_dict)
            true_action = np.asarray(true_action)
            
            # Get the actual batch size from the data
            current_batch_size = true_action.shape[0] if true_action.ndim >= 2 else 1
            
            # Fix image_masks - convert Python bool to numpy arrays
            if "image_mask" in obs_dict:
                obs_dict["image_mask"] = jax.tree.map(
                    lambda x: np.array(x) if isinstance(x, (bool, np.bool_)) else x,
                    obs_dict["image_mask"]
                )
            
            # Create Observation object from the dict (already batched)
            obs_obj = _model.Observation.from_dict(obs_dict)
            
            # Batch inference - process entire batch at once
            start_time = time.time()
            
            # Call sample_actions directly on the batch (skip policy transforms)
            if policy._is_pytorch_model:  # noqa: SLF001
                # PyTorch model - already batched
                obs_obj = jax.tree.map(
                    lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device),  # noqa: SLF001
                    obs_obj
                )
                pred_actions = policy._sample_actions(policy._pytorch_device, obs_obj, **policy._sample_kwargs)  # noqa: SLF001
                pred_actions = np.asarray(pred_actions.detach().cpu())
            else:
                # JAX model - already batched
                obs_obj = jax.tree.map(lambda x: jnp.asarray(x), obs_obj)
                policy._rng, sample_rng = jax.random.split(policy._rng)  # noqa: SLF001
                pred_actions = policy._sample_actions(sample_rng, obs_obj, **policy._sample_kwargs)  # noqa: SLF001
                pred_actions = np.asarray(pred_actions)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Process each sample in the batch for output transforms and storing
            for i in range(current_batch_size):
                if max_samples is not None and num_samples >= max_samples:
                    break
                
                # Extract single sample
                sample_obs = jax.tree.map(
                    lambda x: x[i] if isinstance(x, np.ndarray) and x.ndim > 0 else x,
                    obs_dict
                )
                pred_action = pred_actions[i]
                sample_true_action = true_action[i]
                
                # Apply output transforms
                outputs = {"state": sample_obs.get("state"), "actions": pred_action}
                outputs = policy._output_transform(outputs)  # noqa: SLF001
                pred_action = outputs["actions"]
                
                # Also apply output transform to true_action for fair comparison
                true_outputs = {"state": sample_obs.get("state"), "actions": sample_true_action}
                true_outputs = policy._output_transform(true_outputs)  # noqa: SLF001
                sample_true_action = true_outputs["actions"]
                
                # Store predictions and ground truth
                all_pred_actions.append(pred_action)
                all_true_actions.append(sample_true_action)
                
                num_samples += 1
                pbar.update(1)
            
    finally:
        pbar.close()
    
    logging.info(f"Processed {num_samples} samples")
    
    # Stack all actions
    all_pred_actions = np.stack(all_pred_actions, axis=0)
    all_true_actions = np.stack(all_true_actions, axis=0)
    
    logging.info(f"Prediction shape: {all_pred_actions.shape}")
    logging.info(f"Ground truth shape: {all_true_actions.shape}")
    
    # Compute metrics
    metrics = compute_grouped_mse(all_pred_actions, all_true_actions, action_groups)
    
    # Add timing metrics
    metrics["avg_inference_time_ms"] = (total_inference_time / num_samples) * 1000 if num_samples > 0 else 0
    metrics["num_samples"] = num_samples
    
    return metrics


def main(args: ValidArgs):
    """Main validation function."""
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Config: {args.config_name}")
    logging.info(f"Checkpoint: {args.checkpoint_dir}")
    
    # Parse action groups if provided
    action_groups = None
    if args.action_groups is not None:
        try:
            action_groups = json.loads(args.action_groups)
            logging.info(f"Action groups: {action_groups}")
            
            # Validate action groups
            for group_name, indices in action_groups.items():
                if not isinstance(indices, list):
                    raise ValueError(f"Indices for group '{group_name}' must be a list")
                if not all(isinstance(i, int) for i in indices):
                    raise ValueError(f"All indices for group '{group_name}' must be integers")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse action_groups JSON: {e}")
            raise
    
    # Get config
    config = _config.get_config(args.config_name)
    
    # Create policy from checkpoint
    logging.info("Loading policy from checkpoint...")
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
    logging.info("Policy loaded successfully")
    
    # Create data loader for validation
    logging.info("Creating data loader...")
    
    # Create a modified config for validation
    # We use a minimal number of batches for validation if max_samples is specified
    num_batches = args.max_samples if args.max_samples else None
    
    # Create appropriate data loader
    # We need to create the data config first
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Determine which type of data loader to create
    if data_config.rlds_data_dir is not None:
        # RLDS data loader (e.g., DROID)
        data_loader = _data_loader.create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=args.batch_size,
            shuffle=False,
            num_batches=num_batches,
            skip_norm_stats=False,
        )
    elif data_config.repo_id is not None and data_config.repo_id != "fake":
        # LeRobot dataset via torch
        data_loader = _data_loader.create_torch_data_loader(
            data_config,
            model_config=config.model,
            action_horizon=config.model.action_horizon,
            batch_size=args.batch_size,
            shuffle=False,
            num_batches=num_batches,
            num_workers=0,  # Use single process for validation
            seed=0,
            skip_norm_stats=False,
        )
    else:
        logging.error("Only LeRobot and RLDS datasets are supported for validation")
        raise ValueError("Invalid dataset configuration")
    
    logging.info(f"Data loader created (batch_size={args.batch_size})")
    
    # Run validation
    metrics = validate_policy(
        policy=policy,
        data_loader=data_loader,
        max_samples=args.max_samples,
        action_groups=action_groups,
    )
    
    # Print results
    logging.info("\n" + "=" * 60)
    logging.info("VALIDATION RESULTS")
    logging.info("=" * 60)
    
    for metric_name, metric_value in metrics.items():
        if metric_name == "num_samples":
            logging.info(f"{metric_name}: {metric_value}")
        else:
            logging.info(f"{metric_name}: {metric_value:.6f}")
    
    logging.info("=" * 60)
    
    return metrics


if __name__ == "__main__":
    args = tyro.cli(ValidArgs)
    main(args)
