import random
import dataclasses
import einops
import numpy as np
from typing import ClassVar

from openpi import transforms
from openpi.models import model as _model
from PIL import Image, ImageEnhance
import random


# Used by `LeRobotX2robotDataConfig._individual_keys_repack` in training/config.py.
ALL_COMPONENT_KEYS: tuple[str, ...] = (
    "follow_left_position",
    "follow_left_rotation",
    "follow_left_gripper",
    "follow_right_position",
    "follow_right_rotation",
    "follow_right_gripper",
    "master_left_position",
    "master_left_rotation",
    "master_left_gripper",
    "master_right_position",
    "master_right_rotation",
    "master_right_gripper",
)
PHYSICAL_STATE_DIM=28
TASK_STATE_DIM=3

BASKET_IDX = 28
COL_IDX = 29
ROW_IDX = 30
RAW_STATE_DIM = 31

def make_arx_example() -> dict:
    """Creates a random input example for the ARX policy."""
    return {
        "state": np.random.rand(14),
        "image": {
            "left_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "face_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }



@dataclasses.dataclass(frozen=True)
class ArxDeltaActions(transforms.DataTransformFn):
    """Convert absolute actions to delta actions using the current state.

    Supports:
        state.shape == (D,)
        state.shape == (history + current + future, D)
    """

    mask: tuple[bool, ...]
    current_idx: int = 0

    def __call__(self, data: dict) -> dict:
        delta_reference = data.pop("delta_reference", None)
        if "actions" not in data:
            # During inference input processing, actions are not present.
            return data

        # state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        if delta_reference is not None:
            # 新task-only模式
            current_state = np.asarray(delta_reference)

            if current_state.ndim != 1:
                raise ValueError(
                    f"Expected delta_reference shape (D,), "
                    f"got {current_state.shape}."
                )

        else:
            state = np.asarray(data["state"])
            if state.ndim == 1:
                current_state = state
            elif state.ndim == 2:
                if not 0 <= self.current_idx < state.shape[0]:
                    raise IndexError(
                        f"current_idx={self.current_idx} is out of range for "
                        f"state shape {state.shape}."
                    )
                current_state = state[self.current_idx]
            else:
                raise ValueError(
                    f"Expected state shape (D,) or (T,D), got {state.shape}."
                )

        mask = np.asarray(self.mask, dtype=bool)
        dims = len(mask)

        if actions.shape[-1] < dims:
            raise ValueError(
                f"Action dim {actions.shape[-1]} is smaller than "
                f"delta mask dim {dims}."
            )
        if current_state.shape[-1] < dims:
            raise ValueError(
                f"State dim {current_state.shape[-1]} is smaller than "
                f"delta mask dim {dims}."
            )

        reference = np.where(mask, current_state[:dims], 0.0)

        # actions: (action_horizon, action_dim)
        # reference: (dims,)
        # NumPy broadcasts the same current state across the action horizon.
        actions[..., :dims] -= reference

        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class ArxAbsoluteActions(transforms.DataTransformFn):
    """Convert predicted delta actions back to absolute actions."""

    mask: tuple[bool, ...]
    current_idx: int = 0

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        # state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        if "delta_reference" in data:
            # 新task-only模式
            current_state = np.asarray(
                data["delta_reference"]
            )

            if current_state.ndim != 1:
                raise ValueError(
                    f"Expected delta_reference shape (D,), "
                    f"got {current_state.shape}."
                )
        else:
            state = np.asarray(data["state"])
            if state.ndim == 1:
                current_state = state
            elif state.ndim == 2:
                if not 0 <= self.current_idx < state.shape[0]:
                    raise IndexError(
                        f"current_idx={self.current_idx} is out of range for "
                        f"state shape {state.shape}."
                    )
                current_state = state[self.current_idx]
            else:
                raise ValueError(
                    f"Expected state shape (D,) or (T,D), got {state.shape}."
                )

        mask = np.asarray(self.mask, dtype=bool)
        dims = len(mask)

        if actions.shape[-1] < dims:
            raise ValueError(
                f"Action dim {actions.shape[-1]} is smaller than "
                f"delta mask dim {dims}."
            )
        if current_state.shape[-1] < dims:
            raise ValueError(
                f"State dim {current_state.shape[-1]} is smaller than "
                f"delta mask dim {dims}."
            )

        reference = np.where(mask, current_state[:dims], 0.0)
        actions[..., :dims] += reference

        data["actions"] = actions
        return data


        
@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Transform inputs for the ARX policy."""

    mode: str = "s2s"  # "s2s", "s2m", "sm2m", "sm2sm"
    action_dim: int = 32
    model_type: _model.ModelType = _model.ModelType.PI0
    state_history_size: int = 0
    state_future_size: int = 0
    slave_state_dim: int = 14
    mask_history_slave_states: bool = False
    random_drop_master: float = 0.
    random_drop_history: float = 0.
    random_drop_future: float = 0.
    random_pos_offset: float = 0.
    only_right_obs: bool = False
    unified_input: bool = False
    individual_keys: bool = False
    task_state_only: bool = False
    use_delta_actions: bool = False
    # image augmentation
    random_image_aug: float = 0.0

    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.85, 1.15)
    saturation_range: tuple[float, float] = (0.8, 1.2)
    sharpness_range: tuple[float, float] = (0.8, 1.25)

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist_view", "face_view", "right_wrist_view")

    def __call__(self, data: dict) -> dict:
        delta_reference = None
        if self.use_delta_actions and not self.task_state_only:
            raw_state = np.asarray(data["state"])
            if raw_state.ndim == 1:
                delta_reference = raw_state.copy()
            elif raw_state.ndim == 2:
                if not 0 <= self.state_history_size < raw_state.shape[0]:
                    raise IndexError(
                        f"current_idx={self.state_history_size} is out of range for state shape {raw_state.shape}."
                    )
                delta_reference = raw_state[self.state_history_size].copy()
            else:
                raise ValueError(f"Expected state shape (D,) or (T,D), got {raw_state.shape}.")

        # raw_state = np.asarray(data["state"]).copy()
        if not self.task_state_only:
            model_state = data["state"]
            
            if model_state.ndim == 2:
                assert model_state.shape[0] == self.state_history_size + 1 + self.state_future_size
                model_state, master_mask = self._mask_states(model_state)
                model_state = transforms.pad_to_dim(model_state, self.action_dim)
                model_state[:, -1] = master_mask
            else:
                if random.random() < self.random_drop_master:
                    master_start = self.slave_state_dim
                    master_end = min(
                        self.slave_state_dim * 2,
                        model_state.shape[-1],
                    )
                    master_dim = master_end - master_start
                    if master_dim > 0:
                        model_state[master_start:master_end] = model_state[:master_dim]
                    model_state = transforms.pad_to_dim(model_state, self.action_dim)
                    model_state[-1] = 1.
                
            model_state = transforms.pad_to_dim(model_state, self.action_dim)
        else:
            # add 
            raw_state = np.asarray(data["state"]).copy()

            expected_raw_dim = PHYSICAL_STATE_DIM + TASK_STATE_DIM
            if raw_state.shape[-1] != expected_raw_dim:
                raise ValueError(
                    f"Expected raw state dim {expected_raw_dim} "
                    f"(28 physical + basket/col/row), got {raw_state.shape}."
                )

            if raw_state.ndim == 1:
                # raw_state: (31,)
                current_physical_state = raw_state[:PHYSICAL_STATE_DIM].copy()

                # Only basket / col / row are sent to Pi0.
                model_state = raw_state[
                    PHYSICAL_STATE_DIM:
                    PHYSICAL_STATE_DIM + TASK_STATE_DIM
                ].copy()

            elif raw_state.ndim == 2:
                # raw_state:
                # (history + current + future, 31)
                expected_sequence_length = (
                    self.state_history_size + 1 + self.state_future_size
                )
                if raw_state.shape[0] != expected_sequence_length:
                    raise ValueError(
                        f"Expected state sequence length {expected_sequence_length}, "
                        f"got {raw_state.shape[0]}."
                    )

                current_idx = self.state_history_size

                # Delta 永远以 current frame 的物理状态为参考。
                current_physical_state = raw_state[
                    current_idx, :PHYSICAL_STATE_DIM
                ].copy()

                # 每个 state token 只包含 basket / col / row。
                model_state = raw_state[
                    :,
                    PHYSICAL_STATE_DIM:
                    PHYSICAL_STATE_DIM + TASK_STATE_DIM
                ].copy()

            else:
                raise ValueError(
                    f"Expected raw state shape (31,) or (T,31), "
                    f"got {raw_state.shape}."
                )

            # Pi0Config.action_dim=32，因此将3维任务状态补到32维。
            model_state = transforms.pad_to_dim(
                model_state,
                self.action_dim,
            )
        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel].
            if img.shape[-1] != 3:
                output_image = einops.rearrange(img, "c h w -> h w c")
            else:
                output_image = img
            assert output_image.shape[-1] == 3, f"Image must have 3 channels, got {output_image.shape}."
            return output_image

        # data augmentation
        @staticmethod
        def _data_augmentation(image:np.ndarray, brightness:float, contrast:float, saturation:float, sharpness:float):
            image = np.asarray(image)
            if image.dtype != np.uint8:
                raise TypeError(f"Image type is not uint8, expected type of uint8, got {image.dtype}")
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected RGB image. Current image shape: {image.shape}")
            pil_image = Image.fromarray(image)
            # brightness enhancement
            pil_image = ImageEnhance.Brightness(
                pil_image
            ).enhance(brightness)
            pil_image = ImageEnhance.Contrast(
                pil_image
            ).enhance(contrast)
            pil_image = ImageEnhance.Color(
                pil_image
            ).enhance(saturation)
            pil_image = ImageEnhance.Sharpness(
                pil_image
            ).enhance(sharpness)
            return np.asarray(pil_image, dtype=np.uint8)


        # Convert images to uint8 and rearrange to (H,W,C) format
        for key in self.EXPECTED_CAMERAS:
            assert key in data['images'].keys(), f"Images must contain {key}."
            data['images'][key] = convert_image(data['images'][key])
        if (self.random_image_aug > 0) and (random.random() < self.random_image_aug):
            brightness = random.uniform(*self.brightness_range)
            contrast = random.uniform(*self.contrast_range)
            saturation = random.uniform(*self.saturation_range)
            sharpness = random.uniform(*self.sharpness_range)
            for key in self.EXPECTED_CAMERAS:
                data['images'][key] = _data_augmentation(
                    data['images'][key],
                    brightness,
                    contrast,
                    saturation,
                    sharpness
                )

        inputs = {
            "image": {
                "base_0_rgb": data['images']['face_view'],
                "left_wrist_0_rgb": data['images']['left_wrist_view'],
                "right_wrist_0_rgb": data['images']['right_wrist_view'],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": model_state
        }
        if self.task_state_only:
            inputs["delta_reference"] = current_physical_state
        elif delta_reference is not None:
            inputs["delta_reference"] = delta_reference
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = data["actions_is_pad"]

        # random position offset augmentation
        if (
            self.random_pos_offset > 0.0
            and not self.task_state_only
            and "actions" in inputs
        ):
            pos_offset = (np.random.rand(3) * 2 - 1.) * self.random_pos_offset
            inputs["state"][..., 7:10] += pos_offset
            inputs["actions"][..., 7:10] += pos_offset
            inputs["state"][..., 0:3] += pos_offset
            inputs["actions"][..., 0:3] += pos_offset
            if delta_reference is not None:
                inputs["delta_reference"][7:10] += pos_offset
                inputs["delta_reference"][0:3] += pos_offset
            if self.mode in ["sm2m", "sm2sm"]:
                inputs["state"][..., 21:24] += pos_offset
                inputs["state"][..., 14:17] += pos_offset
                if delta_reference is not None:
                    inputs["delta_reference"][21:24] += pos_offset
                    inputs["delta_reference"][14:17] += pos_offset
            if self.mode == "sm2sm":
                inputs["actions"][..., 21:24] += pos_offset
                inputs["actions"][..., 14:17] += pos_offset
        
        if self.only_right_obs:
            inputs["image_mask"]["base_0_rgb"] = np.False_
            inputs["image_mask"]["left_wrist_0_rgb"] = np.False_

            if self.slave_state_dim == 14:
                # task-only时不能修改basket/col/row
                if not self.task_state_only:
                    inputs["state"][..., :7] = 0.0
                    if delta_reference is not None:
                        inputs["delta_reference"][:7] = 0.0

                    if self.mode in ["sm2m", "sm2sm"]:
                        inputs["state"][..., 14:21] = 0.0
                        if delta_reference is not None:
                            inputs["delta_reference"][14:21] = 0.0

                if "actions" in inputs:
                    inputs["actions"][..., :7] = 0.0

                    if self.mode == "sm2sm":
                        inputs["actions"][..., 14:21] = 0.0
                    
        return inputs

    def _mask_states(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mask future slave states by copying current slave state."""
        state = np.asarray(state).copy()
        current_idx = self.state_history_size
        current_slave = state[current_idx, :self.slave_state_dim]
        current_state = state[current_idx]
        if state.shape[-1] == 32:
            master_mask = state[:, -1]
        else:
            master_mask = np.zeros((state.shape[0],), dtype=state.dtype)

        if self.state_future_size > 0:  # always mask future slave states
            state[current_idx + 1:, :self.slave_state_dim] = current_slave
        if self.mask_history_slave_states and self.state_history_size > 0:
            state[:current_idx, :self.slave_state_dim] = current_slave

        # Data augmentation: randomly drop history/future/master states
        if random.random() < self.random_drop_master:
            master_start = self.slave_state_dim
            master_end = min(
                self.slave_state_dim * 2,
                state.shape[-1],
            )
            master_dim = master_end - master_start

            if master_dim > 0:
                state[
                    :, master_start:master_end
                ] = current_slave[:master_dim]

            master_mask[:] = 1.0
        if random.random() < self.random_drop_history and self.state_history_size > 0:
            state[:current_idx] = current_state
            master_mask[:current_idx] = 1.
        if random.random() < self.random_drop_future and self.state_future_size > 0:
            mask_size = random.randint(1, self.state_future_size)
            state[-mask_size:] = state[-mask_size - 1]
            master_mask[-mask_size:] = 1.
        
        return state, master_mask

@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Outputs for the ARX policy."""
    action_dim: int = 14
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :self.action_dim])
        return {"actions": actions}
