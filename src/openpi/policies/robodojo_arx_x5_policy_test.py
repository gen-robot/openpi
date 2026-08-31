import numpy as np
import pytest

from openpi import transforms
from openpi.policies import robodojo_arx_x5_policy


def test_inputs_preserve_robot_values_and_map_cameras() -> None:
    data = robodojo_arx_x5_policy.make_robodojo_arx_x5_example()
    data["state"] = np.arange(14, dtype=np.float32)
    data["actions"] = np.arange(42, dtype=np.float32).reshape(3, 14)

    result = robodojo_arx_x5_policy.RoboDojoArxX5Inputs()(data)

    np.testing.assert_array_equal(result["state"], data["state"])
    np.testing.assert_array_equal(result["actions"], data["actions"])
    assert result["image"]["base_0_rgb"].shape == (480, 640, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (480, 640, 3)
    assert result["image"]["right_wrist_0_rgb"].shape == (480, 640, 3)
    assert all(bool(value) for value in result["image_mask"].values())


def test_inputs_accept_hwc_images_without_transposing() -> None:
    data = robodojo_arx_x5_policy.make_robodojo_arx_x5_example()
    data["images"] = {
        name: np.full((480, 640, 3), index, dtype=np.uint8)
        for index, name in enumerate(robodojo_arx_x5_policy.RoboDojoArxX5Inputs.EXPECTED_CAMERAS)
    }

    result = robodojo_arx_x5_policy.RoboDojoArxX5Inputs()(data)

    assert result["image"]["base_0_rgb"][0, 0, 0] == 0
    assert result["image"]["left_wrist_0_rgb"][0, 0, 0] == 1
    assert result["image"]["right_wrist_0_rgb"][0, 0, 0] == 2


def test_inputs_reject_wrong_state_dimension() -> None:
    data = robodojo_arx_x5_policy.make_robodojo_arx_x5_example()
    data["state"] = np.zeros((13,), dtype=np.float32)

    with pytest.raises(ValueError, match="state must have shape"):
        robodojo_arx_x5_policy.RoboDojoArxX5Inputs()(data)


def test_outputs_strip_only_model_padding() -> None:
    model_actions = np.arange(96, dtype=np.float32).reshape(3, 32)

    result = robodojo_arx_x5_policy.RoboDojoArxX5Outputs()({"actions": model_actions})

    np.testing.assert_array_equal(result["actions"], model_actions[:, :14])


def test_joint_delta_round_trip_keeps_grippers_absolute() -> None:
    state = np.arange(14, dtype=np.float32)
    absolute = np.stack((state + 1.0, state + 2.0))
    arm_joint_mask = transforms.make_bool_mask(6, -1, 6, -1)

    delta = transforms.DeltaActions(arm_joint_mask)({"state": state.copy(), "actions": absolute.copy()})

    np.testing.assert_array_equal(delta["actions"][:, 6], absolute[:, 6])
    np.testing.assert_array_equal(delta["actions"][:, 13], absolute[:, 13])

    restored = transforms.AbsoluteActions(arm_joint_mask)(delta)
    np.testing.assert_array_equal(restored["actions"], absolute)
