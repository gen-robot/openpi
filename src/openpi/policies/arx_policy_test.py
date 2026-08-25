import numpy as np

from openpi.policies import arx_policy


def test_delta_actions_use_state_before_master_dropout():
    slave_state = np.arange(14, dtype=np.float32) + 1
    master_state = np.arange(14, dtype=np.float32) + 101
    current_state = np.concatenate([slave_state, master_state])
    state = np.stack([current_state - 1, current_state])
    actions = np.stack([current_state + 0.5, current_state + 1.5])
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    data = {
        "state": state,
        "actions": actions,
        "images": {
            "left_wrist_view": image,
            "face_view": image,
            "right_wrist_view": image,
        },
    }

    inputs = arx_policy.ArxInputs(
        mode="sm2sm",
        action_dim=32,
        state_history_size=1,
        slave_state_dim=14,
        random_drop_master=1.0,
        random_pos_offset=0.1,
        use_delta_actions=True,
    )(data)

    # Master state is hidden from the model, while delta_reference retains the
    # real current master state. Position offsets are applied to both actions
    # and the reference, so they cancel when forming delta actions.
    np.testing.assert_allclose(inputs["state"][1, 14:17], inputs["state"][1, :3])
    assert not np.allclose(inputs["delta_reference"][14:17], inputs["state"][1, 14:17])

    transformed = arx_policy.ArxDeltaActions(mask=(True,) * 28, current_idx=1)(inputs)

    np.testing.assert_allclose(transformed["actions"][0, :28], 0.5, atol=1e-6)
    np.testing.assert_allclose(transformed["actions"][1, :28], 1.5, atol=1e-6)
    assert "delta_reference" not in transformed


def test_delta_reference_is_removed_during_inference():
    data = {"state": np.ones(4), "delta_reference": np.ones(4)}

    transformed = arx_policy.ArxDeltaActions(mask=(True,) * 4)(data)

    assert "delta_reference" not in transformed
