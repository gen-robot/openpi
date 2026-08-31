from pathlib import Path

import numpy as np

from openpi import transforms
from openpi.policies import robodojo_arx_x5_policy
from openpi.training import config


def test_robodojo_arx_x5_data_config_wires_transforms(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        config,
        "ModelTransformFactory",
        lambda default_prompt=None: lambda model_config: transforms.Group(),
    )
    factory = config.LeRobotRoboDojoArxX5DataConfig(
        repo_id="robodojo-test",
        assets=config.AssetsConfig(assets_dir=str(tmp_path)),
    )

    data_config = factory.create(tmp_path, config.pi0_config.Pi0Config(pi05=True))

    assert data_config.repo_id == "robodojo-test"
    assert data_config.prompt_from_task is True
    assert data_config.action_sequence_keys == ("action",)
    assert [type(item) for item in data_config.data_transforms.inputs] == [
        robodojo_arx_x5_policy.RoboDojoArxX5Inputs,
        transforms.DeltaActions,
    ]
    assert [type(item) for item in data_config.data_transforms.outputs] == [
        transforms.AbsoluteActions,
        robodojo_arx_x5_policy.RoboDojoArxX5Outputs,
    ]


def test_robodojo_arx_x5_repack_and_delta_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        config,
        "ModelTransformFactory",
        lambda default_prompt=None: lambda model_config: transforms.Group(),
    )
    factory = config.LeRobotRoboDojoArxX5DataConfig(
        repo_id="robodojo-test",
        assets=config.AssetsConfig(assets_dir=str(tmp_path)),
    )
    data_config = factory.create(tmp_path, config.pi0_config.Pi0Config(pi05=True))

    state = np.arange(14, dtype=np.float32)
    absolute_actions = np.stack((state + 1.0, state + 2.0))
    expected_absolute_actions = absolute_actions.copy()
    sample = {
        "observation.images.cam_high": np.zeros((3, 8, 10), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.zeros((3, 8, 10), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.zeros((3, 8, 10), dtype=np.uint8),
        "observation.state": state,
        "action": absolute_actions,
        "prompt": "Stack the three bowls together.",
    }

    repacked = transforms.compose(data_config.repack_transforms.inputs)(sample)
    model_input = transforms.compose(data_config.data_transforms.inputs)(repacked)

    arm_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    np.testing.assert_array_equal(
        model_input["actions"][:, arm_indices],
        expected_absolute_actions[:, arm_indices] - state[arm_indices],
    )
    np.testing.assert_array_equal(
        model_input["actions"][:, [6, 13]], expected_absolute_actions[:, [6, 13]]
    )

    padded = transforms.PadStatesAndActions(32)(model_input)
    restored = transforms.compose(data_config.data_transforms.outputs)(
        {"state": padded["state"], "actions": padded["actions"]}
    )
    np.testing.assert_array_equal(restored["actions"], expected_absolute_actions)


def test_robodojo_arx_x5_train_config_is_registered() -> None:
    train_config = config.get_config("pi05_robodojo_arx_x5_joint")

    assert train_config.model.pi05 is True
    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 50
    assert isinstance(train_config.data, config.LeRobotRoboDojoArxX5DataConfig)


def test_stack_bowls_seed1_config_uses_official_and_seed1_repos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        config,
        "ModelTransformFactory",
        lambda default_prompt=None: lambda model_config: transforms.Group(),
    )
    train_config = config.get_config("pi05_robodojo_stack_bowls_arx_x5_joint")
    repo_ids = [repo_id.strip() for repo_id in train_config.data.repo_id.split(",")]

    assert train_config.exp_name == "official100_dagger_s1"
    assert repo_ids == [
        "robodojo-stack_bowls-official-100ep",
        "robodojo-stack_bowls-dagger-20260830-seed1",
    ]

    data_config = train_config.data.create(tmp_path, train_config.model)
    assert data_config.asset_id == "_".join(repo_ids)


def test_stack_bowls_official_and_dagger_config_uses_both_repos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        config,
        "ModelTransformFactory",
        lambda default_prompt=None: lambda model_config: transforms.Group(),
    )
    train_config = config.get_config("pi05_robodojo_stack_bowls_official100_dagger56")
    repo_ids = [repo_id.strip() for repo_id in train_config.data.repo_id.split(",")]

    assert repo_ids == [
        "robodojo-stack_bowls-official-100ep-xjy-sft",
        "robodojo-stack_bowls-seed0-xjy-0814_225915-dagger",
    ]

    data_config = train_config.data.create(tmp_path, train_config.model)
    assert data_config.asset_id == "_".join(repo_ids)
