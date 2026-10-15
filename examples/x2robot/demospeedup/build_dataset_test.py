import numpy as np

from . import build_dataset


def test_stride_keep_mask_preserves_transitions_and_endpoints():
    precision = np.array([False] * 5 + [True] * 5)
    keep = build_dataset.stride_keep_mask(
        precision,
        precision_stride=2,
        casual_stride=4,
        transition_boundary=1,
    )
    assert keep[0]
    assert keep[-1]
    assert np.all(keep[4:7])
    assert np.max(np.diff(np.flatnonzero(keep)[np.flatnonzero(keep) < 5])) <= 4


def test_motion_limits_add_intermediate_frames():
    actions = np.zeros((5, 14), dtype=np.float64)
    actions[:, 0] = np.arange(5) * 0.02
    keep = np.array([True, False, False, False, True])
    limited, unresolved = build_dataset.enforce_motion_limits(
        keep,
        actions,
        max_position_step=0.03,
        max_rotation_step=None,
        max_gripper_step=None,
    )
    assert np.all(limited)
    assert unresolved == 0
