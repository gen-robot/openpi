import numpy as np

from . import entropy


def test_interpolate_outliers_preserves_alignment():
    values = np.array([0.0, 1.0, 100.0, 3.0, 4.0])
    cleaned = entropy.interpolate_outliers(values, np.array([False, False, True, False, False]))
    np.testing.assert_allclose(cleaned, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cleaned.shape == values.shape


def test_temporal_action_entropy_aligns_overlapping_chunks():
    # Frame 1 is supported by predictions from queries 0 and 1. Their disagreement
    # must make its entropy larger than frame 0, whose samples all agree.
    chunks = np.zeros((2, 2, 2, 1), dtype=np.float32)
    chunks[0, :, 1, 0] = -2.0
    chunks[1, :, 0, 0] = 2.0
    result = entropy.temporal_action_entropy(chunks, action_dims=1, estimator="std")
    assert result[0] == 0.0
    assert result[1] > result[0]


def test_segment_precision_uses_low_entropy_semantics(monkeypatch):
    values = np.concatenate([np.full(20, 0.1), np.full(20, 0.9)])
    monkeypatch.setattr(
        entropy,
        "isolation_forest_clean",
        lambda values, **_: (np.asarray(values), np.zeros(len(values), dtype=bool)),
    )

    class FakeHDBSCAN:
        def __init__(self, **_):
            pass

        def fit_predict(self, features):
            return np.repeat([0, 1], len(features) // 2)

    import sys
    import types

    monkeypatch.setitem(sys.modules, "hdbscan", types.SimpleNamespace(HDBSCAN=FakeHDBSCAN))
    result = entropy.segment_precision(values, min_cluster_size=5, min_precision_run=1)
    assert np.all(result.precision[:20])
    assert not np.any(result.precision[20:])
    np.testing.assert_array_equal(result.casualness, ~result.precision)
