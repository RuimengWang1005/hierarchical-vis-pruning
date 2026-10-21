import numpy as np

from src.generate_mask_gaze_1k import (
    adaptive_indices,
    make_labels,
    token_coverage_and_contribution,
    union_bbox_mask,
)


def test_contribution_sums_to_one_and_matches_visual_token_count():
    mask = np.zeros((73, 119), dtype=bool)
    mask[5:41, 12:87] = True
    coverage, contribution = token_coverage_and_contribution(
        mask, [1, 6, 10], patch_size=14, spatial_merge_size=2
    )
    assert coverage.size == contribution.size == 15
    np.testing.assert_allclose(contribution.sum(), 1.0, atol=1e-6)


def test_adaptive_mapping_reaches_target_and_is_nonempty_for_tiny_mask():
    mask = np.zeros((56, 56), dtype=bool)
    mask[2:4, 2:4] = True
    _, contribution = token_coverage_and_contribution(
        mask, [1, 4, 4], patch_size=14, spatial_merge_size=2
    )
    selected = adaptive_indices(contribution, 0.90)
    assert selected.size >= 1
    assert contribution[selected].sum() >= 0.90


def test_adaptive_mapping_uses_minimum_prefix():
    contribution = np.asarray([0.40, 0.30, 0.22, 0.06, 0.02], dtype=np.float32)
    selected = adaptive_indices(contribution, 0.90)
    np.testing.assert_array_equal(selected, [0, 1, 2])
    assert contribution[selected[:-1]].sum() < 0.90


def test_foreground_label_has_priority_over_background():
    labels = make_labels(np.asarray([2, 4]), np.asarray([0, 2, 3]), 6)
    np.testing.assert_array_equal(labels, [0, -1, 1, 0, 1, -1])


def test_union_bbox_mask_handles_multiple_boxes():
    result = union_bbox_mask((10, 10), [[1, 1, 4, 4], [6, 6, 9, 9]])
    assert int(result.sum()) == 18
    assert result[2, 2]
    assert result[7, 7]
