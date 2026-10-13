import numpy as np

from src.token_labels.mask_to_tokens import assign_token_labels, mask_to_token_coverage


def test_mask_to_tokens_exact_quadrants_and_length():
    mask = np.zeros((56, 56), dtype=bool)
    mask[:28, :28] = True
    coverage = mask_to_token_coverage(mask, [1, 4, 4], patch_size=14, spatial_merge_size=2)
    assert coverage.shape == (4,)
    np.testing.assert_allclose(coverage, [1.0, 0.0, 0.0, 0.0])


def test_four_way_labels_include_boundary_ignore():
    subject = np.array([0.0, 1.0, 1.0, 0.4], dtype=np.float32)
    detail = [np.array([0.0, 0.0, 0.8, 0.0], dtype=np.float32)]
    config = {
        "detail_label": 2,
        "foreground_label": 1,
        "background_label": 0,
        "ignore_label": -1,
        "detail_min_coverage": 0.5,
        "foreground_min_coverage": 0.5,
        "background_max_subject_coverage": 0.05,
        "conflict_min_coverage": 0.2,
    }
    labels, confidence, combined = assign_token_labels(subject, detail, config)
    assert labels.tolist() == [0, 1, 2, -1]
    assert len(confidence) == len(combined) == 4


def test_dynamic_grid_not_assumed_fixed():
    mask = np.ones((101, 203), dtype=bool)
    coverage = mask_to_token_coverage(mask, [1, 6, 10], patch_size=14, spatial_merge_size=2)
    assert coverage.shape == (15,)
    np.testing.assert_allclose(coverage, 1.0)

