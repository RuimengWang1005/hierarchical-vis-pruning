import numpy as np

from src.token_labels.mask_to_tokens import assign_token_labels, mask_to_token_coverage


def test_mask_to_tokens_exact_quadrants_and_length():
    mask = np.zeros((56, 56), dtype=bool)
    mask[:28, :28] = True
    coverage = mask_to_token_coverage(mask, [1, 4, 4], patch_size=14, spatial_merge_size=2)
    assert coverage.shape == (4,)
    np.testing.assert_allclose(coverage, [1.0, 0.0, 0.0, 0.0])


def test_hard_labels_include_boundary_ignore_and_soft_targets():
    subject = np.array([0.0, 1.0, 1.0, 0.04], dtype=np.float32)
    detail = [np.array([0.0, 0.0, 0.8, 0.0], dtype=np.float32)]
    config = {
        "detail_label": 2,
        "foreground_label": 1,
        "background_label": 0,
        "ignore_label": -1,
        "detail_min_coverage": 0.2,
        "foreground_min_coverage": 0.1,
        "background_max_subject_coverage": 0.01,
        "ensure_subject_positive": True,
        "ensure_each_detail_positive": True,
    }
    labels, confidence, combined, soft_targets, diagnostics = assign_token_labels(subject, detail, config)
    assert labels.tolist() == [0, 1, 2, -1]
    assert len(confidence) == len(combined) == 4
    assert soft_targets.shape == (4, 3)
    np.testing.assert_allclose(soft_targets.sum(axis=1), 1.0)
    np.testing.assert_allclose(soft_targets[2], [0.0, 0.2, 0.8])
    assert not diagnostics["subject_positive_fallback"]


def test_positive_fallbacks_rescue_small_subject_and_detail():
    subject = np.array([0.0, 0.04, 0.08, 0.03], dtype=np.float32)
    details = [np.array([0.0, 0.0, 0.02, 0.0], dtype=np.float32)]
    config = {
        "detail_label": 2,
        "foreground_label": 1,
        "background_label": 0,
        "ignore_label": -1,
        "detail_min_coverage": 0.2,
        "foreground_min_coverage": 0.1,
        "background_max_subject_coverage": 0.01,
        "ensure_subject_positive": True,
        "ensure_each_detail_positive": True,
    }
    labels, confidence, _, soft_targets, diagnostics = assign_token_labels(subject, details, config)
    assert labels.tolist() == [0, -1, 2, -1]
    assert not diagnostics["subject_positive_fallback"]
    assert diagnostics["detail_positive_fallback_count"] == 1
    assert diagnostics["detail_mask_positive_indices"] == [2]
    assert confidence[2] == soft_targets[2, 2]


def test_subject_fallback_without_details():
    subject = np.array([0.0, 0.04, 0.08, 0.03], dtype=np.float32)
    config = {
        "detail_label": 2,
        "foreground_label": 1,
        "background_label": 0,
        "ignore_label": -1,
        "detail_min_coverage": 0.2,
        "foreground_min_coverage": 0.1,
        "background_max_subject_coverage": 0.01,
        "ensure_subject_positive": True,
        "ensure_each_detail_positive": True,
    }
    labels, _, _, _, diagnostics = assign_token_labels(subject, [], config)
    assert labels.tolist() == [0, -1, 1, -1]
    assert diagnostics["subject_positive_fallback"]


def test_dynamic_grid_not_assumed_fixed():
    mask = np.ones((101, 203), dtype=bool)
    coverage = mask_to_token_coverage(mask, [1, 6, 10], patch_size=14, spatial_merge_size=2)
    assert coverage.shape == (15,)
    np.testing.assert_allclose(coverage, 1.0)

