import json
from pathlib import Path

import numpy as np

from src.audit_subject_mask_supervision import adaptive_mask_token_indices, recompute_mask_token_metrics
from src.masks import decode_rle, encode_rle


def test_all_one_mask_has_unit_coverage_and_matching_length():
    mask = np.ones((56, 84), dtype=bool)
    coverage, contribution = recompute_mask_token_metrics(mask, [1, 4, 6])
    assert coverage.shape == contribution.shape == (6,)
    np.testing.assert_allclose(coverage, 1.0)


def test_all_zero_mask_has_no_current_or_adaptive_positive():
    mask = np.zeros((56, 56), dtype=bool)
    coverage, contribution = recompute_mask_token_metrics(mask, [1, 4, 4])
    assert not np.flatnonzero(coverage >= 0.5).size
    assert not adaptive_mask_token_indices(contribution).size


def test_small_mask_gets_one_adaptive_token_below_half_coverage():
    mask = np.zeros((56, 56), dtype=bool)
    mask[2:6, 2:6] = True
    coverage, contribution = recompute_mask_token_metrics(mask, [1, 4, 4])
    assert coverage.max() < 0.5
    adaptive = adaptive_mask_token_indices(contribution)
    assert adaptive.size >= 1


def test_nonempty_mask_contribution_sums_to_one():
    mask = np.zeros((73, 119), dtype=bool)
    mask[5:41, 12:87] = True
    _, contribution = recompute_mask_token_metrics(mask, [1, 6, 10])
    np.testing.assert_allclose(contribution.sum(), 1.0, atol=1e-6)


def test_rle_round_trip_is_exact():
    rng = np.random.default_rng(42)
    mask = rng.random((31, 47)) > 0.8
    np.testing.assert_array_equal(decode_rle(encode_rle(mask)), mask)


def test_temporal_grid_length_and_contribution_are_valid():
    mask = np.ones((56, 56), dtype=bool)
    coverage, contribution = recompute_mask_token_metrics(mask, [2, 4, 4])
    assert coverage.shape == contribution.shape == (8,)
    np.testing.assert_allclose(contribution.sum(), 1.0, atol=1e-6)


def test_existing_sample_coverage_matches_independent_recompute():
    dataset_path = Path(__file__).resolve().parents[1] / "outputs/pilot_200/dataset.jsonl"
    with dataset_path.open("r", encoding="utf-8") as handle:
        sample = json.loads(handle.readline())
    mask = decode_rle(sample["subject_mask_rle"])
    coverage, contribution = recompute_mask_token_metrics(mask, sample["image_grid_thw"])
    expected_count = int(sample["visual_token_count"])
    assert coverage.size == contribution.size == expected_count
    existing = np.asarray(sample["subject_token_coverage"], dtype=np.float32)
    assert existing.size == expected_count
    assert float(np.max(np.abs(existing - coverage))) < 1e-6
