import numpy as np

from src.masks import bbox_mask, decode_rle, encode_rle
from src.sam_masks.candidate_generator import _allocate_points


def test_bbox_fallback_round_trip_rle():
    mask = bbox_mask((20, 30), [5.2, 4.1, 15.8, 12.9])
    recovered = decode_rle(encode_rle(mask))
    assert recovered.shape == (20, 30)
    assert np.array_equal(mask, recovered)
    assert recovered.sum() > 0


def test_point_allocation_never_exceeds_global_cap_with_many_instances():
    masks = []
    for index in range(30):
        mask = np.zeros((20, 20), dtype=bool)
        mask[: 1 + index % 19, : 1 + index % 19] = True
        masks.append(mask)
    allocation = _allocate_points(masks, 24)
    assert sum(allocation) == 24
    assert sum(value > 0 for value in allocation) == 24

