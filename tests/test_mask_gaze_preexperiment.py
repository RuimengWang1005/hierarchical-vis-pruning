from __future__ import annotations

import torch

from src.mask_gaze_preexperiment import original_gaze_bbox_labels, three_value_pairwise_margin_loss


def test_original_gaze_bbox_projection_is_inclusive_rectangle() -> None:
    labels = original_gaze_bbox_labels([[20, 10, 60, 50]], 100, 100, [1, 20, 20])
    grid = labels.reshape(10, 10)
    # int([.2,.1,.6,.5] * 10), inclusive end -> 5 rows x 5 columns.
    assert int(grid.sum()) == 25
    assert bool(grid[1, 2])
    assert bool(grid[5, 6])
    assert not bool(grid[0, 2])
    assert not bool(grid[5, 7])


def test_three_value_loss_ignores_minus_one() -> None:
    scores = torch.tensor([0.5, -100.0, 0.1], requires_grad=True)
    labels = torch.tensor([1, -1, 0])
    actual = three_value_pairwise_margin_loss(scores, labels, margin=0.2)
    expected = torch.nn.functional.softplus(torch.tensor(-0.2))
    assert actual is not None
    assert torch.allclose(actual, expected)
    actual.backward()
    assert scores.grad is not None
    assert scores.grad[1].item() == 0.0


def test_three_value_loss_skips_missing_class() -> None:
    assert three_value_pairwise_margin_loss(torch.tensor([1.0]), torch.tensor([1])) is None
    assert three_value_pairwise_margin_loss(torch.tensor([1.0]), torch.tensor([0])) is None
