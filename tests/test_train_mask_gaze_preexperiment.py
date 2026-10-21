from __future__ import annotations

import torch

from src.train_mask_gaze_preexperiment import EfficientTransformerScorer


def test_efficient_scorer_matches_quadratic_definition() -> None:
    torch.manual_seed(7)
    scorer = EfficientTransformerScorer(in_features=8, hidden_dim=4)
    features = torch.randn(9, 8)
    keys = scorer.k_proj(features)
    queries = scorer.q_proj(features)
    reference = (queries @ keys.T / (4**0.5)).mean(dim=-1)
    assert torch.allclose(scorer(features), reference, rtol=1e-5, atol=1e-6)
