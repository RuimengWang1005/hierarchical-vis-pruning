from __future__ import annotations

import json
from pathlib import Path

import torch

from src.mask_vis_online import (
    expected_token_count,
    make_scorer,
    parameter_count,
    pairwise_margin_loss,
    prepare_valid_manifest,
    scan_for_feature_cache,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_chunked_pairwise_loss_is_exact_and_ignores_minus_one() -> None:
    scores = torch.tensor([1.2, -0.4, 0.7, 3.0, -1.1], requires_grad=True)
    labels = torch.tensor([1, 0, 1, -1, 0], dtype=torch.int8)
    positives = scores[torch.tensor([0, 2])]
    negatives = scores[torch.tensor([1, 4])]
    expected = torch.nn.functional.softplus(-(positives[:, None] - negatives[None, :] - 0.2)).mean()
    actual = pairwise_margin_loss(scores, labels, margin=0.2, chunk_size=1)
    assert actual is not None
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert scores.grad is not None
    assert scores.grad[3].item() == 0.0


def test_pairwise_loss_skips_missing_class() -> None:
    assert pairwise_margin_loss(torch.ones(3), torch.tensor([1, 1, -1]), 0.2) is None
    assert pairwise_margin_loss(torch.ones(3), torch.tensor([0, 0, -1]), 0.2) is None


def test_expected_token_count() -> None:
    assert expected_token_count([1, 40, 60]) == 600


def test_original_gaze_scorer_structure_and_seed_are_reproducible() -> None:
    first = make_scorer(42)
    second = make_scorer(42)
    assert parameter_count(first) == 12_848_640
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_prepare_manifest_uses_union_of_exclusions(tmp_path: Path) -> None:
    manifests = []
    masks = []
    tokens = []
    for index in range(20_000):
        sample_id = f"s{index}"
        image = f"/images/{index}.jpg"
        manifests.append({"sample_id": sample_id, "image_path": image, "source": "toy"})
        masks.append({"sample_id": sample_id, "fallback_to_bbox": index in {0, 3}})
        if index == 1:
            labels = [0]
        elif index in {2, 3}:
            labels = [1]
        else:
            labels = [1, 0]
        tokens.append(
            {
                "sample_id": sample_id,
                "image_grid_thw": [1, 2, 2] if len(labels) == 1 else [1, 2, 4],
                "visual_token_count": len(labels),
                "mask_token_labels": labels,
                "mask_positive_count": labels.count(1),
                "reliable_negative_count": labels.count(0),
            }
        )
    manifest_path = tmp_path / "manifest.jsonl"
    masks_path = tmp_path / "masks.jsonl"
    tokens_path = tmp_path / "tokens.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    pilot_path = tmp_path / "pilot.jsonl"
    output = tmp_path / "valid.jsonl"
    _write_jsonl(manifest_path, manifests)
    _write_jsonl(masks_path, masks)
    _write_jsonl(tokens_path, tokens)
    _write_jsonl(failures_path, [])
    _write_jsonl(pilot_path, [])
    config = {
        "paths": {
            "manifest_20k": str(manifest_path),
            "subject_masks_20k": str(masks_path),
            "token_supervision_20k": str(tokens_path),
            "failures_20k": str(failures_path),
            "pilot_manifest": str(pilot_path),
            "valid_training_manifest": str(output),
        }
    }
    summary = prepare_valid_manifest(config)
    assert summary["fallback_to_bbox"] == 2
    assert summary["zero_positive"] == 1
    assert summary["zero_reliable_negative"] == 2
    assert summary["excluded_union"] == 4
    assert summary["valid"] == 19_996
    assert sum(1 for _ in output.open()) == 19_996


def test_feature_cache_scanner_allows_only_training_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "initial_scorer.safetensors").touch()
    (tmp_path / "scorer_epoch_1.safetensors").touch()
    (tmp_path / "checkpoint_last.pt").touch()
    assert scan_for_feature_cache(tmp_path) == []
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (feature_dir / "sample.safetensors").touch()
    assert scan_for_feature_cache(tmp_path) == ["features/sample.safetensors"]

