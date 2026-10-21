from __future__ import annotations

import json

import pytest

from src.report_mask_gaze_preexperiment import _downstream_results


def _write_result(path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": {"docvqa_val": {"anls,none": value}}}), encoding="utf-8")


def test_downstream_results_deduplicates_identical_retries(tmp_path):
    root = tmp_path / "downstream_evaluation" / "A_original" / "budget_30" / "model"
    _write_result(root / "first_results.json", 0.9)
    _write_result(root / "retry_results.json", 0.9)

    rows = _downstream_results(tmp_path)

    assert len(rows) == 1
    assert rows[0]["condition"] == "A_original"
    assert rows[0]["budget"] == 30
    assert rows[0]["anls"] == 0.9
    assert len(rows[0]["duplicate_paths"]) == 1


def test_downstream_results_rejects_conflicting_retries(tmp_path):
    root = tmp_path / "downstream_evaluation" / "A_original" / "budget_30" / "model"
    _write_result(root / "first_results.json", 0.9)
    _write_result(root / "retry_results.json", 0.8)

    with pytest.raises(ValueError, match="conflicting duplicate downstream results"):
        _downstream_results(tmp_path)
