from __future__ import annotations

from typing import Any


def combine_semantic_score(scores: dict[str, float], config: dict[str, Any]) -> float:
    return float(
        float(config["question_weight"]) * scores["question_similarity"]
        + float(config["answer_weight"]) * scores["answer_similarity"]
        + float(config["question_answer_weight"]) * scores["question_answer_similarity"]
    )


def semantic_prescreen(candidates: list[dict[str, Any]], maximum: int) -> list[int]:
    ranking = sorted(
        range(len(candidates)),
        key=lambda index: (-float(candidates[index]["semantic_score"]), candidates[index]["candidate_id"]),
    )
    return ranking[:maximum]

