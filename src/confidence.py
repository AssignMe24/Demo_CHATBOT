from __future__ import annotations

import math

ANSWERED_SUPPORTED = "ANSWERED_SUPPORTED"
ANSWERED_PARTIAL = "ANSWERED_PARTIAL"
REFUSED_INSUFFICIENT_EVIDENCE = "REFUSED_INSUFFICIENT_EVIDENCE"
RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
GENERATION_FAILURE = "GENERATION_FAILURE"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def sigmoid_score(raw: float) -> float:
    return _clamp(1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, float(raw))))))


def is_confident(scores, mode: str = "hybrid", abs_threshold: float | None = None, gap_threshold: float = 0.04) -> bool:
    if len(scores) == 0:
        return False
    threshold = 0.28 if abs_threshold is None else float(abs_threshold)
    best = float(scores[0])
    if best < threshold:
        return False
    if gap_threshold > 0 and len(scores) >= 2 and (best - float(scores[1])) < gap_threshold and best < 0.55:
        return False
    return True


def rerank_is_confident(rerank_scores, floor: float = -1.5) -> bool:
    return bool(rerank_scores and max(float(s) for s in rerank_scores) >= float(floor))


def evidence_confidence(
    *, vector_score: float = 0.0, rrf_score: float = 0.0, rerank_score: float = -8.0,
    keyword_coverage: float = 0.0, source_consistency: float = 0.0,
    answer_support: float = 0.0, heading_match: float = 0.0,
) -> float:
    vector = _clamp((float(vector_score) + 1.0) / 2.0)
    return _clamp(
        0.24 * sigmoid_score(rerank_score)
        + 0.18 * vector
        + 0.16 * _clamp(rrf_score)
        + 0.14 * _clamp(keyword_coverage)
        + 0.10 * _clamp(heading_match)
        + 0.08 * _clamp(source_consistency)
        + 0.10 * _clamp(answer_support)
    )


def confidence_label(score: float) -> str:
    score = _clamp(score)
    if score >= 0.82: return "Very High"
    if score >= 0.68: return "High"
    if score >= 0.52: return "Medium"
    if score >= 0.36: return "Low"
    return "Very Low"


def confidence_color(score: float) -> str:
    score = _clamp(score)
    if score >= 0.82: return "#22c55e"
    if score >= 0.68: return "#84cc16"
    if score >= 0.52: return "#f59e0b"
    if score >= 0.36: return "#f97316"
    return "#ef4444"
