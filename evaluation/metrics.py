"""Metric utilities for distance-based signature verification scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    far: float
    frr: float
    eer: Optional[float] = None


def _is_accept(score: float, threshold: float, smaller_is_more_similar: bool) -> bool:
    if smaller_is_more_similar:
        return score <= threshold
    return score >= threshold


def far_frr(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    smaller_is_more_similar: bool = True,
) -> Tuple[float, float]:
    genuine = 0
    forgery = 0
    false_reject = 0
    false_accept = 0

    for score, label in zip(scores, labels):
        accept = _is_accept(score, threshold, smaller_is_more_similar)
        if label == 0:
            genuine += 1
            if not accept:
                false_reject += 1
        else:
            forgery += 1
            if accept:
                false_accept += 1

    far = false_accept / forgery if forgery else 0.0
    frr = false_reject / genuine if genuine else 0.0
    return far, frr


def eer(
    scores: Sequence[float],
    labels: Sequence[int],
    smaller_is_more_similar: bool = True,
) -> BinaryMetrics:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("at least one score is required")

    genuine_total = sum(1 for label in labels if label == 0)
    forgery_total = len(labels) - genuine_total
    pairs = sorted(
        ((float(score), int(label)) for score, label in zip(scores, labels)),
        reverse=not smaller_is_more_similar,
    )

    best: Optional[BinaryMetrics] = None
    accepted_genuine = 0
    accepted_forgery = 0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            if pairs[index][1] == 0:
                accepted_genuine += 1
            else:
                accepted_forgery += 1
            index += 1

        far = accepted_forgery / forgery_total if forgery_total else 0.0
        frr = (
            (genuine_total - accepted_genuine) / genuine_total
            if genuine_total
            else 0.0
        )
        candidate = BinaryMetrics(
            threshold=threshold,
            far=far,
            frr=frr,
            eer=(far + frr) / 2.0,
        )
        if best is None or abs(candidate.far - candidate.frr) < abs(best.far - best.frr):
            best = candidate

    if best is None:
        raise ValueError("at least one score is required")
    return best


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
