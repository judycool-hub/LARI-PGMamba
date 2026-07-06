#!/usr/bin/env python
"""Recalculate verification metrics from released raw-score CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from metrics import eer, far_frr


def parse_label(row: Dict[str, str]) -> int:
    label_id = row.get("label_id", "").strip()
    if label_id in {"0", "1"}:
        return int(label_id)

    label = row.get("label", "").strip().lower()
    if label in {"genuine", "genuine_probe", "genuine probe"}:
        return 0
    return 1


def parse_direction(row: Dict[str, str], default: str) -> bool:
    direction = row.get("score_direction", default).strip().lower()
    return direction != "larger_is_more_similar"


def summarize_score_file(path: Path, default_direction: str) -> Dict[str, object]:
    scores: List[float] = []
    labels: List[int] = []
    thresholds: List[float] = []
    smaller_is_more_similar = default_direction != "larger_is_more_similar"

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            scores.append(float(row["score"]))
            labels.append(parse_label(row))
            if row.get("threshold", "") != "":
                thresholds.append(float(row["threshold"]))
            smaller_is_more_similar = parse_direction(row, default_direction)

    if not scores:
        raise ValueError(f"No scores found in {path}")

    metric = eer(scores, labels, smaller_is_more_similar)
    released_threshold = thresholds[0] if thresholds else metric.threshold
    released_far, released_frr = far_frr(
        scores,
        labels,
        released_threshold,
        smaller_is_more_similar,
    )

    return {
        "file": path.name,
        "n_trials": len(scores),
        "n_genuine": sum(1 for label in labels if label == 0),
        "n_forgery": sum(1 for label in labels if label != 0),
        "score_direction": "smaller_is_more_similar"
        if smaller_is_more_similar
        else "larger_is_more_similar",
        "released_threshold": released_threshold,
        "released_FAR_percent": released_far * 100.0,
        "released_FRR_percent": released_frr * 100.0,
        "EER_percent": (metric.eer or 0.0) * 100.0,
        "EER_threshold": metric.threshold,
        "EER_FAR_percent": metric.far * 100.0,
        "EER_FRR_percent": metric.frr * 100.0,
    }


def write_summary(rows: List[Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "n_trials",
        "n_genuine",
        "n_forgery",
        "score_direction",
        "released_threshold",
        "released_FAR_percent",
        "released_FRR_percent",
        "EER_percent",
        "EER_threshold",
        "EER_FAR_percent",
        "EER_FRR_percent",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", default="scores")
    parser.add_argument("--output", default="results/score_metrics_summary.csv")
    parser.add_argument(
        "--score-direction",
        default="smaller_is_more_similar",
        choices=["smaller_is_more_similar", "larger_is_more_similar"],
    )
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    paths = sorted(scores_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {scores_dir}")

    rows = [summarize_score_file(path, args.score_direction) for path in paths]
    write_summary(rows, Path(args.output))

    for row in rows:
        print(
            f"{row['file']}: trials={row['n_trials']} "
            f"EER={row['EER_percent']:.4f}% "
            f"released FAR={row['released_FAR_percent']:.4f}% "
            f"released FRR={row['released_FRR_percent']:.4f}%"
        )
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
