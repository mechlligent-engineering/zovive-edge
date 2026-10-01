"""Stage 2: species classification, run only on the crops of a
gate-passed track (never on every frame — the classifier is comparatively
expensive and only needs to run once per event).

Classifies up to `max_crops_per_track` of the best crops
(pipeline/best_snapshot.py) and combines their per-crop results across
multiple frames rather than trusting a single one — this directly
targets the Tiger->Leopard/Elephant confusion from single ambiguous
frames noted in the field report. Below `min_confidence`, reports
"unknown_animal" with the snapshot instead of guessing wrong.

Two combination strategies (`strategy=` / `configs/inference_config.yaml`
`classifier.voting_strategy`), both operating on the same per-crop
`ClassificationResult` list so neither duplicates the classifier call:

- "average" (default, original batch-1 behavior): average the softmax
  score vectors across crops, then take the argmax of the average.
  Smooths out one noisy frame's score vector against the others.
- "majority": each crop casts one vote for its own argmax class; the
  class with the most votes wins, ties broken by summed confidence.
  Reported confidence is the mean confidence of only the crops that
  voted for the winning class. Closer to "5 crops each say Tiger,
  1 says Leopard -> Tiger" than a softmax blend.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from inference.base import ClassificationResult, Classifier
from pipeline.best_snapshot import ScoredCrop

UNKNOWN_ANIMAL = "unknown_animal"
VALID_STRATEGIES = ("average", "majority")


@dataclass
class VerifiedResult:
    species: str
    confidence: float
    crops_used: int
    per_crop_results: list[ClassificationResult] = field(default_factory=list)
    is_unknown: bool = False
    strategy: str = "average"


def _combine_average(valid: list[ClassificationResult], class_names: list[str]) -> tuple[str, float]:
    stacked = np.array([r.all_scores for r in valid])
    avg_scores = stacked.mean(axis=0)
    best_id = int(np.argmax(avg_scores))
    best_conf = float(avg_scores[best_id])
    class_name = class_names[best_id] if best_id < len(class_names) else valid[0].class_name
    return class_name, best_conf


def _combine_majority(valid: list[ClassificationResult], class_names: list[str]) -> tuple[str, float]:
    votes = Counter(r.class_id for r in valid)
    # Tie-break by summed confidence among tied classes, not insertion order.
    top_count = max(votes.values())
    tied = [cid for cid, n in votes.items() if n == top_count]
    if len(tied) > 1:
        summed = {cid: sum(r.confidence for r in valid if r.class_id == cid) for cid in tied}
        best_id = max(tied, key=lambda cid: summed[cid])
    else:
        best_id = tied[0]

    winners = [r for r in valid if r.class_id == best_id]
    mean_conf = float(np.mean([r.confidence for r in winners]))
    class_name = class_names[best_id] if best_id < len(class_names) else winners[0].class_name
    return class_name, mean_conf


_STRATEGIES = {"average": _combine_average, "majority": _combine_majority}


def verify_track(
    classifier: Classifier,
    crops: list[ScoredCrop],
    min_confidence: float = 0.55,
    max_crops: int = 5,
    strategy: str = "average",
) -> VerifiedResult:
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown voting strategy {strategy!r}; expected one of {VALID_STRATEGIES}")

    if not crops:
        return VerifiedResult(
            species=UNKNOWN_ANIMAL, confidence=0.0, crops_used=0, is_unknown=True, strategy=strategy
        )

    used = crops[:max_crops]
    per_crop = [classifier.infer(c.crop) for c in used]
    valid = [r for r in per_crop if r.class_id >= 0 and r.all_scores]

    if not valid:
        return VerifiedResult(
            species=UNKNOWN_ANIMAL,
            confidence=0.0,
            crops_used=0,
            per_crop_results=per_crop,
            is_unknown=True,
            strategy=strategy,
        )

    class_name, confidence = _STRATEGIES[strategy](valid, classifier.class_names)

    if confidence < min_confidence:
        return VerifiedResult(
            species=UNKNOWN_ANIMAL,
            confidence=confidence,
            crops_used=len(valid),
            per_crop_results=per_crop,
            is_unknown=True,
            strategy=strategy,
        )

    return VerifiedResult(
        species=class_name,
        confidence=confidence,
        crops_used=len(valid),
        per_crop_results=per_crop,
        is_unknown=False,
        strategy=strategy,
    )
