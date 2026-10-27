"""Stage 2: the species decision for a gate-passed track, taken from the
detector's own per-frame classes. There is no second model: the 7-class
detector already names the species, and the former crop classifier was
removed because on the clean held-out test boxes it overrode correct
detector species far more often than it fixed them (zovive-ml
reports/detector_classifier_fusion_report.json).

What stays from the classifier design is not trusting a single frame.
The tracker matches boxes by overlap only, so a track's per-frame class
can flip (Tiger -> Leopard on one ambiguous frame). The track's recent
(class_name, score) votes (Track.class_votes, inference/bytetrack_wrapper.py)
are combined, and below `min_confidence` the result is "unknown_animal"
with the snapshot instead of a possibly wrong species.

Two combination strategies (`configs/inference_config.yaml`
`stage2.voting_strategy`):

- "average" (default): each species' summed score divided by the number
  of frames; the highest wins. A frame that voted for another species
  counts as 0, so a track that flip-flops gets a lower confidence and is
  more likely to end as unknown_animal.
- "majority": each frame votes for its species; most votes wins, ties
  broken by summed score. Reported confidence is the mean score of only
  the frames that voted for the winner.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

UNKNOWN_ANIMAL = "unknown_animal"
VALID_STRATEGIES = ("average", "majority")


@dataclass
class VerifiedResult:
    species: str
    confidence: float
    frames_used: int
    is_unknown: bool = False
    strategy: str = "average"


def _combine_average(votes: Sequence[tuple[str, float]]) -> tuple[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for name, score in votes:
        totals[name] += score
    best = max(totals, key=totals.__getitem__)  # a tie goes to the species seen first
    return best, totals[best] / len(votes)


def _combine_majority(votes: Sequence[tuple[str, float]]) -> tuple[str, float]:
    counts = Counter(name for name, _ in votes)
    top_count = max(counts.values())
    # Tie-break by summed score among tied species, not insertion order.
    tied = [name for name, n in counts.items() if n == top_count]
    if len(tied) > 1:
        summed = {name: sum(s for n, s in votes if n == name) for name in tied}
        best = max(tied, key=summed.__getitem__)
    else:
        best = tied[0]
    scores = [s for n, s in votes if n == best]
    return best, sum(scores) / len(scores)


_STRATEGIES = {"average": _combine_average, "majority": _combine_majority}


def verify_track(
    votes: Sequence[tuple[str, float]],
    min_confidence: float = 0.55,
    strategy: str = "average",
) -> VerifiedResult:
    """Combine a track's per-frame detector (class_name, score) votes into one species."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown voting strategy {strategy!r}; expected one of {VALID_STRATEGIES}")

    if not votes:
        return VerifiedResult(
            species=UNKNOWN_ANIMAL, confidence=0.0, frames_used=0, is_unknown=True, strategy=strategy
        )

    species, confidence = _STRATEGIES[strategy](votes)
    unknown = confidence < min_confidence
    return VerifiedResult(
        species=UNKNOWN_ANIMAL if unknown else species,
        confidence=confidence,
        frames_used=len(votes),
        is_unknown=unknown,
        strategy=strategy,
    )
