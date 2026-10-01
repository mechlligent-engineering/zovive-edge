"""SAHI (Slicing Aided Hyper Inference) for the detector.

Listed as deferred in the batch-1 README (`inference/sahi_batched`) —
this is that piece. It targets the same failure mode the field report
called out under "Wild Boar recall" and "small/far animals": a distant
animal that's only a few dozen pixels across gets shrunk past
recognizability by the detector's usual single letterbox-to-640 pass.
Slicing the frame into overlapping tiles and running the *same*
detector on each tile at its native resolution recovers that detail,
at the cost of several detector calls per frame instead of one.

This wraps `inference.base.Detector` rather than replacing it or
branching inside `onnx_inference.py`/`hailo_inference.py`: `SahiDetector`
takes any existing Detector instance (ONNX or Hailo, doesn't care
which) and implements the exact same interface, so
`edge_main.build_detector_and_classifier` can hand back a `SahiDetector`
in place of a plain one and nothing downstream (stage1_gate, the
tracker, stage2_verifier) has to know the difference. That's also why
this module doesn't do its own NMS logic from scratch: it reuses
cv2.dnn.NMSBoxes, the same primitive `inference/yolo_postprocess.py`
already uses to merge a single pass's raw anchors, applied here across
tiles' `Detection` outputs instead of raw anchors.

Off by default (`configs/inference_config.yaml` `detector.sahi.enabled`)
because it does not change behavior on the normal path — a frame that
already fits in one tile is passed straight to the wrapped detector
with zero extra calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from inference.base import Detection, Detector

log = logging.getLogger(__name__)


@dataclass
class SahiConfig:
    tile_size: int = 640
    overlap_ratio: float = 0.2
    run_full_frame_pass: bool = True
    merge_iou_threshold: float = 0.5

    @classmethod
    def from_config(cls, inf_cfg: dict) -> SahiConfig:
        sahi_cfg = (inf_cfg or {}).get("detector", {}).get("sahi") or (inf_cfg or {}).get("sahi") or {}
        return cls(
            tile_size=int(sahi_cfg.get("tile_size", 640)),
            overlap_ratio=float(sahi_cfg.get("overlap_ratio", 0.2)),
            run_full_frame_pass=bool(sahi_cfg.get("run_full_frame_pass", True)),
            merge_iou_threshold=float(sahi_cfg.get("merge_iou_threshold", 0.5)),
        )


def is_sahi_enabled(inf_cfg: dict) -> bool:
    sahi_cfg = (inf_cfg or {}).get("detector", {}).get("sahi") or (inf_cfg or {}).get("sahi") or {}
    return bool(sahi_cfg.get("enabled", False))


def compute_tiles(
    width: int, height: int, tile_size: int, overlap_ratio: float
) -> list[tuple[int, int, int, int]]:
    """Covers the frame with (x1, y1, x2, y2) tile boxes in original-frame
    pixel coords, each `tile_size` square (clipped at the frame edge),
    overlapping neighbors by `overlap_ratio` so an animal straddling a
    tile boundary still lands whole in at least one tile."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    overlap_ratio = max(0.0, min(overlap_ratio, 0.9))
    stride = max(1, int(tile_size * (1 - overlap_ratio)))

    def _starts(dim: int) -> list[int]:
        if dim <= tile_size:
            return [0]
        starts = list(range(0, dim - tile_size + 1, stride))
        if starts[-1] + tile_size < dim:
            starts.append(dim - tile_size)  # flush-align the last tile to the far edge
        return starts

    tiles = []
    for y in _starts(height):
        for x in _starts(width):
            tiles.append((x, y, min(x + tile_size, width), min(y + tile_size, height)))
    return tiles


def _nms_merge(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """NMS per class, so an overlapping-but-different-species pair of
    boxes (e.g. a Tiger tile detection near a Leopard tile detection)
    never suppresses each other — only duplicate detections of the
    *same* class across overlapping tiles / the full-frame pass do."""
    if not detections:
        return []

    by_class: dict[int, list[int]] = {}
    for idx, d in enumerate(detections):
        by_class.setdefault(d.class_id, []).append(idx)

    kept: list[Detection] = []
    for idxs in by_class.values():
        boxes_xywh = [
            [
                detections[i].box[0],
                detections[i].box[1],
                detections[i].box[2] - detections[i].box[0],
                detections[i].box[3] - detections[i].box[1],
            ]
            for i in idxs
        ]
        scores = [detections[i].score for i in idxs]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=iou_threshold)
        if len(indices) == 0:
            continue
        for i in np.array(indices).reshape(-1):
            kept.append(detections[idxs[i]])
    return kept


class SahiDetector(Detector):
    """Drop-in Detector that runs `inner` over overlapping tiles instead
    of (or in addition to) one full-frame pass. See module docstring."""

    def __init__(self, inner: Detector, config: SahiConfig | None = None):
        self.inner = inner
        self.class_names = inner.class_names
        self.config = config or SahiConfig()

    def infer(self, frame: Any) -> list[Detection]:
        h, w = frame.shape[:2]
        cfg = self.config

        # Already fits in one tile: slicing would just re-run the exact
        # same pixels the full-frame pass already covers, for nothing.
        if w <= cfg.tile_size and h <= cfg.tile_size:
            return self.inner.infer(frame)

        all_detections: list[Detection] = []

        if cfg.run_full_frame_pass:
            all_detections.extend(self.inner.infer(frame))

        for x1, y1, x2, y2 in compute_tiles(w, h, cfg.tile_size, cfg.overlap_ratio):
            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            for det in self.inner.infer(tile):
                bx1, by1, bx2, by2 = det.box
                all_detections.append(
                    Detection(
                        box=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                        score=det.score,
                        class_id=det.class_id,
                        class_name=det.class_name,
                    )
                )

        merged = _nms_merge(all_detections, cfg.merge_iou_threshold)
        log.debug(
            "sahi merge",
            extra={"raw_detections": len(all_detections), "merged_detections": len(merged)},
        )
        return merged

    def warmup(self) -> None:
        self.inner.warmup()

    def close(self) -> None:
        self.inner.close()
