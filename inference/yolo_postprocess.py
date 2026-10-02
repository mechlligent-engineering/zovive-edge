"""Decodes raw YOLO11 detection-head output into Detection objects.

Ultralytics YOLO11 detection models exported to ONNX (without the
optional NMS graph baked in) produce a single output tensor shaped
(1, 4 + num_classes, num_anchors) — box in cxcywh (model input pixel
space) followed by one score per class, already sigmoid-activated.
This module transposes that, filters by confidence, runs NMS, and
maps boxes back to the original frame with preprocess.unletterbox_box.

Kept separate from onnx_inference.py / hailo_inference.py so both
backends decode identically — the only difference between them is how
the raw tensor is produced, not how it's interpreted.
"""

from __future__ import annotations

import cv2
import numpy as np

from inference.base import Detection
from inference.preprocess import clip_box, unletterbox_box


def decode_yolo_detections(
    output: np.ndarray,
    class_names: list[str],
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],  # (height, width)
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """`output`: raw model output, shape (1, 4+num_classes, N) or (4+num_classes, N)."""
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] != 4 + len(class_names):
        # Some exports produce (N, 4+num_classes) instead; handle both.
        if arr.shape[-1] == 4 + len(class_names):
            arr = arr.T
        else:
            raise ValueError(f"unexpected YOLO output shape {output.shape} for {len(class_names)} classes")

    boxes_cxcywh = arr[0:4, :].T  # (N, 4)
    class_scores = arr[4:, :].T  # (N, num_classes)

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep_mask = scores >= conf_threshold
    if not np.any(keep_mask):
        return []

    boxes_cxcywh = boxes_cxcywh[keep_mask]
    class_ids = class_ids[keep_mask]
    scores = scores[keep_mask]

    # cxcywh -> xyxy, still in letterboxed-image pixel space
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    boxes_xywh_for_nms = np.stack([x1, y1, w, h], axis=1)

    indices = cv2.dnn.NMSBoxes(boxes_xywh_for_nms.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) == 0:
        return []
    indices = np.array(indices).reshape(-1)

    orig_h, orig_w = orig_shape
    detections: list[Detection] = []
    for i in indices:
        bx1, by1 = x1[i], y1[i]
        bx2, by2 = bx1 + w[i], by1 + h[i]
        mapped = unletterbox_box((bx1, by1, bx2, by2), ratio, pad)
        clipped = clip_box(mapped, orig_w, orig_h)
        cid = int(class_ids[i])
        detections.append(
            Detection(
                box=clipped,
                score=float(scores[i]),
                class_id=cid,
                class_name=class_names[cid] if cid < len(class_names) else str(cid),
            )
        )
    return detections


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
