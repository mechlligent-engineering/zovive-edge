"""Decodes raw YOLO11 detection-head output into Detection objects.

Two raw forms, one interpretation:

- ONNX (onnx_inference.py): Ultralytics YOLO11 detection models exported
  without the optional NMS graph produce a single output tensor shaped
  (1, 4 + num_classes, num_anchors) — box in cxcywh (model input pixel
  space) followed by one score per class, already sigmoid-activated.
  decode_yolo_detections reads that.
- Hailo (hailo_inference.py): the HEF is cut before the head's DFL decode,
  so it has six outputs, NHWC — per stride (8/16/32) a box conv with
  4 * REG_MAX channels and a class conv with one logit per class.
  decode_split_heads turns those into the same rows (cx, cy, w, h and one
  sigmoid score per class, letterbox pixels) that the exported head
  computes.

Both then go through the same selection: confidence filter, boxes centred
in the letterbox padding dropped, NMS, and boxes mapped back to the
original frame with preprocess.unletterbox_box. Kept separate from the
backends so they decode identically — the only difference between them is
how the raw tensors are produced, not how they're interpreted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from inference.base import Detection
from inference.preprocess import clip_box, unletterbox_box

# Distribution Focal Loss bins per box side in Ultralytics YOLO11's head.
REG_MAX = 16


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

    return _select(
        arr[0:4, :].T, arr[4:, :].T, class_names, ratio, pad, orig_shape, conf_threshold, iou_threshold
    )


@dataclass(frozen=True)
class SplitHead:
    """One stride of a split detection head: the names of its two outputs."""

    stride: int
    box: str  # 4 * REG_MAX channels: DFL logits for the left, top, right, bottom distances
    cls: str  # one logit per class


def split_head_layout(
    shapes: Mapping[str, Sequence[int]], num_classes: int, input_size: int
) -> list[SplitHead]:
    """Which outputs of a split-head model (a Hailo HEF cut before the DFL
    decode) form each stride, from their NHWC shapes (a leading batch of 1
    is allowed). Matched by channel count and grid size, not by layer name:
    names like conv71 change whenever a model is recompiled. Raises
    ValueError unless there are exactly three strides, each with one box
    and one class output."""
    if num_classes == 4 * REG_MAX:
        raise ValueError(f"{num_classes} classes cannot be told apart from the box outputs")
    grids: dict[tuple[int, int], dict[str, str]] = {}
    for name, shape in shapes.items():
        dims = [int(d) for d in shape]
        while len(dims) > 3 and dims[0] == 1:
            dims = dims[1:]
        if len(dims) != 3:
            raise ValueError(f"output {name!r} has shape {tuple(shape)}; expected (H, W, C)")
        h, w, c = dims
        kind = "box" if c == 4 * REG_MAX else "cls" if c == num_classes else None
        if kind is None:
            raise ValueError(
                f"output {name!r} has {c} channels; expected {4 * REG_MAX} (box) or {num_classes} (classes)"
            )
        pair = grids.setdefault((h, w), {})
        if kind in pair:
            raise ValueError(f"two {kind} outputs on the {h}x{w} grid: {pair[kind]!r} and {name!r}")
        pair[kind] = name

    heads = []
    for (h, w), pair in grids.items():
        if set(pair) != {"box", "cls"}:
            missing = "class" if "box" in pair else "box"
            raise ValueError(f"the {h}x{w} grid has no {missing} output")
        if input_size % h or input_size % w or input_size // h != input_size // w:
            raise ValueError(f"a {h}x{w} grid does not fit the {input_size}x{input_size} input")
        heads.append(SplitHead(stride=input_size // h, box=pair["box"], cls=pair["cls"]))
    if len(heads) != 3:
        raise ValueError(f"expected 3 strides (8, 16, 32), got {sorted(h.stride for h in heads)}")
    return sorted(heads, key=lambda head: head.stride)


def decode_split_heads(
    outputs: Mapping[str, np.ndarray],
    heads: Sequence[SplitHead],
    class_names: list[str],
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],  # (height, width)
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Decode a split-head model's raw outputs exactly as the exported
    head would: per anchor, sigmoid class scores, and box distances from
    the DFL — a softmax over REG_MAX bins per side, Ultralytics' layout
    (channel = side * REG_MAX + bin, sides left, top, right, bottom) —
    times the stride, around the anchor point (x + 0.5, y + 0.5) * stride.
    Class channel i is class_names[i]. Only anchors that pass
    conf_threshold get the DFL decode."""
    boxes, scores = [], []
    bins = np.arange(REG_MAX, dtype=np.float32)
    for head in heads:
        logits, box = _nhwc(outputs[head.cls]), _nhwc(outputs[head.box])
        h, w = logits.shape[:2]
        logits = logits.reshape(h * w, -1)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        idx = np.flatnonzero(prob.max(axis=1) >= conf_threshold)
        if idx.size == 0:
            continue
        dfl = box.reshape(h * w, 4, REG_MAX)[idx]
        dfl = np.exp(dfl - dfl.max(axis=-1, keepdims=True))
        dist = (dfl / dfl.sum(axis=-1, keepdims=True)) @ bins * head.stride  # (n, 4): l, t, r, b
        ax = (idx % w + 0.5) * head.stride
        ay = (idx // w + 0.5) * head.stride
        x1, y1, x2, y2 = ax - dist[:, 0], ay - dist[:, 1], ax + dist[:, 2], ay + dist[:, 3]
        boxes.append(np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1))
        scores.append(prob[idx])
    if not boxes:
        return []
    return _select(
        np.concatenate(boxes),
        np.concatenate(scores),
        class_names,
        ratio,
        pad,
        orig_shape,
        conf_threshold,
        iou_threshold,
    )


def _nhwc(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output, dtype=np.float32)
    return arr[0] if arr.ndim == 4 and arr.shape[0] == 1 else arr


def _select(
    boxes_cxcywh: np.ndarray,  # (N, 4), letterboxed-image pixels
    class_scores: np.ndarray,  # (N, num_classes), sigmoid scores
    class_names: list[str],
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """Shared by both decoders: confidence filter, NMS, map back to the frame."""
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    # A box centred in the letterbox padding cannot be an animal in the frame.
    orig_h, orig_w = orig_shape
    pad_x, pad_y = pad
    cxs, cys = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1]
    inside = (
        (cxs >= pad_x) & (cxs <= pad_x + orig_w * ratio) & (cys >= pad_y) & (cys <= pad_y + orig_h * ratio)
    )
    keep_mask = (scores >= conf_threshold) & inside
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
