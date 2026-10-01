"""SAHI tile geometry and detection merging.

The NPU call itself lives in hailo_inference.py. Everything here is pure geometry,
so it is testable on a laptop - which matters, because tile coordinate remapping is
the single most common source of "the model works but the boxes are in the wrong
place" bugs.

Flow: cluster ROI -> compute_tiles -> crop each tile -> batch to NPU ->
      remap_tile_detections -> merge_detections (NMS).
"""
from __future__ import annotations

import numpy as np


def compute_tiles(width: int, height: int, tile_w: int = 640, tile_h: int = 640,
                  overlap_w: float = 0.2, overlap_h: float = 0.2):
    """Tile origins covering a WxH region. Returns [(x0, y0, x1, y1), ...].

    Overlap exists so an animal straddling a tile edge is whole in at least one
    tile. With 0 overlap you will systematically miss targets on the seams.
    """
    if width <= tile_w and height <= tile_h:
        return [(0, 0, width, height)]

    step_x = max(1, int(tile_w * (1.0 - overlap_w)))
    step_y = max(1, int(tile_h * (1.0 - overlap_h)))

    def origins(total, tile, step):
        if total <= tile:
            return [0]
        pts = list(range(0, total - tile + 1, step))
        if pts[-1] + tile < total:      # always cover the far edge
            pts.append(total - tile)
        return pts

    tiles = []
    for y0 in origins(height, tile_h, step_y):
        for x0 in origins(width, tile_w, step_x):
            tiles.append((x0, y0, min(x0 + tile_w, width), min(y0 + tile_h, height)))
    return tiles


def remap_tile_detections(boxes: np.ndarray, tile_origin) -> np.ndarray:
    """Shift boxes from tile-local coordinates into region coordinates."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    x0, y0 = tile_origin[0], tile_origin[1]
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] += x0
    out[:, [1, 3]] += y0
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5):
    """Plain greedy NMS. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def merge_detections(per_tile, iou_thr: float = 0.5):
    """Merge [(boxes, scores, classes, tile_origin), ...] into one detection set.

    Overlapping tiles WILL produce duplicates of the same animal - that is the
    point of the overlap - so NMS across the merged set is mandatory, not optional.
    """
    all_b, all_s, all_c = [], [], []
    for boxes, scores, classes, origin in per_tile:
        if len(boxes) == 0:
            continue
        all_b.append(remap_tile_detections(np.asarray(boxes), origin))
        all_s.append(np.asarray(scores))
        all_c.append(np.asarray(classes))
    if not all_b:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.concatenate(all_b)
    scores = np.concatenate(all_s)
    classes = np.concatenate(all_c)
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep], scores[keep], classes[keep]
