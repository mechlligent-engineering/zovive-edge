# Postprocessing pipeline for Hailo split-head YOLO11
from dataclasses import dataclass
import cv2
import numpy as np

REG_MAX = 16

@dataclass
class Detection:
    box: list
    score: float
    class_id: int
    class_name: str

@dataclass(frozen=True)
class YoloHead:
    stride: int
    cls: str
    box: str

def split_head_layout(shapes_by_name, num_classes=None, input_size=640):
    heads_by_stride = {}
    box_channels = REG_MAX * 4
    for name, shape in shapes_by_name.items():
        if len(shape) == 3:
            h, w, c = shape
        elif len(shape) == 4:
            h, w, c = shape[1:4]
        else:
            continue
        stride = input_size // h
        if stride not in (8, 16, 32):
            continue
        if stride not in heads_by_stride:
            heads_by_stride[stride] = {}
        if c == box_channels:
            heads_by_stride[stride]["box"] = name
        else:
            heads_by_stride[stride]["cls"] = name
    heads = []
    for s in (8, 16, 32):
        if s in heads_by_stride and "cls" in heads_by_stride[s] and "box" in heads_by_stride[s]:
            heads.append(YoloHead(stride=s, cls=heads_by_stride[s]["cls"], box=heads_by_stride[s]["box"]))
    return heads

def _nhwc(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor

def _nms_per_class(boxes, scores, class_ids, iou_thresh):
    keep_indices = []
    unique_classes = np.unique(class_ids)
    for c in unique_classes:
        c_mask = (class_ids == c)
        c_idxs = np.where(c_mask)[0]
        c_boxes = boxes[c_idxs]
        c_scores = scores[c_idxs]
        
        # cv2.dnn.NMSBoxes expects boxes as [x, y, w, h]
        rects = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in c_boxes]
        idxs = cv2.dnn.NMSBoxes(rects, c_scores.tolist(), 0.0, iou_thresh)
        if len(idxs) > 0:
            keep_indices.extend(c_idxs[np.array(idxs).reshape(-1)])
    return keep_indices

def decode_split_heads(*args, **kwargs):
    outputs = args[0] if len(args) > 0 else kwargs.get("outputs")
    heads = args[1] if len(args) > 1 else kwargs.get("heads")
    class_names = args[2] if len(args) > 2 else kwargs.get("class_names")
    
    ratio = args[3] if len(args) > 3 else kwargs.get("ratio", 1.0)
    pad = args[4] if len(args) > 4 else kwargs.get("pad", (0.0, 0.0))
    orig_shape = kwargs.get("orig_shape", args[5] if len(args) > 5 else None)
    conf_threshold = kwargs.get("conf_threshold", 0.45)
    iou_threshold = kwargs.get("iou_threshold", 0.45)
    
    all_boxes = []
    all_scores = []
    all_class_ids = []
    bins = np.arange(REG_MAX, dtype=np.float32)
    
    # Safe sigmoid threshold: x >= ln(conf / (1 - conf))
    safe_conf = np.clip(conf_threshold, 1e-4, 1.0 - 1e-4)
    logit_th = float(np.log(safe_conf / (1.0 - safe_conf)))
    
    for head in heads:
        if head.cls not in outputs or head.box not in outputs:
            continue
            
        logits = _nhwc(outputs[head.cls])
        box_raw = _nhwc(outputs[head.box])
        h, w, c = logits.shape
        
        logits_flat = logits.reshape(h * w, c)
        box_flat = box_raw.reshape(h * w, 4, REG_MAX)
        
        cand_mask = logits_flat.max(axis=-1) >= logit_th
        idx = np.flatnonzero(cand_mask)
        if idx.size == 0:
            continue
            
        cand_logits = logits_flat[idx]
        c_ids = np.argmax(cand_logits, axis=-1)
        selected_logits = cand_logits[np.arange(len(idx)), c_ids]
        cand_scores = 1.0 / (1.0 + np.exp(-np.clip(selected_logits, -30.0, 30.0)))
        
        # Softmax over 16 bins for distribution focal loss
        dfl = box_flat[idx]
        dfl_exp = np.exp(dfl - dfl.max(axis=-1, keepdims=True))
        dist = (dfl_exp / dfl_exp.sum(axis=-1, keepdims=True)) @ bins * head.stride
        
        ys, xs = np.divmod(idx, w)
        ax = (xs.astype(np.float32) + 0.5) * head.stride
        ay = (ys.astype(np.float32) + 0.5) * head.stride
        
        x1 = ax - dist[:, 0]
        y1 = ay - dist[:, 1]
        x2 = ax + dist[:, 2]
        y2 = ay + dist[:, 3]
        
        valid = (x2 > x1 + 8.0) & (y2 > y1 + 8.0)
        if not np.any(valid):
            continue
            
        all_boxes.append(np.column_stack([x1[valid], y1[valid], x2[valid], y2[valid]]))
        all_scores.append(cand_scores[valid])
        all_class_ids.append(c_ids[valid])
        
    if not all_boxes:
        return []
        
    boxes = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)
    class_ids = np.concatenate(all_class_ids)
    
    keep = _nms_per_class(boxes, scores, class_ids, iou_threshold)
    if not keep:
        return []
        
    try:
        from inference.base import Detection as OutDetection
    except ImportError:
        OutDetection = Detection
        
    detections = []
    rx, ry = ratio[:2] if isinstance(ratio, (tuple, list)) else (float(ratio), float(ratio))
    pad_x, pad_y = pad[:2] if isinstance(pad, (tuple, list)) else (0.0, 0.0)
    
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        x1 = max(0.0, (x1 - pad_x) / rx)
        y1 = max(0.0, (y1 - pad_y) / ry)
        x2 = (x2 - pad_x) / rx
        y2 = (y2 - pad_y) / ry
        
        if orig_shape is not None:
            x1 = min(x1, float(orig_shape[1]))
            y1 = min(y1, float(orig_shape[0]))
            x2 = min(x2, float(orig_shape[1]))
            y2 = min(y2, float(orig_shape[0]))
            
        cid = int(class_ids[i])
        cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        detections.append(OutDetection(
            box=[x1, y1, x2, y2],
            score=float(scores[i]),
            class_id=cid,
            class_name=cname
        ))
        
    return detections
