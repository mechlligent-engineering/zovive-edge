"""Shared image preprocessing for both the ONNX and Hailo backends, so
letterbox math (and its inverse, used to map boxes back to original
frame coordinates) is written and tested exactly once.
"""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    image: np.ndarray, new_size: int = 640, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize `image` to fit in a `new_size` x `new_size` square, preserving
    aspect ratio, padding the rest with `color`. Returns (padded_image,
    scale_ratio, (pad_x, pad_y)) so boxes predicted on the padded image
    can be mapped back with `unletterbox_box`.
    """
    h, w = image.shape[:2]
    ratio = min(new_size / h, new_size / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = new_size - new_w
    pad_y = new_size - new_h
    top, bottom = pad_y // 2, pad_y - pad_y // 2
    left, right = pad_x // 2, pad_x - pad_x // 2

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, ratio, (left, top)


def unletterbox_box(
    box: tuple[float, float, float, float], ratio: float, pad: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Map a box (x1,y1,x2,y2) predicted on the letterboxed image back to
    the original image's coordinate space."""
    pad_x, pad_y = pad
    x1, y1, x2, y2 = box
    x1 = (x1 - pad_x) / ratio
    y1 = (y1 - pad_y) / ratio
    x2 = (x2 - pad_x) / ratio
    y2 = (y2 - pad_y) / ratio
    return x1, y1, x2, y2


def to_chw_float32(image_bgr: np.ndarray, normalize: bool = True) -> np.ndarray:
    """BGR HxWx3 uint8 -> RGB 1x3xHxW float32, values in [0,1] if normalize."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32)
    if normalize:
        chw /= 255.0
    return np.ascontiguousarray(chw[np.newaxis, ...])


def clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width - 1))
    y2 = max(0.0, min(y2, height - 1))
    return x1, y1, x2, y2


def crop_box(image: np.ndarray, box: tuple[float, float, float, float], pad_ratio: float = 0.1) -> np.ndarray:
    """Crop `box` out of `image` with a small margin so the classifier
    sees a bit of context around the detector's box, not just its
    exact edges (which are often tight/clipped)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
    x1c, y1c, x2c, y2c = clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), w, h)
    x1i, y1i, x2i, y2i = int(x1c), int(y1c), int(x2c), int(y2c)
    if x2i <= x1i or y2i <= y1i:
        return image[0:1, 0:1]  # degenerate box; caller should treat as invalid
    return image[y1i:y2i, x1i:x2i]
