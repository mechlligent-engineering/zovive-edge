"""Compare models/detector.hef (Hailo-8L) with models/detector.onnx (CPU) on
the same images, on the Pi. Both run through zovive-edge's own preprocessing
and decoder (inference/yolo_postprocess.py), so a disagreement comes from the
HEF itself — quantization, a wrong input format or a wrong cut — not from
the code. The only way to verify a compiled HEF; run it after every compile.

    sudo systemctl stop zovive-detect    # the Hailo device takes one process
    .venv/bin/python -m tools.hef_parity --images '/data/parity/*.jpg' --out /tmp/parity.json

Reports, per image and in total: detections from each backend, how many
pair up (IoU >= 0.5), how many of those name the same species, their mean
IoU and mean absolute score difference. Nothing is written except --out.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import paths
from inference.base import Detection
from utils.config_loader import load

MATCH_IOU = 0.5


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(reference: list[Detection], candidate: list[Detection]) -> dict[str, Any]:
    """Pair each reference (ONNX) detection with the unused candidate (HEF)
    detection of highest IoU >= MATCH_IOU, any species."""
    used: set[int] = set()
    pairs = []
    for ref in sorted(reference, key=lambda d: -d.score):
        best, best_iou = None, MATCH_IOU
        for j, cand in enumerate(candidate):
            overlap = _iou(ref.box, cand.box)
            if j not in used and overlap >= best_iou:
                best, best_iou = j, overlap
        if best is not None:
            used.add(best)
            pairs.append((ref, candidate[best], best_iou))
    return {
        "onnx": len(reference),
        "hef": len(candidate),
        "matched": len(pairs),
        "same_species": sum(r.class_name == c.class_name for r, c, _ in pairs),
        "ious": [round(i, 4) for _, _, i in pairs],
        "score_diffs": [round(abs(r.score - c.score), 4) for r, c, _ in pairs],
    }


def summarize(per_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ious = [i for r in per_image.values() for i in r["ious"]]
    diffs = [d for r in per_image.values() for d in r["score_diffs"]]
    total = {k: sum(r[k] for r in per_image.values()) for k in ("onnx", "hef", "matched", "same_species")}
    return {
        "images": len(per_image),
        **total,
        "matched_of_onnx": round(total["matched"] / total["onnx"], 4) if total["onnx"] else None,
        "same_species_of_matched": round(total["same_species"] / total["matched"], 4)
        if total["matched"]
        else None,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "mean_abs_score_diff": round(float(np.mean(diffs)), 4) if diffs else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", nargs="+", required=True, help="image files or glob patterns")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    args = parser.parse_args(argv)

    files = sorted({f for pattern in args.images for f in glob.glob(pattern)})
    if not files:
        print("no images match --images", file=sys.stderr)
        return 2

    # Imported here: hailo_inference needs HailoRT, which only the Pi has.
    from inference.hailo_inference import HailoDetector
    from inference.onnx_inference import OnnxDetector

    det = load(paths.INFERENCE_CONFIG)["detector"]
    common = {
        "conf_threshold": float(det.get("confidence_threshold", 0.45)),
        "iou_threshold": float(det.get("nms_iou_threshold", 0.45)),
    }
    onnx = OnnxDetector(
        str(paths.model_file(det["onnx_path"])),
        det["classes"],
        input_size=int(det.get("input_size", 640)),
        **common,
    )
    hef = HailoDetector(str(paths.model_file(det["hef_path"])), det["classes"], **common)
    per_image: dict[str, dict[str, Any]] = {}
    try:
        for f in files:
            frame = cv2.imread(f)
            if frame is None:
                print(f"skipping unreadable image {f}", file=sys.stderr)
                continue
            per_image[f] = compare(onnx.infer(frame), hef.infer(frame))
    finally:
        hef.close()

    report = {"summary": summarize(per_image), "images": per_image}
    text = json.dumps(report, indent=2)
    print(json.dumps(report["summary"], indent=2))
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
