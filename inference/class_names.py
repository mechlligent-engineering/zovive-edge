"""Species class names: one canonical spelling, and a load-time check that
the detector's class order is the configured one.

The detector checkpoint spells some species with a space ("sloth bear");
the edge always reports the lowercase underscore form ("sloth_bear").
configs/inference_config.yaml `detector.classes` must list the species in
the detector's output order.

An Ultralytics ONNX export embeds the training-time `model.names`
({index: name}) in its metadata, so an ONNX model is checked against the
configured list when it is loaded. A HEF carries no names: on the Hailo
path the configured list is the only index -> species mapping.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Mapping, Sequence

from inference.base import InferenceBackendError

log = logging.getLogger(__name__)


def canonical_class_name(name: str) -> str:
    """'Sloth Bear', 'sloth-bear' and 'sloth_bear' all become 'sloth_bear'."""
    return "_".join(name.strip().lower().replace("-", " ").replace("_", " ").split())


def embedded_class_names(metadata: Mapping[str, str]) -> list[str] | None:
    """Canonical class names in index order from an Ultralytics export's
    `names` metadata (the str() of {0: 'deer', 1: ...}), or None when the
    model has none or it cannot be read."""
    raw = metadata.get("names")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        names = ast.literal_eval(raw)  # literals only; never executes code
    except (ValueError, SyntaxError, RecursionError):
        names = None
    if not isinstance(names, dict) or set(names) != set(range(len(names))):
        log.warning("ignoring unreadable 'names' model metadata", extra={"names": raw[:200]})
        return None
    return [canonical_class_name(str(names[i])) for i in range(len(names))]


def checked_class_names(
    configured: Sequence[str],
    model_path: str,
    *,
    model_names: Sequence[str] | None = None,
) -> list[str]:
    """The configured class names in canonical form, after checking them
    against the model's own embedded names (count, spelling and order).

    Raises InferenceBackendError on a mismatch, so a wrong config or a
    wrong model stops at load time instead of mislabelling species.
    """
    expected = [canonical_class_name(n) for n in configured]
    if model_names is not None and list(model_names) != expected:
        raise InferenceBackendError(
            f"{model_path}: model class order {list(model_names)} != configured {expected}; "
            "set `detector.classes` in configs/inference_config.yaml to the model's order"
        )
    return expected
