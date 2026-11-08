"""Angles for editable translation layers, in Qt's clockwise convention."""

import math


def optional_lettering_angle(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        angle = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return angle if math.isfinite(angle) else None


def render_lettering_angle(block) -> float:
    """Prefer the user's lettering angle, then a rotated OCR box, then Detect."""
    manual = optional_lettering_angle(getattr(block, "manual_lettering_angle", None))
    if manual is not None:
        return manual
    box_angle = optional_lettering_angle(getattr(block, "angle", 0))
    if box_angle:
        return box_angle
    return optional_lettering_angle(getattr(block, "detected_lettering_angle", None)) or 0.0
