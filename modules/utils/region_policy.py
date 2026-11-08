"""Independent, model-free policies for page-wide region operations.

Explicit selected operations may bypass a keep flag after their preview is
accepted. Protection never changes detector classification or an OCR region.
"""
from __future__ import annotations

from collections import Counter
import copy
import math
import logging

import numpy as np
from .lettering_angle import optional_lettering_angle

logger = logging.getLogger(__name__)


_STAGES = {
    'source': 'keep_source', 'ocr': 'keep_source',
    'translation': 'keep_translation', 'layout': 'keep_layout',
    'cleanup': 'skip_auto_cleanup', 'segmentation': 'skip_auto_cleanup',
}


def is_protected(block, key: str) -> bool:
    metadata = getattr(block, 'region_metadata', {})
    flags = metadata.get('protections', {}) if isinstance(metadata, dict) else {}
    return isinstance(flags, dict) and flags.get(key) is True


def permits_region_operation(block, stage: str, *, explicit: bool = False) -> bool:
    """A keep flag affects its own stage only; unknown stages are errors."""
    key = _STAGES[stage]
    return explicit or not is_protected(block, key)


def has_any_protection(block) -> bool:
    return any(is_protected(block, key) for key in set(_STAGES.values()))


def should_preserve_region(block) -> bool:
    # Import locally to avoid a cycle with the backward-compatible merge API.
    from .manual_regions import is_manual_region
    return (is_manual_region(block) or has_any_protection(block)
            or optional_lettering_angle(getattr(block, 'manual_lettering_angle', None)) is not None)


def retained_layout_states(blocks, states, *, retain_all: bool = False) -> dict:
    """Return independent rendered states keyed only by unambiguous region ID.

    Completed hand-drawn regions retain the pre-existing manual-workflow
    safeguard. ``retain_all`` is for incremental publication/new-item fitting.
    Never guess ownership from overlapping or nearby rectangles.
    """
    from .manual_regions import has_manual_translation
    block_counts = Counter(getattr(block, 'region_id', '') for block in blocks)
    state_counts = Counter(state.get('region_id', '') for state in states if isinstance(state, dict))
    eligible = {
        block.region_id for block in blocks
        if getattr(block, 'region_id', '') and block_counts[block.region_id] == 1
        and (retain_all or is_protected(block, 'keep_layout') or has_manual_translation(block))
    }
    return {state['region_id']: copy.deepcopy(state) for state in states
            if isinstance(state, dict) and state.get('region_id') in eligible
            and state_counts[state['region_id']] == 1}


def shifted_region_copy(block, dx=0, dy=0):
    """Copy a logical region into a crop while retaining its identity/decisions."""
    result = copy.deepcopy(block)
    offset = np.asarray([dx, dy, dx, dy], dtype=float)
    for name in ('xyxy', 'bubble_xyxy'):
        value = getattr(result, name, None)
        if value is not None:
            setattr(result, name, np.asarray(value, dtype=float) + offset)
    lines = getattr(result, 'lines', None)
    if lines is not None:
        result.lines = [list(np.asarray(line, dtype=float) + offset) for line in lines]
    points = getattr(result, 'segm_pts', None)
    if points is not None:
        points = np.asarray(points)
        if points.size and points.shape[-1] == 2:
            result.segm_pts = points + np.asarray([dx, dy], dtype=float)
    return result


def cleanup_protection_mask(shape, blocks) -> np.ndarray:
    """Conservative clipped region footprint; separate from segmentation masks.

    Rotated boxes protect their enclosing bounds as well. This is region-based
    automatic-operation protection, not a user-authored protected-pixel mask.
    """
    height, width = shape[:2]
    protected = np.zeros((height, width), dtype=bool)
    for block in blocks or ():
        if not is_protected(block, 'skip_auto_cleanup'):
            continue
        try:
            x1, y1, x2, y2 = map(float, block.xyxy)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                continue
            angle = float(getattr(block, 'angle', 0))
            if math.isfinite(angle) and angle % 360:
                theta = math.radians(angle)
                half_width, half_height = (x2 - x1) / 2, (y2 - y1) / 2
                dx = abs(half_width * math.cos(theta)) + abs(half_height * math.sin(theta))
                dy = abs(half_width * math.sin(theta)) + abs(half_height * math.cos(theta))
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                x1, y1, x2, y2 = min(x1, cx-dx), min(y1, cy-dy), max(x2, cx+dx), max(y2, cy+dy)
            x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
            x2, y2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
            if x2 > x1 and y2 > y1:
                protected[y1:y2, x1:x2] = True
        except (TypeError, ValueError, AttributeError, OverflowError):
            continue
    return protected


def subtract_cleanup_protections(mask, blocks):
    """Subtract all protected regions after any dilation, without mutation."""
    if mask is None:
        return None
    result = mask.copy()
    result[cleanup_protection_mask(mask.shape, blocks)] = 0
    removed = (mask > 0) & (result == 0)
    if removed.ndim == 3:
        removed = np.any(removed, axis=2)
    count = int(np.count_nonzero(removed))
    if count:
        logger.info('Automatic cleanup skipped %d pixels in protected regions', count)
    return result


def composite_cleanup_result(before, result, effective_mask):
    """Keep every pixel outside the effective mask byte-for-byte unchanged."""
    if result is None or effective_mask is None:
        return before.copy()
    if result.shape != before.shape or effective_mask.shape[:2] != before.shape[:2]:
        raise ValueError('Cleanup result, source and mask dimensions must match')
    allowed = effective_mask > 0
    if allowed.ndim == 3:
        allowed = np.any(allowed, axis=2)
    composed = before.copy()
    composed[allowed] = result[allowed]
    return composed
