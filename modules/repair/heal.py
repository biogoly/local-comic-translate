"""Experimental color-adjusted clone preview, using explicit samples only.

This is not a neighborhood-based healing solver: it cannot borrow a panel
border's color. A single immutable clone sample supplies texture; an explicitly
chosen RGB color supplies the target tone. Masks larger than the stamp fail
rather than silently tiling texture or inventing pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from modules.utils.clone_brush import CloneSample, CloneStroke


class HealError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class HealPreview:
    bbox: tuple[int, int, int, int]
    clone_pixels: np.ndarray
    heal_pixels: np.ndarray
    effective_mask: np.ndarray


def _validate_sample(sample):
    if not isinstance(sample, CloneSample):
        raise HealError('invalid_sample')
    pixels, valid = np.asarray(sample.pixels), np.asarray(sample.valid)
    if (not isinstance(sample.size, (int, np.integer)) or isinstance(sample.size, bool)
            or sample.size < 1):
        raise HealError('invalid_sample')
    side = 2 * math.ceil(sample.size / 2) + 1
    if (pixels.dtype != np.uint8 or pixels.shape != (side, side, 3)
            or valid.dtype != np.bool_ or valid.shape != (side, side) or not np.any(valid)):
        raise HealError('invalid_sample')
    half = side // 2
    yy, xx = np.ogrid[-half:half+1, -half:half+1]
    if np.any(valid & (xx**2 + yy**2 >= (sample.size / 2)**2)):
        raise HealError('invalid_sample')


def _unit_value(value, code):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise HealError(code) from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise HealError(code)
    return value


def adjust_sample_color(sample: CloneSample, target_color, strength=1.0) -> CloneSample:
    """Shift valid source pixels toward an explicitly approved target RGB.

    A robust median gives the source tone; one additive RGB offset preserves
    local texture differences except where values clip at 0 or 255. No page
    pixels are consulted here, including those around the repair destination.
    """
    _validate_sample(sample)
    strength = _unit_value(strength, 'invalid_strength')
    try:
        color = np.asarray(target_color, dtype=np.float64)
    except (TypeError, ValueError):
        raise HealError('invalid_color') from None
    if color.shape != (3,) or not np.all(np.isfinite(color)) or np.any((color < 0) | (color > 255)):
        raise HealError('invalid_color')
    valid = np.array(sample.valid, copy=True, order='C')
    pixels = np.array(sample.pixels, copy=True, order='C')
    tone = np.median(pixels[valid].astype(np.float64), axis=0)
    pixels[valid] = np.rint(np.clip(pixels[valid].astype(np.float64) +
                                  strength * (color - tone), 0, 255)).astype(np.uint8)
    pixels[~valid] = 0
    pixels.setflags(write=False)
    valid.setflags(write=False)
    return CloneSample(pixels, valid, sample.size)


def preview_heal(current, mask, sample, target_color, destination, *, strength=1.0,
                 softness=0.25, protection=None) -> HealPreview:
    """Compare one ordinary clone stamp with its adjusted counterpart.

    The painted mask must fit completely within the captured source's valid
    support, even where protected pixels overlap it. Feathering happens first;
    mask alpha and protected pixels are enforced at final compositing. Returned
    RGB patches are ready to display or save, with no extra alpha pass.
    """
    current, mask = np.asarray(current), np.asarray(mask)
    if current.dtype != np.uint8 or current.ndim != 3 or current.shape[2] != 3:
        raise HealError('invalid_image')
    if mask.dtype != np.uint8 or mask.shape != current.shape[:2] or not np.any(mask):
        raise HealError('invalid_mask')
    _validate_sample(sample)
    softness = _unit_value(softness, 'invalid_softness')
    adjusted = adjust_sample_color(sample, target_color, strength)
    try:
        point = np.asarray(destination, dtype=np.float64)
    except (TypeError, ValueError):
        raise HealError('invalid_destination') from None
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise HealError('invalid_destination')
    x, y = map(math.floor, point)
    if not (0 <= x < current.shape[1] and 0 <= y < current.shape[0]):
        raise HealError('invalid_destination')
    protected = np.zeros(mask.shape, dtype=bool)
    if protection is not None:
        protected = np.asarray(protection)
        if protected.dtype not in (np.bool_, np.uint8) or protected.shape != mask.shape:
            raise HealError('invalid_protection')
        protected = protected != 0

    half = sample.pixels.shape[0] // 2
    left, top = x - half, y - half
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(current.shape[1], left + 2 * half + 1), min(current.shape[0], top + 2 * half + 1)
    support = np.zeros(mask.shape, dtype=bool)
    support[y1:y2, x1:x2] = sample.valid[y1-top:y2-top, x1-left:x2-left]
    if np.any((mask != 0) & ~support):
        raise HealError('mask_outside_sample')

    active = (mask != 0) & ~protected
    if not np.any(active):
        raise HealError('fully_protected')
    ys, xs = np.nonzero(active)
    bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    area = np.s_[by1:by2, bx1:bx2]
    effective_mask = np.where(protected[area], 0, mask[area]).astype(np.uint8)
    alpha = effective_mask[..., None].astype(np.float64) / 255
    original = current[area]
    previews = []
    for stamp_sample in (sample, adjusted):
        stroke = CloneStroke(current, stamp_sample, softness)
        stroke.move(x, y)
        pixels = np.rint(original * (1 - alpha) + stroke.result[area] * alpha).astype(np.uint8)
        # Exact assignments also make protected/outside-mask byte identity clear.
        pixels[effective_mask == 0] = original[effective_mask == 0]
        pixels.setflags(write=False)
        previews.append(pixels)
    effective_mask.setflags(write=False)
    return HealPreview((bx1, by1, bx2-bx1, by2-by1), *previews, effective_mask)
