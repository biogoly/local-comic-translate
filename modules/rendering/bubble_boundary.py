"""Conservative, image-local boundaries for typography, never cleanup masks.

The estimator accepts only a well-contained, mostly flat background around the
text. It deliberately declines gradients, open outlines and ambiguous artwork.
Explicit rectangle/ellipse overrides remain available without an image/model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import mahotas as mh
import numpy as np


MAX_ROI_SIDE = 1536
MAX_ROI_PIXELS = 2_000_000
_CONNECTIVITY = np.array([[False, True, False], [True, True, True], [False, True, False]])


@dataclass(frozen=True)
class BubbleBoundary:
    mask: np.ndarray
    origin: tuple[int, int]
    source: str
    reliability: str
    reason: str = ""

    @property
    def available(self) -> bool:
        return bool(self.mask.size and np.any(self.mask))


def _failure(reason, origin=(0, 0)):
    return BubbleBoundary(np.empty((0, 0), dtype=bool), origin, "none", "unavailable", reason)


def _bounds(value, image_shape=None):
    try:
        if value is None or len(value) != 4:
            return None
        values = [float(number) for number in value]
        if not all(math.isfinite(number) for number in values):
            return None
        left, top, right, bottom = values
        if right <= left or bottom <= top:
            return None
        left, top = max(0, math.floor(left)), max(0, math.floor(top))
        right, bottom = max(0, math.ceil(right)), max(0, math.ceil(bottom))
        if image_shape is not None:
            height, width = image_shape[:2]
            left, right = min(left, width), min(right, width)
            top, bottom = min(top, height), min(bottom, height)
        return (left, top, right, bottom) if right > left and bottom > top else None
    except (TypeError, ValueError, OverflowError):
        return None


def _bounded(bounds):
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    return width <= MAX_ROI_SIDE and height <= MAX_ROI_SIDE and width * height <= MAX_ROI_PIXELS


def _distance(mask):
    # Padding supplies a real outside boundary even for an all-True rectangle.
    return mh.distance(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]


def _erode(mask, pixels):
    return mask & (_distance(mask) > pixels * pixels) if pixels else mask.copy()


def _remove_tails(mask, radius):
    """Euclidean opening removes narrow protrusions without a large kernel."""
    core = _erode(mask, radius)
    if not np.any(core):
        return np.zeros_like(mask)
    # Distance to the eroded core gives a bounded-cost circular dilation.
    return mask & (mh.distance(~core) <= radius * radius)


def _seed_slice(bounds, origin, shape):
    x1, y1, x2, y2 = bounds
    ox, oy = origin
    height, width = shape
    return (slice(max(0, y1 - oy), min(height, y2 - oy)),
            slice(max(0, x1 - ox), min(width, x2 - ox)))


def _enclosed_at_crop_edge(image, roi, component, background, color):
    """Verify a clipped narrow tail closes nearby; never add probe pixels to the layout."""
    x1, y1, x2, y2 = roi
    span = min(component.shape)
    limit = min(24, max(3, round(span * 0.12)))
    depth = min(96, max(24, round(span * 0.5)))
    # Each strip starts just outside the crop. Orient all four so its first row
    # meets the touched edge, keeping the edge pixels in the same order.
    edges = (
        (component[0], background[0], image[max(0, y1-depth):y1, x1:x2][::-1]),
        (component[-1], background[-1], image[y2:y2+depth, x1:x2]),
        (component[:, 0], background[:, 0], image[y1:y2, max(0, x1-depth):x1][:, ::-1].swapaxes(0, 1)),
        (component[:, -1], background[:, -1], image[y1:y2, x2:x2+depth].swapaxes(0, 1)),
    )
    touched = [entry for entry in edges if entry[0].any()]
    if not touched:
        return True
    if len(touched) != 1:
        return False
    edge, tone_edge, strip = touched[0]
    contact = np.flatnonzero(edge)
    if not len(strip) or contact[-1] - contact[0] + 1 > limit:
        return False

    # Follow only matching pixels connected to the contact. Reject a channel
    # that stays open, fans into another balloon, or leaves the observed strip.
    near_tone = np.max(np.abs(strip.astype(np.float32) - color), axis=2) <= 14
    labels, _ = mh.label(near_tone, _CONNECTIVITY)
    connected_labels = np.unique(labels[0, contact])
    continuation = np.isin(labels, connected_labels[connected_labels != 0])
    if np.any(continuation[0] & tone_edge & ~edge):
        return False
    if continuation[-1].any() or continuation[:, 0].any() or continuation[:, -1].any():
        return False
    for row in continuation:
        occupied = np.flatnonzero(row)
        if occupied.size and occupied[-1] - occupied[0] + 1 > limit:
            return False
    return True


def estimate_bubble_boundary(
    image, text_bounds, bubble_bounds=None, *, padding=8,
    override="auto", override_bounds=None, page_size=None,
) -> BubbleBoundary:
    """Return a cropped page-space mask, or an explicit unavailable reason.

    Boxes use half-open ``[left, top, right, bottom]`` pixel coordinates. Float
    boxes round outward, then clip to image bounds. With no image, explicit
    overrides clip negative coordinates to zero. Padding is in source pixels.
    The input image is never modified. Reliability labels are qualitative.
    """
    if override not in {"auto", "rectangle", "ellipse"}:
        return _failure("invalid_override")
    if isinstance(padding, bool):
        return _failure("invalid_padding")
    try:
        padding = float(padding)
        if not math.isfinite(padding) or padding < 0:
            return _failure("invalid_padding")
    except (TypeError, ValueError):
        return _failure("invalid_padding")
    valid_image = (isinstance(image, np.ndarray) and image.ndim == 3
                   and image.shape[2] == 3 and image.dtype == np.uint8
                   and image.shape[0] > 0 and image.shape[1] > 0)
    shape = image.shape if valid_image else None
    if override != 'auto' and page_size is not None:
        try:
            if len(page_size) != 2 or not all(math.isfinite(float(v)) and float(v) > 0 for v in page_size):
                return _failure('invalid_bounds')
            shape = (int(page_size[1]), int(page_size[0]), 3)
        except (TypeError, ValueError, OverflowError):
            return _failure('invalid_bounds')
    text = _bounds(text_bounds, shape)
    if text is None:
        return _failure("invalid_bounds")

    if override != "auto":
        chosen = override_bounds if override_bounds is not None else (
            bubble_bounds if bubble_bounds is not None else text_bounds)
        bounds = _bounds(chosen, shape)
        if bounds is None:
            return _failure("invalid_bounds")
        x1, y1, x2, y2 = bounds
        if not _bounded(bounds):
            return _failure("region_too_large", (x1, y1))
        height, width = y2 - y1, x2 - x1
        if override == "rectangle":
            mask = np.ones((height, width), dtype=bool)
        else:
            yy, xx = np.ogrid[:height, :width]
            mask = (((xx + 0.5 - width / 2) / (width / 2)) ** 2
                    + ((yy + 0.5 - height / 2) / (height / 2)) ** 2) <= 1
        mask = _erode(mask, padding)
        if not np.any(mask):
            return _failure("padding_empty", (x1, y1))
        return BubbleBoundary(mask, (x1, y1), override, "explicit")

    if not valid_image:
        return _failure("missing_image" if image is None else "invalid_image")
    if bubble_bounds is not None:
        bubble = _bounds(bubble_bounds, shape)
        if bubble is None:
            return _failure("invalid_bounds")
        margin = max(5, min(32, math.ceil(min(bubble[2] - bubble[0], bubble[3] - bubble[1]) * 0.1)))
        roi = _bounds((bubble[0] - margin, bubble[1] - margin,
                       bubble[2] + margin, bubble[3] + margin), shape)
    else:
        # A bounded local search permits manually drawn text boxes without a
        # detector balloon box. If the outline lies beyond this crop, decline.
        dx = max(24, math.ceil((text[2] - text[0]) * 0.75))
        dy = max(24, math.ceil((text[3] - text[1]) * 0.75))
        roi = _bounds((text[0] - dx, text[1] - dy, text[2] + dx, text[3] + dy), shape)
        bubble = roi
    if roi is None:
        return _failure("invalid_bounds")
    x1, y1, x2, y2 = roi
    origin = (x1, y1)
    if not _bounded(roi):
        return _failure("region_too_large", origin)
    if not (x1 <= text[0] < text[2] <= x2 and y1 <= text[1] < text[3] <= y2):
        return _failure("text_not_contained", origin)
    crop = image[y1:y2, x1:x2]
    seed_slice = _seed_slice(text, origin, crop.shape[:2])
    seed = crop[seed_slice]
    if seed.size < 12:
        return _failure("no_background", origin)

    # Estimate flat RGB tone rather than luminance: equal brightness does not
    # imply the same colored surface. Median is robust to sparse lettering.
    samples = seed.reshape(-1, 3)
    quantized = samples.astype(np.uint16) // 16
    codes = (quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2])
    histogram = np.bincount(codes, minlength=4096)
    dominant = int(histogram.argmax())
    selected = samples[codes == dominant]
    if len(selected) / len(samples) < 0.35:
        return _failure("ambiguous_background", origin)
    color = np.median(selected, axis=0)
    difference = np.max(np.abs(crop.astype(np.float32) - color), axis=2)
    background = difference <= 14
    labels, _ = mh.label(background, _CONNECTIVITY)
    seed_labels = labels[seed_slice].reshape(-1)
    counts = np.bincount(seed_labels)
    counts[0] = 0
    label = int(counts.argmax())
    if label == 0 or counts[label] < len(samples) * 0.35:
        return _failure("no_background", origin)
    component = labels == label
    if not _enclosed_at_crop_edge(image, roi, component, background, color):
        return _failure("open_boundary", origin)

    filled = mh.close_holes(component)
    if np.mean(filled[seed_slice]) < 0.90:
        return _failure("text_not_contained", origin)
    bubble_area = (bubble[2] - bubble[0]) * (bubble[3] - bubble[1])
    if np.count_nonzero(filled) < max(36, bubble_area * 0.12):
        return _failure("insufficient_area", origin)

    # A threshold band in a gradient is not a physical boundary. Require a
    # substantial color edge around the filled outer contour (not text holes).
    shell = mh.dilate(filled, np.ones((3, 3), dtype=bool)) & ~filled
    shell_delta = difference[shell]
    if not len(shell_delta) or np.mean(shell_delta >= 24) < 0.65:
        return _failure("ambiguous_background", origin)

    ys, xs = np.nonzero(filled)
    span = min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    radius = max(2, min(12, round(span * 0.035)))
    body = _remove_tails(filled, radius)
    if not np.any(body):
        return _failure("insufficient_area", origin)
    # Keep only the body connected to most of the text, not detached remnants.
    bodies, _ = mh.label(body, _CONNECTIVITY)
    body_counts = np.bincount(bodies[seed_slice].reshape(-1))
    body_counts[0] = 0
    body_label = int(body_counts.argmax())
    if body_label == 0:
        return _failure("insufficient_area", origin)
    body = bodies == body_label
    if np.mean(body[seed_slice]) < 0.90:
        return _failure("text_not_contained", origin)
    mask = _erode(body, padding)
    if not np.any(mask):
        return _failure("padding_empty", origin)
    return BubbleBoundary(mask, origin, "estimated", "heuristic")
