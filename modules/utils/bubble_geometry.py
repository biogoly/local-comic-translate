"""Small, dependency-light helpers for speech-bubble mask geometry."""

from __future__ import annotations

import numpy as np


def build_fallback_bubble_clip(
    mask_shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
    bubble_xyxy,
    *,
    inset: int,
    image: np.ndarray | None = None,
    seed_bbox=None,
) -> np.ndarray:
    """Build an ellipse or rectangle fallback from the bubble bounding box.

    The detector stores only a bounding box, not the bubble contour. An ellipse
    is conservative for ordinary balloons but incorrectly removes the corners
    of rectangular captions. When three or more inset corners match the
    background sampled around the detected text, use an inset rectangle.
    """
    height, width = mask_shape[:2]
    x1, y1, _x2, _y2 = [int(v) for v in bounds]
    bx1, by1, bx2, by2 = [int(v) for v in bubble_xyxy[:4]]

    bx1_rel = bx1 + inset - x1
    by1_rel = by1 + inset - y1
    bx2_rel = bx2 - inset - x1
    by2_rel = by2 - inset - y1

    yy, xx = np.ogrid[:height, :width]
    center_x = (bx1_rel + bx2_rel) / 2.0
    center_y = (by1_rel + by2_rel) / 2.0
    radius_x = max(1.0, (bx2_rel - bx1_rel) / 2.0)
    radius_y = max(1.0, (by2_rel - by1_rel) / 2.0)
    ellipse = (
        ((xx - center_x) / radius_x) ** 2
        + ((yy - center_y) / radius_y) ** 2
    ) <= 1.0

    if not _looks_like_rectangular_bubble(image, bubble_xyxy, seed_bbox, inset):
        return ellipse

    rectangle = np.zeros((height, width), dtype=bool)
    rx1 = max(0, min(width, bx1_rel))
    ry1 = max(0, min(height, by1_rel))
    rx2 = max(rx1, min(width, bx2_rel))
    ry2 = max(ry1, min(height, by2_rel))
    if rx2 > rx1 and ry2 > ry1:
        rectangle[ry1:ry2, rx1:rx2] = True
    return rectangle if np.any(rectangle) else ellipse


def _looks_like_rectangular_bubble(
    image: np.ndarray | None,
    bubble_xyxy,
    seed_bbox,
    inset: int,
    tolerance: float = 24.0,
) -> bool:
    if image is None or seed_bbox is None or image.size == 0:
        return False

    height, width = image.shape[:2]
    bx1, by1, bx2, by2 = _clipped_box(bubble_xyxy, width, height)
    sx1, sy1, sx2, sy2 = _clipped_box(seed_bbox, width, height)
    if bx2 - bx1 < 8 or by2 - by1 < 8 or sx2 <= sx1 or sy2 <= sy1:
        return False

    gray = _to_gray(image)
    seed = gray[sy1:sy2, sx1:sx2]
    if seed.size == 0:
        return False
    background = _dominant_value(seed)

    bubble_span = min(bx2 - bx1, by2 - by1)
    sample_radius = max(1, min(4, int(round(bubble_span * 0.035))))
    offset = max(1, inset) + sample_radius
    corners = (
        (bx1 + offset, by1 + offset),
        (bx2 - offset - 1, by1 + offset),
        (bx1 + offset, by2 - offset - 1),
        (bx2 - offset - 1, by2 - offset - 1),
    )

    matching = 0
    sampled = 0
    for cx, cy in corners:
        px1 = max(0, cx - sample_radius)
        py1 = max(0, cy - sample_radius)
        px2 = min(width, cx + sample_radius + 1)
        py2 = min(height, cy + sample_radius + 1)
        patch = gray[py1:py2, px1:px2]
        if patch.size == 0:
            continue
        sampled += 1
        if abs(float(np.median(patch)) - background) <= tolerance:
            matching += 1

    return sampled == 4 and matching >= 3


def _clipped_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    return np.mean(image[..., :3].astype(np.float32), axis=2)


def _dominant_value(values: np.ndarray) -> float:
    histogram, edges = np.histogram(values, bins=16, range=(0, 256))
    index = int(np.argmax(histogram))
    return float((edges[index] + edges[index + 1]) / 2.0)
