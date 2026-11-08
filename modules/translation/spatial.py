"""Pixel geometry for optional vision context; never changes OCR or artwork.

Callers supply regions in the coordinate system of the exact delivered image.
The normal page image is unchanged; visible webtoons are mapped before this
boundary. Provider encoding must preserve these dimensions and orientation.
"""
from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from numbers import Real

from .context import normalized_text


class SpatialContextError(ValueError):
    """Do not send misleading coordinates for an invalid region."""


def spatial_context_active(engine):
    return bool(getattr(engine, 'spatial_context_enabled', False) is True
                and getattr(engine, 'img_as_llm_input', False)
                and getattr(engine, 'supports_images', True))


def _numbers(value, count):
    try:
        values = tuple(value)
        if len(values) != count or any(isinstance(v, bool) or not isinstance(v, Real)
                                       or not math.isfinite(v) for v in values):
            raise ValueError
        return tuple(float(v) for v in values)
    except (TypeError, ValueError, OverflowError):
        raise SpatialContextError('Text-box coordinates are invalid. Correct the region or turn off '
                                  'experimental text-box coordinates before translating.') from None


def geometry_signature(block):
    """Canonical unrotated rectangle, clockwise angle and item-local pivot."""
    x1, y1, x2, y2 = _numbers(getattr(block, 'xyxy', None), 4)
    if x2 <= x1 or y2 <= y1:
        raise SpatialContextError('A text box has no usable area. Correct the region or turn off '
                                  'experimental text-box coordinates before translating.')
    angle, = _numbers((getattr(block, 'angle', 0),), 1)
    origin = getattr(block, 'tr_origin_point', None)
    if origin is None or (hasattr(origin, '__len__') and len(origin) == 0):
        origin = ((x2 - x1) / 2, (y2 - y1) / 2)
    ox, oy = _numbers(origin, 2)
    return x1, y1, x2, y2, (angle + 180) % 360 - 180, ox, oy


def region_polygon(block):
    x1, y1, x2, y2, angle, ox, oy = geometry_signature(block)
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    if angle == 0:
        return corners
    cx, cy = x1 + ox, y1 + oy
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    if angle in (-180, -90, 90):
        # A tiny cosine at 90 degrees must not push an exact edge into the next decimal cell.
        cosine, sine = round(cosine), round(sine)
    points = tuple((cx + (x - cx) * cosine - (y - cy) * sine,
                    cy + (x - cx) * sine + (y - cy) * cosine) for x, y in corners)
    if not all(math.isfinite(v) for point in points for v in point):
        raise SpatialContextError('The rotated text box cannot be mapped to the page image.')
    return points


def clip_polygon(points, width, height):
    """Clip the rotated rectangle, not just its bounding envelope."""
    points = list(points)
    for axis, limit, keep_greater in ((0, 0., True), (0, float(width), False),
                                      (1, 0., True), (1, float(height), False)):
        output = []
        if not points:
            break
        previous = points[-1]
        previous_inside = previous[axis] >= limit if keep_greater else previous[axis] <= limit
        for current in points:
            current_inside = current[axis] >= limit if keep_greater else current[axis] <= limit
            if current_inside != previous_inside:
                t = (limit - previous[axis]) / (current[axis] - previous[axis])
                point = tuple(previous[i] + t * (current[i] - previous[i]) for i in (0, 1))
                output.append(point)
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        points = output
    return points


def _normalized_edge(pixel, axis_length, rounding):
    """Round outward for the prompt, with at most 0.1 source pixel per decimal cell."""
    normalized = min(1., max(0., pixel / axis_length))
    # Five decimals cover axes up to 10,000 px. Larger source images need more.
    places = max(5, len(str(axis_length * 10 - 1)))
    # Avoid moving an exact decimal grid edge due to another binary float multiplication.
    return float(Decimal(str(normalized)).quantize(Decimal(1).scaleb(-places), rounding=rounding))


def _position_hint(center_x, center_y):
    horizontal = 'left' if center_x < 1/3 else 'right' if center_x >= 2/3 else 'center'
    vertical = 'top' if center_y < 1/3 else 'bottom' if center_y >= 2/3 else 'middle'
    return 'center' if (horizontal, vertical) == ('center', 'middle') else f'{vertical} {horizontal}'


def build_visual_reference(blocks, image_shape, keys):
    if len(image_shape) not in (2, 3) or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise SpatialContextError('The page image has no usable dimensions for text-box coordinates.')
    blocks, keys = tuple(blocks), tuple(keys)
    if len(blocks) != len(keys) or len(set(keys)) != len(keys):
        raise SpatialContextError('Text-box coordinates require a unique request ID for every region.')
    height, width = map(int, image_shape[:2])
    # Normalize only for this comparison. The translation request retains the exact OCR text.
    phrases = [' '.join(normalized_text(getattr(block, 'text', '')).split()) for block in blocks]
    occurrences = Counter(phrase for phrase in phrases if phrase)
    regions = []
    for index, (block, key) in enumerate(zip(blocks, keys)):
        points = region_polygon(block)
        clipped = any(x < 0 or y < 0 or x > width or y > height for x, y in points)
        visible_points = clip_polygon(points, width, height)
        row = dict(id=key, bbox_normalized=None, angle=geometry_signature(block)[4],
                   reading_order=index, clipped=clipped, visible=False)
        if visible_points:
            xs, ys = zip(*visible_points)
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
            # Touching a border alone does not make any source pixels visible.
            area = abs(sum(x * visible_points[(i + 1) % len(visible_points)][1]
                           - y * visible_points[(i + 1) % len(visible_points)][0]
                           for i, (x, y) in enumerate(visible_points))) / 2
            if right > left and bottom > top and area > 1e-9:
                row['visible'] = True
                row['bbox_normalized'] = [_normalized_edge(left, width, ROUND_FLOOR),
                                          _normalized_edge(top, height, ROUND_FLOOR),
                                          _normalized_edge(right, width, ROUND_CEILING),
                                          _normalized_edge(bottom, height, ROUND_CEILING)]
                if occurrences[phrases[index]] > 1:
                    row['position'] = _position_hint((left/width + right/width)/2, (top/height + bottom/height)/2)
        regions.append(row)
    return dict(image_width=width, image_height=height, coordinate_space='normalized',
                bbox_format='left,top,right,bottom', angle_convention='clockwise_degrees',
                regions=regions)
