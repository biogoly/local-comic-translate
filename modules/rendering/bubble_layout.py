"""Bounded rich-text fitting against a supplied, already-padded bubble mask.

The mask is typography geometry only. No image, cleanup mask or live document
is modified here. The existing editable QTextDocument is retained: candidates
change its soft-wrap width, font scale and placement, not its text engine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QFont, QTextDocument

from app.ui.canvas.text.vertical_layout import VerticalTextDocumentLayout
from .layout import (
    LayoutResult, _horizontal_line_metrics, _join_manual_breaks, _positive,
    _scale_font, _smallest_font_size, _unsupported,
)


class _MaskGeometry:
    """Constant-time full-rectangle checks; holes and concavities are retained."""
    def __init__(self, mask):
        values = np.asarray(mask)
        if values.ndim != 2 or not values.size:
            raise ValueError("Bubble mask must be a nonempty two-dimensional array")
        if values.dtype.kind not in 'buif' or not np.isfinite(values).all():
            raise ValueError("Bubble mask must contain finite numeric or boolean values")
        self.mask = values.astype(bool, copy=False)
        rows = np.count_nonzero(self.mask, axis=1)
        columns = np.count_nonzero(self.mask, axis=0)
        ys, xs = np.flatnonzero(rows), np.flatnonzero(columns)
        self.empty = not len(xs)
        if self.empty:
            self.bounds = self.body = (0, 0, 0, 0)
            self.integral = np.zeros((1, 1), dtype=np.int64)
            return
        x1, y1, x2, y2 = int(xs[0]), int(ys[0]), int(xs[-1] + 1), int(ys[-1] + 1)
        self.bounds = (x1, y1, x2, y2)
        # Search only the broad body, so a narrow tail cannot attract a last
        # line. Current diagnostics still use the complete supplied mask.
        body_rows = np.flatnonzero(rows >= max(1, rows.max() * 0.35))
        body_columns = np.flatnonzero(columns >= max(1, columns.max() * 0.35))
        self.body = (int(body_columns[0]), int(body_rows[0]),
                     int(body_columns[-1] + 1), int(body_rows[-1] + 1))
        forbidden = ~self.mask[y1:y2, x1:x2]
        dtype = np.int32 if forbidden.size < 2 ** 31 else np.int64
        self.integral = np.pad(forbidden.cumsum(axis=0, dtype=dtype).cumsum(axis=1, dtype=dtype),
                               ((1, 0), (1, 0)))

    def contains(self, rectangles, x, y):
        if self.empty or not rectangles:
            return False
        origin_x, origin_y, right, bottom = self.bounds
        table = self.integral
        for rect in rectangles:
            x1, y1 = math.floor(rect.left() + x), math.floor(rect.top() + y)
            x2, y2 = math.ceil(rect.right() + x), math.ceil(rect.bottom() + y)
            if x1 < origin_x or y1 < origin_y or x2 > right or y2 > bottom or x2 <= x1 or y2 <= y1:
                return False
            x1, x2, y1, y2 = x1 - origin_x, x2 - origin_x, y1 - origin_y, y2 - origin_y
            count = int(table[y2, x2]) - int(table[y1, x2]) - int(table[y2, x1]) + int(table[y1, x1])
            if count:
                return False
        return True


@dataclass
class _Shaped:
    result: LayoutResult
    rectangles: list[QRectF]
    extent: QRectF


def _validate(document, font_size, minimum_font_size, outline_width, line_spacing, position):
    if isinstance(document.documentLayout(), VerticalTextDocumentLayout):
        raise ValueError("Bubble fitting currently supports horizontal text only")
    font_size = _positive(font_size, "font_size")
    minimum_font_size = _positive(minimum_font_size, "minimum_font_size")
    _positive(line_spacing, "line_spacing")
    outline_width = float(outline_width)
    if not math.isfinite(outline_width) or outline_width < 0:
        raise ValueError("outline_width must be finite and non-negative")
    if len(position) != 2 or not all(math.isfinite(float(value)) for value in position):
        raise ValueError("current_position must contain two finite coordinates")
    return font_size, minimum_font_size, outline_width, tuple(float(value) for value in position)


def _shape(document, font_size, minimum_font_size, outline_width):
    rectangles, widths, illegal_break = _horizontal_line_metrics(document)
    rectangles = [rect.adjusted(-outline_width, -outline_width, outline_width, outline_width)
                  for rect in rectangles if rect.width() > 0 and rect.height() > 0]
    extent = QRectF()
    for rect in rectangles:
        extent = extent.united(rect)
    reasons = []
    if not document.toPlainText().strip():
        reasons.append('empty')
    if _unsupported(document):
        reasons.append('unsupported_content')
    if illegal_break:
        reasons.append('unbreakable_word')
    mean = sum(widths) / len(widths) if widths else 0
    score = (sum((width - mean) ** 2 for width in widths) / len(widths) / mean ** 2
             if mean > 0 else 0)
    if len(widths) > 1 and widths[-1] < mean * 0.35:
        score += 0.5
    size = document.size()
    result = LayoutResult(
        key='candidate', html=document.toHtml(), font_size=font_size,
        width=size.width(), height=size.height(), content_width=extent.width(),
        content_height=extent.height(), fits=not reasons, overflow_reasons=tuple(reasons),
        below_minimum=_smallest_font_size(document, font_size) < minimum_font_size - 0.01,
        default_font=QFont(document.defaultFont()), line_count=len(widths), balance_score=score,
    )
    return _Shaped(result, rectangles, extent)


def _positioned(shaped, geometry, x, y, *, key='candidate'):
    reasons = shaped.result.overflow_reasons
    if not geometry.contains(shaped.rectangles, x, y):
        reasons = (*reasons, 'boundary')
    return replace(shaped.result, key=key, offset_x=x, offset_y=y,
                   fits=not reasons, overflow_reasons=reasons)


def _current(document, geometry, font_size, minimum_font_size, outline_width, position):
    clone = document.clone()
    if clone.textWidth() <= 0:
        clone.setTextWidth(document.size().width())
    return _positioned(_shape(clone, font_size, minimum_font_size, outline_width),
                       geometry, *position, key='current')


def measure_bubble_layout(
    document: QTextDocument, *, mask, current_position=(0, 0), font_size: float,
    minimum_font_size: float = 10, outline_width: float = 0, line_spacing: float = 1.2,
) -> LayoutResult:
    """Check the current shaped text at its actual position, with no reflow."""
    font_size, minimum_font_size, outline_width, position = _validate(
        document, font_size, minimum_font_size, outline_width, line_spacing, current_position)
    return _current(document, _MaskGeometry(mask), font_size, minimum_font_size, outline_width, position)


def _find_position(shaped, geometry):
    if not shaped.result.fits or not shaped.rectangles or geometry.empty:
        return None
    left, top, right, bottom = geometry.body
    extent = shaped.extent
    min_y, max_y = top - extent.top(), bottom - extent.bottom()
    if max_y < min_y or extent.width() > right - left:
        return None
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    target_x = center_x - extent.center().x()
    target_y = center_y - extent.center().y()
    x_offsets = (target_x, target_x - (right - left) * 0.06, target_x + (right - left) * 0.06)
    # Position candidates are separate from text layouts. Include nearby pixels
    # around grid samples to avoid skipping narrow valid vertical placements.
    grid = np.linspace(min_y, max_y, min(33, max(1, math.ceil(max_y - min_y) + 1)))
    y_offsets = {target_y, min_y, max_y}
    for value in grid:
        y_offsets.update((float(value), max(min_y, float(value) - 1), min(max_y, float(value) + 1)))
    ordered = sorted(((x, y) for x in x_offsets for y in y_offsets),
                     key=lambda pair: ((pair[0] - target_x) / max(right-left, 1)) ** 2
                                    + ((pair[1] - target_y) / max(bottom-top, 1)) ** 2)
    for x, y in ordered:
        if geometry.contains(shaped.rectangles, x, y):
            result = _positioned(shaped, geometry, x, y)
            displacement = abs(x-target_x) / max(right-left, 1) + abs(y-target_y) / max(bottom-top, 1)
            return replace(result, balance_score=result.balance_score + displacement * 0.1)
    return None


def build_bubble_candidates(
    document: QTextDocument, *, mask, current_position=(0, 0), font_size: float,
    minimum_font_size: float = 10, maximum_font_size: float = 72,
    outline_width: float = 0, line_spacing: float = 1.2, keep_manual_breaks: bool = True,
) -> list[LayoutResult]:
    """Return contour-validated Current/Balanced/Larger editable layouts.

    At most 7 widths × 20 font sizes are shaped. Samples span the size range,
    then refine near observed fits: irregular contours need not be monotonic
    across line-wrap transitions. Every accepted placement is checked against
    every mask pixel covered by each shaped line rectangle, including outlines.
    """
    font_size, minimum_font_size, outline_width, position = _validate(
        document, font_size, minimum_font_size, outline_width, line_spacing, current_position)
    maximum_font_size = _positive(maximum_font_size, 'maximum_font_size')
    if maximum_font_size < minimum_font_size:
        raise ValueError('maximum_font_size must not be below minimum_font_size')
    geometry = _MaskGeometry(mask)
    current = _current(document, geometry, font_size, minimum_font_size, outline_width, position)
    if 'empty' in current.overflow_reasons or 'unsupported_content' in current.overflow_reasons:
        return [current]
    if geometry.empty:
        return [current, replace(current, key='balanced'), replace(current, key='larger')]
    prepared = document.clone()
    if not keep_manual_breaks:
        _join_manual_breaks(prepared)
    left, top, right, bottom = geometry.body
    maximum_width = max(1, right - left)
    widths = [max(1, maximum_width * ratio) for ratio in (1, 0.94, 0.86, 0.78, 0.70, 0.62, 0.54)]
    coarse = {float(value) for value in np.linspace(minimum_font_size, maximum_font_size, 9)}
    coarse.add(min(maximum_font_size, max(minimum_font_size, font_size)))
    interval = (maximum_font_size - minimum_font_size) / 8
    fitted = []
    fallback = None
    selected_shapes = {}
    for wrap_width in widths:
        seen = {}

        def evaluate(size):
            nonlocal fallback
            size = round(min(maximum_font_size, max(minimum_font_size, size)), 4)
            if size in seen:
                return seen[size]
            doc = prepared.clone()
            _scale_font(doc, size / font_size)
            doc.setTextWidth(wrap_width)
            shaped = _shape(doc, size, minimum_font_size, outline_width)
            result = _find_position(shaped, geometry)
            if fallback is None:
                fallback = _positioned(shaped, geometry,
                                       (left + right - shaped.result.width) / 2,
                                       (top + bottom - shaped.result.height) / 2)
                if fallback.fits:
                    fallback = replace(fallback, fits=False, overflow_reasons=('boundary',))
            if result is not None:
                fitted.append(result)
                selected_shapes[id(result)] = shaped
            seen[size] = result
            return result

        for size in sorted(coarse):
            evaluate(size)
        local_fits = [size for size, result in seen.items() if result is not None]
        if local_fits and interval > 0:
            best = max(local_fits)
            for scale in (1, 0.25):
                start = best
                for fraction in (0.25, 0.5, 0.75, 1):
                    result = evaluate(start + interval * scale * fraction)
                    if result is not None:
                        best = max(best, result.font_size)
    if not fitted:
        return [current, replace(fallback, key='balanced'), replace(fallback, key='larger')]
    largest = max(fitted, key=lambda result: (result.font_size, -result.balance_score))
    near_largest = [result for result in fitted if result.font_size >= largest.font_size * 0.9]
    balanced = min(near_largest, key=lambda result: (result.balance_score, -result.font_size))
    # Explicit final validation protects future changes to ranking/refinement.
    for result in (balanced, largest):
        if not geometry.contains(selected_shapes[id(result)].rectangles, result.offset_x, result.offset_y):
            raise RuntimeError('Bubble fit failed final boundary validation')
    return [current, replace(balanced, key='balanced'), replace(largest, key='larger')]
