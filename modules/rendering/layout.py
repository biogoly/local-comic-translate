"""Reversible, rich-document text fitting with truthful fit diagnostics.

All dimensions are in the text item's unrotated local coordinates. Callers
resolve the desired box and retain the item's scene transform. Nothing here
mutates a live document, inserts automatic hard line breaks, or edits pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from PySide6.QtCore import QRectF, QTextBoundaryFinder
from PySide6.QtGui import QFont, QFontInfo, QTextCharFormat, QTextCursor, QTextDocument

from app.ui.canvas.text.vertical_layout import VerticalTextDocumentLayout


@dataclass(frozen=True)
class LayoutResult:
    key: str
    html: str
    font_size: float
    width: float
    height: float
    content_width: float
    content_height: float
    fits: bool
    overflow_reasons: tuple[str, ...]
    below_minimum: bool
    # Retained separately from HTML; Qt does not serialize document defaults.
    default_font: QFont
    line_count: int = 0
    balance_score: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _fragments(document):
    block = document.begin()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid():
                yield fragment
            iterator += 1
        block = block.next()


def _scale_font(document: QTextDocument, factor: float) -> None:
    """Change only size, preserving all other character and paragraph formats."""
    if abs(factor - 1.0) < 1e-9:
        return
    default_font = QFont(document.defaultFont())
    original_points = default_font.pointSizeF()
    original_pixels = default_font.pixelSize()
    # Gather ranges before merging formats; QTextFragments are live handles.
    ranges = []
    for fragment in _fragments(document):
        fmt = fragment.charFormat()
        points = fmt.fontPointSize()
        pixels = fmt.font().pixelSize()
        ranges.append((fragment.position(), fragment.length(), points, pixels))
    if original_points > 0:
        default_font.setPointSizeF(max(0.5, original_points * factor))
    elif original_pixels > 0:
        default_font.setPixelSize(max(1, round(original_pixels * factor)))
    document.setDefaultFont(default_font)
    cursor = QTextCursor(document)
    cursor.beginEditBlock()
    for start, length, points, pixels in ranges:
        fmt = QTextCharFormat()
        if points > 0:
            fmt.setFontPointSize(max(0.5, points * factor))
        elif pixels > 0:
            fmt.setProperty(QTextCharFormat.Property.FontPixelSize, max(1, round(pixels * factor)))
        else:
            continue  # Inherited size follows the document default.
        cursor.setPosition(start)
        cursor.setPosition(start + length, QTextCursor.MoveMode.KeepAnchor)
        cursor.mergeCharFormat(fmt)
    cursor.endEditBlock()


def _join_manual_breaks(document: QTextDocument) -> None:
    """Explicit reflow choice: replace separators with spaces, preserving ranges.

    Each separator is one UTF-16 unit, as is its replacement, so selection
    outline offsets and character formatting remain attached to the same text.
    The terminal paragraph separator is never removed.
    """
    positions = [i for i in range(document.characterCount() - 1)
                 if document.characterAt(i) in ("\n", "\r", "\u2028", "\u2029")]
    cursor = QTextCursor(document)
    cursor.beginEditBlock()
    for position in reversed(positions):
        cursor.setPosition(position)
        cursor.setPosition(position + 1, QTextCursor.MoveMode.KeepAnchor)
        fmt = cursor.charFormat()
        cursor.insertText(" ", fmt)
    cursor.endEditBlock()


def _line_boundaries(text: str) -> set[int]:
    finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Line, text)
    allowed = {0}
    while True:
        position = finder.toNextBoundary()
        if position < 0:
            return allowed
        allowed.add(position)


def _horizontal_line_metrics(document: QTextDocument):
    """Actual shaped per-line extents, shared by rectangular and bubble fitting."""
    document.size()  # Force layout before obtaining QTextLines.
    rectangles = []
    widths = []
    illegal_break = False
    block = document.begin()
    while block.isValid():
        layout = block.layout()
        permitted = _line_boundaries(block.text())
        for index in range(layout.lineCount()):
            line = layout.lineAt(index)
            rect = line.naturalTextRect().translated(layout.position())
            # Natural rect retains underline/line-height space; glyph bounds
            # additionally include italic and accented ink overhang.
            for run in line.glyphRuns():
                raw = run.rawFont()
                for glyph, position in zip(run.glyphIndexes(), run.positions()):
                    rect = rect.united(raw.boundingRect(glyph).translated(position + layout.position()))
            rectangles.append(rect)
            widths.append(line.naturalTextWidth())
            endpoint = line.textStart() + line.textLength()
            if index < layout.lineCount() - 1 and endpoint not in permitted:
                illegal_break = True
        block = block.next()
    return rectangles, widths, illegal_break


def _horizontal_metrics(document: QTextDocument):
    rectangles, widths, illegal_break = _horizontal_line_metrics(document)
    bounds = QRectF()
    for rect in rectangles:
        bounds = bounds.united(rect)
    return bounds, widths, illegal_break


def _unsupported(document: QTextDocument) -> bool:
    return bool(document.rootFrame().childFrames()) or any(
        fragment.charFormat().isImageFormat() for fragment in _fragments(document)
    )


def _smallest_font_size(document: QTextDocument, fallback: float) -> float:
    """Readability reflects small inline words too, not just the item default."""
    sizes = []
    for fragment in _fragments(document):
        if not fragment.text().strip():
            continue
        font = fragment.charFormat().font().resolve(document.defaultFont())
        size = font.pointSizeF()
        if size <= 0 and font.pixelSize() > 0:
            size = QFontInfo(font).pointSizeF()
        if math.isfinite(size) and size > 0:
            sizes.append(size)
    return min(sizes) if sizes else fallback


def _measure(
    document: QTextDocument, *, key: str, font_size: float, width: float,
    height: float, minimum_font_size: float, outline_width: float,
    vertical: bool, line_spacing: float,
) -> LayoutResult:
    reasons = []
    if not document.toPlainText().strip():
        reasons.append("empty")
    if _unsupported(document):
        reasons.append("unsupported_content")
    if vertical:
        layout = document.documentLayout()
        if not isinstance(layout, VerticalTextDocumentLayout):
            layout = VerticalTextDocumentLayout(document, line_spacing=line_spacing)
            document.setDocumentLayout(layout)
            layout.set_max_size(document.textWidth(), height)
        else:
            layout.update_layout()
        state = layout._layout_state
        bounds = QRectF(state.content_left_x, document.documentMargin(),
                        state.content_width, max(state.content_height, state.min_required_height))
        widths = [node.width() for node in state.nodes]
        illegal_break = False
    else:
        bounds, widths, illegal_break = _horizontal_metrics(document)

    occupied = bounds.adjusted(-outline_width, -outline_width, outline_width, outline_width)
    size = document.size()
    tolerance = 0.5
    if occupied.left() < -tolerance or occupied.right() > width + tolerance:
        reasons.append("width")
    if occupied.top() < -tolerance or occupied.bottom() > height + tolerance or size.height() > height + tolerance:
        reasons.append("height")
    if vertical and size.width() > width + tolerance:
        reasons.append("width")
    if illegal_break:
        reasons.append("unbreakable_word")
    # Compare only real line widths; a perfectly equal set is most balanced.
    average = sum(widths) / len(widths) if widths else 0.0
    balance = (sum((value - average) ** 2 for value in widths) / len(widths) / average ** 2
               if average > 0 else 0.0)
    if len(widths) > 1 and widths[-1] < average * 0.35:
        balance += 0.5
    reasons = tuple(dict.fromkeys(reasons))
    return LayoutResult(
        key=key, html=document.toHtml(), font_size=font_size,
        width=float(size.width()), height=float(size.height()),
        content_width=max(0.0, occupied.width()), content_height=max(0.0, occupied.height()),
        fits=not reasons, overflow_reasons=reasons,
        below_minimum=_smallest_font_size(document, font_size) < minimum_font_size - 0.01,
        default_font=QFont(document.defaultFont()), line_count=len(widths), balance_score=balance,
    )


def measure_layout(
    document: QTextDocument, *, width: float, height: float, font_size: float,
    minimum_font_size: float = 10, outline_width: float = 0,
    vertical: bool = False, line_spacing: float = 1.2,
) -> LayoutResult:
    """Measure only the current layout; inexpensive enough for debounced UI."""
    width = _positive(width, "width")
    height = _positive(height, "height")
    font_size = _positive(font_size, "font_size")
    minimum_font_size = _positive(minimum_font_size, "minimum_font_size")
    outline_width = float(outline_width)
    if not math.isfinite(outline_width) or outline_width < 0:
        raise ValueError("outline_width must be finite and non-negative")
    line_spacing = _positive(line_spacing, "line_spacing")
    clone = document.clone()
    source_layout = document.documentLayout()
    if vertical:
        current_layout = VerticalTextDocumentLayout(clone, line_spacing=line_spacing)
        clone.setDocumentLayout(current_layout)
        current_layout.set_max_size(getattr(source_layout, "max_width", width),
                                    getattr(source_layout, "max_height", height))
    elif clone.textWidth() <= 0:
        clone.setTextWidth(document.size().width())
    return _measure(clone, key="current", font_size=font_size, width=width,
                    height=height, minimum_font_size=minimum_font_size,
                    outline_width=outline_width, vertical=vertical, line_spacing=line_spacing)


def build_layout_candidates(
    document: QTextDocument, *, width: float, height: float, font_size: float,
    minimum_font_size: float = 10, maximum_font_size: float | None = None,
    outline_width: float = 0, vertical: bool = False, line_spacing: float = 1.2,
    keep_manual_breaks: bool = True,
) -> list[LayoutResult]:
    """Return Current, Balanced and Larger candidates without changing source.

    ``width`` and ``height`` constrain the desired box. Current retains its
    existing wrap width. Alternatives adjust only the wrap width and font size;
    all normal automatic wrapping stays soft. Large rich text/table documents
    are diagnosed but not automatically refit.
    """
    width = _positive(width, "width")
    height = _positive(height, "height")
    font_size = _positive(font_size, "font_size")
    minimum_font_size = _positive(minimum_font_size, "minimum_font_size")
    maximum_font_size = _positive(maximum_font_size if maximum_font_size is not None else max(font_size, 72), "maximum_font_size")
    if maximum_font_size < minimum_font_size:
        raise ValueError("maximum_font_size must not be below minimum_font_size")
    outline_width = float(outline_width)
    if not math.isfinite(outline_width) or outline_width < 0:
        raise ValueError("outline_width must be finite and non-negative")
    line_spacing = _positive(line_spacing, "line_spacing")
    common = dict(width=width, height=height, minimum_font_size=minimum_font_size,
                  outline_width=outline_width, vertical=vertical, line_spacing=line_spacing)
    current = measure_layout(document, font_size=font_size, **common)
    if "empty" in current.overflow_reasons or "unsupported_content" in current.overflow_reasons:
        return [current]

    prepared = document.clone()
    if not keep_manual_breaks:
        _join_manual_breaks(prepared)

    # A small, deterministic search; no per-keystroke work or unbounded DP.
    candidate_widths = [width] if vertical else [width * ratio for ratio in (1.0, 0.94, 0.86, 0.78, 0.70)]
    results = []

    def evaluate(size, wrap_width):
        doc = prepared.clone()
        _scale_font(doc, size / font_size)
        if vertical:
            layout = VerticalTextDocumentLayout(doc, line_spacing=line_spacing)
            doc.setDocumentLayout(layout)
            layout.set_max_size(wrap_width, height)
        else:
            doc.setTextWidth(wrap_width)
        return _measure(doc, key="candidate", font_size=size, **common)

    for wrap_width in candidate_widths:
        low, high = minimum_font_size, maximum_font_size
        smallest = evaluate(low, wrap_width)
        best = smallest
        if smallest.fits:
            # Search quarter-point resolution, then validate the chosen result.
            for _ in range(12):
                if high - low < 0.25:
                    break
                mid = math.floor((low + high) * 2) / 4
                result = evaluate(mid, wrap_width)
                if result.fits:
                    low, best = mid, result
                else:
                    high = mid
            # Include the endpoint when exactly on the largest permitted size.
            endpoint = evaluate(maximum_font_size, wrap_width)
            if endpoint.fits:
                best = endpoint
        results.append(best)

    fitting = [result for result in results if result.fits]
    if fitting:
        largest = max(fitting, key=lambda result: (result.font_size, -result.balance_score))
        # Keep balance within 10% of the largest readable candidate.
        near_largest = [result for result in fitting if result.font_size >= largest.font_size * 0.9]
        balanced = min(near_largest, key=lambda result: (result.balance_score, -result.font_size))
    else:
        largest = balanced = results[0]  # Explicit no-fit at minimum, never success.
    return [current, replace(balanced, key="balanced"), replace(largest, key="larger")]
