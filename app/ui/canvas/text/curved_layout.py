"""Editable circular lettering built from Qt's shaped glyph clusters.

The source QTextDocument and its QTextLines remain the editing model. Painting,
selection, caret placement and hit testing all use the same cluster transforms;
no bitmap or detached replacement text is involved.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QPointF, QRectF, QSizeF, QLineF, QTextBoundaryFinder
from PySide6.QtGui import (QAbstractTextDocumentLayout, QFontMetricsF, QGlyphRun,
                          QPainterPath, QPalette, QPen, QTextCursor, QTextLayout,
                          QTextOption, QTransform)


@dataclass
class CurvedCluster:
    start: int
    end: int
    cell: QRectF
    transform: QTransform
    runs: list = field(default_factory=list)
    format: object = None


@dataclass
class CurvedLine:
    block_position: int
    qt_line: object
    placements: list = field(default_factory=list)
    carets: list = field(default_factory=list)
    bounds: QRectF = field(default_factory=QRectF)


class CurvedTextDocumentLayout(QAbstractTextDocumentLayout):
    """A bounded arc for horizontal text, retaining native shaping and wrapping."""

    def __init__(self, document, arc_angle, line_spacing=1.2):
        super().__init__(document)
        self.arc_angle = float(arc_angle)
        self.line_spacing = line_spacing
        self.lines = []
        self._size = QSizeF(1, 1)
        self._block_bounds = {}
        self._updating = False
        self._offset = QPointF()
        self._curvature = 0.0
        self._center_x = 0.0

    def set_arc_angle(self, angle):
        if self.arc_angle != angle:
            self.arc_angle = float(angle)
            self.update_layout()

    def documentChanged(self, position, removed, added):
        self.update_layout()

    def _map_baseline(self, x, baseline):
        s = x - self._center_x
        k = self._curvature
        if abs(k) < 1e-10:
            return QPointF(x, baseline)
        return QPointF(self._center_x + math.sin(k * s) / k,
                       baseline + (1.0 - math.cos(k * s)) / k)

    def _cluster_transform(self, x, baseline):
        point = self._map_baseline(x, baseline)
        transform = QTransform()
        transform.translate(point.x(), point.y())
        transform.rotate(math.degrees(self._curvature * (x - self._center_x)))
        transform.translate(-x, -baseline)
        return transform

    @staticmethod
    def _boundaries(text):
        finder = QTextBoundaryFinder(QTextBoundaryFinder.Grapheme, text)
        result = [0]
        while True:
            position = finder.toNextBoundary()
            if position < 0:
                break
            result.append(position)
        return result

    @staticmethod
    def _shaped_clusters(line, boundaries):
        """Associate shaped glyphs with source graphemes without run indexes.

        Qt can return zero stringIndexes for an entire run after a font
        fallback (for example an accented letter in a comic font). Querying
        each source range retains native shaping and avoids those indexes.
        Partial ligature queries share a glyph: join their source ranges and
        paint that glyph only once, instead of splitting or duplicating it.
        """
        start, end = line.textStart(), line.textStart() + line.textLength()
        positions = [p for p in boundaries if start <= p <= end]
        parents, shared, entries = {}, {}, []
        flags = (QTextLayout.GlyphRunRetrievalFlag.RetrieveGlyphIndexes
                 | QTextLayout.GlyphRunRetrievalFlag.RetrieveGlyphPositions)

        def root(position):
            while parents[position] != position:
                parents[position] = parents[parents[position]]
                position = parents[position]
            return position

        for lower, upper in zip(positions, positions[1:]):
            parents[lower] = lower
            occurrences = {}
            for run in line.glyphRuns(lower, upper - lower, flags):
                font = run.rawFont()
                split = bool(run.flags() & QGlyphRun.SplitLigature)
                for glyph, position in zip(run.glyphIndexes(), run.positions()):
                    key = (font, glyph, position.x(), position.y())
                    ordinal = occurrences.get(key, 0)
                    occurrences[key] = ordinal + 1
                    key += (ordinal,)
                    if split and key in shared:
                        first, current = root(shared[key]), root(lower)
                        parents[max(first, current)] = min(first, current)
                    else:
                        entries.append((lower, glyph, position, font))
                        if split:
                            shared[key] = lower
        return [(root(position), glyph, point, font)
                for position, glyph, point, font in entries]

    def update_layout(self):
        if self._updating:
            return
        self._updating = True
        try:
            document = self.document()
            margin = document.documentMargin()
            logical_width = document.textWidth()
            # Unbounded text is laid out once at its natural width. Existing
            # manually drawn boxes always supply an explicit positive width.
            available = max(1.0, logical_width - margin * 2) if logical_width > 0 else 1000000.0
            self.lines = []
            y = margin
            block = document.begin()
            widest = 1.0
            while block.isValid():
                layout = block.layout()
                layout.clearLayout()
                layout.setFont(document.defaultFont())
                option = QTextOption(document.defaultTextOption())
                option.setAlignment(block.blockFormat().alignment())
                option.setTextDirection(block.blockFormat().layoutDirection())
                option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
                layout.setTextOption(option)
                formats = []
                iterator = block.begin()
                while not iterator.atEnd():
                    fragment = iterator.fragment()
                    if fragment.isValid():
                        entry = QTextLayout.FormatRange()
                        entry.start = fragment.position() - block.position()
                        entry.length = fragment.length()
                        entry.format = fragment.charFormat()
                        formats.append(entry)
                    iterator += 1
                layout.setFormats(formats)
                y += max(0.0, block.blockFormat().topMargin())
                layout.beginLayout()
                while True:
                    line = layout.createLine()
                    if not line.isValid():
                        break
                    line.setLineWidth(available)
                    line.setPosition(QPointF(margin, y))
                    self.lines.append(CurvedLine(block.position(), line))
                    widest = max(widest, line.naturalTextWidth())
                    bf = block.blockFormat()
                    height = line.height()
                    if bf.lineHeightType() == 1 and bf.lineHeight() > 0:
                        height *= bf.lineHeight() / 100.0
                    elif bf.lineHeightType() == 2 and bf.lineHeight() > 0:
                        height = bf.lineHeight()
                    elif bf.lineHeightType() == 3:
                        height = max(height, bf.lineHeight())
                    elif bf.lineHeightType() == 4:
                        height += bf.lineHeight()
                    y += max(1.0, height)
                layout.endLayout()
                y += max(0.0, block.blockFormat().bottomMargin())
                block = block.next()
            if logical_width <= 0:
                available = widest
                logical_width = widest + margin * 2
            self._curvature = math.radians(self.arc_angle) / max(available, 1.0)
            self._center_x = margin + available / 2.0
            bounds = QRectF(0, 0, max(1.0, logical_width), max(1.0, y + margin))
            for record in self.lines:
                line = record.qt_line
                block = document.findBlock(record.block_position)
                start, end = line.textStart(), line.textStart() + line.textLength()
                boundaries = self._boundaries(block.text())
                entries = self._shaped_clusters(line, boundaries)
                starts = sorted({entry[0] for entry in entries} | {end})
                groups = {}
                baseline = line.y() + line.ascent()
                # A bidi boundary can have two visual cursor locations. Derive
                # each cluster's advance cell from the shaped glyph positions,
                # not cursorToX(end), which may jump across an entire RTL run.
                advance_cells = {}
                for cluster_start, glyph, position, font in entries:
                    advance = font.advancesForGlyphIndexes([glyph])[0].x()
                    left, right = sorted((position.x(), position.x() + advance))
                    if cluster_start in advance_cells:
                        old_left, old_right = advance_cells[cluster_start]
                        left, right = min(left, old_left), max(right, old_right)
                    advance_cells[cluster_start] = left, right
                for cluster_start, glyph, position, font in entries:
                    cluster_end = starts[bisect.bisect_right(starts, cluster_start)]
                    if cluster_start not in groups:
                        x1, x2 = advance_cells[cluster_start]
                        cell = QRectF(min(x1, x2), line.y(), max(abs(x2 - x1), 0.5), line.height())
                        transform = self._cluster_transform(cell.center().x(), baseline)
                        cursor = QTextCursor(document)
                        cursor.setPosition(record.block_position + cluster_start)
                        cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
                        groups[cluster_start] = CurvedCluster(record.block_position + cluster_start,
                            record.block_position + cluster_end, cell, transform, format=cursor.charFormat())
                    placement = groups[cluster_start]
                    glyph_run = QGlyphRun()
                    glyph_run.setRawFont(font)
                    glyph_run.setGlyphIndexes([glyph])
                    glyph_run.setPositions([position])
                    placement.runs.append(glyph_run)
                    glyph_bounds = font.boundingRect(glyph).translated(position)
                    bounds = bounds.united(placement.transform.mapRect(glyph_bounds))
                record.placements = list(groups.values())
                for placement in record.placements:
                    rectangle = placement.transform.mapRect(placement.cell)
                    record.bounds = record.bounds.united(rectangle)
                    bounds = bounds.united(rectangle)
                # Native cursorToX supplies correct bidi and ligature positions;
                # only grapheme boundaries are exposed for mouse hit testing.
                positions = [p for p in boundaries if start <= p <= end]
                if not positions:
                    positions = [start]
                for position in positions:
                    x = line.cursorToX(position)[0]
                    transform = self._cluster_transform(x, baseline)
                    caret = QLineF(transform.map(QPointF(x, line.y())),
                                   transform.map(QPointF(x, line.y() + line.height())))
                    record.carets.append((record.block_position + position, caret))
                    bounds = bounds.united(QRectF(caret.p1(), caret.p2()).normalized())
                if not record.placements:
                    record.bounds = QRectF(margin, line.y(), max(1.0, available), line.height())
            self._offset = QPointF(max(0.0, -bounds.left()), max(0.0, -bounds.top()))
            shift = QTransform.fromTranslate(self._offset.x(), self._offset.y())
            self._block_bounds = {}
            for record in self.lines:
                for placement in record.placements:
                    placement.transform = placement.transform * shift
                record.carets = [(position, QLineF(caret.p1() + self._offset, caret.p2() + self._offset))
                                 for position, caret in record.carets]
                record.bounds.translate(self._offset)
                self._block_bounds[record.block_position] = self._block_bounds.get(record.block_position, QRectF()).united(record.bounds)
            self._size = QSizeF(max(logical_width, bounds.right() + self._offset.x()),
                                max(1.0, bounds.bottom() + self._offset.y()))
            self.documentSizeChanged.emit(self._size)
            self.update.emit()
        finally:
            self._updating = False

    def documentSize(self):
        return self._size

    def pageCount(self):
        return 1

    def frameBoundingRect(self, frame):
        return QRectF(QPointF(), self._size)

    def blockBoundingRect(self, block):
        return self._block_bounds.get(block.position(), QRectF())

    def line_for_position(self, position):
        # A wrap boundary belongs to the following line, except document end.
        candidates = [record for record in self.lines if record.block_position + record.qt_line.textStart() <= position]
        return candidates[-1] if candidates else (self.lines[0] if self.lines else None)

    def caret_line(self, position):
        record = self.line_for_position(position)
        if record is None or not record.carets:
            return QLineF()
        return QLineF(min(record.carets, key=lambda entry: abs(entry[0] - position))[1])

    @staticmethod
    def _distance(point, line):
        dx, dy = line.dx(), line.dy()
        length = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((point.x() - line.x1()) * dx + (point.y() - line.y1()) * dy) / length)) if length else 0.0
        return (point.x() - line.x1() - t * dx) ** 2 + (point.y() - line.y1() - t * dy) ** 2

    def hitTest(self, point, accuracy):
        if accuracy == Qt.ExactHit:
            inside = False
            for record in self.lines:
                for placement in record.placements:
                    path = QPainterPath()
                    path.addRect(placement.cell)
                    if placement.transform.map(path).contains(point):
                        inside = True
                        break
            if not inside:
                return -1
        candidates = [(position, caret) for record in self.lines for position, caret in record.carets]
        return min(candidates, key=lambda entry: self._distance(point, entry[1]))[0] if candidates else 0

    def draw(self, painter, context):
        painter.save()
        for record in self.lines:
            for placement in record.placements:
                fmt = placement.format
                selected = next((selection for selection in context.selections
                                 if selection.cursor.selectionStart() < placement.end
                                 and selection.cursor.selectionEnd() > placement.start), None)
                painter.save()
                painter.setTransform(placement.transform, True)
                background = selected.format.background() if selected else fmt.background()
                if background.style() != Qt.NoBrush:
                    painter.fillRect(placement.cell, background)
                brush = selected.format.foreground() if selected else fmt.foreground()
                color = brush.color() if brush.style() != Qt.NoBrush else context.palette.color(QPalette.Text)
                painter.setPen(color)
                for run in placement.runs:
                    painter.drawGlyphRun(QPointF(), run)
                font = fmt.font().resolve(self.document().defaultFont())
                metrics = QFontMetricsF(font)
                baseline = record.qt_line.y() + record.qt_line.ascent()
                pen = QPen(color, max(0.5, metrics.lineWidth()))
                painter.setPen(pen)
                if fmt.fontUnderline():
                    painter.drawLine(QPointF(placement.cell.left(), baseline + metrics.underlinePos()),
                                     QPointF(placement.cell.right(), baseline + metrics.underlinePos()))
                if fmt.fontStrikeOut():
                    painter.drawLine(QPointF(placement.cell.left(), baseline - metrics.strikeOutPos()),
                                     QPointF(placement.cell.right(), baseline - metrics.strikeOutPos()))
                if fmt.fontOverline():
                    painter.drawLine(QPointF(placement.cell.left(), baseline - metrics.ascent()),
                                     QPointF(placement.cell.right(), baseline - metrics.ascent()))
                painter.restore()
        if context.cursorPosition >= 0:
            painter.setPen(QPen(context.palette.color(QPalette.Text), 1.0))
            painter.drawLine(self.caret_line(context.cursorPosition))
        painter.restore()
