"""Project-serializable paint settings shared by page and webtoon repair masks."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen


def stroke_record(path, pen, brush):
    record = {
        'path': path,
        'pen': pen.color().name(QColor.HexArgb),
        'brush': brush.color().name(QColor.HexArgb),
        'width': pen.widthF(),
        'pen_style': pen.style().value,
        'cap_style': pen.capStyle().value,
        'join_style': pen.joinStyle().value,
        'brush_style': brush.style().value,
        'fill_rule': path.fillRule().value,
    }
    if pen.style() == Qt.CustomDashLine:
        record['dash_pattern'] = pen.dashPattern()
    if pen.dashOffset():
        record['dash_offset'] = pen.dashOffset()
    if pen.isCosmetic():
        record['cosmetic'] = True
    if pen.joinStyle() in (Qt.MiterJoin, Qt.SvgMiterJoin):
        record['miter_limit'] = pen.miterLimit()
    return record


def add_stored_stroke(scene, path, record):
    pen = QPen(QColor(record['pen']))
    pen.setWidthF(record['width'])
    pen.setStyle(Qt.PenStyle(record.get('pen_style', Qt.SolidLine.value)))
    pen.setCapStyle(Qt.PenCapStyle(record.get('cap_style', Qt.RoundCap.value)))
    pen.setJoinStyle(Qt.PenJoinStyle(record.get('join_style', Qt.RoundJoin.value)))
    if 'dash_pattern' in record:
        pen.setDashPattern(record['dash_pattern'])
    if 'dash_offset' in record:
        pen.setDashOffset(record['dash_offset'])
    if 'cosmetic' in record:
        pen.setCosmetic(record['cosmetic'])
    if 'miter_limit' in record:
        pen.setMiterLimit(record['miter_limit'])

    color = QColor(record['brush'])
    if 'brush_style' in record:
        brush_style = Qt.BrushStyle(record['brush_style'])
    else:
        # Older projects filled only generated segmentation paths.
        brush_style = Qt.SolidPattern if color == QColor('#80ff0000') else Qt.NoBrush
    brush = QBrush(color, brush_style)
    if 'fill_rule' in record:
        path = QPainterPath(path)
        path.setFillRule(Qt.FillRule(record['fill_rule']))
    return scene.addPath(path, pen, brush)
