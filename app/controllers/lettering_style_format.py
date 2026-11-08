"""Rich-text style previews and exact item restoration without live selections.

Only the requested format properties are merged. In particular, changing a
font family must never replace the whole QFont and flatten word emphasis.
"""
from __future__ import annotations

import copy
import unicodedata

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from modules.utils.lettering_styles import normalize_style_properties


DEFAULT_STYLE_FIELDS = frozenset({
    'font_family', 'font_size', 'text_color', 'outline', 'outline_color',
    'outline_width', 'line_spacing', 'alignment',
})


def _color(value, fallback='#000000'):
    color = QtGui.QColor(value) if value is not None else QtGui.QColor(fallback)
    return color if color.isValid() else QtGui.QColor(fallback)


def _color_text(value, fallback='#000000'):
    color = _color(value, fallback)
    return color.name(QtGui.QColor.NameFormat.HexArgb if color.alpha() != 255 else QtGui.QColor.NameFormat.HexRgb)


def style_from_properties(properties: dict) -> dict:
    """Capture base style, not the current word selection, as portable values."""
    props = TextItemProperties.from_dict(properties)
    family = props.font_family or QtWidgets.QApplication.font().family()
    horizontal = int(props.alignment) & int(QtCore.Qt.AlignmentFlag.AlignHorizontal_Mask)
    return normalize_style_properties({
        'font_family': family, 'font_size': float(props.font_size),
        'text_color': _color_text(props.text_color),
        'outline': props.outline,
        'outline_color': _color_text(props.outline_color, '#ffffff'),
        'outline_width': float(props.outline_width or 0),
        'line_spacing': float(props.line_spacing), 'alignment': horizontal,
        'bold': bool(props.bold), 'italic': bool(props.italic),
        'underline': bool(props.underline),
    })


def capture_style(item) -> dict:
    return style_from_properties(TextItemProperties.from_text_item(item).to_dict())


def _default_font(props):
    font = QtGui.QFont()
    if props.document_font and font.fromString(props.document_font):
        return font
    font.setFamily(props.font_family or QtWidgets.QApplication.font().family())
    font.setPointSizeF(max(1, float(props.font_size)))
    font.setBold(bool(props.bold))
    font.setItalic(bool(props.italic))
    font.setUnderline(bool(props.underline))
    return font


def _document(properties):
    props = TextItemProperties.from_dict(properties)
    document = QtGui.QTextDocument()
    document.setDefaultFont(_default_font(props))
    document.setHtml(props.text)
    if props.width is not None:
        document.setTextWidth(props.width)
    return document


def _outline_info(value):
    if isinstance(value, OutlineInfo):
        return copy.deepcopy(value)
    if not isinstance(value, dict):
        raise ValueError('Invalid text outline')
    outline_type = value.get('type', OutlineType.Selection)
    if not isinstance(outline_type, OutlineType):
        outline_type = OutlineType(outline_type)
    return OutlineInfo(int(value['start']), int(value['end']),
                       _color(value.get('color'), '#ffffff'), float(value['width']), outline_type)


def styled_properties(before: dict, style: dict, fields=DEFAULT_STYLE_FIELDS) -> dict:
    """Build an independent full snapshot; the live item is never consulted."""
    after = copy.deepcopy(before)
    style = normalize_style_properties(style, strict=True)
    selected = set(fields) & style.keys()
    if not selected:
        return after
    props = TextItemProperties.from_dict(before)
    doc = _document(before)
    font = doc.defaultFont()
    char_format = QtGui.QTextCharFormat()
    if 'font_family' in selected:
        char_format.setFontFamilies([style['font_family']])
        font.setFamily(style['font_family'])
    if 'font_size' in selected:
        char_format.setFontPointSize(style['font_size'])
        font.setPointSizeF(style['font_size'])
    if 'text_color' in selected:
        char_format.setForeground(QtGui.QColor(style['text_color']))
    if 'bold' in selected:
        char_format.setFontWeight(QtGui.QFont.Weight.Bold if style['bold'] else QtGui.QFont.Weight.Normal)
        font.setBold(style['bold'])
    if 'italic' in selected:
        char_format.setFontItalic(style['italic'])
        font.setItalic(style['italic'])
    if 'underline' in selected:
        char_format.setFontUnderline(style['underline'])
        font.setUnderline(style['underline'])
    cursor = QtGui.QTextCursor(doc)
    cursor.select(QtGui.QTextCursor.SelectionType.Document)
    if char_format.properties():
        cursor.mergeCharFormat(char_format)
    doc.setDefaultFont(font)
    block_format = QtGui.QTextBlockFormat()
    if 'line_spacing' in selected:
        block_format.setLineHeight(style['line_spacing'] * 100,
                                  QtGui.QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
    if 'alignment' in selected:
        vertical_bits = int(props.alignment) & ~int(QtCore.Qt.AlignmentFlag.AlignHorizontal_Mask)
        block_format.setAlignment(QtCore.Qt.AlignmentFlag(style['alignment'] | vertical_bits))
    if block_format.properties():
        cursor.mergeBlockFormat(block_format)
    for key in selected:
        if key in {'text_color', 'outline_color'}:
            after[key] = QtGui.QColor(style[key])
        elif key == 'alignment':
            after[key] = block_format.alignment()
        else:
            after[key] = style[key]
    after['text'] = doc.toHtml()
    after['document_font'] = font.toString()
    if selected & {'outline', 'outline_color', 'outline_width'}:
        outlines = [_outline_info(value) for value in props.selection_outlines]
        outlines = [value for value in outlines if value.type != OutlineType.Full_Document]
        if after.get('outline', props.outline):
            outlines.append(OutlineInfo(0, max(0, doc.characterCount() - 1),
                _color(after.get('outline_color'), '#ffffff'),
                float(after.get('outline_width', 1) or 0), OutlineType.Full_Document))
        after['selection_outlines'] = outlines
    return after


def apply_style_properties(item, properties: dict):
    """Restore every base/style field and HTML while keeping editor interaction.

    Position, width and transforms come from the snapshot. Horizontal height is
    document-computed and may naturally change when a different style reflows.
    """
    props = TextItemProperties.from_dict(properties)
    old_cursor = item.textCursor()
    anchor, position = old_cursor.anchor(), old_cursor.position()
    remembered_selection = copy.deepcopy(getattr(item, '_format_selection', None))
    interaction = item.textInteractionFlags()
    insertion_guard = getattr(item, '_restoring_insertion_format', False)
    with QtCore.QSignalBlocker(item):
        item._restoring_insertion_format = True
        try:
            for name in ('font_family', 'font_size', 'alignment', 'line_spacing',
                         'outline', 'outline_width', 'bold', 'italic', 'underline', 'direction'):
                setattr(item, name, getattr(props, name))
            item.text_color = _color(props.text_color)
            item.outline_color = QtGui.QColor(props.outline_color) if props.outline_color is not None else None
            item.region_id = props.region_id
            item.layout_preferences = copy.deepcopy(props.layout_preferences)
            # No selected-word-aware setters: this is a complete saved document.
            item.setHtml(props.text)
            item.document().setDefaultFont(_default_font(props))
            item.setDefaultTextColor(item.text_color)
            item._apply_text_direction()
            item.set_vertical(bool(props.vertical))
            if props.width is not None:
                item.setTextWidth(props.width)
            if item.vertical and item.layout:
                item.layout.set_line_spacing(props.line_spacing)
                item.layout.set_max_size(props.width or item.boundingRect().width(),
                    props.layout_preferences.get('bound_height', props.height or item.boundingRect().height()))
            item.selection_outlines = [_outline_info(value) for value in props.selection_outlines]
            item.setPos(QtCore.QPointF(*props.position))
            item.setRotation(props.rotation)
            item.setScale(props.scale)
            item.set_lettering_transform(props.lettering_transform)
            if props.transform_origin is not None:
                item.setTransformOriginPoint(QtCore.QPointF(*props.transform_origin))
            limit = max(0, item.document().characterCount() - 1)
            cursor = QtGui.QTextCursor(item.document())
            cursor.setPosition(min(anchor, limit))
            cursor.setPosition(min(position, limit), QtGui.QTextCursor.MoveMode.KeepAnchor)
            item.setTextCursor(cursor)
            item._format_selection = remembered_selection
            item.setTextInteractionFlags(interaction)
            item.update()
        finally:
            item._restoring_insertion_format = insertion_guard


def _missing_characters(text, font):
    missing = []
    for character in dict.fromkeys(text):
        if character.isspace() or unicodedata.category(character) in {'Cc', 'Cf', 'Cs'}:
            continue
        # QTextLayout uses Qt's font fallback, unlike testing only the primary
        # QRawFont. A supported CJK fallback must not be reported as missing.
        layout = QtGui.QTextLayout(character, font)
        layout.beginLayout()
        line = layout.createLine()
        if line.isValid():
            line.setLineWidth(100000)
        layout.endLayout()
        runs = layout.glyphRuns()
        glyphs = [index for run in runs for index in run.glyphIndexes()]
        if not glyphs or 0 in glyphs:
            missing.append(character)
    return ''.join(missing)


def font_warnings(properties: dict) -> list[dict]:
    """Return structured warnings for requested families and actual missing glyphs."""
    doc = _document(properties)
    known = {family.casefold() for family in QtGui.QFontDatabase.families()}
    requested = {}
    requested.setdefault(properties.get('font_family') or doc.defaultFont().family(), '')
    block = doc.firstBlock()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid():
                font = fragment.charFormat().font()
                family = font.family() or doc.defaultFont().family()
                key = (family, font.toString())
                requested.setdefault(key, '')
                requested[key] += fragment.text()
            iterator += 1
        block = block.next()
    warnings, absent = [], set()
    missing_by_font = {}
    for key, text in requested.items():
        family = key[0] if isinstance(key, tuple) else key
        if family.casefold() not in known and family not in absent:
            warnings.append({'code': 'missing_font', 'font': family, 'characters': ''})
            absent.add(family)
        if text:
            font = QtGui.QFont()
            font.fromString(key[1])
            missing = _missing_characters(text, font)
            if missing:
                missing_by_font.setdefault(family, '')
                missing_by_font[family] += missing
    for family, text in missing_by_font.items():
        warnings.append({'code': 'missing_glyphs', 'font': family,
                         'characters': ''.join(dict.fromkeys(text))[:32]})
    return warnings
