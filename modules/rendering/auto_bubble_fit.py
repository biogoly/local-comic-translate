"""Optional bubble fitting for newly rendered text, shared by all pipelines.

Inputs and outputs are page-local. The existing rectangular result is kept on
every unavailable/unsupported/no-fit path. This module never edits blocks,
source images, cleanup masks, or existing rendered items and creates no widget.
"""

from __future__ import annotations

import copy
import logging
import math
import re

import numpy as np
from PySide6.QtGui import QColor, QFont, QGuiApplication, QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument

from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from .bubble_boundary import estimate_bubble_boundary
from .bubble_layout import build_bubble_candidates


logger = logging.getLogger(__name__)
_HTML = re.compile(r'<(div|span|p|br|html|body|style)[^>]*>', re.IGNORECASE)


def load_original_for_fitting(path, expected_shape=None):
    from app.path_materialization import load_original_rgb
    return load_original_rgb(path, expected_shape)


def _valid_image(image):
    return (isinstance(image, np.ndarray) and image.dtype == np.uint8
            and image.ndim == 3 and image.shape[2] == 3
            and image.shape[0] > 0 and image.shape[1] > 0)


def _finite_number(value):
    return not isinstance(value, bool) and math.isfinite(float(value))


def _inside_page(bounds, width, height):
    try:
        return (len(bounds) == 4 and all(_finite_number(v) for v in bounds)
                and 0 <= bounds[0] < bounds[2] <= width
                and 0 <= bounds[1] < bounds[3] <= height)
    except (TypeError, ValueError, OverflowError):
        return False


def _outline_value(outline, name, default=None):
    return outline.get(name, default) if isinstance(outline, dict) else getattr(outline, name, default)


def _is_full_outline(outline):
    kind = _outline_value(outline, 'type')
    return kind == OutlineType.Full_Document or getattr(kind, 'value', kind) == 'full_document'


def _document(properties, source_text):
    """Match the canvas's plain/rich-document initialization without widgets."""
    document = QTextDocument()
    family = properties.font_family.strip() if isinstance(properties.font_family, str) else ''
    font = QFont(family or QGuiApplication.font().family())
    font.setPointSizeF(float(properties.font_size))
    font.setBold(bool(properties.bold))
    font.setItalic(bool(properties.italic))
    font.setUnderline(bool(properties.underline))
    rich = bool(_HTML.search(properties.text))
    if rich and properties.document_font:
        saved = QFont()
        if saved.fromString(properties.document_font):
            font = saved
    document.setDefaultFont(font)
    if rich:
        # An existing rich document is authoritative. A plain translation must
        # not flatten its character spans or explicit paragraph formatting.
        document.setHtml(properties.text)
    else:
        document.setPlainText(source_text if source_text is not None else properties.text)
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        char_format = QTextCharFormat()
        char_format.setFont(font)
        if properties.text_color is not None and properties.text_color.isValid():
            char_format.setForeground(properties.text_color)
        cursor.mergeCharFormat(char_format)
        paragraph = QTextBlockFormat()
        paragraph.setAlignment(properties.alignment)
        paragraph.setLineHeight(float(properties.line_spacing) * 100,
                                QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        cursor.mergeBlockFormat(paragraph)
    option = document.defaultTextOption()
    option.setTextDirection(properties.direction)
    document.setDefaultTextOption(option)
    if properties.width is not None and float(properties.width) > 0:
        document.setTextWidth(float(properties.width))
    return document, rich


def _outlines(properties, length):
    outlines = []
    for original in properties.selection_outlines:
        if isinstance(original, dict):
            kind = original.get('type', 'selection')
            outlines.append(OutlineInfo(int(original['start']), int(original['end']),
                                        QColor(original['color']), float(original['width']),
                                        kind if isinstance(kind, OutlineType) else OutlineType(kind)))
        else:
            outlines.append(copy.deepcopy(original))
    has_full = False
    for outline in outlines:
        if _is_full_outline(outline):
            has_full = True
            outline.start, outline.end = 0, length
    color = properties.outline_color
    if not has_full and color is not None and color.isValid():
        outlines.append(OutlineInfo(0, length, copy.copy(color), float(properties.outline_width),
                                    OutlineType.Full_Document))
    return outlines


def fit_new_text_item(
    properties: TextItemProperties, block, original_image, settings, *, source_text=None,
) -> TextItemProperties:
    """Fit one new horizontal item; retain the existing Box result on failure.

    ``block`` must be the original page-space block, before render-area
    adjustment. ``source_text`` is the logical target before the old wrapper
    inserts line breaks. Explicit newlines are kept; no prose is rewritten.
    Callers retain existing rendered regions rather than passing them here.
    """
    unchanged = copy.deepcopy(properties)

    def fallback(reason):
        logger.debug('Automatic bubble fitting retained Box layout: %s', reason)
        return unchanged

    if not getattr(settings, 'auto_bubble_fit', False):
        return unchanged
    if not _valid_image(original_image):
        return fallback('original image unavailable')
    try:
        if (properties.vertical or not _finite_number(properties.rotation)
                or not math.isclose(float(properties.rotation) % 360, 0, abs_tol=1e-6)
                or not _finite_number(properties.scale)
                or not math.isclose(float(properties.scale), 1, abs_tol=1e-6)):
            return fallback('unsupported transform')
        angle = getattr(block, 'angle', 0)
        if not _finite_number(angle) or not math.isclose(float(angle) % 360, 0, abs_tol=1e-6):
            return fallback('rotated source region')
        if properties.layout_preferences:
            return fallback('existing explicit layout')
        metadata = getattr(block, 'region_metadata', {})
        if isinstance(metadata, dict) and metadata.get('protections', {}).get('keep_layout', False):
            return fallback('layout protected')
        height, width = original_image.shape[:2]
        source_bounds = getattr(block, 'xyxy', None)
        if not _inside_page(source_bounds, width, height):
            return fallback('region crosses a page boundary')
        if len(properties.position) != 2 or not all(_finite_number(v) for v in properties.position):
            return fallback('invalid item position')
        if source_text is not None and not isinstance(source_text, str):
            return fallback('invalid logical text')
        # Partial outline positions cannot safely be reused when replacing the
        # wrapper's text with a differently indexed logical source.
        rich = bool(_HTML.search(properties.text))
        if (not rich and source_text is not None and source_text != properties.text
                and any(not _is_full_outline(outline) for outline in properties.selection_outlines)):
            return fallback('partial outline needs explicit review')
        boundary = estimate_bubble_boundary(
            original_image, source_bounds, getattr(block, 'bubble_xyxy', None),
            padding=getattr(settings, 'bubble_padding', 8),
        )
        if not boundary.available:
            return fallback(boundary.reason)
        document, _ = _document(properties, source_text)
        if not document.toPlainText().strip():
            return fallback('empty text')
        outlines = _outlines(properties, max(0, document.characterCount() - 1))
        if any(not _finite_number(outline.width) or outline.width < 0 for outline in outlines):
            return fallback('invalid outline width')
        extent = max([float(_outline_value(outline, 'width', 0)) for outline in outlines] + [0])
        candidates = build_bubble_candidates(
            document, mask=boundary.mask,
            current_position=(float(properties.position[0]) - boundary.origin[0],
                              float(properties.position[1]) - boundary.origin[1]),
            font_size=float(properties.font_size),
            minimum_font_size=float(getattr(settings, 'min_font_size', 10)),
            maximum_font_size=float(getattr(settings, 'max_font_size', 72)),
            outline_width=extent, line_spacing=float(properties.line_spacing),
            keep_manual_breaks=True,
        )
        chosen = next((candidate for key in ('balanced', 'larger') for candidate in candidates
                       if candidate.key == key and candidate.fits and not candidate.below_minimum), None)
        if chosen is None:
            return fallback('no suitable readable layout')
        position = (boundary.origin[0] + chosen.offset_x, boundary.origin[1] + chosen.offset_y)
        if not _inside_page((*position, position[0] + chosen.width, position[1] + chosen.height), width, height):
            return fallback('fitted item crosses a page boundary')
        fitted = copy.deepcopy(properties)
        fitted.text = chosen.html
        fitted.font_size = chosen.font_size
        fitted.document_font = chosen.default_font.toString()
        fitted.width, fitted.height = chosen.width, chosen.height
        fitted.position = position
        fitted.selection_outlines = outlines
        fitted.outline = any(_is_full_outline(outline) for outline in outlines)
        fallback_width = float(properties.width) if properties.width is not None and float(properties.width) > 0 else float(source_bounds[2] - source_bounds[0])
        fallback_height = float(properties.height) if properties.height is not None and float(properties.height) > 0 else float(source_bounds[3] - source_bounds[1])
        fitted.layout_preferences = {
            'mode': 'bubble', 'boundary_shape': 'auto',
            'padding': float(getattr(settings, 'bubble_padding', 8)),
            'bound_width': fallback_width, 'bound_height': fallback_height,
            'min_font_size': float(getattr(settings, 'min_font_size', 10)),
            'keep_manual_breaks': True,
            'boundary_box': [float(v) for v in (getattr(block, 'bubble_xyxy', None)
                                               if getattr(block, 'bubble_xyxy', None) is not None else source_bounds)],
        }
        return fitted
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError, RuntimeError):
        logger.debug('Automatic bubble fitting failed; retaining Box layout', exc_info=True)
        return unchanged
