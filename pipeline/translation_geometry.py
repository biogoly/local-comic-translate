"""Image-relative translation snapshots, independent of OCR/repair geometry."""
from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui

from modules.translation.spatial import geometry_signature, region_polygon


def loaded_page_geometry(main, path, identities):
    """Read loaded scene copies by stable ID, expressed in physical-page pixels.

    Absence is not an edit: webtoon pages and their scene copies can be unloaded
    or published asynchronously. Saved page state guards those cases separately.
    """
    if not getattr(main, 'webtoon_mode', False) or path not in main.image_files:
        return {}
    index = main.image_files.index(path)
    manager = main.image_viewer.webtoon_manager
    if index not in manager.loaded_pages:
        return {}
    blocks = [block for block in main.blk_list if block.region_id in identities]
    if not blocks:
        return {}
    converter = getattr(manager, 'coordinate_converter', None)
    if converter is not None:
        origin = converter.page_local_to_scene_position(QtCore.QPointF(), index)
        dx, dy = origin.x(), origin.y()
    else:
        image = manager.image_loader.get_image_data(index)
        if image is None:
            raise ValueError('A loaded translation page has no original geometry.')
        dx, dy = (manager.webtoon_width - image.shape[1]) / 2, manager.image_positions[index]
    result = {}
    for block in blocks:
        if block.region_id in result:
            raise ValueError('A loaded translation region has duplicate identities.')
        x1, y1, x2, y2, angle, px, py = geometry_signature(block)
        result[block.region_id] = (x1 - dx, y1 - dy, x2 - dx, y2 - dy, angle, px, py)
    return result


def compose_visible_translation(pages, blocks):
    """Keep scene spacing/centering so every region uses one translation.

    Pages contain original RGB pixels and exact integer row crops. Blank gaps
    remain blank; unlike the legacy visible-area image, pages are not squeezed
    together. Fractional centered origins are rendered at their actual position.
    Neither the source pixels nor live TextBlocks are modified.
    """
    if not pages:
        return None, []
    left = math.floor(min(page['scene_x'] for page in pages))
    top = math.floor(min(page['scene_y'] + page['crop_top'] for page in pages))
    right = math.ceil(max(page['scene_x'] + page['image'].shape[1] for page in pages))
    bottom = math.ceil(max(page['scene_y'] + page['crop_bottom'] for page in pages))
    canvas = QtGui.QImage(right - left, bottom - top, QtGui.QImage.Format_RGB888)
    if canvas.isNull():
        raise ValueError('The visible translation image could not be allocated.')
    canvas.fill(QtCore.Qt.white)
    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
    try:
        for page in pages:
            pixels = np.ascontiguousarray(page['image'][page['crop_top']:page['crop_bottom']])
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError('The visible translation image must contain RGB pixels.')
            h, w = pixels.shape[:2]
            image = QtGui.QImage(pixels.data, w, h, pixels.strides[0], QtGui.QImage.Format_RGB888)
            painter.drawImage(QtCore.QPointF(page['scene_x'] - left,
                page['scene_y'] + page['crop_top'] - top), image)
    finally:
        painter.end()
    height, width = canvas.height(), canvas.width()
    pixels = np.frombuffer(canvas.constBits(), dtype=np.uint8).reshape(height, canvas.bytesPerLine())
    result = pixels[:, :width * 3].reshape(height, width, 3).copy()
    copies = []
    for block in blocks:
        copied = block.deep_copy()
        for name in ('xyxy', 'bubble_xyxy'):
            value = getattr(copied, name, None)
            if value is not None:
                setattr(copied, name, np.asarray(value, dtype=float) - [left, top, left, top])
        if isinstance(getattr(block, '_page_id', None), str):
            copied._page_id = block._page_id
        copies.append(copied)
    return result, copies


def capture_visible_translation(main, pipeline, single_block=False):
    """Capture original page portions and stable region ownership on the GUI thread."""
    viewer = main.image_viewer
    manager = viewer.webtoon_manager
    layout = manager.layout_manager
    bounds = viewer.mapToScene(viewer.viewport().rect()).boundingRect()
    pages = []
    for index in sorted(layout.get_pages_for_scene_bounds(bounds)):
        image = manager.image_loader.get_image_data(index)
        if image is None:
            raise ValueError('A visible original page is unavailable for translation.')
        h, w = image.shape[:2]
        y = float(manager.image_positions[index])
        crop_top = max(0, min(h, int(max(0, bounds.top() - y))))
        crop_bottom = max(0, min(h, int(min(h, bounds.bottom() - y))))
        if crop_bottom <= crop_top:
            continue
        path = main.image_files[index]
        pages.append(dict(index=index, image=image, scene_x=(manager.webtoon_width - w) / 2,
                          scene_y=y, crop_top=crop_top, crop_bottom=crop_bottom,
                          page_id=main.image_states.get(path, {}).get('page_id')))
    if not pages:
        return None, []
    selected = pipeline.get_selected_block() if single_block else None
    candidates = [selected] if selected is not None else ([] if single_block else main.blk_list)
    blocks = []
    owners = {}
    for path in main.image_files:
        state = main.image_states.get(path, {})
        for block in state.get('blk_list', []):
            owners.setdefault(block.region_id, state.get('page_id'))
    for block in candidates:
        polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in region_polygon(block)])
        shape = QtGui.QPainterPath()
        shape.addPolygon(polygon)
        shape.closeSubpath()
        intersections = [page for page in pages if shape.intersects(QtCore.QRectF(
            page['scene_x'], page['scene_y'] + page['crop_top'], page['image'].shape[1],
            page['crop_bottom'] - page['crop_top']))]
        # Explicit selected translation keeps its text even if the box is
        # outside the viewport; its visual reference will say it is outside.
        if not intersections and not single_block:
            continue
        copied = block.deep_copy()
        copied._page_id = owners.get(block.region_id) or getattr(block, '_page_id', None)
        if not copied._page_id and intersections:
            copied._page_id = intersections[0]['page_id']
        blocks.append(copied)
    return compose_visible_translation(pages, blocks)
