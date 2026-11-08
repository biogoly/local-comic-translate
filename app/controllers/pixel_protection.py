"""Page-local, undoable protection masks; their overlay is never comic artwork."""
from copy import deepcopy
import logging
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.local_repair import rasterize_paths
from modules.repair.protection import decode_protection, encode_protection


logger = logging.getLogger(__name__)
PROTECTION_TOOLS = {"protect_pixels", "unprotect_pixels"}


class ProtectionOverlay(QtWidgets.QGraphicsItem):
    """Neither a brush path nor a patch pixmap, so serializers cannot capture it."""

    def __init__(self, mask):
        super().__init__()
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = (164, 92, 235)
        rgba[..., 3] = np.where(mask != 0, 95, 0)
        self.image = QtGui.QImage(rgba.data, w, h, w * 4, QtGui.QImage.Format_RGBA8888).copy()
        self.setZValue(8)
        self.setAcceptedMouseButtons(QtCore.Qt.NoButton)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.image.width(), self.image.height())

    def shape(self):
        # Persistent protection must not intercept itemAt() hit testing for
        # text selection, resizing, or context menus underneath the tint.
        return QtGui.QPainterPath()

    def paint(self, painter, option, widget=None):
        painter.drawImage(QtCore.QPointF(), self.image)


class PixelProtectionCommand(QtGui.QUndoCommand):
    """Change only one page's mask; do not alter its image/cleanup/text history."""

    def __init__(self, controller, path, before, after, *, had_before, page_id, epoch, text):
        super().__init__(text)
        self.controller = controller
        self.path = path
        self.before = deepcopy(before)
        self.after = deepcopy(after)
        self.had_before = had_before
        self.page_id = page_id
        self.epoch = epoch

    def _apply(self, value, present):
        main = self.controller.main
        if getattr(main, "_translation_project_epoch", None) != self.epoch:
            return
        if self.path not in main.image_files:
            return
        state = main.image_states.get(self.path)
        if state is None or state.get("page_id") != self.page_id:
            return
        if present:
            state["pixel_protection"] = deepcopy(value)
        else:
            state.pop("pixel_protection", None)
        self.controller.refresh_overlay()

    def redo(self):
        self._apply(self.after, self.after is not None)

    def undo(self):
        self._apply(self.before, self.had_before)


class PixelProtectionController(QtCore.QObject):
    def __init__(self, main):
        super().__init__(main.image_viewer)
        self.main = main
        self.viewer = main.image_viewer
        self.panel = main.local_repair_panel
        self.stroke = None
        self.preview = None
        self.overlays = {}
        self.viewer.protection_controller = self
        self.panel.protect.clicked.connect(lambda checked: self.select_tool("protect_pixels" if checked else None))
        self.panel.unprotect.clicked.connect(lambda checked: self.select_tool("unprotect_pixels" if checked else None))
        self.panel.show_protected.toggled.connect(self.refresh_overlay)
        self.viewer.page_changed.connect(self.refresh_overlay)

    @property
    def is_drawing(self):
        return self.stroke is not None

    def select_tool(self, tool):
        self.main.set_tool(tool)
        if tool in PROTECTION_TOOLS:
            self.main.set_slider_size(self.viewer.brush_size)
            self.panel.protection_toggle.setChecked(True)
            self.panel.show_protected.setChecked(True)
            self.refresh_overlay()

    def transient_items(self):
        return list(self.overlays.values()) + ([self.preview] if self.preview is not None else [])

    def _remove(self, item):
        if item is not None:
            try:
                if item.scene() is self.viewer._scene:
                    self.viewer._scene.removeItem(item)
            except RuntimeError:  # scene.clear() already deleted the C++ item
                pass

    def cancel(self):
        self._remove(self.preview)
        self.preview = None
        self.stroke = None

    def reset_page(self):
        self.cancel()
        for item in self.overlays.values():
            self._remove(item)
        self.overlays.clear()

    def refresh_overlay(self, *_):
        for item in self.overlays.values():
            self._remove(item)
        self.overlays.clear()
        if not self.panel.show_protected.isChecked() or not self.viewer.hasPhoto():
            return
        if self.viewer.webtoon_mode:
            indices = sorted(self.viewer.webtoon_manager.loaded_pages)
        else:
            indices = [self.main.curr_img_idx]
        for index in indices:
            if not 0 <= index < len(self.main.image_files):
                continue
            path = self.main.image_files[index]
            state = getattr(self.main, "image_states", {}).get(path, {})
            try:
                mask = decode_protection(state.get("pixel_protection"))
                if mask is None:
                    continue
                origin = (self.viewer.page_to_scene_coordinates(index, QtCore.QPointF())
                          if self.viewer.webtoon_mode else QtCore.QPointF())
                if origin is None:
                    continue
                overlay = ProtectionOverlay(mask)
                overlay.setPos(origin)
                self.viewer._scene.addItem(overlay)
                self.overlays[path] = overlay
            except ValueError as exc:
                logger.warning("Cannot display protection mask for %s: %s", path, exc)

    def press(self, scene_pos):
        self.cancel()
        repair = getattr(self.viewer, "repair_controller", None)
        if repair is None or not repair._available():
            return
        page = repair._page(scene_pos)
        if page is None:
            return
        path, origin = page
        try:
            image = self.main.image_ctrl.get_composited_page_image(path)
            if image is None:
                return
            h, w = image.shape[:2]
            local = scene_pos - origin
            if not (0 <= local.x() < w and 0 <= local.y() < h):
                return
            state = self.main.image_states.setdefault(path, {})
            before = deepcopy(state.get("pixel_protection"))
            mask = decode_protection(before, image.shape)
            if mask is None:
                mask = np.zeros((h, w), np.uint8)
            painter_path = QtGui.QPainterPath(scene_pos)
            painter_path.lineTo(scene_pos + QtCore.QPointF(0.001, 0))
            stack = self.main.undo_stacks[path]
            self.stroke = dict(path=path, origin=origin, shape=image.shape, mask=mask,
                               before=before, had_before="pixel_protection" in state,
                               page_id=state.get("page_id"),
                               epoch=getattr(self.main, "_translation_project_epoch", None),
                               stack=stack, revision=stack.index(), painter_path=painter_path,
                               diameter=max(1, self.viewer.brush_size),
                               protect=self.viewer.current_tool == "protect_pixels")
            self.preview = QtWidgets.QGraphicsPolygonItem()
            self.preview.setZValue(9)
            self.preview.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            self.preview.setBrush(QtGui.QColor(164, 92, 235, 150) if self.stroke["protect"]
                                  else QtGui.QColor(0, 210, 210, 150))
            self.preview.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self.viewer._scene.addItem(self.preview)
            self._refresh_preview()
        except Exception as exc:
            self.cancel()
            self.main.default_error_handler((type(exc), exc, traceback.format_exc()))

    def _refresh_preview(self):
        stroke = self.stroke
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(stroke["diameter"])
        stroker.setCapStyle(QtCore.Qt.RoundCap)
        stroker.setJoinStyle(QtCore.Qt.RoundJoin)
        outline = stroker.createStroke(stroke["painter_path"])
        page_bounds = QtGui.QPainterPath()
        h, w = stroke["shape"][:2]
        page_bounds.addRect(QtCore.QRectF(stroke["origin"], QtCore.QSizeF(w, h)))
        self.preview.setPolygon(outline.intersected(page_bounds).toFillPolygon())

    def move(self, scene_pos):
        if self.stroke is None:
            return
        stroke = self.stroke
        h, w = stroke["shape"][:2]
        origin = stroke["origin"]
        point = QtCore.QPointF(min(max(scene_pos.x(), origin.x()), origin.x() + w - 0.001),
                              min(max(scene_pos.y(), origin.y()), origin.y() + h - 0.001))
        stroke["painter_path"].lineTo(point)
        self._refresh_preview()

    def release(self):
        stroke = self.stroke
        if stroke is None:
            return
        self.cancel()
        try:
            path = stroke["path"]
            state = self.main.image_states.get(path)
            stack = self.main.undo_stacks.get(path)
            repair = getattr(self.viewer, "repair_controller", None)
            current_page = repair._page() if repair is not None else None
            if (state is None or path not in self.main.image_files
                    or stack is not stroke["stack"] or stack.index() != stroke["revision"]
                    or state.get("page_id") != stroke["page_id"]
                    or getattr(self.main, "_translation_project_epoch", None) != stroke["epoch"]
                    or state.get("pixel_protection") != stroke["before"]
                    or (not self.viewer.webtoon_mode and (current_page is None or current_page[0] != path))):
                return
            pen = QtGui.QPen(QtCore.Qt.white, stroke["diameter"], QtCore.Qt.SolidLine,
                             QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            footprint = rasterize_paths(stroke["shape"],
                                        [(stroke["painter_path"], pen, QtGui.QBrush(QtCore.Qt.NoBrush))],
                                        stroke["origin"]) != 0
            after_mask = stroke["mask"].copy()
            after_mask[footprint] = 255 if stroke["protect"] else 0
            if np.array_equal(after_mask, stroke["mask"]):
                return
            command = PixelProtectionCommand(
                self, path, stroke["before"], encode_protection(after_mask),
                had_before=stroke["had_before"], page_id=stroke["page_id"], epoch=stroke["epoch"],
                text=self.tr("Protect artwork") if stroke["protect"] else self.tr("Unprotect artwork"))
            stack.push(command)
        except Exception as exc:
            self.main.default_error_handler((type(exc), exc, traceback.format_exc()))
