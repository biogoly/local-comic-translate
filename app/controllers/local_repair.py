"""Undoable, page-local repairs; no model calls or destructive base-image edits."""
import math
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.commands.brush import ClearBrushStrokesCommand
from app.ui.commands.base import PathCommandBase
from app.ui.commands.inpaint import PatchInsertCommand
from app.ui.dayu_widgets.message import MMessage
from app.controllers.clone_brush import CloneBrushController
from app.ui.dialogs.gradient_repair import GradientRepairDialog
from modules.repair.protection import get_protection


def rasterize_paths(shape, paths, origin=QtCore.QPointF()):
    """Render exactly the visible mask, with no hidden dilation."""
    height, width = shape[:2]
    image = QtGui.QImage(width, height, QtGui.QImage.Format_Grayscale8)
    image.fill(0)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.translate(-origin)
    for path, pen, brush in paths:
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawPath(path)
    painter.end()
    return np.array(image.constBits()).reshape(height, image.bytesPerLine())[:, :width].copy()


def repair_patch(current, replacement, mask):
    """Composite only masked pixels; retain all other existing edits in the patch."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    before = current[y1:y2, x1:x2]
    source = replacement[y1:y2, x1:x2] if isinstance(replacement, np.ndarray) and replacement.ndim == 3 else replacement
    alpha = mask[y1:y2, x1:x2, None].astype(np.float32) / 255
    after = np.rint(before.astype(np.float32) * (1 - alpha) + np.asarray(source, dtype=np.float32) * alpha).astype(np.uint8)
    if np.array_equal(before, after):
        return None
    return {"bbox": [x1, y1, x2 - x1, y2 - y1], "image": after}


class ClearPageMaskCommand(ClearBrushStrokesCommand):
    """Consume one page's mask without deleting the rest of a cross-page stroke."""
    def __init__(self, viewer, rect):
        super().__init__(viewer, [rect])
        self.cut = QtGui.QPainterPath()
        self.cut.addRect(rect)
        self.remainders = []

    def redo(self):
        super().redo()
        self.remainders = []
        for props in self.properties_list:
            shape = QtGui.QPainterPath()
            if props['brush'] == '#80ff0000':
                shape = QtGui.QPainterPath(props['path'])
            settings = props['pen_settings']
            if settings['style'] != QtCore.Qt.NoPen:
                stroker = QtGui.QPainterPathStroker()
                stroker.setWidth(max(1, settings['width']))
                stroker.setCapStyle(settings['cap'])
                stroker.setJoinStyle(settings['join'])
                shape = shape.united(stroker.createStroke(props['path']))
            remaining = shape.subtracted(self.cut)
            if remaining.isEmpty():
                continue
            item = self.scene.addPath(remaining, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(QtGui.QColor(255, 0, 0, 128)))
            self.remainders.append(PathCommandBase.save_path_properties(item))

    def undo(self):
        for props in self.remainders:
            item = PathCommandBase.find_matching_item(self.scene, props)
            if item is not None:
                self.scene.removeItem(item)
        super().undo()


class LocalRepairController(QtCore.QObject):
    def __init__(self, main):
        super().__init__(main if isinstance(main, QtCore.QObject) else None)
        self.main = main
        self.viewer = main.image_viewer
        self.panel = main.local_repair_panel
        self.color = None
        self.stroke = None
        self.preview = None
        self.clone = CloneBrushController(self)
        self.panel.restore.clicked.connect(lambda checked: self.select_tool("restore" if checked else None))
        self.panel.pick.clicked.connect(lambda checked: self.select_tool("color_pick" if checked else None))
        self.panel.clone.clicked.connect(lambda checked: self.select_tool("clone" if checked else None))
        self.panel.fill.clicked.connect(self.fill_mask)
        self.viewer.repair_controller = self

    def select_tool(self, tool):
        self.main.set_tool(tool)
        if tool in {"restore", "clone"}:
            self.main.set_slider_size(self.viewer.brush_size)

    @property
    def is_drawing(self):
        return self.stroke is not None or self.clone.stroke is not None

    def transient_items(self):
        items = [item for item in (self.preview, self.clone.preview, self.clone.marker, self.clone.destination) if item is not None]
        protection = getattr(self.main, 'pixel_protection_ctrl', None)
        if protection is not None:
            items.extend(protection.transient_items())
        return items

    def reset_page(self):
        self.cancel()
        self.clone.reset()

    def _available(self):
        pool = getattr(self.main, "threadpool", None)
        artistic = getattr(self.main, "artistic_edit_ctrl", None)
        if (pool is not None and pool.activeThreadCount()) or (artistic is not None and artistic.panel.is_busy()):
            self._notice(self.tr("Wait for processing to finish before repairing the page."))
            return False
        return self.viewer.hasPhoto()

    def _page(self, scene_pos=None):
        index = self.main.curr_img_idx
        if self.viewer.webtoon_mode and scene_pos is not None:
            index, _ = self.viewer.scene_to_page_coordinates(scene_pos)
        if index is None or not 0 <= index < len(self.main.image_files):
            return None
        if not self.viewer.webtoon_mode:
            page_list = getattr(self.main, "page_list", None)
            if page_list is not None and page_list.currentRow() != index:
                return None  # asynchronous page navigation has not finished
        origin = self.viewer.page_to_scene_coordinates(index, QtCore.QPointF()) if self.viewer.webtoon_mode else QtCore.QPointF()
        if origin is None:
            return None
        file_path = self.main.image_files[index]
        if file_path not in self.main.undo_stacks:
            return None
        return file_path, origin

    def press(self, scene_pos, set_source=False):
        self.cancel()
        if self.viewer.current_tool == "clone":
            try:
                self.clone.press(scene_pos, set_source)
            except Exception as exc:
                self.cancel()
                self.main.default_error_handler((type(exc), exc, traceback.format_exc()))
            return
        if not self._available():
            return
        page = self._page(scene_pos)
        if page is None:
            return
        file_path, origin = page
        try:
            current = self.main.image_ctrl.get_composited_page_image(file_path)
            if current is None:
                return
            local = scene_pos - origin
            x, y = math.floor(local.x()), math.floor(local.y())
            if not (0 <= y < current.shape[0] and 0 <= x < current.shape[1]):
                return
            if self.viewer.current_tool == "color_pick":
                self.color = tuple(int(v) for v in current[y, x, :3])
                self.panel.set_color(QtGui.QColor(*self.color))
                self.main.set_tool("brush")
                self.main.set_slider_size(self.viewer.brush_size)
                return
            original = self.main.image_ctrl.load_original_image(file_path)
            if original is None or original.shape != current.shape:
                self._notice(self.tr("The original page image is unavailable."))
                return
            path = QtGui.QPainterPath(scene_pos)
            path.lineTo(scene_pos + QtCore.QPointF(0.01, 0))
            size = self.viewer.brush_size
            self.stroke = (file_path, origin, current.copy(), original.copy(), path, size,
                           self.main.undo_stacks[file_path].index())
            # A polygon overlay is deliberately not a QGraphicsPathItem: it
            # must never become a saved cleanup mask during autosave or export.
            self.preview = self.viewer._scene.addPolygon(QtGui.QPolygonF(), QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(QtGui.QColor(0, 190, 220, 150)))
            self.preview.setZValue(10)
            self.preview.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
            self._update_preview()
        except Exception as exc:
            self.cancel()
            self.main.default_error_handler((type(exc), exc, traceback.format_exc()))

    def move(self, scene_pos):
        if self.clone.stroke is not None:
            self.clone.move(scene_pos)
            return
        if self.stroke is None:
            return
        _, origin, current, _, path, _, _ = self.stroke
        local = scene_pos - origin
        # Clamp to the starting page; a gesture never edits neighbouring pages.
        local.setX(max(0, min(current.shape[1] - 1, local.x())))
        local.setY(max(0, min(current.shape[0] - 1, local.y())))
        path.lineTo(local + origin)
        self._update_preview()

    def _update_preview(self):
        _, origin, current, _, path, size, _ = self.stroke
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(size)
        stroker.setCapStyle(QtCore.Qt.RoundCap)
        stroker.setJoinStyle(QtCore.Qt.RoundJoin)
        page = QtGui.QPainterPath()
        page.addRect(QtCore.QRectF(origin.x(), origin.y(), current.shape[1], current.shape[0]))
        self.preview.setPolygon(stroker.createStroke(path).intersected(page).toFillPolygon())

    def release(self):
        if self.clone.stroke is not None:
            try:
                self.clone.release()
            except Exception as exc:
                self.cancel()
                self.main.default_error_handler((type(exc), exc, traceback.format_exc()))
            return
        stroke = self.stroke
        self.cancel()
        if stroke is None:
            return
        file_path, origin, current, original, path, size, revision = stroke
        stack = self.main.undo_stacks.get(file_path)
        if stack is None or stack.index() != revision or not self._available():
            return
        if not self.viewer.webtoon_mode:
            page = self._page()
            if page is None or page[0] != file_path:
                return
        try:
            pen = QtGui.QPen(QtCore.Qt.white, size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            mask = rasterize_paths(current.shape, [(path, pen, QtCore.Qt.NoBrush)], origin)
            patch = repair_patch(current, original, mask)
            if patch is not None:
                stack.push(PatchInsertCommand(self.main, [patch], file_path, kind="inpaint", text=self.tr("Restore area")))
        except Exception as exc:
            self.main.default_error_handler((type(exc), exc, traceback.format_exc()))

    def cancel(self):
        self.clone.cancel()
        if self.preview is not None:
            try:
                if self.preview.scene() is not None:
                    self.viewer._scene.removeItem(self.preview)
            except RuntimeError:  # scene already cleared during navigation
                pass
        self.preview = None
        self.stroke = None

    def fill_mask(self):
        if self.panel.fill_mode.currentData() == "heal":
            self.fill_heal_mask()
            return
        if self.panel.fill_mode.currentData() == "gradient":
            self.fill_gradient_mask()
            return
        self.cancel()
        if not self._available():
            return
        if self.color is None:
            self._notice(self.tr("Pick a background color first."))
            return
        page = self._page()
        if page is None:
            return
        file_path, origin = page
        try:
            current = self.main.image_ctrl.get_composited_page_image(file_path)
            if current is None:
                return
            mask = self._current_mask(current.shape, origin)
            if not np.any(mask):
                self._notice(self.tr("Paint or segment the area to fill first."))
                return
            patch = repair_patch(current, self.color, mask)
            if patch is None:
                return
            self._commit_fill(patch, file_path, origin, current.shape, self.tr("Sampled color fill"))
        except Exception as exc:
            self.main.default_error_handler((type(exc), exc, traceback.format_exc()))

    def _current_mask(self, shape, origin):
        paths = []
        for item in self.viewer._scene.items():
            if not isinstance(item, QtWidgets.QGraphicsPathItem):
                continue
            pen = QtGui.QPen(item.pen())
            pen.setColor(QtCore.Qt.white)
            brush = QtGui.QBrush(QtCore.Qt.white) if item.brush().style() != QtCore.Qt.NoBrush else QtCore.Qt.NoBrush
            paths.append((item.mapToScene(item.path()), pen, brush))
        return rasterize_paths(shape, paths, origin)

    def _commit_fill(self, patch, file_path, origin, shape, text, metadata=None):
        stack = self.main.undo_stacks[file_path]
        rect = QtCore.QRectF(origin.x(), origin.y(), shape[1], shape[0])
        command = PatchInsertCommand(self.main, [patch], file_path, kind="inpaint", text=text, metadata=metadata)
        stack.beginMacro(text)
        try:
            stack.push(command)
            stack.push(ClearPageMaskCommand(self.viewer, rect) if self.viewer.webtoon_mode else ClearBrushStrokesCommand(self.viewer))
        finally:
            stack.endMacro()

    def fill_gradient_mask(self):
        self.cancel()
        if not self._available():
            return
        page = self._page()
        if page is None:
            return
        file_path, origin = page
        try:
            current = self.main.image_ctrl.get_composited_page_image(file_path)
            if current is None:
                return
            current = current.copy()
            mask = self._current_mask(current.shape, origin)
            if not np.any(mask):
                self._notice(self.tr("Paint or segment the area to fill first."))
                return
            # Original sampling is optional; an unavailable source must not
            # prevent repairing the already loaded, composed background.
            try:
                original = self.main.image_ctrl.load_original_image(file_path)
            except Exception:
                original = None
            original = original.copy() if original is not None and original.shape == current.shape else None
            stack = self.main.undo_stacks[file_path]
            revision = stack.index()
            webtoon_mode = self.viewer.webtoon_mode
            project_epoch = getattr(self.main, "_translation_project_epoch", None)
            page_id = getattr(self.main, "image_states", {}).get(file_path, {}).get("page_id")
            parent = self.main if isinstance(self.main, QtWidgets.QWidget) else None
            protection = get_protection(self.main, file_path, current.shape)
            options = {'protection': protection} if protection is not None else {}
            dialog = GradientRepairDialog(current, original, mask, parent=parent, **options)
            try:
                if dialog.exec() != QtWidgets.QDialog.Accepted or dialog.patch is None:
                    return
                if not self._available():
                    return
                page_now = self._page()
                valid = (page_now is not None and page_now[0] == file_path and page_now[1] == origin
                         and self.viewer.webtoon_mode == webtoon_mode
                         and self.main.undo_stacks.get(file_path) is stack and stack.index() == revision
                         and getattr(self.main, "_translation_project_epoch", None) == project_epoch
                         and getattr(self.main, "image_states", {}).get(file_path, {}).get("page_id") == page_id)
                if valid:
                    latest = self.main.image_ctrl.get_composited_page_image(file_path)
                    valid = latest is not None and np.array_equal(latest, current)
                if valid:
                    valid = np.array_equal(self._current_mask(current.shape, origin), mask)
                if valid:
                    valid = np.array_equal(get_protection(self.main, file_path, current.shape), protection)
                if valid and dialog.metadata.get("source") == "original":
                    try:
                        latest_original = self.main.image_ctrl.load_original_image(file_path)
                    except Exception:
                        latest_original = None
                    valid = original is not None and latest_original is not None and np.array_equal(latest_original, original)
                if not valid:
                    self._notice(self.tr("The page or mask changed. Open a new gradient preview before applying the repair."))
                    return
                self._commit_fill(dialog.patch, file_path, origin, current.shape,
                                  self.tr("Sampled gradient fill"), dialog.metadata)
            finally:
                dialog.deleteLater()
        except Exception as exc:
            self.main.default_error_handler((type(exc), exc, traceback.format_exc()))

    def fill_heal_mask(self):
        """Experimental one-stamp comparison; never infer a texture/color source."""
        self.cancel()
        if not self._available():
            return
        page = self._page()
        if page is None:
            return
        file_path, origin = page
        source = self.clone.source
        stack = self.main.undo_stacks[file_path]
        if source is None or source['file'] != file_path or source['stack'] is not stack:
            self._notice(self.tr("Alt-click a clean area with Clone to capture a sample first."))
            return
        if self.color is None:
            self._notice(self.tr("Pick a background color first."))
            return
        try:
            from app.ui.dialogs.heal_repair import HealRepairDialog
            current = self.main.image_ctrl.get_composited_page_image(file_path)
            if current is None or current.shape != source['shape']:
                return
            current = current.copy()
            mask = self._current_mask(current.shape, origin)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                self._notice(self.tr("Paint or segment the area to fill first."))
                return
            # Center one fixed stamp on the explicitly painted repair area.
            # The engine rejects masks larger than the captured sample.
            destination = ((int(xs.min()) + int(xs.max())) / 2,
                           (int(ys.min()) + int(ys.max())) / 2)
            protection = get_protection(self.main, file_path, current.shape)
            revision, webtoon = stack.index(), self.viewer.webtoon_mode
            epoch = getattr(self.main, '_translation_project_epoch', None)
            page_id = getattr(self.main, 'image_states', {}).get(file_path, {}).get('page_id')
            parent = self.main if isinstance(self.main, QtWidgets.QWidget) else None
            dialog = HealRepairDialog(current, mask, source['sample'], self.color, destination,
                parent=parent, protection=protection, softness=self.panel.clone_softness.value() / 100)
            try:
                if dialog.exec() != QtWidgets.QDialog.Accepted or dialog.patch is None:
                    return
                if not self._available():
                    return
                page_now = self._page()
                valid = (page_now is not None and page_now == page and self.viewer.webtoon_mode == webtoon
                         and self.main.undo_stacks.get(file_path) is stack and stack.index() == revision
                         and getattr(self.main, '_translation_project_epoch', None) == epoch
                         and getattr(self.main, 'image_states', {}).get(file_path, {}).get('page_id') == page_id)
                if valid:
                    latest = self.main.image_ctrl.get_composited_page_image(file_path)
                    valid = (latest is not None and np.array_equal(latest, current)
                             and np.array_equal(self._current_mask(current.shape, origin), mask)
                             and np.array_equal(get_protection(self.main, file_path, current.shape), protection))
                if not valid:
                    self._notice(self.tr("The page or mask changed. Open a new repair preview before applying."))
                    return
                self._commit_fill(dialog.patch, file_path, origin, current.shape,
                                  self.tr("Heal repair"), dialog.metadata)
            finally:
                dialog.deleteLater()
        except Exception as exc:
            self.main.default_error_handler((type(exc), exc, traceback.format_exc()))

    def _notice(self, text):
        MMessage.info(text, parent=self.main, duration=5)
