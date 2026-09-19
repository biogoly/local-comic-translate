"""Fixed-source clone stamps with distinct source and destination indicators."""
import math

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.commands.inpaint import PatchInsertCommand
from modules.utils.clone_brush import CloneStroke, capture_sample


class CloneBrushController(QtCore.QObject):
    def __init__(self, repair):
        super().__init__(repair)
        self.repair = repair
        self.main = repair.main
        self.viewer = repair.viewer
        self.panel = repair.panel
        self.source = None
        self.stroke = None
        self.preview = None
        self.marker = None
        self.destination = None
        self.last_scene_pos = None
        self.panel.clone_original.toggled.connect(self.reset)
        self.viewer.viewport().setMouseTracking(True)
        self.viewer.viewport().installEventFilter(self)
        self.viewer.horizontalScrollBar().valueChanged.connect(self.refresh_cursor)
        self.viewer.verticalScrollBar().valueChanged.connect(self.refresh_cursor)

    def reset(self, *_):
        self.cancel()
        self.source = None

    def cancel(self):
        for item in (self.preview, self.marker, self.destination):
            if item is not None:
                try:
                    if item.scene() is not None:
                        self.viewer._scene.removeItem(item)
                except RuntimeError:
                    pass
        self.preview = self.marker = self.destination = None
        self.stroke = None

    def press(self, scene_pos, set_source=False):
        self.cancel()
        if not self.repair._available():
            return
        page = self.repair._page(scene_pos)
        if page is None:
            return
        file_path, origin = page
        local = scene_pos - origin
        local = QtCore.QPointF(math.floor(local.x()) + 0.5, math.floor(local.y()) + 0.5)
        stack = self.main.undo_stacks[file_path]
        if set_source:
            image = (self.main.image_ctrl.load_image(file_path) if self.panel.clone_original.isChecked()
                     else self.main.image_ctrl.get_composited_page_image(file_path))
            if image is None:
                self.repair._notice(self.tr("The clone source image is unavailable."))
                return
            if not (0 <= local.x() < image.shape[1] and 0 <= local.y() < image.shape[0]):
                return
            sample = capture_sample(image, local.x(), local.y(), self.viewer.brush_size)
            self.source = dict(file=file_path, stack=stack, sample=sample, anchor=local, shape=image.shape)
            self.hover(scene_pos)
            return
        if (self.source is None or self.source['file'] != file_path or self.source['stack'] is not stack
                or self.source['sample'].size != self.viewer.brush_size):
            self.repair._notice(self.tr("Alt-click a clean area on this page to set the clone source."))
            return
        current = self.main.image_ctrl.get_composited_page_image(file_path)
        if current is None or current.shape != self.source['shape']:
            self.reset()
            return
        if not (0 <= local.x() < current.shape[1] and 0 <= local.y() < current.shape[0]):
            return
        engine = CloneStroke(current.copy(), self.source['sample'], self.panel.clone_softness.value() / 100)
        self.stroke = dict(file=file_path, origin=origin, stack=stack, revision=stack.index(), engine=engine)
        self.move(scene_pos)

    def hover(self, scene_pos):
        self.last_scene_pos = QtCore.QPointF(scene_pos)
        page = self.repair._page(scene_pos)
        if page is None:
            self.hide_indicators()
            return
        _, origin = page
        if self.destination is None:
            pen = QtGui.QPen(QtCore.Qt.black, 3)
            pen.setCosmetic(True)
            self.destination = self.viewer._scene.addEllipse(QtCore.QRectF(), pen)
            self.destination.setZValue(11)
            self.destination.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            outline = QtWidgets.QGraphicsEllipseItem(self.destination)
            pen = QtGui.QPen(QtCore.Qt.white, 1)
            pen.setCosmetic(True)
            outline.setPen(pen)
            outline.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        size = self.stroke['engine'].sample.size if self.stroke else self.viewer.brush_size
        center = origin + QtCore.QPointF(math.floor(scene_pos.x()-origin.x())+0.5,
                                        math.floor(scene_pos.y()-origin.y())+0.5)
        rect = QtCore.QRectF(center.x()-size/2, center.y()-size/2, size, size)
        self.destination.setRect(rect)
        self.destination.childItems()[0].setRect(rect)
        self.destination.show()
        if self.source is None or page[0] != self.source['file']:
            if self.marker is not None:
                self.marker.hide()
            return
        center = origin + self.source['anchor']
        if self.marker is None:
            pen = QtGui.QPen(QtGui.QColor(0, 190, 220), 1.5, QtCore.Qt.DashLine)
            pen.setCosmetic(True)
            self.marker = self.viewer._scene.addEllipse(QtCore.QRectF(), pen)
            self.marker.setZValue(10)
            self.marker.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        radius = self.source['sample'].size / 2
        self.marker.setRect(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        self.marker.show()

    def hide_indicators(self):
        for item in (self.marker, self.destination):
            if item is not None:
                item.hide()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Leave:
            self.hide_indicators()
        return False

    def refresh_cursor(self, *_):
        if self.viewer.current_tool != 'clone':
            return
        pos = self.viewer.viewport().mapFromGlobal(QtGui.QCursor.pos())
        if self.viewer.viewport().rect().contains(pos):
            self.hover(self.viewer.mapToScene(pos))
        else:
            self.hide_indicators()

    def move(self, scene_pos):
        if self.stroke is None:
            return
        point = scene_pos - self.stroke['origin']
        engine = self.stroke['engine']
        engine.move(math.floor(point.x()) + 0.5, math.floor(point.y()) + 0.5)
        patch = engine.patch()
        if patch is not None:
            if self.preview is None:
                self.preview = QtWidgets.QGraphicsPixmapItem()
                self.preview.setZValue(0.6)  # above saved patches, below editable text
                self.preview.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                self.viewer._scene.addItem(self.preview)
            self.preview.setPixmap(QtGui.QPixmap.fromImage(self.viewer.qimage_from_array(patch['image'])))
            x, y, _, _ = patch['bbox']
            self.preview.setPos(self.stroke['origin'] + QtCore.QPointF(x, y))
        self.hover(scene_pos)

    def release(self):
        stroke = self.stroke
        self.cancel()
        if stroke is None or not self.repair._available():
            return
        page = self.repair._page()
        if (page is None or page[0] != stroke['file']
                or self.main.undo_stacks.get(stroke['file']) is not stroke['stack']
                or stroke['stack'].index() != stroke['revision']):
            return
        patch = stroke['engine'].patch()
        if patch is not None:
            stroke['stack'].push(PatchInsertCommand(self.main, [patch], stroke['file'], kind="inpaint", text=self.tr("Clone brush")))
        if self.viewer.current_tool == 'clone' and self.last_scene_pos is not None:
            self.hover(self.last_scene_pos)
