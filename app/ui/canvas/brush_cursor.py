"""Brush footprint in scene pixels, so zoom and DPI cannot distort its size."""
from PySide6 import QtCore, QtGui, QtWidgets


class BrushCursorOverlay(QtCore.QObject):
    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.item = None
        self._refreshing = False
        viewer.viewport().setMouseTracking(True)
        viewer.viewport().installEventFilter(self)
        viewer.horizontalScrollBar().valueChanged.connect(self.refresh)
        viewer.verticalScrollBar().valueChanged.connect(self.refresh)

    def hide(self):
        if self.item is not None:
            self.item.hide()

    def clear(self):
        if self.item is not None:
            self.viewer._scene.removeItem(self.item)
        self.item = None

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Leave:
            self.hide()
        elif event.type() == QtCore.QEvent.MouseMove:
            self.refresh(scene_pos=self.viewer.mapToScene(event.position().toPoint()))
        return False

    def refresh(self, *_, scene_pos=None):
        if self._refreshing:
            return
        viewer = self.viewer
        if viewer.current_tool not in {'brush', 'eraser', 'restore'} or not viewer.hasPhoto():
            self.hide()
            return
        if scene_pos is None:
            pos = viewer.viewport().mapFromGlobal(QtGui.QCursor.pos())
            if not viewer.viewport().rect().contains(pos):
                self.hide()
                return
            scene_pos = viewer.mapToScene(pos)
        self._refreshing = True
        try:
            if self.item is None:
                pen = QtGui.QPen(QtCore.Qt.black, 3)
                pen.setCosmetic(True)
                self.item = viewer._scene.addEllipse(QtCore.QRectF(), pen)
                self.item.setZValue(11)
                self.item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                inner = QtWidgets.QGraphicsEllipseItem(self.item)
                pen = QtGui.QPen(QtCore.Qt.white, 1)
                pen.setCosmetic(True)
                inner.setPen(pen)
                inner.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            # erase_at() treats eraser_size as a radius; other tools use a pen diameter.
            size = 2 * viewer.eraser_size if viewer.current_tool == 'eraser' else viewer.brush_size
            rect = QtCore.QRectF(scene_pos.x()-size/2, scene_pos.y()-size/2, size, size)
            self.item.setRect(rect)
            self.item.childItems()[0].setRect(rect)
            self.item.show()
            viewer.setCursor(QtCore.Qt.CrossCursor)
        finally:
            self._refreshing = False
