from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class _PreviewView(QtWidgets.QGraphicsView):
    zoom_changed = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = self._scene.addPixmap(QtGui.QPixmap())
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QtGui.QColor(28, 28, 28))

    @staticmethod
    def _pixmap(image: np.ndarray) -> QtGui.QPixmap:
        rgb = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = rgb.shape[:2]
        qimage = QtGui.QImage(
            rgb.data, width, height, width * 3, QtGui.QImage.Format.Format_RGB888
        ).copy()
        return QtGui.QPixmap.fromImage(qimage)

    def set_image(self, image: np.ndarray) -> None:
        self._item.setPixmap(self._pixmap(image))
        self._scene.setSceneRect(self._item.boundingRect())
        self.fit_image()

    def fit_image(self) -> None:
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_changed.emit(self.transform().m11())

    def set_zoom(self, scale: float) -> None:
        current = self.transform().m11()
        if current > 0 and scale > 0:
            self.scale(scale / current, scale / current)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else (1.0 / 1.2)
        target = max(0.02, min(20.0, self.transform().m11() * factor))
        self.set_zoom(target)
        self.zoom_changed.emit(target)


class ArtisticEditPreviewDialog(QtWidgets.QDialog):
    REGENERATE_SAME_SEED = 2
    REGENERATE_NEW_SEED = 3

    def __init__(
        self,
        before: np.ndarray,
        after: np.ndarray,
        summary: str,
        *,
        apply_available: bool,
        stale_reason: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Artistic Edit Preview"))
        self.resize(1180, 760)
        layout = QtWidgets.QVBoxLayout(self)

        panes = QtWidgets.QHBoxLayout()
        self.before_view = _PreviewView()
        self.after_view = _PreviewView()
        for title, view, image in (
            (self.tr("Before"), self.before_view, before),
            (self.tr("After"), self.after_view, after),
        ):
            column = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(title)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            column.addWidget(label)
            column.addWidget(view, 1)
            panes.addLayout(column, 1)
            view.set_image(image)
        layout.addLayout(panes, 1)

        self.before_view.zoom_changed.connect(self.after_view.set_zoom)
        self.after_view.zoom_changed.connect(self.before_view.set_zoom)
        self._sync_scrollbars(self.before_view, self.after_view)
        self._sync_scrollbars(self.after_view, self.before_view)

        summary_label = QtWidgets.QLabel(summary)
        summary_label.setWordWrap(True)
        summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary_label)

        self.stale_label = QtWidgets.QLabel(stale_reason)
        self.stale_label.setWordWrap(True)
        self.stale_label.setStyleSheet("color: #d97b7b;")
        self.stale_label.setVisible(bool(stale_reason))
        layout.addWidget(self.stale_label)

        buttons = QtWidgets.QHBoxLayout()
        fit_button = QtWidgets.QPushButton(self.tr("Fit"))
        same_button = QtWidgets.QPushButton(self.tr("Regenerate Same Seed"))
        new_button = QtWidgets.QPushButton(self.tr("Regenerate New Seed"))
        discard_button = QtWidgets.QPushButton(self.tr("Discard"))
        self.apply_button = QtWidgets.QPushButton(self.tr("Apply"))
        self.apply_button.setDefault(True)
        self.apply_button.setEnabled(apply_available)
        buttons.addWidget(fit_button)
        buttons.addStretch()
        buttons.addWidget(same_button)
        buttons.addWidget(new_button)
        buttons.addWidget(discard_button)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

        fit_button.clicked.connect(self._fit_both)
        same_button.clicked.connect(lambda: self.done(self.REGENERATE_SAME_SEED))
        new_button.clicked.connect(lambda: self.done(self.REGENERATE_NEW_SEED))
        discard_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.accept)

    @staticmethod
    def _sync_scrollbars(source: _PreviewView, target: _PreviewView) -> None:
        source.horizontalScrollBar().valueChanged.connect(target.horizontalScrollBar().setValue)
        source.verticalScrollBar().valueChanged.connect(target.verticalScrollBar().setValue)

    def _fit_both(self) -> None:
        self.before_view.fit_image()
        self.after_view.fit_image()

    def block_apply(self, reason: str) -> None:
        self.apply_button.setEnabled(False)
        self.stale_label.setText(reason)
        self.stale_label.show()
