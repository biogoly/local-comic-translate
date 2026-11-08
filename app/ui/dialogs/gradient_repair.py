"""Preview a sampled background repair without changing the page or its mask."""
from __future__ import annotations

import json
import logging
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.dayu_widgets import dayu_theme
from modules.repair.gradient import (
    GradientFitError, GradientSampleLine, fit_gradient, sample_center, sample_coordinates,
)

logger = logging.getLogger(__name__)


def _pixmap(pixels):
    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]
    format_ = (QtGui.QImage.Format.Format_RGBA8888 if pixels.shape[2] == 4
               else QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(QtGui.QImage(pixels.data, width, height,
        pixels.strides[0], format_).copy())


class _RepairView(QtWidgets.QGraphicsView):
    point_clicked = QtCore.Signal(int, int)
    line_sampled = QtCore.Signal(object)
    zoom_requested = QtCore.Signal(float)

    def __init__(self, sampling=False, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.sampling = sampling
        self._press = None
        self._line_start = None
        self._line_preview = None
        self.image_item = self.scene().addPixmap(QtGui.QPixmap())
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setMinimumSize(220, 220)

    def set_image(self, pixels):
        self.image_item.setPixmap(_pixmap(pixels))
        self.setSceneRect(self.image_item.boundingRect())

    def zoom(self, factor):
        scale = self.transform().m11()
        target = min(24.0, max(0.01, scale * factor))
        self.scale(target / scale, target / scale)

    def wheelEvent(self, event):
        if event.angleDelta().y():
            self.zoom_requested.emit(1.25 if event.angleDelta().y() > 0 else 0.8)
        event.accept()

    def mousePressEvent(self, event):
        if (self.sampling and event.button() == QtCore.Qt.MouseButton.LeftButton
                and event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
            self.cancel_line()
            self._press = None
            self._line_start = self._pixel_at(event)
            pen = QtGui.QPen(QtGui.QColor('#00b8e8'), 2, QtCore.Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._line_preview = self.scene().addLine(QtCore.QLineF(), pen)
            self._line_preview.setZValue(3)
            self._update_line(self._line_start)
            event.accept()
            return
        self._press = event.position() if event.button() == QtCore.Qt.MouseButton.LeftButton else None
        super().mousePressEvent(event)

    def _pixel_at(self, event):
        point = self.mapToScene(event.position().toPoint())
        return math.floor(point.x()), math.floor(point.y())

    def _update_line(self, end):
        x, y = self._line_start
        self._line_preview.setLine(x + 0.5, y + 0.5, end[0] + 0.5, end[1] + 0.5)

    def cancel_line(self):
        if self._line_preview is not None:
            self.scene().removeItem(self._line_preview)
        self._line_preview = self._line_start = None

    def mouseMoveEvent(self, event):
        if self._line_start is not None:
            self._update_line(self._pixel_at(event))
            event.accept()
            return
        if self._press is not None and (event.position() - self._press).manhattanLength() > 4:
            self._press = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._line_start is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            start, end = self._line_start, self._pixel_at(event)
            self.cancel_line()
            if self.viewport().rect().contains(event.position().toPoint()) and start != end:
                self.line_sampled.emit(GradientSampleLine(start, end))
            event.accept()
            return
        start, self._press = self._press, None
        super().mouseReleaseEvent(event)
        if (self.sampling and start is not None and event.button() == QtCore.Qt.MouseButton.LeftButton
                and (event.position() - start).manhattanLength() <= 4):
            point = self.mapToScene(event.position().toPoint())
            self.point_clicked.emit(math.floor(point.x()), math.floor(point.y()))

    def focusOutEvent(self, event):
        self.cancel_line()
        self._press = None
        super().focusOutEvent(event)

    def hideEvent(self, event):
        self.cancel_line()
        super().hideEvent(event)


class GradientRepairDialog(QtWidgets.QDialog):
    """Fixed snapshots in, accepted patch out; persistence belongs to the caller."""

    def __init__(self, current, original, mask, parent=None, protection=None):
        super().__init__(parent)
        current = np.asarray(current)
        mask = np.asarray(mask)
        if current.dtype != np.uint8 or current.ndim != 3 or current.shape[2] != 3:
            raise ValueError('Gradient repair requires an RGB uint8 image.')
        if mask.dtype != np.uint8 or mask.shape != current.shape[:2] or not np.any(mask):
            raise ValueError('Gradient repair requires a nonempty uint8 mask matching the image.')
        self.current = np.array(current, copy=True, order='C')
        self.mask = np.array(mask, copy=True, order='C')
        self.protection = None if protection is None else np.array(protection, copy=True)
        self.original = None
        if original is not None:
            original = np.asarray(original)
            if original.dtype == np.uint8 and original.shape == current.shape:
                self.original = np.array(original, copy=True, order='C')
        for pixels in (self.current, self.original, self.mask, self.protection):
            if pixels is not None:
                pixels.setflags(write=False)
        self.sample_points = []
        self._markers = []
        self.patch = None
        self.metadata = {}
        self.setWindowTitle(self.tr('Gradient repair'))
        self.resize(1120, 740)
        self.setMinimumSize(760, 520)
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)
        instructions = QtWidgets.QLabel(self.tr(
            'Repair one smooth background area at a time. Click at least four clean spots around the entire mask, away from text and borders. Six or more samples improve reliability.')
            + ' ' + self.tr('Shift-drag to sample a line of background color.'))
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        source_row = QtWidgets.QHBoxLayout()
        source_label = QtWidgets.QLabel(self.tr('Sample from'))
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem(self.tr('Edited page'), 'current')
        self.source_combo.addItem(self.tr('Original page'), 'original')
        self.source_combo.model().item(1).setEnabled(self.original is not None)
        self.source_combo.setAccessibleName(self.tr('Sample from'))
        self.source_combo.setToolTip(self.tr('Changing the source clears samples. Preview pixels are never used as samples.'))
        source_label.setBuddy(self.source_combo)
        source_row.addWidget(source_label)
        source_row.addWidget(self.source_combo)
        self.mask_toggle = QtWidgets.QCheckBox(self.tr('Show mask'))
        self.mask_toggle.setChecked(True)
        source_row.addWidget(self.mask_toggle)
        source_row.addStretch()
        self.fit_area_button = QtWidgets.QPushButton(self.tr('Fit repair area'))
        self.fit_page_button = QtWidgets.QPushButton(self.tr('Fit page'))
        self.zoom_out_button = QtWidgets.QPushButton('−')
        self.zoom_out_button.setAccessibleName(self.tr('Zoom out'))
        self.zoom_out_button.setToolTip(self.tr('Zoom out'))
        self.zoom_in_button = QtWidgets.QPushButton('+')
        self.zoom_in_button.setAccessibleName(self.tr('Zoom in'))
        self.zoom_in_button.setToolTip(self.tr('Zoom in'))
        for button in (self.fit_area_button, self.fit_page_button, self.zoom_out_button, self.zoom_in_button):
            source_row.addWidget(button)
        layout.addLayout(source_row)
        views = QtWidgets.QHBoxLayout()
        self.sample_view = _RepairView(sampling=True)
        self.preview_view = _RepairView()
        self.sample_view.setAccessibleName(self.tr('Sample background'))
        self.preview_view.setAccessibleName(self.tr('Repair preview'))
        for view, label in ((self.sample_view, self.tr('Sample background')),
                            (self.preview_view, self.tr('Repair preview'))):
            column = QtWidgets.QVBoxLayout()
            column.addWidget(QtWidgets.QLabel(label))
            column.addWidget(view, 1)
            views.addLayout(column, 1)
        layout.addLayout(views, 1)
        self.sample_view.set_image(self.current)
        self.preview_view.set_image(self.current)
        overlay = np.zeros((*self.mask.shape, 4), np.uint8)
        overlay[:, :, 0] = 255
        overlay[:, :, 3] = np.rint(self.mask.astype(np.float32) * 0.35).astype(np.uint8)
        self.mask_item = self.sample_view.scene().addPixmap(_pixmap(overlay))
        self.mask_item.setZValue(1)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName('gradientPreviewStatus')
        self.status_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(35)
        layout.addWidget(self.status_label)
        sample_row = QtWidgets.QHBoxLayout()
        self.sample_count = QtWidgets.QLabel()
        self.remove_sample_button = QtWidgets.QPushButton(self.tr('Remove last sample'))
        self.clear_samples_button = QtWidgets.QPushButton(self.tr('Clear samples'))
        self.generate_button = QtWidgets.QPushButton(self.tr('Generate preview'))
        sample_row.addWidget(self.sample_count)
        sample_row.addStretch()
        for button in (self.remove_sample_button, self.clear_samples_button, self.generate_button):
            sample_row.addWidget(button)
        layout.addLayout(sample_row)
        hint = QtWidgets.QLabel(self.tr('Drag to pan; use the mouse wheel to zoom. Only pixels inside the painted mask will change.'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        self.apply_button = QtWidgets.QPushButton(self.tr('Apply'))
        self.cancel_button = QtWidgets.QPushButton(self.tr('Cancel'))
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.mask_toggle.toggled.connect(self.mask_item.setVisible)
        self.sample_view.point_clicked.connect(self.add_sample)
        self.sample_view.line_sampled.connect(self._add_observation)
        self.remove_sample_button.clicked.connect(self.remove_last_sample)
        self.clear_samples_button.clicked.connect(self.clear_samples)
        self.generate_button.clicked.connect(self.generate_preview)
        self.apply_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.fit_area_button.clicked.connect(self.fit_area)
        self.fit_page_button.clicked.connect(self.fit_page)
        self.zoom_in_button.clicked.connect(lambda: self.zoom(1.25))
        self.zoom_out_button.clicked.connect(lambda: self.zoom(0.8))
        self.sample_view.zoom_requested.connect(self.zoom)
        self.preview_view.zoom_requested.connect(self.zoom)
        self._theme_ready = True
        self._apply_theme()
        self._invalidate()

    @property
    def source(self):
        return self.original if self.source_combo.currentData() == 'original' else self.current

    def _apply_theme(self):
        outline = dayu_theme.disable_color if dayu_theme.is_dark else dayu_theme.border_color
        style = f'''
            QDialog {{ background-color: {dayu_theme.background_color}; color: {dayu_theme.primary_text_color}; }}
            QLabel, QCheckBox {{ color: {dayu_theme.primary_text_color}; }}
            QPushButton, QComboBox {{ background-color: {dayu_theme.background_in_color};
                color: {dayu_theme.primary_text_color}; border: 1px solid {outline};
                border-radius: 4px; padding: 5px 10px; min-height: 18px; }}
            QPushButton:hover, QPushButton:focus {{ border-color: {dayu_theme.primary_5};
                background-color: {dayu_theme.background_out_color}; }}
            QPushButton:disabled {{ color: {dayu_theme.disable_color}; border-color: {dayu_theme.border_color}; }}
            QGraphicsView {{ border: 1px solid {outline}; }}
            QLabel#gradientPreviewStatus {{ background-color: {dayu_theme.background_in_color};
                border: 1px solid {outline}; border-left: 4px solid {dayu_theme.primary_color};
                border-radius: 4px; padding: 6px 10px; }}
            QLabel#gradientPreviewStatus[severity="error"] {{ border-left-color: {dayu_theme.error_color}; }}
        '''
        if self.styleSheet() != style:
            self.setStyleSheet(style)
        for view in (self.sample_view, self.preview_view):
            view.setBackgroundBrush(QtGui.QColor(dayu_theme.background_out_color))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.StyleChange and getattr(self, '_theme_ready', False):
            self._apply_theme()

    def showEvent(self, event):
        self._apply_theme()
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.fit_area)

    def zoom(self, factor):
        for view in (self.sample_view, self.preview_view):
            view.zoom(factor)

    def fit_area(self):
        ys, xs = np.nonzero(self.mask)
        rect = QtCore.QRectF(float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1),
                            float(ys.max() - ys.min() + 1))
        padding = max(32.0, max(rect.width(), rect.height()) * 0.6)
        rect = rect.adjusted(-padding, -padding, padding, padding).intersected(self.sample_view.sceneRect())
        for view in (self.sample_view, self.preview_view):
            view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def fit_page(self):
        for view in (self.sample_view, self.preview_view):
            view.fitInView(view.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _invalidate(self):
        self.patch = None
        self.metadata = {}
        self.apply_button.setEnabled(False)
        self.generate_button.setEnabled(len(self.sample_points) >= 4)
        self.remove_sample_button.setEnabled(bool(self.sample_points))
        self.clear_samples_button.setEnabled(bool(self.sample_points))
        self.sample_count.setText(self.tr('Samples: {count}').format(count=len(self.sample_points)))
        self._set_status(self.tr('Ready to generate a preview.') if len(self.sample_points) >= 4
                         else self.tr('Choose at least four distinct clean spots around the repair.'))
        self.preview_view.set_image(self.current)

    def _set_status(self, text, severity='info'):
        self.status_label.setText(text)
        self.status_label.setProperty('severity', severity)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.status_label.update()

    def _source_changed(self):
        self.clear_samples()
        self.sample_view.set_image(self.source)

    def add_sample(self, x, y):
        self._add_observation((x, y))

    def _add_observation(self, sample):
        try:
            sample_coordinates(sample, self.mask)
        except GradientFitError as exc:
            self._set_status(self._fit_error(exc.code), 'error')
            return
        center = sample_center(sample)
        if any(sample_center(previous) == center for previous in self.sample_points):
            self._set_status(self.tr('That pixel is already sampled. Choose another clean spot.'), 'error')
            return
        if len(self.sample_points) >= 64:
            self._set_status(self.tr('Use at most 64 samples. Remove a sample before adding another.'), 'error')
            return
        self.sample_points.append(sample if isinstance(sample, GradientSampleLine) else center)
        self._invalidate()
        self._draw_markers()

    def _draw_markers(self):
        for marker in self._markers:
            self.sample_view.scene().removeItem(marker)
        self._markers = []
        for number, sample in enumerate(self.sample_points, 1):
            if isinstance(sample, GradientSampleLine):
                x1, y1 = sample.start
                x2, y2 = sample.end
                pen = QtGui.QPen(QtGui.QColor('#00b8e8'), 2)
                pen.setCosmetic(True)
                segment = self.sample_view.scene().addLine(x1 + 0.5, y1 + 0.5, x2 + 0.5, y2 + 0.5, pen)
                segment.setZValue(2)
                self._markers.append(segment)
            x, y = sample_center(sample)
            marker = QtWidgets.QGraphicsEllipseItem(-5, -5, 10, 10)
            marker.setPos(x + 0.5, y + 0.5)
            marker.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            marker.setPen(QtGui.QPen(QtGui.QColor('#00b8e8'), 2))
            marker.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 170)))
            marker.setZValue(2)
            label = QtWidgets.QGraphicsSimpleTextItem(str(number), marker)
            label.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.white))
            label.setPos(7, -10)
            self.sample_view.scene().addItem(marker)
            self._markers.append(marker)

    def remove_last_sample(self):
        if self.sample_points:
            self.sample_points.pop()
        self._invalidate()
        self._draw_markers()

    def clear_samples(self):
        self.sample_view.cancel_line()
        self.sample_points.clear()
        self._invalidate()
        self._draw_markers()

    def _fit_error(self, code):
        messages = {
            'too_few_samples': self.tr('Choose at least four distinct clean spots around the repair.'),
            'too_many_samples': self.tr('Use at most 64 samples. Remove a sample before adding another.'),
            'sample_inside_mask': self.tr('Choose a clean spot outside the painted mask.'),
            'sample_outside_image': self.tr('Click inside the page image.'),
            'degenerate_samples': self.tr('Spread samples in both directions; do not place them along one line.'),
            'insufficient_coverage': self.tr('Place clean samples around all sides of the painted mask.'),
            'inconsistent_samples': self.tr('The samples do not describe a smooth background. Remove spots on text or borders, or add more clean samples.'),
        }
        return messages.get(code, self.tr('Cannot build this repair from the image, mask, and samples.'))

    def generate_preview(self):
        self.patch = None
        self.metadata = {}
        self.apply_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self._set_status(self.tr('Generating preview…'))
        self.status_label.repaint()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.preview_view.set_image(self.current)
            result = fit_gradient(self.source, self.mask, self.sample_points, protection=self.protection)
            # Import lazily: LocalRepairController opens this dialog itself.
            from app.controllers.local_repair import repair_patch
            from modules.repair.protection import protect_image
            replacement = (protect_image(self.current, result.replacement, self.protection)
                           if self.protection is not None else result.replacement)
            patch = repair_patch(self.current, replacement, self.mask)
            if patch is None:
                self._set_status(self.tr('The preview would not change any pixels.'))
                return
            patch['image'].setflags(write=False)
            points = [s for s in self.sample_points if not isinstance(s, GradientSampleLine)]
            lines = [(s.start, s.end) for s in self.sample_points if isinstance(s, GradientSampleLine)]
            metadata = {'repair_method': 'gradient', 'source': self.source_combo.currentData(),
                        'sample_points_json': json.dumps(points, separators=(',', ':')),
                        'sample_count': int(result.sample_count), 'fit_inliers': int(result.inlier_count),
                        'fit_rms': float(result.rms_error)}
            if lines:
                metadata['sample_lines_json'] = json.dumps(lines, separators=(',', ':'))
            preview = self.current.copy()
            x, y, width, height = patch['bbox']
            preview[y:y + height, x:x + width] = patch['image']
            self.preview_view.set_image(preview)
            # Publish only after the complete preview has been constructed.
            self.patch, self.metadata = patch, metadata
            self.apply_button.setEnabled(True)
            self._set_status(self.tr('Preview ready: {used} of {total} samples used.').format(
                used=result.inlier_count, total=result.sample_count))
        except GradientFitError as exc:
            if exc.code == 'fully_protected':
                self._set_status(self.tr('The preview would not change any pixels.'))
            else:
                self._set_status(self.tr('Preview unavailable: {reason}').format(reason=self._fit_error(exc.code)), 'error')
        except Exception as exc:
            logger.exception('Gradient repair preview failed')
            self.patch, self.metadata = None, {}
            self.apply_button.setEnabled(False)
            self._set_status(self.tr('Unable to generate preview: {error}').format(
                error=str(exc) or type(exc).__name__), 'error')
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.generate_button.setEnabled(len(self.sample_points) >= 4)

    def accept(self):
        if self.patch is not None and self.apply_button.isEnabled():
            super().accept()

    def reject(self):
        self.patch = None
        self.metadata = {}
        super().reject()
