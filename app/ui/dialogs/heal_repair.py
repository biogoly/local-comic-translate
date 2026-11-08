"""A bounded experimental Heal comparison; accepting returns a patch only."""
from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.dayu_widgets import dayu_theme
from app.ui.dialogs.gradient_repair import _RepairView
from modules.repair.heal import HealError, adjust_sample_color, preview_heal


logger = logging.getLogger(__name__)


class HealRepairDialog(QtWidgets.QDialog):
    """Freeze page, mask, sample and target; caller owns commit and stale checks."""

    def __init__(self, current, mask, sample, target_rgb, destination,
                 protection=None, parent=None, softness=0.25):
        super().__init__(parent)
        self.current = np.array(current, copy=True, order='C')
        self.mask = np.array(mask, copy=True, order='C')
        self.sample = adjust_sample_color(sample, target_rgb, strength=0)
        self.target_rgb = tuple(target_rgb)
        self.destination = tuple(destination)
        self.protection = None if protection is None else np.array(protection, copy=True, order='C')
        self.softness = softness
        for pixels in (self.current, self.mask, self.protection):
            if pixels is not None:
                pixels.setflags(write=False)
        self.patch = None
        self.metadata = {}
        self._candidate = None
        self.setWindowTitle(self.tr('Experimental Heal preview'))
        self.setModal(True)
        self.resize(1160, 660)
        self.setMinimumSize(780, 480)
        layout = QtWidgets.QVBoxLayout(self)
        instructions = QtWidgets.QLabel(self.tr(
            'Compare one fixed clone sample with a color-adjusted copy. This prototype preserves sampled texture and shifts its color toward your picked color. It does not reconstruct detail or borrow nearby pixels.'))
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        limits = QtWidgets.QLabel(self.tr(
            'The entire painted mask must fit inside one sample centered on the mask. Use a larger clean clone sample or a smaller mask. Only painted, unprotected pixels can change; texture will not be tiled.'))
        limits.setWordWrap(True)
        layout.addWidget(limits)
        controls = QtWidgets.QHBoxLayout()
        self.color_label = QtWidgets.QLabel(self.tr('Picked target color: RGB {0}').format(', '.join(str(int(c)) for c in self.target_rgb)))
        self.swatch = QtWidgets.QLabel()
        self.swatch.setFixedSize(24, 24)
        self.swatch.setStyleSheet('background-color: rgb(%d, %d, %d); border: 1px solid #808080;' % self.target_rgb)
        controls.addWidget(self.color_label)
        controls.addWidget(self.swatch)
        controls.addStretch()
        self.fit_button = QtWidgets.QPushButton(self.tr('Fit repair area'))
        self.zoom_out_button = QtWidgets.QPushButton('−')
        self.zoom_in_button = QtWidgets.QPushButton('+')
        self.zoom_out_button.setAccessibleName(self.tr('Zoom out'))
        self.zoom_in_button.setAccessibleName(self.tr('Zoom in'))
        for button in (self.fit_button, self.zoom_out_button, self.zoom_in_button):
            controls.addWidget(button)
        layout.addLayout(controls)
        comparisons = QtWidgets.QHBoxLayout()
        self.current_view, self.clone_view, self.heal_view = (_RepairView() for _ in range(3))
        self.views = (self.current_view, self.clone_view, self.heal_view)
        for view, label in zip(self.views, (self.tr('Current page'), self.tr('Ordinary Clone'), self.tr('Heal preview'))):
            column = QtWidgets.QVBoxLayout()
            column.addWidget(QtWidgets.QLabel(label))
            view.setAccessibleName(label)
            view.set_image(self.current)
            view.zoom_requested.connect(self._zoom)
            column.addWidget(view, 1)
            comparisons.addLayout(column, 1)
        layout.addLayout(comparisons, 1)
        strength_row = QtWidgets.QHBoxLayout()
        strength_label = QtWidgets.QLabel(self.tr('Color adjustment'))
        self.strength_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_slider.setAccessibleName(self.tr('Color adjustment'))
        strength_label.setBuddy(self.strength_slider)
        self.strength_label = QtWidgets.QLabel('100%')
        strength_row.addWidget(strength_label)
        strength_row.addWidget(self.strength_slider, 1)
        strength_row.addWidget(self.strength_label)
        layout.addLayout(strength_row)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status_label.setMinimumHeight(30)
        layout.addWidget(self.status_label)
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch()
        self.apply_button = QtWidgets.QPushButton(self.tr('Apply Heal'))
        self.cancel_button = QtWidgets.QPushButton(self.tr('Cancel'))
        actions.addWidget(self.apply_button)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
        self.apply_button.clicked.connect(self._apply)
        self.cancel_button.clicked.connect(self.reject)
        self.fit_button.clicked.connect(self._fit)
        self.zoom_out_button.clicked.connect(lambda: self._zoom(0.8))
        self.zoom_in_button.clicked.connect(lambda: self._zoom(1.25))
        self.strength_slider.valueChanged.connect(self._generate)
        dayu_theme.apply(self)
        self._theme_ready = True
        self._apply_theme()
        self._generate()
        QtCore.QTimer.singleShot(0, self._fit)

    def _apply_theme(self):
        outline = dayu_theme.disable_color if dayu_theme.is_dark else dayu_theme.border_color
        style = f'''
            QDialog {{ background-color: {dayu_theme.background_color}; color: {dayu_theme.primary_text_color}; }}
            QLabel {{ color: {dayu_theme.primary_text_color}; }}
            QPushButton {{ background-color: {dayu_theme.background_in_color};
                color: {dayu_theme.primary_text_color}; border: 1px solid {outline};
                border-radius: 4px; padding: 5px 10px; min-height: 18px; }}
            QPushButton:hover, QPushButton:focus {{ border-color: {dayu_theme.primary_5};
                background-color: {dayu_theme.background_out_color}; }}
            QPushButton:disabled {{ color: {dayu_theme.disable_color}; border-color: {dayu_theme.border_color}; }}
            QGraphicsView {{ border: 1px solid {outline}; }}
        '''
        if self.styleSheet() != style:
            self.setStyleSheet(style)
        for view in self.views:
            view.setBackgroundBrush(QtGui.QColor(dayu_theme.background_out_color))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.StyleChange and getattr(self, '_theme_ready', False):
            self._apply_theme()

    def showEvent(self, event):
        self._apply_theme()
        super().showEvent(event)

    def _zoom(self, factor):
        for view in self.views:
            view.zoom(factor)

    def _fit(self):
        ys, xs = np.nonzero(self.mask)
        if len(xs):
            margin = max(12, self.sample.size // 4)
            rect = QtCore.QRectF(int(xs.min()) - margin, int(ys.min()) - margin,
                                 int(xs.max() - xs.min() + 1) + 2 * margin,
                                 int(ys.max() - ys.min() + 1) + 2 * margin)
            for view in self.views:
                view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _generate(self, *_):
        self._candidate = None
        self.patch = None
        self.metadata = {}
        self.apply_button.setEnabled(False)
        strength = self.strength_slider.value() / 100
        self.strength_label.setText(f'{self.strength_slider.value()}%')
        try:
            preview = preview_heal(self.current, self.mask, self.sample, self.target_rgb,
                                   self.destination, strength=strength, softness=self.softness,
                                   protection=self.protection)
            x, y, w, h = preview.bbox
            for view, pixels in ((self.clone_view, preview.clone_pixels), (self.heal_view, preview.heal_pixels)):
                displayed = self.current.copy()
                displayed[y:y+h, x:x+w] = pixels
                view.set_image(displayed)
            if np.array_equal(self.current[y:y+h, x:x+w], preview.heal_pixels):
                self.status_label.setText(self.tr('This preview would not change any pixels.'))
                return
            self._candidate = {'bbox': list(preview.bbox), 'image': preview.heal_pixels.copy()}
            self.status_label.setText(self.tr('Preview ready. Compare the texture and color before applying.'))
            self.apply_button.setEnabled(True)
        except HealError as exc:
            for view in (self.clone_view, self.heal_view):
                view.set_image(self.current)
            messages = {
                'mask_outside_sample': self.tr('The mask extends beyond the clean clone sample. Cancel and use a larger clean sample or a smaller mask.'),
                'fully_protected': self.tr('All masked pixels are protected. No repair can be applied.'),
                'invalid_mask': self.tr('Paint a repair mask before opening Heal preview.'),
            }
            self.status_label.setText(messages.get(exc.code, self.tr('The repair inputs are invalid. Cancel and sample again.')))
        except Exception:
            logger.exception('Experimental Heal preview failed')
            for view in (self.clone_view, self.heal_view):
                view.set_image(self.current)
            self.status_label.setText(self.tr('Could not generate the preview. No changes were applied.'))

    def _apply(self):
        if self._candidate is None:
            return
        self.patch = {'bbox': list(self._candidate['bbox']), 'image': self._candidate['image'].copy()}
        self.metadata = {'operation': 'heal', 'repair_method': 'heal', 'experimental': True,
                         'source_size': int(self.sample.size), 'target_rgb': list(self.target_rgb),
                         'destination': list(self.destination),
                         'strength': self.strength_slider.value() / 100, 'softness': float(self.softness)}
        self.accept()
