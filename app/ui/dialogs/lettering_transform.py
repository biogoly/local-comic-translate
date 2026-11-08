"""Preview native editable text transforms without modifying the live page."""
from __future__ import annotations

import copy

from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.lettering_style_format import apply_style_properties
from app.ui.canvas.text_item import TextBlockItem
from app.ui.dayu_widgets import dayu_theme
from modules.utils.lettering_transform import normalize_lettering_transform


class LetteringTransformDialog(QtWidgets.QDialog):
    def __init__(self, properties, parent=None):
        super().__init__(parent)
        self.before = copy.deepcopy(properties)
        self.candidate = None
        self.result_properties = None
        self._preview_items = []
        self.setWindowTitle(self.tr('Text shape'))
        self.resize(1000, 650)
        self.setMinimumSize(760, 520)
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(self.tr(
            'Stretch, skew, curve or add perspective while keeping text and individual-word formatting editable.'))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        controls = QtWidgets.QGridLayout()
        settings = normalize_lettering_transform(properties.get('lettering_transform'))
        self._original_settings = settings
        self._initial_values = {}
        self._has_reset = False
        self.controls = {}
        specs = (
            ('scale_x', self.tr('Width'), 25, 400, 1, 100, ' %'),
            ('scale_y', self.tr('Height'), 25, 400, 1, 100, ' %'),
            ('skew_x', self.tr('Skew'), -45, 45, 0, 1, '°'),
            ('perspective_x', self.tr('Horizontal perspective'), -60, 60, 0, 100, ' %'),
            ('perspective_y', self.tr('Vertical perspective'), -60, 60, 0, 100, ' %'),
            ('arc_angle', self.tr('Curve'), -120, 120, 0, 1, '°'),
        )
        self._factors = {}
        for index, (key, label, low, high, default, factor, suffix) in enumerate(specs):
            caption = QtWidgets.QLabel(label)
            control = QtWidgets.QDoubleSpinBox()
            control.setRange(low, high)
            control.setDecimals(1)
            control.setSingleStep(5 if factor == 100 else 1)
            control.setSuffix(suffix)
            control.setKeyboardTracking(False)
            control.setAccessibleName(label)
            caption.setBuddy(control)
            control.setValue(settings.get(key, default) * factor)
            self.controls[key] = control
            self._factors[key] = factor
            self._initial_values[key] = control.value()
            controls.addWidget(caption, index // 3 * 2, index % 3)
            controls.addWidget(control, index // 3 * 2 + 1, index % 3)
        self.reset_button = QtWidgets.QPushButton(self.tr('Reset distortion'))
        controls.addWidget(self.reset_button, 4, 2)
        self.curve_note = QtWidgets.QLabel(self.tr(
            'Positive curves arch upward; negative curves bend downward. Each line keeps its text and formatting.'))
        self.curve_note.setWordWrap(True)
        controls.addWidget(self.curve_note, 4, 0, 1, 2)
        self.controls['arc_angle'].setSingleStep(5)
        if properties.get('vertical'):
            self.controls['arc_angle'].setEnabled(False)
            self.curve_note.setText(self.tr('Curves are available for horizontal lettering.'))
        layout.addLayout(controls)
        note = QtWidgets.QLabel(self.tr(
            'Width and height stretch the lettering without changing line wrapping. Use the canvas handles to resize the text box or rotate it.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        comparison = QtWidgets.QHBoxLayout()
        self.current_view, self.transformed_view = QtWidgets.QGraphicsView(), QtWidgets.QGraphicsView()
        for view, label in ((self.current_view, self.tr('Current')), (self.transformed_view, self.tr('Transformed'))):
            column = QtWidgets.QVBoxLayout()
            column.addWidget(QtWidgets.QLabel(label))
            view.setAccessibleName(label)
            view.setScene(QtWidgets.QGraphicsScene(view))
            view.setInteractive(False)
            view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
            column.addWidget(view, 1)
            comparison.addLayout(column, 1)
        layout.addLayout(comparison, 1)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.export_note = QtWidgets.QLabel()
        self.export_note.setWordWrap(True)
        layout.addWidget(self.export_note)
        buttons = QtWidgets.QHBoxLayout()
        self.fit_button = QtWidgets.QPushButton(self.tr('Fit preview'))
        self.apply_button = QtWidgets.QPushButton(self.tr('Apply'))
        self.cancel_button = QtWidgets.QPushButton(self.tr('Cancel'))
        buttons.addWidget(self.fit_button)
        buttons.addStretch()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
        for control in self.controls.values():
            control.valueChanged.connect(self.rebuild)
        self.reset_button.clicked.connect(self.reset_distortion)
        self.fit_button.clicked.connect(self.fit_preview)
        self.apply_button.clicked.connect(self.apply_selection)
        self.cancel_button.clicked.connect(self.reject)
        self._apply_theme()
        self.rebuild()

    def reset_distortion(self):
        self._has_reset = True
        for key, control in self.controls.items():
            with QtCore.QSignalBlocker(control):
                control.setValue(100 if key.startswith('scale_') else 0)
        self.rebuild()

    def rebuild(self, *_):
        self.candidate = None
        self._preview_items = []
        for view in (self.current_view, self.transformed_view):
            view.scene().clear()
        try:
            settings = normalize_lettering_transform({
                key: (self._original_settings.get(key, 1 if key.startswith('scale_') else 0)
                      if not self._has_reset and control.value() == self._initial_values[key]
                      else control.value() / self._factors[key])
                for key, control in self.controls.items()
            })
            candidate = copy.deepcopy(self.before)
            candidate['lettering_transform'] = settings
            for view, props in ((self.current_view, self.before), (self.transformed_view, candidate)):
                item = TextBlockItem()
                apply_style_properties(item, props)
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
                item.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
                view.scene().addItem(item)
                self._preview_items.append(item)
            self.candidate = candidate
            self.status_label.setText(self.tr('Preview only. Apply creates one undoable change.'))
            perspective = bool(settings.get('perspective_x') or settings.get('perspective_y'))
            curved = bool(settings.get('arc_angle'))
            self.export_note.setText(self.tr(
                'PSD: curved lettering exports as a labeled image layer. Keep the project file to edit its text and shape.') if curved else self.tr(
                'PSD: perspective lettering exports as a labeled image layer. Keep the project file to edit its text and shape.') if perspective else self.tr(
                'PSD: this lettering exports as an editable text layer.'))
            self.fit_preview()
        except (ValueError, TypeError, RuntimeError):
            self.status_label.setText(self.tr('This transform could not be previewed. Reset the distortion and try again.'))
            self.export_note.clear()
        self.apply_button.setEnabled(self.candidate is not None)

    def fit_preview(self):
        if len(self._preview_items) != 2:
            return
        bounds = QtCore.QRectF()
        for item in self._preview_items:
            bounds = bounds.united(item.sceneBoundingRect())
        bounds = bounds.adjusted(-20, -20, 20, 20)
        for view in (self.current_view, self.transformed_view):
            view.setSceneRect(bounds)
            view.fitInView(bounds, QtCore.Qt.KeepAspectRatio)

    def apply_selection(self):
        if self.candidate is not None:
            self.result_properties = copy.deepcopy(self.candidate)
            self.accept()

    def reject(self):
        self.result_properties = None
        super().reject()

    def _apply_theme(self):
        border = dayu_theme.disable_color if dayu_theme.is_dark else dayu_theme.border_color
        self.setStyleSheet(f'''
            QDialog {{ background-color: {dayu_theme.background_color}; color: {dayu_theme.primary_text_color}; }}
            QLabel {{ color: {dayu_theme.primary_text_color}; }}
            QPushButton, QDoubleSpinBox {{ background-color: {dayu_theme.background_in_color};
                color: {dayu_theme.primary_text_color}; border: 1px solid {border};
                border-radius: 4px; padding: 4px 8px; min-height: 20px; }}
            QPushButton:hover, QPushButton:focus, QDoubleSpinBox:focus {{ border-color: {dayu_theme.primary_5}; }}
            QPushButton:disabled {{ color: {dayu_theme.disable_color}; }}
            QGraphicsView {{ border: 1px solid {border}; }}
        ''')
        for view in (self.current_view, self.transformed_view):
            view.setBackgroundBrush(QtGui.QColor('#aeb5bf'))

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_theme()
        QtCore.QTimer.singleShot(0, self.fit_preview)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_preview_items'):
            self.fit_preview()
