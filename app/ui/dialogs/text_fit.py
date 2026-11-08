"""A non-destructive preview of rectangular rich-text fitting."""
from __future__ import annotations

import copy
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.vertical_layout import VerticalTextDocumentLayout
from modules.rendering.layout import build_layout_candidates


def boundary_message(reason):
    # Keep the literals visible to Qt's source extractor.
    if reason == 'unsupported_transform':
        return QtCore.QCoreApplication.translate('TextFitDialog', 'Bubble fitting requires unrotated horizontal text at normal scale. Using Box.')
    if reason in ('missing_image', 'invalid_image', 'image_dimensions'):
        return QtCore.QCoreApplication.translate('TextFitDialog', 'Original image unavailable or its dimensions changed. Using Box.')
    if reason == 'padding_empty':
        return QtCore.QCoreApplication.translate('TextFitDialog', 'Padding leaves no usable interior. Using Box.')
    if reason == 'region_too_large':
        return QtCore.QCoreApplication.translate('TextFitDialog', 'Fitting area is too large. Using Box.')
    if reason == 'invalid_bounds':
        return QtCore.QCoreApplication.translate('TextFitDialog', 'The boundary is outside the page or has no usable area. Using Box.')
    return QtCore.QCoreApplication.translate('TextFitDialog', 'No reliable enclosed interior was found. Using Box. Try Rectangle or Ellipse.')


class TextFitDialog(QtWidgets.QDialog):
    def __init__(self, item, width, height, minimum_font_size=10, maximum_font_size=72, parent=None, *, boundary_context=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Fit Text"))
        self.resize(760, 620)
        self.document = item.document().clone()
        self.font_size = item.font_size
        self.properties = dict(font_family=item.font_family, render_color=item.text_color,
                               alignment=item.alignment, line_spacing=item.line_spacing,
                               outline_color=item.outline_color, outline_width=item.outline_width,
                               bold=item.bold, italic=item.italic, underline=item.underline,
                               direction=item.direction)
        self.vertical = item.vertical
        self.current_vertical_size = None
        if self.vertical and item.layout:
            self.current_vertical_size = (item.layout.max_width, item.layout.max_height)
            snapshot_layout = VerticalTextDocumentLayout(self.document, line_spacing=item.line_spacing)
            self.document.setDocumentLayout(snapshot_layout)
            snapshot_layout.set_max_size(*self.current_vertical_size)
        self.outlines = copy.deepcopy(item.selection_outlines)
        self.outline_extent = max([float(o.width) for o in self.outlines] + [0.0])
        self.candidates = []
        self.preview_item = None
        self.boundary_context = boundary_context or {}
        self.boundary = None
        self.boundary_note = ''
        self._boundary_cache = {}
        self._original_loaded = False
        self._original_image = None
        self._original_reason = ''
        self.item_position = tuple(self.boundary_context.get('position', (0, 0)))
        self.bubble_supported = (not item.vertical and abs(item.rotation() % 360) < 1e-6
                                 and abs(item.scale() - 1) < 1e-6 and item.transform().isIdentity())
        prefs = getattr(item, 'layout_preferences', {})
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(self.tr("Preview text layout inside the fitting area. Artwork and cleanup masks are unchanged."))
        info.setWordWrap(True)
        layout.addWidget(info)
        form = QtWidgets.QFormLayout()
        self.form = form
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem(self.tr('Box'), 'box')
        self.mode.addItem(self.tr('Bubble interior'), 'bubble')
        self.mode.setCurrentIndex(1 if prefs.get('mode') == 'bubble' else 0)
        form.addRow(self.tr('Fit mode'), self.mode)
        self.boundary_shape = QtWidgets.QComboBox()
        self.boundary_shape.addItem(self.tr('Estimate from original'), 'auto')
        self.boundary_shape.addItem(self.tr('Rectangle'), 'rectangle')
        self.boundary_shape.addItem(self.tr('Ellipse'), 'ellipse')
        self.boundary_shape.setCurrentIndex(max(0, self.boundary_shape.findData(prefs.get('boundary_shape', 'auto'))))
        form.addRow(self.tr('Boundary'), self.boundary_shape)
        bounds = prefs.get('boundary_box') or self.boundary_context.get('bubble_bounds')
        if bounds is None:
            bounds = (*self.item_position, self.item_position[0] + width, self.item_position[1] + height)
        self.boundary_left = QtWidgets.QDoubleSpinBox()
        self.boundary_top = QtWidgets.QDoubleSpinBox()
        origin_row = QtWidgets.QHBoxLayout()
        for spin, value, label in ((self.boundary_left, bounds[0], self.tr('Left')),
                                   (self.boundary_top, bounds[1], self.tr('Top'))):
            spin.setRange(-100000, 100000)
            spin.setValue(value)
            spin.setAccessibleName(label)
            origin_row.addWidget(spin)
        form.addRow(self.tr('Boundary origin: X × Y'), origin_row)
        self._box_size = (width, height)
        self._boundary_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        area = QtWidgets.QHBoxLayout()
        self.area_width = QtWidgets.QDoubleSpinBox()
        self.area_height = QtWidgets.QDoubleSpinBox()
        for spin, value in ((self.area_width, width), (self.area_height, height)):
            spin.setRange(1, 100000)
            spin.setDecimals(1)
            spin.setValue(value)
            spin.setSuffix(" px")
            area.addWidget(spin)
        self.area_width.setAccessibleName(self.tr("Width"))
        self.area_height.setAccessibleName(self.tr("Height"))
        form.addRow(self.tr("Fitting area: width × height"), area)
        self.padding = QtWidgets.QDoubleSpinBox()
        self.padding.setRange(0, 200)
        self.padding.setValue(prefs.get('padding', 8))
        self.padding.setSuffix(' px')
        form.addRow(self.tr('Padding'), self.padding)
        self.boundary_status = QtWidgets.QLabel()
        self.boundary_status.setWordWrap(True)
        form.addRow(self.boundary_status)
        self.minimum = QtWidgets.QDoubleSpinBox()
        self.maximum = QtWidgets.QDoubleSpinBox()
        for spin, value in ((self.minimum, minimum_font_size), (self.maximum, maximum_font_size)):
            spin.setRange(1, 300)
            spin.setDecimals(1)
            spin.setValue(value)
        form.addRow(self.tr("Minimum font size"), self.minimum)
        form.addRow(self.tr("Maximum font size"), self.maximum)
        self.keep_breaks = QtWidgets.QCheckBox(self.tr("Keep manual line breaks"))
        self.keep_breaks.setChecked(getattr(item, 'layout_preferences', {}).get('keep_manual_breaks', True))
        form.addRow(self.keep_breaks)
        self.choice = QtWidgets.QComboBox()
        self.choice.addItems([self.tr("Current"), self.tr("Balanced"), self.tr("Larger text")])
        form.addRow(self.tr("Layout"), self.choice)
        layout.addLayout(form)
        self.view = QtWidgets.QGraphicsView()
        self.view.setMinimumHeight(220)
        self.scene = QtWidgets.QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        self.view.setInteractive(False)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        layout.addWidget(self.view, 1)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QDialogButtonBox()
        self.apply_button = buttons.addButton(self.tr("Apply"), QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_button = buttons.addButton(self.tr("Cancel"), QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        self.apply_button.setAutoDefault(False)
        self.cancel_button.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(180)
        self.timer.timeout.connect(self.rebuild)
        for spin in (self.area_width, self.area_height, self.minimum, self.maximum):
            spin.valueChanged.connect(self.schedule)
        for spin in (self.boundary_left, self.boundary_top, self.padding):
            spin.valueChanged.connect(self.schedule)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.boundary_shape.currentIndexChanged.connect(self.schedule)
        self.keep_breaks.toggled.connect(self.schedule)
        self.choice.currentIndexChanged.connect(self.show_candidate)
        self.mode_changed()
        self.choice.setCurrentIndex(1)

    def mode_changed(self, *_):
        bubble = self.mode.currentData() == 'bubble'
        previous = getattr(self, '_previous_mode', None)
        if previous == 'box':
            self._box_size = (self.area_width.value(), self.area_height.value())
        elif previous == 'bubble':
            self._boundary_size = (self.area_width.value(), self.area_height.value())
        self._previous_mode = self.mode.currentData()
        values = self._boundary_size if bubble else self._box_size
        for spin, value in zip((self.area_width, self.area_height), values):
            with QtCore.QSignalBlocker(spin):
                spin.setValue(value)
        self.rebuild()

    def resolve_boundary(self):
        self.boundary = None
        self.boundary_note = ''
        bubble = self.mode.currentData() == 'bubble'
        manual = self.boundary_shape.currentData() != 'auto'
        # Keep the ordinary Box workflow compact; expose geometry overrides
        # only when a manual boundary is selected.
        for row, visible in ((1, bubble), (2, bubble and manual),
                             (3, not bubble or manual), (4, bubble), (5, bubble)):
            self.form.setRowVisible(row, visible)
        self.boundary_shape.setEnabled(bubble)
        self.padding.setEnabled(bubble)
        for spin in (self.boundary_left, self.boundary_top):
            spin.setEnabled(bubble and manual)
        for spin in (self.area_width, self.area_height):
            spin.setEnabled(not bubble or manual)
        if not bubble:
            return
        from modules.rendering.bubble_boundary import estimate_bubble_boundary
        if not self.bubble_supported:
            self.boundary_note = boundary_message('unsupported_transform')
            return
        if not manual and not self._original_loaded:
            self._original_loaded = True
            loader = self.boundary_context.get('image_loader')
            if loader:
                try:
                    self._original_image, self._original_reason = loader()
                except (OSError, ValueError, RuntimeError):
                    self._original_reason = 'missing_image'
        if not manual and self._original_reason:
            self.boundary_note = boundary_message(self._original_reason)
            return
        override_bounds = (self.boundary_left.value(), self.boundary_top.value(),
                           self.boundary_left.value() + self.area_width.value(),
                           self.boundary_top.value() + self.area_height.value())
        key = (self.boundary_shape.currentData(), override_bounds, self.padding.value())
        if key not in self._boundary_cache:
            if len(self._boundary_cache) >= 8:
                self._boundary_cache.clear()
            self._boundary_cache[key] = estimate_bubble_boundary(
                None if manual else self._original_image, self.boundary_context.get('text_bounds', override_bounds),
                self.boundary_context.get('bubble_bounds'), padding=self.padding.value(),
                override=key[0], override_bounds=override_bounds, page_size=self.boundary_context.get('page_size'))
        boundary = self._boundary_cache[key]
        if boundary.available:
            self.boundary = boundary
            self.boundary_note = (self.tr('Estimated interior. Check the boundary before applying.')
                                  if boundary.source == 'estimated' else self.tr('Manual boundary.'))
        else:
            self.boundary_note = boundary_message(boundary.reason)

    def schedule(self, *_):
        self.apply_button.setEnabled(False)
        self.timer.start()

    def rebuild(self):
        self.timer.stop()
        if self.minimum.value() > self.maximum.value():
            self.status.setText(self.tr("Minimum font size must not exceed maximum font size."))
            self.candidates = []
            self.apply_button.setEnabled(False)
            return
        self.resolve_boundary()
        self.boundary_status.setText(self.boundary_note)
        common = dict(
            font_size=self.font_size, minimum_font_size=self.minimum.value(),
            maximum_font_size=self.maximum.value(), outline_width=self.outline_extent,
            line_spacing=self.properties['line_spacing'],
            keep_manual_breaks=self.keep_breaks.isChecked())
        if self.boundary is not None:
            from modules.rendering.bubble_layout import build_bubble_candidates
            current = tuple(p - o for p, o in zip(self.item_position, self.boundary.origin))
            self.candidates = build_bubble_candidates(self.document, mask=self.boundary.mask,
                                                       current_position=current, **common)
        else:
            # Failure uses the established rectangular constraints, not an
            # unverified detector-sized approximation of the bubble.
            width, height = ((self.area_width.value(), self.area_height.value())
                             if self.mode.currentData() == 'box' else self._box_size)
            self.candidates = build_layout_candidates(self.document, width=width, height=height,
                                                       vertical=self.vertical, **common)
        self.show_candidate()

    def show_candidate(self, *_):
        candidate = self.selected_candidate()
        if candidate is None:
            return
        self.scene.clear()
        self.preview_item = TextBlockItem(font_size=candidate.font_size, **self.properties)
        self.preview_item.setHtml(candidate.html)
        self.preview_item.document().setDefaultFont(candidate.default_font)
        self.preview_item.setTextWidth(candidate.width)
        self.preview_item.set_vertical(self.vertical)
        if self.vertical and self.preview_item.layout:
            limits = (self.current_vertical_size if candidate.key == 'current' and self.current_vertical_size
                      else (candidate.width, self.area_height.value() if self.mode.currentData() == 'box' else self._box_size[1]))
            self.preview_item.layout.set_max_size(*limits)
        self.preview_item.selection_outlines = copy.deepcopy(self.outlines)
        self.scene.addItem(self.preview_item)
        if self.boundary is not None:
            self.preview_item.setPos(candidate.offset_x, candidate.offset_y)
        color = self.properties['render_color']
        background = QtGui.QColor('#252525' if color is not None and color.lightness() > 150 else '#ffffff')
        self.view.setBackgroundBrush(background)
        if self.boundary is not None:
            boundary = QtCore.QRectF(0, 0, self.boundary.mask.shape[1], self.boundary.mask.shape[0])
            # A translucent overlay of the actual padded mask, including holes.
            rgba = np.zeros((*self.boundary.mask.shape, 4), dtype=np.uint8)
            rgba[self.boundary.mask] = (50, 150, 210, 55)
            h, w = self.boundary.mask.shape
            qimage = QtGui.QImage(rgba.data, w, h, w * 4, QtGui.QImage.Format.Format_RGBA8888).copy()
            overlay = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimage))
            overlay.setZValue(-1)
        else:
            width, height = ((self.area_width.value(), self.area_height.value())
                             if self.mode.currentData() == 'box' else self._box_size)
            boundary = QtCore.QRectF(0, 0, width, height)
        pen = QtGui.QPen(QtGui.QColor('#888888'), 1, QtCore.Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        if self.boundary is None:
            self.scene.addRect(boundary, pen)
        self.scene.setSceneRect(boundary.united(self.preview_item.sceneBoundingRect()).adjusted(-8, -8, 8, 8))
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        status = self.tr("Fits within the area.") if candidate.fits else self.tr("Does not fit. Enlarge the area or shorten the text.")
        if candidate.below_minimum:
            status += ' ' + self.tr("Text is below the minimum readable size.")
        size = f'{candidate.font_size:.2f}'.rstrip('0').rstrip('.')
        self.status.setText(self.tr("Font size: {size} — {status}").format(size=size, status=status))
        self.apply_button.setEnabled(candidate.fits and candidate.key != 'current' and not self.timer.isActive())

    def selected_candidate(self):
        index = self.choice.currentIndex()
        return self.candidates[index] if 0 <= index < len(self.candidates) else None

    def preferences(self):
        bubble = self.boundary is not None
        width, height = ((self.area_width.value(), self.area_height.value())
                         if self.mode.currentData() == 'box' else self._box_size)
        prefs = {'mode': 'bubble' if bubble else 'box', 'bound_width': width, 'bound_height': height,
                 'min_font_size': self.minimum.value(), 'keep_manual_breaks': self.keep_breaks.isChecked()}
        if bubble:
            prefs.update(padding=self.padding.value(), boundary_shape=self.boundary_shape.currentData(),
                         boundary_box=[self.boundary_left.value(), self.boundary_top.value(),
                                       self.boundary_left.value() + self.area_width.value(),
                                       self.boundary_top.value() + self.area_height.value()])
        return prefs

    def candidate_page_position(self):
        candidate = self.selected_candidate()
        if self.boundary is not None and candidate is not None:
            return (self.boundary.origin[0] + candidate.offset_x, self.boundary.origin[1] + candidate.offset_y)
        return self.item_position

    def accept(self):
        if self.timer.isActive():
            self.rebuild()
        if self.selected_candidate() is not None and self.apply_button.isEnabled():
            super().accept()

    def showEvent(self, event):
        super().showEvent(event)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
