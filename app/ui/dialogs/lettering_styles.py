"""Project lettering presets with isolated, editable-renderer comparisons."""
from __future__ import annotations

import copy
import logging
from uuid import uuid4

from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.lettering_style_format import (
    apply_style_properties, font_warnings, style_from_properties, styled_properties,
)
from app.ui.canvas.text_item import TextBlockItem
from app.ui.dayu_widgets import dayu_theme
from modules.utils.lettering_style_exchange import (
    StyleExchangeError, merge_imported_styles, read_style_file, write_style_file,
)
from modules.utils.lettering_styles import normalize_style_library


logger = logging.getLogger(__name__)


class LetteringStylesDialog(QtWidgets.QDialog):
    """Only publish a draft library and prepared candidates on acceptance."""

    def __init__(self, library, selections, parent=None):
        super().__init__(parent)
        self._library = normalize_style_library(copy.deepcopy(library))
        self._selections = copy.deepcopy(selections)
        self._candidates = []
        self._warnings = []
        self._preview_items = []
        self.library_result = None
        self.candidates = []
        self.setWindowTitle(self.tr('Lettering styles'))
        self.setModal(True)
        self.resize(1040, 740)
        self.setMinimumSize(780, 580)
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(self.tr(
            'Reuse a project style or change emphasis directly. Bold, Italic and Underline affect whole selected text boxes; Keep existing preserves individual-word formatting.'))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        manager = QtWidgets.QHBoxLayout()
        self.preset_list = QtWidgets.QListWidget()
        self.preset_list.setAccessibleName(self.tr('Project styles'))
        self.preset_list.setMinimumWidth(170)
        self.preset_list.setMaximumHeight(142)
        library_column = QtWidgets.QVBoxLayout()
        library_column.addWidget(self.preset_list, 1)
        exchange_row = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton(self.tr('Import presets…'))
        self.import_button.setToolTip(self.tr('Existing presets are kept. Name conflicts receive a numbered suffix.'))
        self.export_button = QtWidgets.QPushButton(self.tr('Export presets…'))
        self.export_button.setToolTip(self.tr('Export all presets in this list to a file.'))
        exchange_row.addWidget(self.import_button)
        exchange_row.addWidget(self.export_button)
        library_column.addLayout(exchange_row)
        manager.addLayout(library_column, 1)
        edit = QtWidgets.QVBoxLayout()
        name_row = QtWidgets.QHBoxLayout()
        name_label = QtWidgets.QLabel(self.tr('Style name'))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setMaxLength(80)
        self.name_edit.setPlaceholderText(self.tr('Dialogue, narration, sound effects…'))
        name_label.setBuddy(self.name_edit)
        name_row.addWidget(name_label)
        name_row.addWidget(self.name_edit, 1)
        edit.addLayout(name_row)
        self.capture_button = QtWidgets.QPushButton(self.tr('Save from previewed text box'))
        self.capture_button.setToolTip(self.tr('Creates a new style from the current text box and your emphasis choices.'))
        self.rename_button = QtWidgets.QPushButton(self.tr('Rename'))
        self.delete_button = QtWidgets.QPushButton(self.tr('Delete style'))
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(self.capture_button, 2)
        action_row.addWidget(self.rename_button)
        action_row.addWidget(self.delete_button)
        edit.addLayout(action_row)
        self.replace_button = QtWidgets.QPushButton(self.tr('Replace style from previewed text box'))
        self.replace_button.setToolTip(self.tr('Replaces the selected preset with the current text box style and your emphasis choices.'))
        edit.addWidget(self.replace_button)
        self.library_status = QtWidgets.QLabel()
        self.library_status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.library_status.setWordWrap(True)
        edit.addWidget(self.library_status)
        edit.addStretch()
        manager.addLayout(edit, 3)
        layout.addLayout(manager)

        fields_box = QtWidgets.QGroupBox(self.tr('Properties to apply'))
        fields_layout = QtWidgets.QGridLayout(fields_box)
        self.field_checks = {}
        labels = [
            ('font_family', self.tr('Font')), ('font_size', self.tr('Font size')),
            ('text_color', self.tr('Text color')), ('outline', self.tr('Outline')),
            ('line_spacing', self.tr('Line spacing')), ('alignment', self.tr('Alignment')),
        ]
        for index, (key, label) in enumerate(labels):
            check = QtWidgets.QCheckBox(label)
            check.setChecked(True)
            check.toggled.connect(self.rebuild)
            self.field_checks[key] = check
            fields_layout.addWidget(check, index // 6, index % 6)
        self.emphasis_combos = {}
        for column, (key, label) in enumerate((('bold', self.tr('Bold')),
                                              ('italic', self.tr('Italic')),
                                              ('underline', self.tr('Underline')))):
            row = QtWidgets.QHBoxLayout()
            caption = QtWidgets.QLabel(label)
            combo = QtWidgets.QComboBox()
            combo.setAccessibleName(label)
            caption.setBuddy(combo)
            combo.addItem(self.tr('Keep existing'), 'keep')
            combo.addItem(self.tr('On'), 'on')
            combo.addItem(self.tr('Off'), 'off')
            combo.addItem(self.tr('Use preset'), 'preset')
            combo.setToolTip(self.tr('On and Off change this emphasis for each entire selected box. Keep existing preserves word-level formatting. Use preset copies the saved value.'))
            combo.currentIndexChanged.connect(self.rebuild)
            self.emphasis_combos[key] = combo
            row.addWidget(caption)
            row.addWidget(combo, 1)
            fields_layout.addLayout(row, 1, column * 2, 1, 2)
        layout.addWidget(fields_box)

        choice_row = QtWidgets.QHBoxLayout()
        choice_label = QtWidgets.QLabel(self.tr('Preview text box'))
        self.selection_combo = QtWidgets.QComboBox()
        self.selection_combo.setAccessibleName(self.tr('Preview text box'))
        choice_label.setBuddy(self.selection_combo)
        for index, props in enumerate(self._selections):
            document = QtGui.QTextDocument()
            document.setHtml(props.get('text', ''))
            excerpt = ' '.join(document.toPlainText().split())[:48]
            self.selection_combo.addItem(self.tr('{0}: {1}').format(index + 1, excerpt), index)
        choice_row.addWidget(choice_label)
        choice_row.addWidget(self.selection_combo, 1)
        self.count_label = QtWidgets.QLabel(self.tr('{0} selected text boxes').format(len(self._selections)))
        choice_row.addWidget(self.count_label)
        self.fit_button = QtWidgets.QPushButton(self.tr('Fit preview'))
        choice_row.addWidget(self.fit_button)
        layout.addLayout(choice_row)
        comparison = QtWidgets.QHBoxLayout()
        self.current_view, self.styled_view = QtWidgets.QGraphicsView(), QtWidgets.QGraphicsView()
        for view, label in ((self.current_view, self.tr('Current')), (self.styled_view, self.tr('Styled'))):
            column = QtWidgets.QVBoxLayout()
            column.addWidget(QtWidgets.QLabel(label))
            view.setScene(QtWidgets.QGraphicsScene(view))
            view.setInteractive(False)
            view.setAccessibleName(label)
            view.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
            view.setMinimumHeight(170)
            column.addWidget(view, 1)
            comparison.addLayout(column, 1)
        layout.addLayout(comparison, 1)
        note = QtWidgets.QLabel(self.tr('Preview uses a neutral background. Inspect text fit after changing font or size.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(30)
        layout.addWidget(self.status_label)
        self.warning_ack = QtWidgets.QCheckBox(self.tr('Apply despite these font warnings'))
        self.warning_ack.hide()
        self.warning_ack.toggled.connect(self._update_apply)
        layout.addWidget(self.warning_ack)
        buttons = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton(self.tr('Save presets only'))
        self.apply_button = QtWidgets.QPushButton(self.tr('Apply to selection'))
        self.cancel_button = QtWidgets.QPushButton(self.tr('Cancel'))
        buttons.addWidget(self.save_button)
        buttons.addStretch()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
        self.capture_button.clicked.connect(self.capture_preset)
        self.replace_button.clicked.connect(self.replace_preset)
        self.rename_button.clicked.connect(self.rename_preset)
        self.delete_button.clicked.connect(self.delete_preset)
        self.import_button.clicked.connect(self.import_presets)
        self.export_button.clicked.connect(self.export_presets)
        self.save_button.clicked.connect(self.save_presets)
        self.apply_button.clicked.connect(self.apply_selection)
        self.cancel_button.clicked.connect(self.reject)
        self.preset_list.currentRowChanged.connect(self.preset_changed)
        self.selection_combo.currentIndexChanged.connect(self.show_selection)
        self.fit_button.clicked.connect(self.fit_preview)
        self._theme_ready = True
        self._apply_theme()
        self._refresh_list()
        self.rebuild()

    @property
    def fields(self):
        fields = {key for key, check in self.field_checks.items() if check.isChecked()}
        if 'outline' in fields:
            fields.update(('outline_color', 'outline_width'))
        fields.update(key for key, combo in self.emphasis_combos.items()
                      if combo.currentData() != 'keep')
        return fields

    def _emphasis_overrides(self):
        return {key: combo.currentData() == 'on'
                for key, combo in self.emphasis_combos.items()
                if combo.currentData() in {'on', 'off'}}

    def _capture_properties(self, index):
        properties = style_from_properties(self._selections[index])
        preset = self._preset()
        for key, combo in self.emphasis_combos.items():
            if (combo.currentData() == 'preset' and preset is not None
                    and key in preset['properties']):
                properties[key] = preset['properties'][key]
        properties.update(self._emphasis_overrides())
        return properties

    def _preset(self):
        index = self.preset_list.currentRow()
        return self._library['presets'][index] if 0 <= index < len(self._library['presets']) else None

    def _refresh_list(self, selected_id=None):
        with QtCore.QSignalBlocker(self.preset_list):
            self.preset_list.clear()
            selected = -1
            for index, preset in enumerate(self._library['presets']):
                self.preset_list.addItem(preset['name'])
                if preset['id'] == selected_id:
                    selected = index
            if selected < 0 and self._library['presets']:
                selected = 0
            self.preset_list.setCurrentRow(selected)
        self.preset_changed()

    def preset_changed(self, *_):
        preset = self._preset()
        self.name_edit.setText(preset['name'] if preset else '')
        self.rename_button.setEnabled(preset is not None)
        self.delete_button.setEnabled(preset is not None)
        self.capture_button.setEnabled(bool(self._selections) and len(self._library['presets']) < 100)
        self.replace_button.setEnabled(bool(self._selections) and preset is not None)
        self.export_button.setEnabled(bool(self._library['presets']))
        self.library_status.setText(self.tr('Changes to presets are saved only when you choose Save presets only or Apply to selection.'))
        self.rebuild()

    def _name(self, excluding_id=None):
        name = self.name_edit.text().strip()
        if not name:
            self.library_status.setText(self.tr('Enter a style name.'))
            return None
        if any(p['name'].casefold() == name.casefold() and p['id'] != excluding_id for p in self._library['presets']):
            self.library_status.setText(self.tr('A style with this name already exists. Choose another name.'))
            return None
        return name

    def capture_preset(self):
        index = self.selection_combo.currentIndex()
        if not 0 <= index < len(self._selections) or len(self._library['presets']) >= 100:
            return
        name = self._name()
        if name is None:
            return
        try:
            preset = {'id': str(uuid4()), 'name': name,
                      'properties': self._capture_properties(index)}
            self._library['presets'].append(preset)
            self._refresh_list(preset['id'])
        except Exception:
            logger.exception('Unable to capture lettering style')
            self.library_status.setText(self.tr('Could not capture this text style. No preset was added.'))

    def rename_preset(self):
        preset = self._preset()
        if preset is None:
            return
        name = self._name(preset['id'])
        if name is not None:
            preset['name'] = name
            self._refresh_list(preset['id'])

    def replace_preset(self):
        preset = self._preset()
        index = self.selection_combo.currentIndex()
        if preset is None or not 0 <= index < len(self._selections):
            return
        try:
            properties = self._capture_properties(index)
            preset['properties'] = properties
            self.rebuild()
        except Exception:
            logger.exception('Unable to replace lettering style')
            self.library_status.setText(self.tr('Could not capture this text style. No preset was added.'))

    def delete_preset(self):
        index = self.preset_list.currentRow()
        if 0 <= index < len(self._library['presets']):
            del self._library['presets'][index]
            self._refresh_list()

    def import_presets(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr('Import presets…'), '', self.tr('Lettering presets (*.json)'))
        if not path:
            return
        try:
            result = merge_imported_styles(self._library, read_style_file(path))
        except StyleExchangeError as error:
            self._show_exchange_error(error)
            return
        if result.added:
            self._library = result.library
            first_added = self._library['presets'][-result.added]
            self._refresh_list(first_added['id'])
        self.library_status.setText(self.tr('Imported: {0}. Renamed: {1}. Skipped duplicates: {2}.').format(
            result.added, result.renamed, result.skipped))

    def export_presets(self):
        picker = QtWidgets.QFileDialog(self, self.tr('Export presets…'))
        picker.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        picker.setNameFilter(self.tr('Lettering presets (*.json)'))
        picker.setDefaultSuffix('json')
        picker.selectFile('lettering-presets.json')
        try:
            if picker.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            path = picker.selectedFiles()[0]
        finally:
            picker.deleteLater()
        try:
            write_style_file(path, self._library)
        except StyleExchangeError as error:
            self._show_exchange_error(error)
            return
        self.library_status.setText(self.tr('Exported {0} presets.').format(len(self._library['presets'])))

    def _show_exchange_error(self, error):
        messages = {
            'read': self.tr('Could not read the preset file.'),
            'write': self.tr('Could not write the preset file.'),
            'oversized': self.tr('The preset file is too large.'),
            'format': self.tr('This file is not a lettering preset library.'),
            'version': self.tr('This preset file version is unsupported.'),
            'invalid': self.tr('The preset file contains invalid styles.'),
            'capacity': self.tr('The import would exceed the project limit of 100 presets.'),
            'empty': self.tr('The preset library is empty.'),
        }
        self.library_status.setText(messages[error.code])

    def rebuild(self, *_):
        self._candidates = []
        self._warnings = []
        with QtCore.QSignalBlocker(self.warning_ack):
            self.warning_ack.setChecked(False)
        self.warning_ack.hide()
        preset = self._preset()
        properties = dict(preset['properties']) if preset else {}
        properties.update(self._emphasis_overrides())
        for key, combo in self.emphasis_combos.items():
            available = preset is not None and key in preset['properties']
            combo.model().item(combo.findData('preset')).setEnabled(available)
            if not available and combo.currentData() == 'preset':
                with QtCore.QSignalBlocker(combo):
                    combo.setCurrentIndex(combo.findData('keep'))
        try:
            if self.fields & properties.keys() and self._selections:
                candidates = [styled_properties(props, properties, self.fields) for props in self._selections]
                warnings = []
                for index, candidate in enumerate(candidates):
                    for warning in font_warnings(candidate):
                        warnings.append((index + 1, warning))
                self._candidates, self._warnings = candidates, warnings
                if warnings:
                    self.status_label.setText('\n'.join(self._warning_text(index, warning) for index, warning in warnings)[:3000])
                    self.warning_ack.show()
                else:
                    self.status_label.setText(self.tr('Ready to apply formatting to {0} text boxes.').format(len(candidates)))
            elif not self._selections:
                self.status_label.setText(self.tr('Select text boxes on one page to capture or apply a style. You can still manage project presets.'))
            elif preset is None:
                self.status_label.setText(self.tr('Choose a project style, save a new style, or set Bold, Italic or Underline to On or Off.'))
            else:
                self.status_label.setText(self.tr('Choose at least one property to apply.'))
            self.show_selection()
        except Exception:
            logger.exception('Lettering style preview failed')
            self._candidates = []
            self.status_label.setText(self.tr('Could not build the style preview. No changes were applied.'))
            for view in (self.current_view, self.styled_view):
                view.scene().clear()
        self._update_apply()

    def _warning_text(self, index, warning):
        if warning.get('code') == 'missing_font':
            return self.tr('Text box {0}: font "{1}" is unavailable. A fallback font will be used.').format(index, warning.get('font', ''))
        return self.tr('Text box {0}: font "{1}" lacks these characters: {2}. Fallback glyphs may differ.').format(
            index, warning.get('font', ''), warning.get('characters', ''))

    def _update_apply(self, *_):
        self.apply_button.setEnabled(bool(self._candidates) and (not self._warnings or self.warning_ack.isChecked()))

    def show_selection(self, *_):
        self._preview_items = []
        for view in (self.current_view, self.styled_view):
            view.scene().clear()
        index = self.selection_combo.currentIndex()
        if not 0 <= index < len(self._selections):
            return
        styled = self._candidates[index] if len(self._candidates) == len(self._selections) else self._selections[index]
        for view, props in ((self.current_view, self._selections[index]), (self.styled_view, styled)):
            item = TextBlockItem()
            apply_style_properties(item, copy.deepcopy(props))
            item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            item.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
            # Move the preview by translation only; keep its actual rotation/scale.
            view.scene().addItem(item)
            item.setPos(item.pos() - item.sceneBoundingRect().topLeft())
            self._preview_items.append(item)
        self.fit_preview()

    def fit_preview(self):
        if not self._preview_items:
            return
        bounds = QtCore.QRectF()
        for item in self._preview_items:
            bounds = bounds.united(item.sceneBoundingRect())
        bounds = bounds.adjusted(-16, -16, 16, 16)
        for view in (self.current_view, self.styled_view):
            view.setSceneRect(bounds)
            view.fitInView(bounds, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def save_presets(self):
        self.library_result = copy.deepcopy(self._library)
        self.candidates = []
        super().accept()

    def apply_selection(self):
        if not self.apply_button.isEnabled():
            return
        self.library_result = copy.deepcopy(self._library)
        self.candidates = copy.deepcopy(self._candidates)
        super().accept()

    def reject(self):
        self.library_result = None
        self.candidates = []
        super().reject()

    def _apply_theme(self):
        outline = dayu_theme.disable_color if dayu_theme.is_dark else dayu_theme.border_color
        style = f'''
            QDialog {{ background-color: {dayu_theme.background_color}; color: {dayu_theme.primary_text_color}; }}
            QLabel, QCheckBox, QGroupBox {{ color: {dayu_theme.primary_text_color}; }}
            QPushButton, QComboBox, QLineEdit, QListWidget {{ background-color: {dayu_theme.background_in_color};
                color: {dayu_theme.primary_text_color}; border: 1px solid {outline};
                border-radius: 4px; padding: 4px 8px; min-height: 18px; }}
            QPushButton:hover, QPushButton:focus {{ border-color: {dayu_theme.primary_5};
                background-color: {dayu_theme.background_out_color}; }}
            QPushButton:disabled {{ color: {dayu_theme.disable_color}; border-color: {dayu_theme.border_color}; }}
            QListWidget::item:selected {{ background-color: {dayu_theme.background_out_color};
                color: {dayu_theme.primary_text_color}; border: 1px solid {dayu_theme.primary_5}; border-radius: 3px; }}
            QGraphicsView {{ border: 1px solid {outline}; }}
            QGroupBox {{ border: 1px solid {outline}; border-radius: 4px; margin-top: 8px; padding-top: 10px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 9px; padding: 0 5px; }}
        '''
        if self.styleSheet() != style:
            self.setStyleSheet(style)
        for view in (self.current_view, self.styled_view):
            view.setBackgroundBrush(QtGui.QColor('#aeb5bf'))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.StyleChange and getattr(self, '_theme_ready', False):
            self._apply_theme()

    def showEvent(self, event):
        self._apply_theme()
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.fit_preview)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_preview_items'):
            self.fit_preview()
