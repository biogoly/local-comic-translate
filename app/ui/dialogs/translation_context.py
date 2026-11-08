"""Project-scoped glossary editing; previewing never sends a model request."""
from __future__ import annotations

import copy
import uuid

from PySide6 import QtCore, QtWidgets

from app.ui.dayu_widgets import dayu_theme
from modules.translation.context import normalize_translation_profile
from modules.translation.glossary_suggestion import glossary_source_conflicts


class _ContextDialog(QtWidgets.QDialog):
    """Small theme-aware outlines scoped to context-editor windows."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._context_theme_ready = True
        self._apply_context_theme()

    def _apply_context_theme(self):
        outline = dayu_theme.disable_color if dayu_theme.is_dark else dayu_theme.border_color
        style = f'''
            QPushButton, QToolButton {{
                background-color: {dayu_theme.background_in_color};
                color: {dayu_theme.primary_text_color};
                border: 1px solid {outline};
                border-radius: 4px;
                padding: 5px 10px;
                min-height: 18px;
            }}
            QPushButton:hover, QPushButton:focus, QToolButton:hover, QToolButton:focus {{
                background-color: {dayu_theme.background_out_color};
                border-color: {dayu_theme.primary_5};
            }}
            QPushButton:pressed, QToolButton:pressed, QToolButton:checked {{
                background-color: {dayu_theme.background_selected_color};
                border-color: {dayu_theme.primary_color};
            }}
            QPushButton:disabled, QToolButton:disabled {{
                background-color: {dayu_theme.background_color};
                color: {dayu_theme.disable_color};
                border-color: {dayu_theme.border_color};
            }}
            QTabWidget::pane {{
                border: 1px solid {dayu_theme.border_color};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: {dayu_theme.background_in_color};
                color: {dayu_theme.primary_text_color};
                border: 1px solid {outline};
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 7px 16px;
                min-width: 85px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {dayu_theme.background_out_color};
                border-bottom: 2px solid {dayu_theme.primary_color};
            }}
            QTabBar::tab:hover {{ border-color: {dayu_theme.primary_5}; }}
        '''
        if self.styleSheet() != style:
            self.setStyleSheet(style)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.StyleChange and getattr(self, '_context_theme_ready', False):
            self._apply_context_theme()

    def showEvent(self, event):
        self._apply_context_theme()
        super().showEvent(event)


class ContextEntryDialog(_ContextDialog):
    def __init__(self, kind, entry=None, languages=None, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.entry = copy.deepcopy(entry or {})
        self.setWindowTitle(self.tr('Glossary entry') if kind == 'glossary' else self.tr('Character'))
        self.resize(560, 540)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.source = QtWidgets.QLineEdit(self.entry.get('source' if kind == 'glossary' else 'name', ''))
        self.target = QtWidgets.QLineEdit(self.entry.get('target', ''))
        self.aliases = QtWidgets.QPlainTextEdit('\n'.join(self.entry.get('aliases', [])))
        self.aliases.setMaximumHeight(85)
        self.aliases.setPlaceholderText(self.tr('One alias per line'))
        form.addRow(self.tr('Source term') if kind == 'glossary' else self.tr('Name'), self.source)
        form.addRow(self.tr('Preferred translation'), self.target)
        form.addRow(self.tr('Aliases'), self.aliases)
        self.source_lang = self._language_combo(languages, self.entry.get('source_lang', ''))
        self.target_lang = self._language_combo(languages, self.entry.get('target_lang', ''))
        form.addRow(self.tr('Source language'), self.source_lang)
        form.addRow(self.tr('Target language'), self.target_lang)
        self.match = QtWidgets.QComboBox()
        self.match.addItem(self.tr('Automatic'), 'auto')
        self.match.addItem(self.tr('Whole word'), 'whole_word')
        self.match.addItem(self.tr('Substring'), 'substring')
        self.match.setCurrentIndex(max(0, self.match.findData(self.entry.get('match', 'auto'))))
        if kind == 'glossary':
            form.addRow(self.tr('Matching rule'), self.match)
        self.notes = QtWidgets.QPlainTextEdit(self.entry.get('note' if kind == 'glossary' else 'notes', ''))
        self.notes.setMaximumHeight(130)
        form.addRow(self.tr('Notes') if kind == 'glossary' else self.tr('Voice and formality notes'), self.notes)
        self.enabled = QtWidgets.QCheckBox(self.tr('Enabled'))
        self.enabled.setChecked(self.entry.get('enabled', True))
        self.approved = QtWidgets.QCheckBox(self.tr('Approved for translation'))
        self.approved.setChecked(self.entry.get('approved', True))
        form.addRow(self.enabled)
        form.addRow(self.approved)
        layout.addLayout(form)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for label, field in ((self.tr('Source term') if kind == 'glossary' else self.tr('Name'), self.source),
                             (self.tr('Preferred translation'), self.target), (self.tr('Aliases'), self.aliases),
                             (self.tr('Source language'), self.source_lang), (self.tr('Target language'), self.target_lang),
                             (self.tr('Matching rule'), self.match), (self.tr('Notes'), self.notes)):
            field.setAccessibleName(label)

    def _language_combo(self, languages, selected):
        combo = QtWidgets.QComboBox()
        combo.addItem(self.tr('Any language'), '')
        for canonical, label in (languages or []):
            if canonical and canonical != 'Auto':
                combo.addItem(label, canonical)
        index = combo.findData(selected)
        if selected and index < 0:
            combo.addItem(selected, selected)
            index = combo.count() - 1
        combo.setCurrentIndex(max(0, index))
        return combo

    def value(self):
        result = dict(id=self.entry.get('id') or str(uuid.uuid4()),
                      aliases=[line.strip() for line in self.aliases.toPlainText().splitlines() if line.strip()],
                      target=self.target.text().strip(), source_lang=self.source_lang.currentData(),
                      target_lang=self.target_lang.currentData(), enabled=self.enabled.isChecked(),
                      approved=self.approved.isChecked())
        if self.kind == 'glossary':
            result.update(source=self.source.text().strip(), note=self.notes.toPlainText(), match=self.match.currentData())
        else:
            result.update(name=self.source.text().strip(), notes=self.notes.toPlainText())
        return result

    def accept(self):
        if not self.source.text().strip() or (self.kind == 'glossary' and not self.target.text().strip()):
            self.error.setText(self.tr('Enter a source term and preferred translation.') if self.kind == 'glossary'
                               else self.tr('Enter a character name.'))
            return
        super().accept()


class _SuggestionDialog(_ContextDialog):
    """Shared disclosure, explicit generation and review-before-accept controls."""
    generate_requested = QtCore.Signal()

    def __init__(self, request, source, target, parent=None):
        super().__init__(parent)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.resize(620, 510)
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel()
        info.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        info.setWordWrap(True)
        route = (QtCore.QCoreApplication.translate('SummarySuggestionDialog', 'Generate locally with {provider}.') if request.local else
                 QtCore.QCoreApplication.translate('SummarySuggestionDialog', 'Generate sends the displayed material to {destination} using {provider}.'))
        info.setText(route.format(provider=request.provider, destination=request.destination))
        layout.addWidget(info)
        self.inputs_toggle = QtWidgets.QToolButton()
        self.inputs_toggle.setCheckable(True)
        self.inputs_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.inputs_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.inputs_toggle.setText(QtCore.QCoreApplication.translate('SummarySuggestionDialog', 'Approved translations ({count}) · {source} → {target}').format(
            count=len(request.examples), source=source, target=target))
        layout.addWidget(self.inputs_toggle)
        self.inputs_edit = QtWidgets.QPlainTextEdit()
        self.inputs_edit.setReadOnly(True)
        self.inputs_edit.setAccessibleName(self.inputs_toggle.text())
        self.inputs_edit.setMaximumHeight(140)
        self.inputs_edit.setPlainText('\n\n'.join(f'{example.source}\n{example.target}' for example in request.examples))
        self.inputs_edit.hide()
        self.inputs_toggle.toggled.connect(self._toggle_inputs)
        layout.addWidget(self.inputs_edit)
        self.content = QtWidgets.QVBoxLayout()
        layout.addLayout(self.content, 1)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        self.generate_button = QtWidgets.QPushButton(QtCore.QCoreApplication.translate('SummarySuggestionDialog', 'Generate'))
        self.generate_button.clicked.connect(self.generate_requested.emit)
        self.use_button = QtWidgets.QPushButton()
        self.use_button.clicked.connect(self.accept)
        self.use_button.setEnabled(False)
        row.addWidget(self.generate_button)
        row.addWidget(self.use_button)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        layout.addLayout(row)

    def _toggle_inputs(self, visible):
        self.inputs_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow)
        self.inputs_edit.setVisible(visible)

    def start_generation(self):
        self.generate_button.setEnabled(False)
        self.status_label.setText(QtCore.QCoreApplication.translate('SummarySuggestionDialog', 'Generating suggestion…'))

    def show_error(self, text):
        self.generate_button.setEnabled(False)
        self.suggestion_edit.setEnabled(False)
        self.use_button.setEnabled(False)
        self.status_label.setText(text)


class SummarySuggestionDialog(_SuggestionDialog):
    def __init__(self, request, source, target, parent=None):
        super().__init__(request, source, target, parent)
        self.setWindowTitle(self.tr('Suggest story summary'))
        self.use_button.setText(self.tr('Use suggestion'))
        self.content.addWidget(QtWidgets.QLabel(self.tr('Current summary')))
        self.current_edit = QtWidgets.QPlainTextEdit()
        self.current_edit.setAccessibleName(self.tr('Current summary'))
        self.current_edit.setPlainText(request.summary)
        self.current_edit.setReadOnly(True)
        self.current_edit.setMaximumHeight(100)
        self.content.addWidget(self.current_edit)
        self.content.addWidget(QtWidgets.QLabel(self.tr('Suggested summary')))
        self.suggestion_edit = QtWidgets.QPlainTextEdit()
        self.suggestion_edit.setAccessibleName(self.tr('Suggested summary'))
        self.suggestion_edit.setEnabled(False)
        self.content.addWidget(self.suggestion_edit, 1)
        self.suggestion_edit.textChanged.connect(self._update_use)

    def _update_use(self):
        self.use_button.setEnabled(self.suggestion_edit.isEnabled() and bool(self.suggestion_edit.toPlainText().strip()))

    def show_result(self, text):
        self.suggestion_edit.setEnabled(True)
        self.suggestion_edit.setPlainText(text)
        self.status_label.setText('')
        self.suggestion_edit.setFocus()


class GlossarySuggestionDialog(_SuggestionDialog):
    def __init__(self, request, source, target, glossary, languages, parent=None):
        super().__init__(request, source, target, parent)
        self.setWindowTitle(self.tr('Suggest glossary terms'))
        self.use_button.setText(self.tr('Add selected terms'))
        self.examples = request.examples
        self.existing_glossary = glossary
        self.languages = languages
        self.suggestion_edit = QtWidgets.QTableWidget(0, 3)
        self.suggestion_edit.setAccessibleName(self.windowTitle())
        self.suggestion_edit.setHorizontalHeaderLabels([
            QtCore.QCoreApplication.translate('TranslationContextDialog', 'Add'),
            QtCore.QCoreApplication.translate('TranslationContextDialog', 'Source term'),
            QtCore.QCoreApplication.translate('TranslationContextDialog', 'Preferred translation')])
        self.suggestion_edit.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.suggestion_edit.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.suggestion_edit.verticalHeader().hide()
        self.suggestion_edit.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.suggestion_edit.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.suggestion_edit.setEnabled(False)
        self.content.addWidget(self.suggestion_edit, 1)
        self.evidence_label = QtWidgets.QLabel(self.tr('Approved example'))
        self.evidence_edit = QtWidgets.QPlainTextEdit()
        self.evidence_edit.setAccessibleName(self.evidence_label.text())
        self.evidence_edit.setReadOnly(True)
        self.evidence_edit.setMaximumHeight(95)
        self.evidence_label.hide()
        self.evidence_edit.hide()
        self.content.addWidget(self.evidence_label)
        self.content.addWidget(self.evidence_edit)
        self.suggestion_edit.currentCellChanged.connect(self._show_evidence)
        self.suggestion_edit.itemChanged.connect(self._update_use)

    def show_result(self, candidates):
        table = self.suggestion_edit
        with QtCore.QSignalBlocker(table):
            table.setRowCount(len(candidates))
            for row, candidate in enumerate(candidates):
                check = QtWidgets.QTableWidgetItem()
                check.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable |
                               QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                check.setCheckState(QtCore.Qt.CheckState.Unchecked)
                check.setData(QtCore.Qt.ItemDataRole.UserRole, candidate.example_index)
                table.setItem(row, 0, check)
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(candidate.source))
                table.setItem(row, 2, QtWidgets.QTableWidgetItem(candidate.target))
        table.setEnabled(bool(candidates))
        if candidates:
            table.setCurrentCell(0, 1)
            self._update_use()
        else:
            self.status_label.setText(self.tr('No new terms found in the approved translations.'))

    def _show_evidence(self, row, *_):
        check = self.suggestion_edit.item(row, 0) if row >= 0 else None
        if check is not None:
            example = self.examples[check.data(QtCore.Qt.ItemDataRole.UserRole)]
            self.evidence_edit.setPlainText(f'{example.source}\n{example.target}')
        self.evidence_label.setVisible(check is not None)
        self.evidence_edit.setVisible(check is not None)

    def selected_entries(self):
        table = self.suggestion_edit
        return [dict(source=table.item(row, 1).text().strip(), target=table.item(row, 2).text().strip(),
                     source_lang=self.languages[0], target_lang=self.languages[1])
                for row in range(table.rowCount()) if table.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked]

    def accept(self):
        # Clicking Add or activating it by keyboard can arrive before the
        # delegate's focus-out commit. Validate the wording the user sees.
        table = self.suggestion_edit
        if table.state() == QtWidgets.QAbstractItemView.State.EditingState:
            for editor in table.findChildren(QtWidgets.QLineEdit):
                if editor.isVisibleTo(table):
                    table.commitData(editor)
                    table.closeEditor(editor, QtWidgets.QAbstractItemDelegate.EndEditHint.NoHint)
                    break
        self._update_use()
        if self.use_button.isEnabled():
            super().accept()

    def _update_use(self, *_):
        entries = self.selected_entries()
        known = list(self.existing_glossary)
        message = ''
        for entry in entries:
            if not entry['source'] or not entry['target']:
                message = self.tr('Enter a source term and preferred translation for each selected row.')
                break
            if glossary_source_conflicts(entry['source'], known, *self.languages):
                message = self.tr('A selected source term already exists in this language pair or appears twice in your selection.')
                break
            known.append(entry)
        self.use_button.setEnabled(self.suggestion_edit.isEnabled() and bool(entries) and not message)
        self.status_label.setText(message if entries else self.tr('Select terms to add to the draft glossary.'))


class TranslationContextDialog(_ContextDialog):
    apply_requested = QtCore.Signal(object)
    preview_requested = QtCore.Signal(object, str)
    add_selected_requested = QtCore.Signal()
    undo_requested = QtCore.Signal()
    redo_requested = QtCore.Signal()
    suggest_summary_requested = QtCore.Signal()
    suggest_glossary_requested = QtCore.Signal()

    def __init__(self, profile, languages=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Translation context'))
        self.setModal(False)
        self.resize(840, 700)
        self.languages = languages or []
        self._loading = False
        self._busy = False
        self._selection = None
        self._can_undo = self._can_redo = False
        self._dirty = False
        layout = QtWidgets.QVBoxLayout(self)
        self.support_label = QtWidgets.QLabel()
        self.support_label.setWordWrap(True)
        layout.addWidget(self.support_label)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.tables, self.entry_buttons = {}, {}
        for kind, label in (('glossary', self.tr('Glossary')), ('characters', self.tr('Characters'))):
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            table = QtWidgets.QTableWidget(0, 4)
            table.setHorizontalHeaderLabels([
                self.tr('Source term') if kind == 'glossary' else self.tr('Name'),
                self.tr('Preferred translation'), self.tr('Language pair'), self.tr('Status')])
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            table.verticalHeader().hide()
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
            table.setAccessibleName(label)
            table.itemDoubleClicked.connect(lambda _item, k=kind: self.edit_entry(k))
            page_layout.addWidget(table, 1)
            row = QtWidgets.QHBoxLayout()
            buttons = []
            for title, callback in ((self.tr('Add'), lambda _=False, k=kind: self.edit_entry(k, new=True)),
                                    (self.tr('Edit'), lambda _=False, k=kind: self.edit_entry(k)),
                                    (self.tr('Remove'), lambda _=False, k=kind: self.remove_entry(k))):
                button = QtWidgets.QPushButton(title)
                button.clicked.connect(callback)
                row.addWidget(button)
                buttons.append(button)
            row.addStretch()
            if kind == 'glossary':
                self.add_selected_button = QtWidgets.QPushButton(self.tr('Add selected source/translation'))
                self.add_selected_button.clicked.connect(self.add_selected_requested.emit)
                row.addWidget(self.add_selected_button)
                self.suggest_glossary_button = QtWidgets.QPushButton(self.tr('Suggest glossary terms…'))
                self.suggest_glossary_button.clicked.connect(self.suggest_glossary_requested.emit)
                row.addWidget(self.suggest_glossary_button)
            page_layout.addLayout(row)
            self.tables[kind], self.entry_buttons[kind] = table, buttons
            table.itemSelectionChanged.connect(self._update_actions)
            if kind == 'characters':
                self.selection_label = QtWidgets.QLabel()
                self.selection_label.setWordWrap(True)
                page_layout.addWidget(self.selection_label)
                assignment = QtWidgets.QHBoxLayout()
                self.speaker_combo = QtWidgets.QComboBox()
                self.speaker_combo.setAccessibleName(self.tr('Speaker'))
                assignment.addWidget(self.speaker_combo, 1)
                self.assign_button = QtWidgets.QPushButton(self.tr('Assign selected region'))
                self.assign_button.clicked.connect(self.assign_selected)
                assignment.addWidget(self.assign_button)
                page_layout.addLayout(assignment)
            self.tabs.addTab(page, label)
        notes_page = QtWidgets.QWidget()
        notes_layout = QtWidgets.QFormLayout(notes_page)
        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setAccessibleName(self.tr('Project notes'))
        self.notes_edit.setPlaceholderText(self.tr('Setting, tone, terminology guidance, and other context for this work.'))
        self.summary_edit = QtWidgets.QPlainTextEdit()
        self.summary_edit.setAccessibleName(self.tr('Story summary'))
        notes_layout.addRow(self.tr('Project notes'), self.notes_edit)
        notes_layout.addRow(self.tr('Story summary'), self.summary_edit)
        self.suggest_summary_button = QtWidgets.QPushButton(self.tr('Suggest story summary…'))
        self.suggest_summary_button.clicked.connect(self.suggest_summary_requested.emit)
        notes_layout.addRow('', self.suggest_summary_button)
        self.history_checkbox = QtWidgets.QCheckBox(self.tr('Include approved translations from earlier pages'))
        notes_layout.addRow(self.history_checkbox)
        self.budget_spin = QtWidgets.QSpinBox()
        self.budget_spin.setRange(128, 32768)
        self.budget_spin.setSingleStep(128)
        self.budget_spin.setAccessibleName(self.tr('Context token limit'))
        notes_layout.addRow(self.tr('Context token limit'), self.budget_spin)
        budget_note = QtWidgets.QLabel(self.tr('Uses a conservative estimate. Required context that exceeds the available budget must be shortened.'))
        budget_note.setWordWrap(True)
        notes_layout.addRow(budget_note)
        self.tabs.addTab(notes_page, self.tr('Notes'))
        self.notes_edit.textChanged.connect(self._mark_changed)
        self.summary_edit.textChanged.connect(self._mark_changed)
        self.history_checkbox.toggled.connect(self._mark_changed)
        self.budget_spin.valueChanged.connect(self._mark_changed)

        preview_row = QtWidgets.QHBoxLayout()
        self.preview_toggle = QtWidgets.QToolButton()
        self.preview_toggle.setText(self.tr('Request preview'))
        self.preview_toggle.setCheckable(True)
        self.preview_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.preview_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.preview_toggle.toggled.connect(self._show_preview)
        preview_row.addWidget(self.preview_toggle)
        self.preview_scope = QtWidgets.QComboBox()
        self.preview_scope.addItem(self.tr('Current page'), 'page')
        self.preview_scope.addItem(self.tr('Selected region'), 'selected')
        self.preview_scope.setAccessibleName(self.tr('Preview scope'))
        preview_row.addWidget(self.preview_scope)
        self.preview_button = QtWidgets.QPushButton(self.tr('Build preview'))
        self.preview_button.clicked.connect(lambda: self.preview_requested.emit(self.profile(), self.preview_scope.currentData()))
        preview_row.addWidget(self.preview_button)
        preview_row.addStretch()
        layout.addLayout(preview_row)
        self.preview_text = QtWidgets.QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setAccessibleName(self.tr('Request preview'))
        self.preview_text.setMaximumHeight(230)
        self.preview_text.hide()
        self.preview_scope.currentIndexChanged.connect(lambda: self.set_preview(''))
        layout.addWidget(self.preview_text)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        bottom = QtWidgets.QHBoxLayout()
        self.undo_button = QtWidgets.QPushButton(self.tr('Undo context'))
        self.redo_button = QtWidgets.QPushButton(self.tr('Redo context'))
        self.reset_button = QtWidgets.QPushButton(self.tr('Revert draft'))
        self.apply_button = QtWidgets.QPushButton(self.tr('Apply'))
        self.close_button = QtWidgets.QPushButton(self.tr('Close'))
        self.undo_button.clicked.connect(self.undo_requested.emit)
        self.redo_button.clicked.connect(self.redo_requested.emit)
        self.reset_button.clicked.connect(lambda: self.set_profile(self._saved))
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self.profile()))
        self.close_button.clicked.connect(self.close)
        for button in (self.undo_button, self.redo_button, self.reset_button):
            bottom.addWidget(button)
        bottom.addStretch()
        bottom.addWidget(self.apply_button)
        bottom.addWidget(self.close_button)
        layout.addLayout(bottom)
        self.set_profile(profile)
        self.set_supported(True)

    def profile(self):
        result = copy.deepcopy(self._draft)
        result.update(notes=self.notes_edit.toPlainText(), summary=self.summary_edit.toPlainText(),
                      include_approved_history=self.history_checkbox.isChecked(),
                      context_token_limit=self.budget_spin.value())
        return normalize_translation_profile(result)

    def set_profile(self, profile):
        self._loading = True
        self._saved = normalize_translation_profile(profile)
        self._draft = copy.deepcopy(self._saved)
        self.notes_edit.setPlainText(self._draft['notes'])
        self.summary_edit.setPlainText(self._draft['summary'])
        self.history_checkbox.setChecked(self._draft.get('include_approved_history') is True)
        self.budget_spin.setValue(self._draft['context_token_limit'])
        self._fill_tables()
        self._loading = False
        self._dirty = False
        self.status_label.setText(self.tr('Changes apply to this project. The existing LLM context setting also remains in use.'))
        self.set_preview('')
        self._update_actions()

    def _fill_tables(self):
        for kind, table in self.tables.items():
            with QtCore.QSignalBlocker(table):
                table.setRowCount(len(self._draft[kind]))
                for row, entry in enumerate(self._draft[kind]):
                    names = dict(self.languages)
                    source = names.get(entry.get('source_lang'), entry.get('source_lang')) or self.tr('Any language')
                    target = names.get(entry.get('target_lang'), entry.get('target_lang')) or self.tr('Any language')
                    language_pair = '{} → {}'.format(source, target)
                    status = self.tr('Disabled') if not entry.get('enabled') else (
                        self.tr('Approved') if entry.get('approved') else self.tr('Draft'))
                    values = (entry['source' if kind == 'glossary' else 'name'], entry.get('target', ''), language_pair, status)
                    for column, value in enumerate(values):
                        item = QtWidgets.QTableWidgetItem(value)
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, entry['id'])
                        item.setToolTip(value)
                        table.setItem(row, column, item)
        self._fill_speakers()

    def _fill_speakers(self):
        with QtCore.QSignalBlocker(self.speaker_combo):
            self.speaker_combo.clear()
            self.speaker_combo.addItem(self.tr('No speaker'), '')
            for entry in self._draft['characters']:
                self.speaker_combo.addItem(entry['name'], entry['id'])
            chosen = ''
            if self._selection:
                chosen = next((a['character_id'] for a in self._draft['speakers']
                               if a['page_id'] == self._selection['page_id'] and a['region_id'] == self._selection['region_id']), '')
            self.speaker_combo.setCurrentIndex(max(0, self.speaker_combo.findData(chosen)))

    def edit_entry(self, kind, new=False, initial=None):
        if self._busy:
            return
        row = self.tables[kind].currentRow()
        if not new and row < 0:
            return
        entry = initial or ({} if new else self._draft[kind][row])
        editor = ContextEntryDialog('glossary' if kind == 'glossary' else 'characters', entry, self.languages, self)
        accepted = editor.exec() == QtWidgets.QDialog.DialogCode.Accepted
        value = editor.value() if accepted else None
        editor.deleteLater()
        if not accepted:
            return
        if new:
            self._draft[kind].append(value)
        else:
            self._draft[kind][row] = value
        self._fill_tables()
        self._mark_changed()

    def remove_entry(self, kind):
        row = self.tables[kind].currentRow()
        if self._busy or row < 0:
            return
        removed = self._draft[kind].pop(row)
        if kind == 'characters':
            self._draft['speakers'] = [a for a in self._draft['speakers'] if a['character_id'] != removed['id']]
        self._fill_tables()
        self._mark_changed()

    def add_suggested_terms(self, entries):
        # Normalize just the new rows so existing draft entries and IDs survive.
        new = normalize_translation_profile({'glossary': entries})['glossary']
        self._draft['glossary'].extend(new)
        self._fill_tables()
        self._mark_changed()

    def set_selection(self, selection):
        if self._selection != selection:
            self._selection = copy.deepcopy(selection)
            self.set_preview('')
            text = selection.get('text', '') if selection else ''
            self.selection_label.setText(self.tr('Selected region: {text}').format(text=text[:160]) if selection
                                         else self.tr('Select a text region on the page to assign a speaker.'))
            self._fill_speakers()
        self._update_actions()

    def assign_selected(self):
        if self._busy or not self._selection:
            return
        page_id, region_id = self._selection['page_id'], self._selection['region_id']
        self._draft['speakers'] = [a for a in self._draft['speakers']
                                  if (a['page_id'], a['region_id']) != (page_id, region_id)]
        if self.speaker_combo.currentData():
            self._draft['speakers'].append(dict(page_id=page_id, region_id=region_id,
                                               character_id=self.speaker_combo.currentData()))
        self._mark_changed()

    def set_busy(self, busy):
        self._busy = bool(busy)
        self._update_actions()

    def set_undo_state(self, undo, redo):
        self._can_undo, self._can_redo = bool(undo), bool(redo)
        self._update_actions()

    def set_supported(self, supported):
        self._supported = bool(supported)
        self.support_label.setText(self.tr('Approved glossary entries, assigned character notes, and project notes guide LLM translation.')
            if supported else self.tr('This translator does not support structured LLM context. Project entries remain editable for use with an LLM.'))
        self._update_actions()

    def _mark_changed(self, *_):
        if self._loading:
            return
        self._dirty = self.profile() != self._saved
        self.status_label.setText(self.tr('Unapplied changes. Apply to use them for translation.') if self._dirty
                                 else self.tr('No unapplied changes.'))
        self.set_preview('')
        self._update_actions()

    def _update_actions(self):
        if not hasattr(self, 'apply_button'):
            return
        ready = not self._busy
        for kind, buttons in self.entry_buttons.items():
            selected = self.tables[kind].currentRow() >= 0
            buttons[0].setEnabled(ready)
            buttons[1].setEnabled(ready and selected)
            buttons[2].setEnabled(ready and selected)
        self.tabs.setEnabled(ready)
        llm_ready = ready and getattr(self, '_supported', True)
        self.suggest_summary_button.setEnabled(llm_ready)
        self.suggest_glossary_button.setEnabled(llm_ready)
        self.history_checkbox.setEnabled(llm_ready)
        self.add_selected_button.setEnabled(ready and self._selection is not None)
        self.assign_button.setEnabled(ready and self._selection is not None)
        self.apply_button.setEnabled(ready and self._dirty)
        self.reset_button.setEnabled(ready and self._dirty)
        self.undo_button.setEnabled(ready and not self._dirty and self._can_undo)
        self.redo_button.setEnabled(ready and not self._dirty and self._can_redo)
        self.preview_button.setEnabled(ready and getattr(self, '_supported', True))
        self.preview_scope.setEnabled(ready and getattr(self, '_supported', True))

    def _show_preview(self, shown):
        self.preview_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.RightArrow)
        self.preview_text.setVisible(shown)

    def set_preview(self, text):
        self.preview_text.setPlainText(text)
        if text:
            self.preview_toggle.setChecked(True)
