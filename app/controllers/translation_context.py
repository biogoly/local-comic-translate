"""Project-only context commands and an optional, modeless context editor."""
from __future__ import annotations

import copy
from threading import Event

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.dialogs.translation_context import GlossarySuggestionDialog, SummarySuggestionDialog, TranslationContextDialog
from modules.translation.context import normalize_translation_profile
from modules.translation.history import capture_approved_history
from modules.translation.glossary_suggestion import prepare_glossary_suggestion
from modules.translation.processor import Translator
from modules.translation.summary_suggestion import SummarySuggestionError, prepare_summary_suggestion


class TranslationProfileCommand(QtGui.QUndoCommand):
    def __init__(self, controller, before, after):
        super().__init__(QtCore.QCoreApplication.translate('TranslationContextController', 'Change translation context'))
        self.controller = controller
        self.before, self.after = copy.deepcopy(before), copy.deepcopy(after)

    def undo(self):
        self.controller._replace(self.before)

    def redo(self):
        self.controller._replace(self.after)


class TranslationContextController(QtCore.QObject):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.dialog = None
        self.suggestion_dialog = None
        self.undo_stack = QtGui.QUndoStack(self)
        self.undo_stack.indexChanged.connect(self._update_state)
        main.translation_profile = normalize_translation_profile(getattr(main, 'translation_profile', None))
        main.translation_context_button.clicked.connect(self.show_editor)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self._update_state)

    def busy(self):
        return bool(getattr(self.main, '_batch_active', False)
                    or getattr(getattr(self.main, 'task_runner_ctrl', None), 'is_processing_queue', False)
                    or getattr(getattr(self.main, 'review_ctrl', None), '_preview_running', False))

    def show_editor(self):
        if self.dialog is None:
            languages = [(value, label) for label, value in self.main.lang_mapping.items()]
            self.dialog = TranslationContextDialog(self.main.translation_profile, languages, self.main)
            self.dialog.apply_requested.connect(self.apply_profile)
            self.dialog.preview_requested.connect(self.preview_request)
            self.dialog.add_selected_requested.connect(self.add_selected)
            self.dialog.undo_requested.connect(self.undo)
            self.dialog.redo_requested.connect(self.redo)
            self.dialog.suggest_summary_requested.connect(self.suggest_summary)
            self.dialog.suggest_glossary_requested.connect(self.suggest_glossary)
            self.dialog.finished.connect(self.timer.stop)
            self.dialog.finished.connect(self._close_suggestion)
        elif not self.dialog.isVisible() and self.dialog._saved != self.main.translation_profile:
            self.dialog.set_profile(self.main.translation_profile)
        self._update_state()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        self.timer.start()

    def reset_for_project(self):
        """Persistence owns data replacement; this clears only editor state."""
        self.timer.stop()
        self._close_suggestion()
        self.undo_stack.clear()
        if self.dialog is not None:
            self.dialog.close()
            self.dialog.set_profile(getattr(self.main, 'translation_profile', None))

    def shutdown(self):
        self.timer.stop()
        self._close_suggestion()
        if self.dialog is not None:
            self.dialog.close()

    def _selection(self):
        review = getattr(self.main, 'review_ctrl', None)
        selected = review.selected_region() if review is not None else None
        if not selected:
            return None
        refs = review.region_refs(*selected)
        if not refs:
            return None
        block = refs[-1]
        return dict(page_id=selected[0], region_id=selected[1], text=block.text or '',
                    translation=block.translation or '')

    def _languages(self):
        mapping = self.main.lang_mapping
        return mapping.get(self.main.s_combo.currentText(), self.main.s_combo.currentText()), \
            mapping.get(self.main.t_combo.currentText(), self.main.t_combo.currentText())

    def _supports_context(self):
        key = self.main.settings_page.get_tool_selection('translator')
        traditional = {'Google Translate', 'Microsoft Translator', 'DeepL', 'Yandex'}
        traditional.update(self.main.settings_page.ui.tr(name) for name in tuple(traditional))
        return key not in traditional

    def _update_state(self, *_):
        if self.dialog is None:
            return
        self.dialog.set_busy(self.busy())
        self.dialog.set_undo_state(self.undo_stack.canUndo(), self.undo_stack.canRedo())
        self.dialog.set_selection(self._selection())
        self.dialog.set_supported(self._supports_context())

    def apply_profile(self, value):
        if self.busy():
            return
        before = normalize_translation_profile(self.main.translation_profile)
        after = normalize_translation_profile(value)
        if before != after:
            self.undo_stack.push(TranslationProfileCommand(self, before, after))
        elif self.dialog is not None:
            self.dialog.set_profile(after)

    def _replace(self, profile):
        self.main.translation_profile = copy.deepcopy(profile)
        self.main._translation_profile_revision = getattr(self.main, '_translation_profile_revision', 0) + 1
        self.main.mark_project_dirty()
        if self.dialog is not None:
            self.dialog.set_profile(profile)
        self._update_state()

    def undo(self):
        if not self.busy():
            self.undo_stack.undo()

    def redo(self):
        if not self.busy():
            self.undo_stack.redo()

    def add_selected(self):
        selection = self._selection()
        if selection is None or self.busy() or self.dialog is None:
            return
        source_lang, target_lang = self._languages()
        entry = dict(source=selection['text'], target=selection['translation'],
                     source_lang='' if source_lang == 'Auto' else source_lang, target_lang=target_lang)
        self.dialog.tabs.setCurrentIndex(0)
        self.dialog.edit_entry('glossary', new=True, initial=entry)

    def _close_suggestion(self, *_):
        if self.suggestion_dialog is not None:
            self.suggestion_dialog.reject()

    def _prepare_suggestion(self, kind):
        source, target = self._languages()
        if not self._supports_context():
            raise SummarySuggestionError('unsupported')
        examples = capture_approved_history(self.main).for_pair(source, target)
        if not examples:
            raise SummarySuggestionError('no_examples')
        profile = self.dialog.profile()
        translator = Translator(self.main, source, target)
        if kind == 'glossary':
            return prepare_glossary_suggestion(translator, profile['glossary'], examples,
                                               context_token_limit=profile['context_token_limit'])
        return prepare_summary_suggestion(translator, profile['summary'], examples,
                                          context_token_limit=profile['context_token_limit'])

    def _suggestion_error(self, exc, kind):
        messages = {
            'unsupported': self.tr('Choose an LLM translator to suggest a summary.'),
            'no_examples': self.tr('In Review, mark a translation reviewed for this source and target language pair.'),
            'budget': self.tr('Increase the context token limit or shorten the story summary to fit approved translations.'),
            'credentials': self.tr('Check the selected translator’s connection settings.'),
            'response': self.tr('The model returned an unusable summary.'),
        }
        if kind == 'glossary':
            messages.update(unsupported=self.tr('Choose an LLM translator to suggest glossary terms.'),
                            budget=self.tr('Increase the context token limit to fit approved translations.'),
                            response=self.tr('The model returned unusable glossary suggestions.'))
            default = self.tr('The glossary request failed. Check the translator settings and try again.')
        else:
            default = self.tr('The summary request failed. Check the translator settings and try again.')
        return messages.get(getattr(exc, 'code', ''), default)

    def _suggestion_scope(self):
        index = self.main.curr_img_idx
        path = self.main.image_files[index] if 0 <= index < len(self.main.image_files) else None
        return (getattr(self.main, '_translation_project_epoch', 0), getattr(self.main, '_translation_cancel_epoch', 0),
                getattr(self.main, '_translation_profile_revision', 0), path, self._languages())

    def suggest_summary(self):
        self._open_suggestion('summary')

    def suggest_glossary(self):
        self._open_suggestion('glossary')

    def _open_suggestion(self, kind):
        if self.busy() or self.dialog is None:
            return
        if self.suggestion_dialog is not None:
            self.suggestion_dialog.raise_()
            return
        try:
            request = self._prepare_suggestion(kind)
        except Exception as exc:
            self.dialog.status_label.setText(self._suggestion_error(exc, kind))
            return

        source, target = self.main.s_combo.currentText(), self.main.t_combo.currentText()
        if kind == 'glossary':
            form = GlossarySuggestionDialog(request, source, target, self.dialog.profile()['glossary'],
                                            self._languages(), self.dialog)
            stale = self.tr('The project or glossary inputs changed. Open the suggestion again.')
        else:
            form = SummarySuggestionDialog(request, source, target, self.dialog)
            stale = self.tr('The project or summary inputs changed. Open the suggestion again.')
        self.suggestion_dialog = form
        discarded = Event()
        scope = self._suggestion_scope()

        def current():
            if self.suggestion_dialog is not form or self._suggestion_scope() != scope:
                return False
            try:
                return request.fingerprint == self._prepare_suggestion(kind).fingerprint
            except Exception:
                return False

        def ready(result):
            if self.suggestion_dialog is not form:
                return
            if current():
                form.show_result(result)
            else:
                form.show_error(stale)

        def error(failure):
            if self.suggestion_dialog is form:
                form.show_error(self._suggestion_error(failure[1], kind))

        def finished():
            # A task canceled by the main window can finish without a result.
            if self.suggestion_dialog is form and self._suggestion_scope() != scope:
                form.show_error(stale)

        def generate():
            if not form.generate_button.isEnabled() or self.busy() or not current():
                form.show_error(stale)
                return
            form.start_generation()
            def run():
                if not discarded.is_set():
                    return request.run()
            try:
                self.main.run_threaded(run, ready, error, finished)
            except Exception as exc:
                form.show_error(self._suggestion_error(exc, kind))

        def closed(result):
            discarded.set()
            if self.suggestion_dialog is form:
                if result == QtWidgets.QDialog.DialogCode.Accepted and form.use_button.isEnabled():
                    if not current():
                        self.dialog.status_label.setText(stale)
                    elif kind == 'glossary':
                        self.dialog.add_suggested_terms(form.selected_entries())
                    else:
                        cursor = self.dialog.summary_edit.textCursor()
                        cursor.select(QtGui.QTextCursor.SelectionType.Document)
                        cursor.insertText(form.suggestion_edit.toPlainText().strip())
                        self.dialog.summary_edit.setTextCursor(cursor)
                self.suggestion_dialog = None
                self._update_state()
            form.deleteLater()

        form.generate_requested.connect(generate)
        form.finished.connect(closed)
        form.open()

    def preview_request(self, profile, scope):
        if self.dialog is None or self.busy():
            return
        if not self._supports_context():
            self.dialog.set_supported(False)
            return
        try:
            review = self.main.review_ctrl
            snapshot = review.snapshot()
            selected = review.selected_region()
            if scope == 'selected':
                if selected not in snapshot:
                    self.dialog.set_preview(self.tr('Select a text region to preview its request.'))
                    return
                records = [snapshot[selected]]
            else:
                index = self.main.curr_img_idx
                path = self.main.image_files[index] if 0 <= index < len(self.main.image_files) else None
                records = [snap for snap in snapshot.values() if snap['path'] == path]
                from pipeline.translation_job import eligible_translation_blocks
                eligible_ids = {block.region_id for block in eligible_translation_blocks([snap['block'] for snap in records])}
                records = [snap for snap in records if snap['block'].region_id in eligible_ids]
            if not records:
                self.dialog.set_preview(self.tr('The current page has no text regions to preview.'))
                return
            blocks = [snap['block'].deep_copy() for snap in records]
            if self.main.image_viewer.webtoon_mode:
                from modules.rendering.auto_bubble_fit import load_original_for_fitting
                image = load_original_for_fitting(records[0]['path'])
            else:
                image = self.main.image_viewer.get_image_array(include_patches=False)
            source_lang, target_lang = self._languages()
            translator = Translator(self.main, source_lang, target_lang)
            history_options = {'history': capture_approved_history(self.main)} if profile.get('include_approved_history') else {}
            request = translator.prepare_request(blocks, image,
                self.main.settings_page.get_llm_settings().get('extra_context', ''),
                profile=profile, page_id=records[0]['page_id'], **history_options)
            self.dialog.set_preview(request.preview)
        except Exception as exc:
            self.dialog.set_preview(self.tr('Cannot build request preview: {error}').format(error=str(exc)))
