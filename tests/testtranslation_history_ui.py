"""Project history and explicit summary suggestions through the Qt context editor."""
import copy
import json
import time
import unittest
from threading import get_ident
from unittest.mock import Mock, patch

from PySide6 import QtCore, QtWidgets

from app.controllers.task_runner import TaskRunnerController
from app.controllers.translation_context import TranslationContextController
from modules.review.issues import fingerprint_for_stage, mark_reviewed
from modules.translation.context import empty_translation_profile
from modules.translation.llm.local import LocalLLMTranslation
from modules.utils.textblock import TextBlock
from testtranslation_context_ui import ContextWindow


class HistorySummaryUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = ContextWindow()
        self.main.image_files.insert(0, 'earlier.png')
        self.main.curr_img_idx = 1
        self.main.blk_list = [self.main.block]
        languages = {'source': 'Vietnamese', 'target': 'Brazilian Portuguese'}
        self.earlier = TextBlock(text_bbox=[1, 2, 40, 30], text='Đây là Hà.', translation='Esta é Hà.')
        self.earlier.region_metadata = mark_reviewed(self.earlier.region_metadata, 'translation',
            fingerprint_for_stage('translation', self.earlier, languages=languages))
        self.unreviewed = TextBlock(text_bbox=[2, 5, 30, 50], text='Private work in progress', translation='Tentative words')
        self.main.image_states = {
            'earlier.png': dict(page_id='earlier', blk_list=[self.earlier, self.unreviewed],
                                source_lang='Vietnamese', target_lang='Brazilian Portuguese'),
            'page.png': dict(page_id='p1', blk_list=[self.main.block], source_lang='Vietnamese',
                             target_lang='Brazilian Portuguese')}
        self.engine = LocalLLMTranslation()
        self.engine.runtime_mode = 'external'
        self.engine.api_base_url = 'http://127.0.0.1:8080/v1'
        self.engine.model = 'test-local'
        self.engine.complete_text_once = Mock(return_value=json.dumps({'summary': 'Hà chegou à cidade.'}))
        factory = patch('modules.translation.processor.TranslationFactory.create_engine', return_value=self.engine)
        factory.start()
        self.addCleanup(factory.stop)
        self.jobs = []
        self.main.run_threaded = lambda *callbacks: self.jobs.append(callbacks)
        self.ctrl = TranslationContextController(self.main)
        self.ctrl.show_editor()
        self.ctrl.timer.stop()
        self.editor = self.ctrl.dialog

    def tearDown(self):
        self.ctrl.shutdown()
        self.main.close()
        self.main.deleteLater()
        self.app.processEvents()

    def open_form(self):
        self.editor.suggest_summary_button.click()
        form = self.ctrl.suggestion_dialog
        self.assertIsNotNone(form, self.editor.status_label.text())
        return form

    def finish_job(self):
        run, ready, error, finished = self.jobs.pop(0)
        try:
            result = run()
        except Exception as exc:
            error((type(exc), exc, 'traceback'))
        else:
            ready(result)
        finished()

    def test_unapplied_history_checkbox_alters_preview_then_supports_project_undo(self):
        self.assertFalse(self.editor.history_checkbox.isChecked())
        self.editor.history_checkbox.click()
        self.ctrl.preview_request(self.editor.profile(), 'selected')
        self.assertIn(self.earlier.translation, self.editor.preview_text.toPlainText())
        self.assertNotIn(self.unreviewed.translation, self.editor.preview_text.toPlainText())
        self.assertFalse(self.main.translation_profile['include_approved_history'])
        self.editor.apply_button.click()
        self.assertTrue(self.main.translation_profile['include_approved_history'])
        self.editor.undo_button.click()
        self.assertFalse(self.main.translation_profile['include_approved_history'])
        self.assertFalse(self.editor.history_checkbox.isChecked())

    def test_summary_displays_approved_inputs_and_requires_generate_use_apply(self):
        self.editor.notes_edit.setPlainText('My unsaved note')
        self.editor.summary_edit.setPlainText('Previous approved events')
        original = copy.deepcopy(self.main.translation_profile)
        form = self.open_form()
        self.assertIn('locally', form.findChildren(QtWidgets.QLabel)[0].text())
        self.assertIn('Brazilian Portuguese', form.inputs_toggle.text())
        self.assertIn('(1)', form.inputs_toggle.text())
        self.assertIn(self.earlier.translation, form.inputs_edit.toPlainText())
        self.assertNotIn(self.unreviewed.text, form.inputs_edit.toPlainText())
        self.assertEqual(form.current_edit.toPlainText(), 'Previous approved events')
        self.assertFalse(form.use_button.isEnabled())
        self.engine.complete_text_once.assert_not_called()
        form.generate_button.click()
        self.finish_job()
        self.assertEqual(form.suggestion_edit.toPlainText(), 'Hà chegou à cidade.')
        self.assertEqual(self.editor.summary_edit.toPlainText(), 'Previous approved events')
        form.suggestion_edit.setPlainText('Hà chegou de trem.')
        form.use_button.click()
        self.assertIsNone(self.ctrl.suggestion_dialog)
        self.assertEqual(self.editor.summary_edit.toPlainText(), 'Hà chegou de trem.')
        self.assertEqual(self.editor.notes_edit.toPlainText(), 'My unsaved note')
        self.assertEqual(self.main.translation_profile, original)
        self.editor.apply_button.click()
        self.assertEqual(self.main.translation_profile['summary'], 'Hà chegou de trem.')
        self.editor.undo_button.click()
        self.assertEqual(self.main.translation_profile, original)
        self.assertEqual(self.main.page_stack.index(), 1)

    def test_remote_endpoint_names_destination_and_cancel_sends_nothing(self):
        self.engine.api_base_url = 'https://remote.example/v1'
        form = self.open_form()
        self.assertIn('https://remote.example', form.findChildren(QtWidgets.QLabel)[0].text())
        form.reject()
        self.engine.complete_text_once.assert_not_called()
        self.assertEqual(self.jobs, [])

    def test_approval_change_after_consent_discards_response(self):
        form = self.open_form()
        form.generate_button.click()
        self.earlier.translation = 'New unapproved words'
        self.finish_job()
        self.assertFalse(form.use_button.isEnabled())
        self.assertIn('inputs changed', form.status_label.text())
        self.assertEqual(self.editor.summary_edit.toPlainText(), '')

    def test_provider_changed_before_generate_requires_fresh_disclosure(self):
        form = self.open_form()
        self.engine.api_base_url = 'https://remote.example/v1'
        form.generate_button.click()
        self.assertFalse(form.generate_button.isEnabled())
        self.assertIn('inputs changed', form.status_label.text())
        self.assertEqual(self.jobs, [])

    def test_already_running_reply_cannot_enter_a_reopened_editor(self):
        form = self.open_form()
        form.generate_button.click()
        run, ready, _, finished = self.jobs.pop()
        reply = run()
        self.editor.close()
        self.ctrl.show_editor()
        replacement = self.open_form()
        ready(reply)
        finished()
        self.assertIs(self.ctrl.suggestion_dialog, replacement)
        self.assertFalse(replacement.use_button.isEnabled())
        self.assertEqual(replacement.suggestion_edit.toPlainText(), '')
        self.assertEqual(self.editor.summary_edit.toPlainText(), '')

    def test_language_change_after_result_prevents_using_draft(self):
        form = self.open_form()
        form.generate_button.click()
        self.finish_job()
        self.main.t_combo.addItem('English')
        self.main.t_combo.setCurrentText('English')
        form.use_button.click()
        self.assertEqual(self.editor.summary_edit.toPlainText(), '')
        self.assertIn('inputs changed', self.editor.status_label.text())

    def test_discard_or_project_reset_ignores_late_response(self):
        for action in ('cancel', 'reset'):
            with self.subTest(action=action):
                form = self.open_form()
                form.generate_button.click()
                if action == 'cancel':
                    form.reject()
                else:
                    self.main.translation_profile = empty_translation_profile()
                    self.ctrl.reset_for_project()
                self.finish_job()
                self.assertIsNone(self.ctrl.suggestion_dialog)
                self.engine.complete_text_once.assert_not_called()
                self.assertEqual(self.main.translation_profile['summary'], '')
                self.assertFalse(self.ctrl.undo_stack.canUndo())

    def test_errors_are_safe_and_missing_approvals_do_not_call_model(self):
        self.engine.complete_text_once.side_effect = ValueError('API secret')
        form = self.open_form()
        form.generate_button.click()
        self.finish_job()
        self.assertNotIn('secret', form.status_label.text())
        self.assertIn('failed', form.status_label.text())
        self.assertFalse(form.use_button.isEnabled())
        form.reject()
        self.earlier.translation += ' (changed)'
        self.editor.suggest_summary_button.click()
        self.assertIsNone(self.ctrl.suggestion_dialog)
        self.assertIn('mark a translation reviewed', self.editor.status_label.text())
        self.assertEqual(self.engine.complete_text_once.call_count, 1)

    def test_queued_request_runs_off_the_gui_thread(self):
        thread_ids = []
        self.main.threadpool = QtCore.QThreadPool(self.main)
        self.main.task_runner_ctrl = TaskRunnerController(self.main)
        self.main.run_threaded = self.main.task_runner_ctrl.run_threaded
        def complete(*_):
            thread_ids.append(get_ident())
            return '{"summary":"Draft"}'
        self.engine.complete_text_once.side_effect = complete
        form = self.open_form()
        form.generate_button.click()
        stop = time.monotonic() + 3
        while not form.use_button.isEnabled() and time.monotonic() < stop:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(form.use_button.isEnabled(), form.status_label.text())
        self.assertEqual(len(thread_ids), 1)
        self.assertNotEqual(thread_ids[0], get_ident())
