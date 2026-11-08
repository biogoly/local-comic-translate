"""Real queued check jobs stay independent of preview Apply and project state."""
import os
from dataclasses import replace
from threading import Event, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets
from shiboken6 import isValid

import testartistic_edit_controller as fixtures
from app.ui.dialogs.artistic_edit_preview import ArtisticEditPreviewDialog
from modules.utils.textblock import TextBlock


class LetteringCheckControllerTests(unittest.TestCase):
    _session = fixtures.ArtisticEditControllerTests._session
    _mark_window_update = fixtures.ArtisticEditControllerTests._mark_window_update

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        fixtures.ArtisticEditControllerTests.setUp(self)
        self.main.settings_page = SimpleNamespace(
            get_tool_selection=lambda _: 'Default', is_gpu_enabled=lambda: False,
            get_credentials=lambda _: {}, ui=SimpleNamespace(tr=lambda text: text))
        self.main.lang_mapping = {'Français': 'French'}
        self.main.t_combo = QtWidgets.QComboBox()
        self.main.t_combo.addItems(['Français', 'English'])
        self.addCleanup(self.main.t_combo.deleteLater)
        self.main.close = Mock()
        self.main.image_states = {self.page_path: {}}
        self.block = TextBlock(text_bbox=[20, 20, 40, 36], text='MY DEAR!',
                               translation=' MA CHÉRIE!\n\n ')
        self.main.blk_list = [self.block]
        self.main.rect_item_ctrl = SimpleNamespace(
            find_corresponding_text_block=lambda *_: self.block)
        rect = self.viewer.add_rectangle(QtCore.QRectF(0, 0, 20, 16), QtCore.QPointF(20, 20))
        self.viewer.select_rectangle(rect)

    def wait_until(self, predicate, timeout=2000):
        timer = QtCore.QElapsedTimer()
        timer.start()
        while not predicate() and timer.elapsed() < timeout:
            self.app.processEvents()
            QtTest.QTest.qWait(2)
        self.assertTrue(predicate(), 'Queued GUI/worker operation did not finish')

    def checked_session(self):
        session = self._session()
        session.expected_lettering = 'MA CHÉRIE!'
        session.lettering_language = 'French'
        self.controller._session = session
        return session

    def dialog_for(self, session):
        dialog = ArtisticEditPreviewDialog(session.prepared.source_crop,
            session.prepared.source_crop.copy(), 'Test lettering', apply_available=True)
        self.addCleanup(dialog.deleteLater)
        self.controller._configure_lettering_preview(dialog, session, True, '')
        return dialog

    def blocking_recognizer(self, reading='MA CHERIE!'):
        entered, release = Event(), Event()
        calls = []
        # Runs before the fixture's shutdown cleanup, even if an assertion fails.
        self.addCleanup(release.set)
        def recognize(pixels, language, settings):
            calls.append((get_ident(), pixels.copy(), language, settings))
            entered.set()
            if not release.wait(4):
                raise RuntimeError('Test worker was not released')
            pixels[:] = 91  # The worker must own its image.
            return reading
        return recognize, entered, release, calls

    def test_helper_captures_exact_wording_and_prompt_edits_cannot_infer_binding(self):
        self.controller.use_selected_translation()
        prompt = self.panel.options().prompt
        self.assertEqual(self.controller._expected_lettering(prompt), 'MA CHÉRIE!')
        self.assertIn('with "MA CHÉRIE!"', prompt)
        self.panel.prompt_edit.insertPlainText(' Change the style.')
        self.assertIsNone(self.controller._expected_lettering(self.panel.options().prompt))
        self.panel.set_prompt(prompt)
        self.assertIsNone(self.controller._expected_lettering(prompt))
        self.controller.use_selected_translation()
        self.panel.prompt_template_combo.setCurrentIndex(1)
        self.assertIsNone(self.controller._expected_lettering(prompt))
        self.controller.use_selected_translation()
        self.assertIn('Do not put any text', self.panel.options().prompt)
        self.assertIsNone(self.controller._expected_lettering(self.panel.options().prompt))
        self.panel.prompt_template_combo.setCurrentIndex(0)
        self.assertIsNone(self.controller._expected_lettering(prompt))

    def test_generation_freezes_exact_wording_and_target_language_without_inference(self):
        self.controller.worker_factory = fixtures._FakeWorker
        self.controller.configured_worker_python = lambda: os.path.abspath(__file__)
        self.controller.use_selected_translation()
        self.controller.update_availability()
        self.controller.generate_preview()
        session = self.controller._session
        self.assertIsNotNone(session)
        self.assertEqual(session.expected_lettering, 'MA CHÉRIE!')
        self.assertEqual(session.lettering_language, 'French')
        self.assertTrue(session.lettering_case_sensitive)
        self.block.translation = 'different target'
        self.main.t_combo.setCurrentText('English')
        self.panel.prompt_edit.insertPlainText(' extra instruction')
        self.assertEqual(session.expected_lettering, 'MA CHÉRIE!')
        self.assertEqual(session.lettering_language, 'French')
        self.assertTrue(session.lettering_case_sensitive)
        self.assertEqual(self.stack.count(), 0)

    def test_freeform_generation_has_no_check_target(self):
        self.controller.worker_factory = fixtures._FakeWorker
        self.controller.configured_worker_python = lambda: os.path.abspath(__file__)
        self.panel.set_prompt('Replace lettering with exactly "Hello"')
        self.controller.update_availability()
        self.controller.generate_preview()
        session = self.controller._session
        self.assertIsNone(session.expected_lettering)
        self.assertIsNone(session.lettering_language)
        dialog = self.dialog_for(session)
        self.assertFalse(dialog.lettering_check_button.isEnabled())

    def test_generated_prompt_matches_source_case_without_rewriting_target(self):
        self.block.text = 'Color de la ilustración de cubierta BRUSCO'
        self.block.translation = 'COVER ILLUSTRATION COLORING: BRUSCO'
        self.controller.use_selected_translation()
        prompt = self.panel.options().prompt
        self.assertIn('with "Cover illustration coloring: BRUSCO"', prompt)
        self.assertNotIn('COVER ILLUSTRATION COLORING', prompt)
        self.assertEqual(self.block.translation, 'COVER ILLUSTRATION COLORING: BRUSCO')
        self.assertEqual(self.controller._expected_lettering(prompt), 'Cover illustration coloring: BRUSCO')

    def test_queued_check_uses_the_captured_case_policy(self):
        for case_sensitive, status in ((False, 'match'), (True, 'mismatch')):
            with self.subTest(case_sensitive=case_sensitive):
                session = self.checked_session()
                session.lettering_case_sensitive = case_sensitive
                dialog = self.dialog_for(session)
                results = Mock(wraps=dialog.show_lettering_check_result)
                dialog.show_lettering_check_result = results
                with patch('app.controllers.artistic_edit.run_lettering_ocr', return_value='Ma chérie!'):
                    dialog.lettering_check_button.click()
                    self.wait_until(lambda: self.controller._lettering_job is None)
                results.assert_called_once()
                result = results.call_args.args[0]
                self.assertEqual(result.status, status)
                self.assertEqual(result.expected, 'MA CHÉRIE!')
                self.assertEqual(result.recognized, 'Ma chérie!')
                self.assertTrue(dialog.apply_button.isEnabled())

    def test_real_queued_job_uses_target_language_and_no_expected_text_in_ocr(self):
        session = self.checked_session()
        dialog = self.dialog_for(session)
        before_page, before_patch = self.page.copy(), session.patch.image.copy()
        before_regions = [(b.text, b.translation, list(b.xyxy)) for b in self.main.blk_list]
        recognize, entered, release, calls = self.blocking_recognizer()
        results = []
        show_result = dialog.show_lettering_check_result
        dialog.show_lettering_check_result = lambda value: (results.append((get_ident(), value)), show_result(value))
        with patch('app.controllers.artistic_edit.run_lettering_ocr', side_effect=recognize) as run:
            dialog.lettering_check_button.click()
            self.wait_until(entered.is_set)
            ticks = []
            QtCore.QTimer.singleShot(0, lambda: ticks.append(True))
            self.wait_until(lambda: bool(ticks))
            self.assertFalse(dialog.lettering_check_button.isEnabled())
            self.assertTrue(dialog.apply_button.isEnabled())
            self.assertNotEqual(calls[0][0], get_ident())
            self.assertEqual(calls[0][2], 'French')
            self.assertEqual(calls[0][3].ocr_key, 'Default')
            self.assertEqual(len(run.call_args.args), 3)
            self.assertFalse(run.call_args.kwargs)
            release.set()
            self.wait_until(lambda: self.controller._lettering_job is None)
        self.assertEqual(results[0][0], get_ident())
        self.assertEqual(results[0][1].status, 'mismatch')
        np.testing.assert_array_equal(self.page, before_page)
        np.testing.assert_array_equal(session.patch.image, before_patch)
        self.assertEqual([(b.text, b.translation, list(b.xyxy)) for b in self.main.blk_list], before_regions)
        self.assertEqual(self.stack.count(), 0)
        self.assertEqual(self.main.image_patches, {})
        self.assertEqual(self.window_modified_updates, 0)
        # A warning does not turn into an automatic edit or prohibit explicit Apply.
        self.assertTrue(dialog.apply_button.isEnabled())
        self.controller._apply_session(session)
        self.assertEqual(self.stack.count(), 1)
        self.stack.undo()
        self.assertEqual(self.main.image_patches[self.page_path], [])

    def test_source_change_during_check_discards_result_and_blocks_stale_apply(self):
        session = self.checked_session()
        dialog = self.dialog_for(session)
        results = Mock(wraps=dialog.show_lettering_check_result)
        dialog.show_lettering_check_result = results
        recognize, entered, release, _ = self.blocking_recognizer('MA CHÉRIE!')
        with patch('app.controllers.artistic_edit.run_lettering_ocr', side_effect=recognize):
            dialog.lettering_check_button.click()
            self.wait_until(entered.is_set)
            x, y, _, _ = session.prepared.context_bbox
            self.main.image_ctrl.page[y, x] = 70
            release.set()
            self.wait_until(lambda: self.controller._lettering_job is None)
        results.assert_not_called()
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertIn('source pixels changed', dialog.stale_label.text().lower())
        self.assertEqual(self.stack.count(), 0)

    def test_ocr_error_is_advisory_redacted_and_can_be_retried(self):
        session = self.checked_session()
        dialog = self.dialog_for(session)
        error = RuntimeError('https://provider.example?api_key=private-secret')
        with patch('app.controllers.artistic_edit.run_lettering_ocr', side_effect=[error, 'MA CHÉRIE!']) as run:
            dialog.lettering_check_button.click()
            self.wait_until(lambda: self.controller._lettering_job is None)
            self.assertIn('RuntimeError', dialog.lettering_status_label.text())
            self.assertNotIn('private-secret', dialog.lettering_status_label.text())
            self.assertNotIn('provider.example', dialog.lettering_status_label.text())
            self.assertTrue(dialog.apply_button.isEnabled())
            self.assertTrue(dialog.lettering_check_button.isEnabled())
            dialog.lettering_check_button.click()
            self.wait_until(lambda: self.controller._lettering_job is None)
            self.assertEqual(run.call_count, 2)
        self.assertIn('matches the requested wording', dialog.lettering_status_label.text())
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertEqual(self.stack.count(), 0)

    def test_changed_patch_discards_late_match(self):
        for changed in ('image', 'alpha', 'bbox', 'wording', 'language', 'case_policy', 'prompt'):
            with self.subTest(changed=changed):
                session = self.checked_session()
                dialog = self.dialog_for(session)
                results = Mock(wraps=dialog.show_lettering_check_result)
                dialog.show_lettering_check_result = results
                recognize, entered, release, _ = self.blocking_recognizer('MA CHÉRIE!')
                with patch('app.controllers.artistic_edit.run_lettering_ocr', side_effect=recognize):
                    dialog.lettering_check_button.click()
                    self.wait_until(entered.is_set)
                    if changed == 'image':
                        session.patch.image[:] = 90
                    elif changed == 'alpha':
                        session.patch.alpha[0, 0] ^= 255
                    elif changed == 'bbox':
                        x, y, w, h = session.patch.bbox
                        session.patch = replace(session.patch, bbox=(x + 1, y, w, h))
                    elif changed == 'wording':
                        session.expected_lettering = 'NEW WORDING'
                    elif changed == 'language':
                        session.lettering_language = 'English'
                    elif changed == 'case_policy':
                        session.lettering_case_sensitive = not session.lettering_case_sensitive
                    else:
                        session.request = replace(session.request, prompt='Draw different lettering')
                    release.set()
                    self.wait_until(lambda: self.controller._lettering_job is None)
                results.assert_not_called()
                self.assertIn('preview changed', dialog.lettering_status_label.text().lower())
                self.assertEqual(self.stack.count(), 0)

    def test_replaced_preview_ignores_old_result_and_rebinds_one_current_route(self):
        session = self.checked_session()
        old_dialog = self.dialog_for(session)
        old_result = Mock(wraps=old_dialog.show_lettering_check_result)
        old_dialog.show_lettering_check_result = old_result
        recognize, entered, release, _ = self.blocking_recognizer('MA CHÉRIE!')
        with patch('app.controllers.artistic_edit.run_lettering_ocr', side_effect=recognize):
            old_dialog.lettering_check_button.click()
            self.wait_until(entered.is_set)
            replacement = self.checked_session()
            new_dialog = self.dialog_for(replacement)
            new_result = Mock(wraps=new_dialog.show_lettering_check_result)
            new_dialog.show_lettering_check_result = new_result
            self.assertFalse(new_dialog.lettering_check_button.isEnabled())
            self.main.settings_page.get_tool_selection = lambda _: 'Microsoft OCR'
            release.set()
            self.wait_until(lambda: self.controller._lettering_job is None)
        old_result.assert_not_called()
        new_result.assert_not_called()
        self.assertIn('Microsoft OCR', new_dialog.lettering_route_label.text())
        self.assertTrue(new_dialog.lettering_check_button.isEnabled())
        with patch('app.controllers.artistic_edit.run_lettering_ocr', return_value='MA CHÉRIE!') as run:
            new_dialog.lettering_check_button.click()
            self.wait_until(lambda: self.controller._lettering_job is None)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[2].ocr_key, 'Microsoft OCR')
        self.assertEqual(self.stack.count(), 0)

    def test_discard_while_check_runs_retires_dialog_and_preserves_project(self):
        session = self.checked_session()
        recognize, entered, release, _ = self.blocking_recognizer()
        dialogs = []
        def factory(*args, **kwargs):
            kwargs.pop('parent', None)
            dialog = ArtisticEditPreviewDialog(*args, **kwargs)
            dialogs.append(dialog)
            self.addCleanup(lambda: dialog.deleteLater() if isValid(dialog) else None)
            dialog.show_lettering_check_result = Mock(wraps=dialog.show_lettering_check_result)
            QtCore.QTimer.singleShot(0, dialog.lettering_check_button.click)
            poll = QtCore.QTimer(dialog)
            poll.timeout.connect(lambda: dialog.reject() if entered.is_set() else None)
            poll.start(2)
            watchdog = QtCore.QTimer(dialog)
            watchdog.setSingleShot(True)
            watchdog.timeout.connect(dialog.reject)
            watchdog.start(2000)
            dialog.finished.connect(poll.stop)
            dialog.finished.connect(watchdog.stop)
            return dialog
        self.controller.preview_factory = factory
        with patch('app.controllers.artistic_edit.run_lettering_ocr', side_effect=recognize):
            self.controller._show_preview(SimpleNamespace(seed=42, elapsed_seconds=.1))
            self.assertTrue(entered.is_set())
            self.assertIsNone(self.controller._session)
            self.assertIsNone(self.controller._lettering_dialog)
            self.assertIsNone(self.controller._lettering_context)
            release.set()
            self.wait_until(lambda: self.controller._lettering_job is None)
        self.assertEqual(self.stack.count(), 0)
        self.assertEqual(self.window_modified_updates, 0)
        dialogs[0].show_lettering_check_result.assert_not_called()

    def test_prepare_close_waits_for_worker_and_reschedules_clean_close(self):
        session = self.checked_session()
        dialog = self.dialog_for(session)
        recognize, entered, release, _ = self.blocking_recognizer()
        results = Mock(wraps=dialog.show_lettering_check_result)
        dialog.show_lettering_check_result = results
        with patch('app.controllers.artistic_edit.run_lettering_ocr', side_effect=recognize):
            dialog.lettering_check_button.click()
            self.wait_until(entered.is_set)
            self.assertFalse(self.controller.prepare_close())
            self.assertTrue(self.controller._close_pending)
            self.main.close.assert_not_called()
            release.set()
            self.wait_until(lambda: self.controller._lettering_job is None and self.main.close.called)
        self.main.close.assert_called_once()
        self.assertTrue(self.controller.prepare_close())
        self.assertFalse(self.controller._close_pending)
        results.assert_not_called()
        self.assertEqual(self.stack.count(), 0)


if __name__ == '__main__':
    unittest.main()
