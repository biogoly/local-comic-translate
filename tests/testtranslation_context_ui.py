"""Project context UI and undo using real Qt widgets and no model calls."""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from app.controllers.translation_context import TranslationContextController
from app.ui.dialogs.translation_context import ContextEntryDialog, TranslationContextDialog
from modules.translation.context import empty_translation_profile, normalize_translation_profile
from modules.translation.llm.local import LocalLLMTranslation
from modules.utils.textblock import TextBlock


def profile_with_entries():
    return normalize_translation_profile({
        'glossary': [{'id': 'term1', 'source': 'Thảo', 'aliases': ['Thao'], 'target': 'Thảo',
                      'source_lang': 'Vietnamese', 'target_lang': 'Brazilian Portuguese'}],
        'characters': [{'id': 'char1', 'name': '先生', 'target': 'Professor', 'notes': 'Formal speech'}],
        'notes': 'Keep the story tone.', 'speakers': [{'page_id': 'p1', 'region_id': 'r1', 'character_id': 'char1'}],
    })


class ContextWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.translation_profile = profile_with_entries()
        self.translation_context_button = QtWidgets.QPushButton(self)
        self.lang_mapping = {'English': 'English', 'Japanese': 'Japanese', 'Vietnamese': 'Vietnamese',
                             'Portuguese': 'Portuguese', 'Brazilian Portuguese': 'Brazilian Portuguese'}
        self.s_combo, self.t_combo = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.s_combo.addItem('Vietnamese')
        self.t_combo.addItem('Brazilian Portuguese')
        self.settings_page = SimpleNamespace(get_tool_selection=Mock(return_value='Local LLM'),
            get_llm_settings=Mock(return_value={'extra_context': 'General directions'}),
            ui=SimpleNamespace(tr=lambda text: text))
        self.mark_project_dirty = Mock()
        self._batch_active = False
        self.task_runner_ctrl = SimpleNamespace(is_processing_queue=False)
        self.undo_group = QtGui.QUndoGroup(self)
        self.page_stack = QtGui.QUndoStack(self)
        self.undo_group.addStack(self.page_stack)
        self.undo_group.setActiveStack(self.page_stack)
        self.page_stack.push(QtGui.QUndoCommand('Page edit'))
        self.block = TextBlock(text_bbox=[0, 0, 30, 25], text='Thảo', translation='Olá')
        self.translation_profile['speakers'][0]['region_id'] = self.block.region_id
        self.image_files = ['page.png']
        self.curr_img_idx = 0
        self.image_viewer = SimpleNamespace(webtoon_mode=False,
            get_image_array=Mock(return_value=np.zeros((30, 40, 3), dtype=np.uint8)))
        self.review_ctrl = SimpleNamespace(_preview_running=False, selected_region=Mock(return_value=('p1', self.block.region_id)),
            region_refs=Mock(return_value=[self.block]), snapshot=Mock(return_value={('p1', self.block.region_id):
                {'block': self.block, 'path': 'page.png', 'page_id': 'p1'}}))


class TranslationContextUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = ContextWindow()
        self.ctrl = TranslationContextController(self.main)
        self.ctrl.show_editor()
        self.ctrl.timer.stop()
        self.dialog = self.ctrl.dialog

    def tearDown(self):
        self.ctrl.shutdown()
        self.main.close()
        self.main.deleteLater()
        self.app.processEvents()

    def test_drafts_apply_and_context_undo_leave_page_undo_untouched(self):
        initial = copy.deepcopy(self.main.translation_profile)
        self.dialog.notes_edit.setPlainText('New setting notes.')
        self.assertEqual(self.main.translation_profile, initial)
        self.assertTrue(self.dialog.apply_button.isEnabled())
        self.dialog.apply_button.click()
        self.assertEqual(self.main.translation_profile['notes'], 'New setting notes.')
        self.assertTrue(self.dialog.undo_button.isEnabled())
        self.assertEqual(self.main.page_stack.index(), 1)
        self.assertIs(self.main.undo_group.activeStack(), self.main.page_stack)
        self.dialog.undo_button.click()
        self.assertEqual(self.main.translation_profile, initial)
        self.assertEqual(self.main.page_stack.index(), 1)
        self.dialog.redo_button.click()
        self.assertEqual(self.main.translation_profile['notes'], 'New setting notes.')
        self.assertEqual(self.main.mark_project_dirty.call_count, 3)

    def test_project_reset_discards_drafts_and_cross_project_undo(self):
        self.dialog.notes_edit.setPlainText('Saved A')
        self.dialog.apply_button.click()
        self.dialog.summary_edit.setPlainText('Unapplied A')
        project_b = normalize_translation_profile({'notes': 'Only B'})
        self.main.translation_profile = project_b
        self.ctrl.reset_for_project()
        self.assertEqual(self.main.translation_profile, project_b)
        self.assertFalse(self.ctrl.undo_stack.canUndo())
        self.ctrl.show_editor()
        self.ctrl.timer.stop()
        self.assertEqual(self.dialog.profile(), project_b)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.ctrl.undo()
        self.assertEqual(self.main.translation_profile, project_b)

    def test_closing_and_reopening_same_project_retains_unapplied_draft(self):
        self.dialog.notes_edit.setPlainText('Unapplied note')
        self.dialog.close()
        self.ctrl.show_editor()
        self.ctrl.timer.stop()
        self.assertEqual(self.dialog.notes_edit.toPlainText(), 'Unapplied note')
        self.assertTrue(self.dialog.apply_button.isEnabled())
        self.assertNotEqual(self.main.translation_profile['notes'], 'Unapplied note')

    def test_preview_scope_or_region_changes_clear_snapshot_without_losing_draft(self):
        self.dialog.notes_edit.setPlainText('Unapplied note')
        self.dialog.set_preview('Previously built preview')
        self.dialog.preview_scope.setCurrentIndex(1)
        self.assertEqual(self.dialog.preview_text.toPlainText(), '')
        self.dialog.set_preview('Selected preview')
        self.dialog.set_selection(None)
        self.assertEqual(self.dialog.preview_text.toPlainText(), '')
        self.assertEqual(self.dialog.notes_edit.toPlainText(), 'Unapplied note')
        self.dialog.set_preview('Previous context')
        self.dialog.summary_edit.setPlainText('Changed summary')
        self.assertEqual(self.dialog.preview_text.toPlainText(), '')

    def test_busy_blocks_mutation_and_no_page_source_changes(self):
        self.dialog.notes_edit.setPlainText('New draft')
        before = copy.deepcopy(self.main.translation_profile)
        self.main.task_runner_ctrl.is_processing_queue = True
        self.ctrl._update_state()
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertFalse(self.dialog.tabs.isEnabled())
        self.ctrl.apply_profile(self.dialog.profile())
        self.assertEqual(self.main.translation_profile, before)
        self.assertEqual(self.main.block.text, 'Thảo')
        self.assertEqual(self.main.block.translation, 'Olá')

    def test_selected_glossary_prefill_preserves_language_dialect(self):
        with patch.object(self.dialog, 'edit_entry') as edit:
            self.ctrl.add_selected()
        self.assertEqual(edit.call_args.args, ('glossary',))
        seed = edit.call_args.kwargs['initial']
        self.assertEqual(seed, {'source': 'Thảo', 'target': 'Olá', 'source_lang': 'Vietnamese',
                                'target_lang': 'Brazilian Portuguese'})
        self.assertTrue(edit.call_args.kwargs['new'])

    def test_speaker_assignment_uses_stable_ids_and_character_removal_is_draft(self):
        self.dialog.speaker_combo.setCurrentIndex(0)
        self.dialog.assign_button.click()
        self.assertEqual(self.dialog.profile()['speakers'], [])
        self.assertEqual(len(self.main.translation_profile['speakers']), 1)
        self.dialog.speaker_combo.setCurrentIndex(self.dialog.speaker_combo.findData('char1'))
        self.dialog.assign_button.click()
        self.assertEqual(self.dialog.profile()['speakers'], [{'page_id': 'p1', 'region_id': self.main.block.region_id, 'character_id': 'char1'}])
        self.dialog.tables['characters'].selectRow(0)
        self.dialog.remove_entry('characters')
        self.assertEqual(self.dialog.profile()['speakers'], [])
        self.assertEqual(self.dialog.profile()['characters'], [])
        self.assertEqual(len(self.main.translation_profile['characters']), 1)
        self.dialog.reset_button.click()
        self.assertEqual(len(self.dialog.profile()['characters']), 1)

    def test_preview_uses_draft_without_applying_or_sending_model_request(self):
        self.dialog.notes_edit.setPlainText('Preview draft')
        with patch('app.controllers.translation_context.Translator') as factory:
            factory.return_value.prepare_request.return_value = SimpleNamespace(preview='Compiled context preview')
            self.ctrl.preview_request(self.dialog.profile(), 'selected')
        factory.return_value.translate.assert_not_called()
        request = factory.return_value.prepare_request.call_args
        self.assertEqual(request.kwargs['profile']['notes'], 'Preview draft')
        self.assertEqual(request.kwargs['page_id'], 'p1')
        self.assertIsNot(request.args[0][0], self.main.block)
        self.assertEqual(self.dialog.preview_text.toPlainText(), 'Compiled context preview')
        self.assertNotEqual(self.main.translation_profile['notes'], 'Preview draft')
        self.assertEqual(self.main.mark_project_dirty.call_count, 0)

    def test_traditional_backend_marks_context_unavailable_without_network(self):
        self.main.settings_page.get_tool_selection.return_value = 'DeepL'
        self.ctrl._update_state()
        self.assertFalse(self.dialog.preview_button.isEnabled())
        self.assertTrue(self.dialog.tabs.isEnabled())
        with patch('app.controllers.translation_context.Translator') as factory:
            self.ctrl.preview_request(self.dialog.profile(), 'page')
        factory.assert_not_called()

    def test_real_request_preview_includes_matching_terms_and_assigned_speaker_without_transport(self):
        engine = LocalLLMTranslation()
        engine.runtime_mode = 'external'
        engine.model = 'fixture-model'
        engine._perform_translation = Mock(side_effect=AssertionError('Preview must never send a request'))
        with patch('modules.translation.processor.TranslationFactory.create_engine', return_value=engine):
            self.ctrl.preview_request(self.dialog.profile(), 'selected')
        preview = self.dialog.preview_text.toPlainText()
        self.assertIn('Thảo', preview)
        self.assertIn('Formal speech', preview)
        self.assertIn('Professor', preview)
        self.assertIn('Conservative token estimate', preview)
        self.assertNotIn('Cannot build', preview)
        engine._perform_translation.assert_not_called()
        self.assertEqual(self.main.mark_project_dirty.call_count, 0)

    def test_context_budget_failure_is_visible_and_preserves_project(self):
        engine = LocalLLMTranslation()
        engine.runtime_mode = 'external'
        engine._perform_translation = Mock(side_effect=AssertionError('No transport for oversized context'))
        self.dialog.notes_edit.setPlainText('Large context ' * 1000)
        self.dialog.budget_spin.setValue(128)
        before = copy.deepcopy(self.main.translation_profile)
        with patch('modules.translation.processor.TranslationFactory.create_engine', return_value=engine):
            self.ctrl.preview_request(self.dialog.profile(), 'selected')
        self.assertIn('Cannot build request preview', self.dialog.preview_text.toPlainText())
        self.assertEqual(self.main.translation_profile, before)
        engine._perform_translation.assert_not_called()

    def test_page_preview_omits_kept_target_selected_preview_can_include_it(self):
        self.main.block.region_metadata['protections']['keep_translation'] = True
        with patch('app.controllers.translation_context.Translator') as factory:
            factory.return_value.prepare_request.return_value = SimpleNamespace(preview='Explicit selected preview')
            self.ctrl.preview_request(self.dialog.profile(), 'page')
            factory.assert_not_called()
            self.ctrl.preview_request(self.dialog.profile(), 'selected')
            self.assertEqual(factory.call_count, 1)
        self.assertEqual(self.dialog.preview_text.toPlainText(), 'Explicit selected preview')

    def test_entry_roundtrip_aliases_unicode_and_language_pair(self):
        entry = profile_with_entries()['glossary'][0]
        editor = ContextEntryDialog('glossary', entry, [('Vietnamese', 'Vietnamese'),
            ('Portuguese', 'Portuguese'), ('Brazilian Portuguese', 'Brazilian Portuguese')])
        self.addCleanup(editor.deleteLater)
        editor.aliases.setPlainText('Thao\nThảo Nguyễn\n')
        editor.notes.setPlainText('Keep accents.')
        editor.match.setCurrentIndex(editor.match.findData('whole_word'))
        editor.approved.setChecked(False)
        value = editor.value()
        self.assertEqual(value['aliases'], ['Thao', 'Thảo Nguyễn'])
        self.assertEqual(value['target_lang'], 'Brazilian Portuguese')
        self.assertEqual(value['source_lang'], 'Vietnamese')
        self.assertEqual(value['match'], 'whole_word')
        self.assertFalse(value['approved'])
        editor.target.clear()
        editor.accept()
        self.assertNotEqual(editor.result(), QtWidgets.QDialog.DialogCode.Accepted)
