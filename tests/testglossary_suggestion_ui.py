"""Review glossary suggestions against approved text before changing a project."""
import copy
import json
import unittest

from PySide6 import QtCore, QtTest, QtWidgets

from modules.review.issues import fingerprint_for_stage, mark_reviewed
from modules.utils.textblock import TextBlock
import testtranslation_history_ui as fixtures


class GlossarySuggestionUITests(unittest.TestCase):
    tearDown = fixtures.HistorySummaryUITests.tearDown
    finish_job = fixtures.HistorySummaryUITests.finish_job

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        fixtures.HistorySummaryUITests.setUp(self)
        self.wharf = TextBlock(text_bbox=[1, 20, 80, 38], text='Tới Bến Sao Đỏ.', translation='Ao Cais da Estrela Rubra.')
        languages = {'source': 'Vietnamese', 'target': 'Brazilian Portuguese'}
        self.wharf.region_metadata = mark_reviewed(self.wharf.region_metadata, 'translation',
            fingerprint_for_stage('translation', self.wharf, languages=languages))
        self.main.image_states['earlier.png']['blk_list'].append(self.wharf)
        self.engine.complete_text_once.return_value = json.dumps({'terms': [
            {'source': 'Hà', 'target': 'Hà', 'example': 1},
            {'source': 'Bến Sao Đỏ', 'target': 'Cais da Estrela Rubra', 'example': 2}]}, ensure_ascii=False)

    def open_form(self):
        self.editor.suggest_glossary_button.click()
        form = self.ctrl.suggestion_dialog
        self.assertIsNotNone(form, self.editor.status_label.text())
        return form

    def generate(self):
        form = self.open_form()
        form.generate_button.click()
        self.finish_job()
        return form

    def test_edit_and_choose_one_term_with_approved_evidence_then_apply_and_undo(self):
        original = copy.deepcopy(self.main.translation_profile)
        self.editor.notes_edit.setPlainText('My unapplied note.')
        form = self.open_form()
        self.assertIn('locally', form.findChildren(QtWidgets.QLabel)[0].text())
        self.assertIn('(2)', form.inputs_toggle.text())
        self.assertNotIn(self.unreviewed.text, form.inputs_edit.toPlainText())
        self.engine.complete_text_once.assert_not_called()
        form.generate_button.click()
        self.finish_job()
        table = form.suggestion_edit
        self.assertEqual(table.rowCount(), 2, form.status_label.text())
        self.assertFalse(form.use_button.isEnabled())
        self.assertEqual(form.evidence_edit.toPlainText(), self.earlier.text + '\n' + self.earlier.translation)
        table.setCurrentCell(1, 1)
        self.assertEqual(form.evidence_edit.toPlainText(), self.wharf.text + '\n' + self.wharf.translation)
        table.item(1, 2).setText('Porto da Estrela Rubra')
        table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Checked)
        form.use_button.click()
        self.assertEqual(self.main.translation_profile, original)
        profile = self.editor.profile()
        self.assertEqual(profile['glossary'][:-1], original['glossary'])
        entry = profile['glossary'][-1]
        self.assertEqual((entry['source'], entry['target']), ('Bến Sao Đỏ', 'Porto da Estrela Rubra'))
        self.assertEqual((entry['source_lang'], entry['target_lang']), ('Vietnamese', 'Brazilian Portuguese'))
        self.assertTrue(entry['id'] and entry['enabled'] and entry['approved'])
        self.assertEqual(profile['notes'], 'My unapplied note.')
        self.editor.apply_button.click()
        self.assertEqual(self.main.translation_profile, profile)
        self.editor.undo_button.click()
        self.assertEqual(self.main.translation_profile, original)
        self.assertEqual(self.main.page_stack.index(), 1)

    def test_edited_rows_cannot_collide_with_an_alias_wildcard_or_each_other(self):
        # Draft entries count even if disabled or still awaiting approval.
        self.editor._draft['glossary'].extend([
            dict(id='wild', source='Mar', target='sea', aliases=['Biển'], source_lang='', target_lang='', enabled=False),
            dict(id='other', source='Outros', target='outros', source_lang='Vietnamese', target_lang='Portuguese')])
        form = self.generate()
        table = form.suggestion_edit
        table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
        table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Checked)
        for word in ('thao', 'BIỂN', 'bến sao đỏ', ''):
            with self.subTest(source=word):
                table.item(0, 1).setText(word)
                self.assertFalse(form.use_button.isEnabled())
                self.assertTrue(form.status_label.text())
        # European Portuguese does not conflict with this Brazilian Portuguese term.
        table.item(0, 1).setText('Outros')
        self.assertTrue(form.use_button.isEnabled())
        table.item(1, 2).setText(' ')
        self.assertFalse(form.use_button.isEnabled())
        table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.assertTrue(form.use_button.isEnabled())

    def test_no_candidates_or_no_approved_examples(self):
        self.engine.complete_text_once.return_value = '{"terms":[]}'
        form = self.generate()
        self.assertEqual(form.suggestion_edit.rowCount(), 0)
        self.assertFalse(form.use_button.isEnabled())
        self.assertIn('No new terms', form.status_label.text())
        form.reject()
        self.earlier.translation += ' x'
        self.wharf.translation += ' y'
        self.editor.suggest_glossary_button.click()
        self.assertIsNone(self.ctrl.suggestion_dialog)
        self.assertIn('mark a translation reviewed', self.editor.status_label.text())
        self.assertEqual(self.engine.complete_text_once.call_count, 1)

    def test_add_commits_the_active_editor_and_rechecks_its_new_source_for_conflict(self):
        form = self.generate()
        table = form.suggestion_edit
        table.setCurrentCell(1, 0)
        table.setFocus()
        QtTest.QTest.keyClick(table, QtCore.Qt.Key.Key_Space)
        self.assertTrue(form.use_button.isEnabled())
        table.editItem(table.item(1, 1))
        field = table.findChild(QtWidgets.QLineEdit)
        self.assertIsNotNone(field)
        field.selectAll()
        QtTest.QTest.keyClicks(field, 'Thao')  # An alias of a preexisting term.
        QtTest.QTest.mouseClick(form.use_button, QtCore.Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertIs(self.ctrl.suggestion_dialog, form)
        self.assertFalse(form.use_button.isEnabled())
        self.assertEqual(self.editor.profile()['glossary'], self.main.translation_profile['glossary'])

    def test_remote_consent_and_changed_route_before_generate(self):
        self.engine.api_base_url = 'https://example.test/v1'
        form = self.open_form()
        self.assertIn('https://example.test', form.findChildren(QtWidgets.QLabel)[0].text())
        self.engine.api_base_url = 'https://another.example/v1'
        form.generate_button.click()
        self.assertEqual(self.jobs, [])
        self.assertFalse(form.generate_button.isEnabled())
        self.engine.complete_text_once.assert_not_called()

    def test_revoked_approval_and_changed_glossary_input_discard_a_reply(self):
        original = copy.deepcopy(self.editor.profile())
        for change in ('approval', 'glossary'):
            with self.subTest(change=change):
                form = self.open_form()
                form.generate_button.click()
                if change == 'approval':
                    self.earlier.text += ' changed'
                else:
                    self.editor._draft['glossary'].append(dict(id='new', source='Hà', target='Hà',
                                                               source_lang='Vietnamese', target_lang='Brazilian Portuguese'))
                self.finish_job()
                self.assertFalse(form.use_button.isEnabled())
                self.assertIn('inputs changed', form.status_label.text())
                self.assertEqual(self.main.translation_profile, original)
                form.reject()
                self.earlier.text = 'Đây là Hà.'

    def test_canceled_jobs_skip_unsent_work_and_ignore_already_dispatched_results(self):
        form = self.open_form()
        form.generate_button.click()
        form.reject()
        self.finish_job()
        self.engine.complete_text_once.assert_not_called()
        form = self.open_form()
        form.generate_button.click()
        run, ready, _, finished = self.jobs.pop()
        result = run()
        self.ctrl.reset_for_project()
        self.ctrl.show_editor()
        replacement = self.open_form()
        ready(result)
        finished()
        self.assertIs(self.ctrl.suggestion_dialog, replacement)
        self.assertEqual(replacement.suggestion_edit.rowCount(), 0)
        self.assertEqual(self.editor.profile()['glossary'], self.main.translation_profile['glossary'])

    def test_language_change_before_add_prevents_acceptance(self):
        original = self.editor.profile()
        form = self.generate()
        form.suggestion_edit.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
        self.main.t_combo.addItem('English')
        self.main.t_combo.setCurrentText('English')
        form.use_button.click()
        self.assertEqual(self.editor.profile(), original)
        self.assertIn('inputs changed', self.editor.status_label.text())


if __name__ == '__main__':
    unittest.main()
