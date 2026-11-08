"""Isolated style-library editing and multi-item preview acceptance boundaries."""
import copy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.dayu_widgets import dayu_theme
from app.ui.dialogs.lettering_styles import LetteringStylesDialog
from modules.utils.lettering_style_exchange import read_style_file, write_style_file


class LetteringStylesUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.live_items = []
        self.selections = []
        for index, size in enumerate((16, 22)):
            item = TextBlockItem('One emphasized word', font_family='Arial', font_size=size,
                                 render_color=QtGui.QColor('#102030'), outline_color=None)
            item.setTextWidth(180)
            item.setPos(20 + index * 210, 35)
            cursor = QtGui.QTextCursor(item.document())
            cursor.setPosition(4)
            cursor.setPosition(14, QtGui.QTextCursor.MoveMode.KeepAnchor)
            emphasis = QtGui.QTextCharFormat()
            emphasis.setFontWeight(QtGui.QFont.Weight.Bold)
            cursor.mergeCharFormat(emphasis)
            self.live_items.append(item)
            self.selections.append(TextItemProperties.from_text_item(item).to_dict())
        self.library = {'schema_version': 1, 'presets': [{
            'id': '37a94b52-fb2b-4f16-97cb-4cc87bf8b72b', 'name': 'Dialogue',
            'properties': {'font_family': 'Arial', 'font_size': 26, 'text_color': '#202040',
                           'outline': False, 'outline_color': '#ffffff', 'outline_width': 1,
                           'line_spacing': 1.1, 'alignment': 4,
                           'bold': False, 'italic': False, 'underline': False}}]}
        self.dialog = LetteringStylesDialog(self.library, self.selections)

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def test_preview_owns_snapshots_and_preserves_emphasis_by_default(self):
        for combo in self.dialog.emphasis_combos.values():
            self.assertEqual(combo.currentData(), 'keep')
        self.assertEqual(len(self.dialog._candidates), 2)
        for before, after in zip(self.selections, self.dialog._candidates):
            self.assertEqual(before['position'], after['position'])
            self.assertEqual(after['font_size'], 26)
        self.selections[0]['font_size'] = 99
        self.library['presets'][0]['name'] = 'Changed outside'
        self.dialog.rebuild()
        self.assertEqual(self.dialog._selections[0]['font_size'], 16)
        self.assertEqual(self.dialog._library['presets'][0]['name'], 'Dialogue')
        self.assertEqual(self.live_items[0].font_size, 16)
        self.assertIsNone(self.dialog.library_result)
        self.assertEqual(self.dialog.candidates, [])

    def test_capture_uses_explicitly_previewed_text_box(self):
        self.dialog.selection_combo.setCurrentIndex(1)
        self.dialog.name_edit.setText('Narration')
        self.dialog.capture_button.click()
        self.assertEqual(len(self.dialog._library['presets']), 2)
        captured = self.dialog._library['presets'][1]
        self.assertEqual(captured['name'], 'Narration')
        self.assertEqual(captured['properties']['font_size'], 22)
        self.assertEqual(len(self.library['presets']), 1)
        self.assertEqual(self.dialog.preset_list.currentRow(), 1)

    def test_cancel_discards_draft_library_edits_and_candidates(self):
        before = copy.deepcopy(self.library)
        self.dialog.name_edit.setText('Renamed')
        self.dialog.rename_button.click()
        self.dialog.delete_button.click()
        self.dialog.cancel_button.click()
        self.assertEqual(self.library, before)
        self.assertEqual(self.dialog.result(), QtWidgets.QDialog.DialogCode.Rejected)
        self.assertIsNone(self.dialog.library_result)
        self.assertEqual(self.dialog.candidates, [])

    def test_replace_keeps_preset_identity_and_is_only_a_draft(self):
        identity = self.library['presets'][0]['id']
        self.dialog.selection_combo.setCurrentIndex(1)
        self.dialog.replace_button.click()
        preset = self.dialog._library['presets'][0]
        self.assertEqual(preset['id'], identity)
        self.assertEqual(preset['name'], 'Dialogue')
        self.assertEqual(preset['properties']['font_size'], 22)
        self.assertEqual(self.library['presets'][0]['properties']['font_size'], 26)

    def test_save_presets_only_accepts_library_without_formatting_items(self):
        self.dialog.name_edit.setText('Narration')
        self.dialog.rename_button.click()
        self.dialog.save_button.click()
        self.assertEqual(self.dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)
        self.assertEqual(self.dialog.library_result['presets'][0]['name'], 'Narration')
        self.assertEqual(self.dialog.candidates, [])
        self.assertEqual(self.live_items[0].font_size, 16)

    def test_apply_returns_every_prepared_candidate_together(self):
        with patch('app.ui.dialogs.lettering_styles.font_warnings', return_value=[]):
            self.dialog.rebuild()
        self.dialog.apply_button.click()
        self.assertEqual(self.dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)
        self.assertEqual(len(self.dialog.candidates), 2)
        self.assertEqual(self.dialog.candidates[1]['font_size'], 26)
        self.assertEqual(self.live_items[1].font_size, 22)
        self.dialog._candidates[0]['font_size'] = 100
        self.assertEqual(self.dialog.candidates[0]['font_size'], 26)

    def test_font_warning_requires_acknowledgement_and_changes_reset_it(self):
        warning = {'code': 'missing_font', 'font': 'Not Installed Font', 'characters': ''}
        with patch('app.ui.dialogs.lettering_styles.font_warnings', return_value=[warning]):
            self.dialog.rebuild()
            self.assertFalse(self.dialog.apply_button.isEnabled())
            self.assertIn('fallback font', self.dialog.status_label.text())
            self.dialog.warning_ack.setChecked(True)
            self.assertTrue(self.dialog.apply_button.isEnabled())
            combo = self.dialog.emphasis_combos['italic']
            combo.setCurrentIndex(combo.findData('on'))
            self.assertFalse(self.dialog.warning_ack.isChecked())
            self.assertFalse(self.dialog.apply_button.isEnabled())

    def test_glyph_warning_lists_affected_characters(self):
        warning = {'code': 'missing_glyphs', 'font': 'Arial', 'characters': '漢字'}
        with patch('app.ui.dialogs.lettering_styles.font_warnings', return_value=[warning]):
            self.dialog.rebuild()
        self.assertIn('漢字', self.dialog.status_label.text())
        self.assertFalse(self.dialog.apply_button.isEnabled())

    def _emphasis(self, item, key, position):
        cursor = QtGui.QTextCursor(item.document())
        cursor.setPosition(position)
        fmt = cursor.charFormat()
        return {'bold': fmt.fontWeight() >= QtGui.QFont.Weight.Bold,
                'italic': fmt.fontItalic(), 'underline': fmt.fontUnderline()}[key]

    def test_each_emphasis_control_enables_removes_and_preserves_actual_text(self):
        for key, combo in self.dialog.emphasis_combos.items():
            with self.subTest(property=key):
                for mode, expected in (('on', True), ('off', False), ('preset', False)):
                    combo.setCurrentIndex(combo.findData(mode))
                    preview = self.dialog._preview_items[1]
                    for position in (2, 8):
                        self.assertEqual(self._emphasis(preview, key, position), expected)
                    self.assertEqual(self.dialog._candidates[0][key], expected)
                combo.setCurrentIndex(combo.findData('keep'))
                preview = self.dialog._preview_items[1]
                self.assertFalse(self._emphasis(preview, key, 2))
                self.assertEqual(self._emphasis(preview, key, 8), key == 'bold')
        self.assertEqual(self.dialog._library['presets'][0]['properties'],
                         self.library['presets'][0]['properties'])

    def test_direct_emphasis_without_a_saved_preset_can_be_previewed_and_applied(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.dialog = LetteringStylesDialog(None, self.selections)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        for key, combo in self.dialog.emphasis_combos.items():
            self.assertFalse(combo.model().item(combo.findData('preset')).isEnabled())
            combo.setCurrentIndex(combo.findData('on'))
        self.assertTrue(self.dialog.apply_button.isEnabled())
        for index, candidate in enumerate(self.dialog._candidates):
            self.assertEqual(candidate['font_size'], self.selections[index]['font_size'])
            self.assertTrue(all(candidate[key] for key in ('bold', 'italic', 'underline')))
        self.dialog.apply_button.click()
        self.assertEqual(len(self.dialog.candidates), 2)
        self.assertEqual(self.dialog.library_result['presets'], [])
        self.assertFalse(self._emphasis(self.live_items[0], 'italic', 2))

    def test_capture_and_replace_save_explicit_emphasis_choices(self):
        for key, combo in self.dialog.emphasis_combos.items():
            combo.setCurrentIndex(combo.findData('on'))
        self.dialog.name_edit.setText('Emphasized')
        self.dialog.capture_button.click()
        captured = self.dialog._preset()
        self.assertTrue(all(captured['properties'][key] for key in self.dialog.emphasis_combos))
        for combo in self.dialog.emphasis_combos.values():
            combo.setCurrentIndex(combo.findData('off'))
        self.dialog.replace_button.click()
        self.assertFalse(any(captured['properties'][key] for key in self.dialog.emphasis_combos))
        # Preview overrides alone never rewrite another saved preset.
        self.assertEqual(self.dialog._library['presets'][0], self.library['presets'][0])

    def test_use_preset_restores_saved_true_values_and_clears_when_preset_deleted(self):
        self.dialog._preset()['properties'].update(bold=True, italic=True, underline=True)
        for key, combo in self.dialog.emphasis_combos.items():
            combo.setCurrentIndex(combo.findData('preset'))
            self.assertTrue(self._emphasis(self.dialog._preview_items[1], key, 2))
        self.dialog.delete_button.click()
        self.assertTrue(all(combo.currentData() == 'keep' for combo in self.dialog.emphasis_combos.values()))
        self.assertFalse(self.dialog.apply_button.isEnabled())

    def test_partial_preview_failure_never_leaves_apply_enabled(self):
        ready = copy.deepcopy(self.dialog._candidates[0])
        with patch('app.ui.dialogs.lettering_styles.styled_properties', side_effect=[ready, ValueError('fixture')]):
            with self.assertLogs('app.ui.dialogs.lettering_styles', level='ERROR'):
                self.dialog.rebuild()
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertEqual(self.dialog._candidates, [])
        self.assertEqual(self.dialog.candidates, [])

    def test_without_selection_library_can_be_managed_but_not_applied(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.dialog = LetteringStylesDialog(self.library, [])
        self.assertFalse(self.dialog.capture_button.isEnabled())
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.dialog.name_edit.setText('Narration')
        self.dialog.rename_button.click()
        self.dialog.save_button.click()
        self.assertEqual(self.dialog.library_result['presets'][0]['name'], 'Narration')

    def test_export_writes_entire_draft_and_cancel_discards_project_edits(self):
        self.dialog.name_edit.setText('Narration')
        self.dialog.capture_button.click()
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'test.json'
            with patch.object(QtWidgets.QFileDialog, 'exec', return_value=QtWidgets.QDialog.Accepted), \
                 patch.object(QtWidgets.QFileDialog, 'selectedFiles', return_value=[str(path)]):
                self.dialog.export_button.click()
            exported = read_style_file(path)
        self.assertEqual([entry['name'] for entry in exported], ['Dialogue', 'Narration'])
        self.assertEqual(self.dialog.library_status.text(), 'Exported 2 presets.')
        self.dialog.cancel_button.click()
        self.assertIsNone(self.dialog.library_result)
        self.assertEqual(len(self.library['presets']), 1)

    def test_import_collision_selects_preview_and_checks_missing_fonts(self):
        incoming = copy.deepcopy(self.library)
        incoming['presets'][0]['properties']['font_family'] = 'Missing Style Exchange Test Font'
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'incoming.json'
            write_style_file(path, incoming)
            with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(path), '')):
                self.dialog.import_button.click()
                self.assertEqual(self.dialog._preset()['name'], 'Dialogue (2)')
                self.assertIn('Renamed: 1.', self.dialog.library_status.text())
                self.assertIn('Missing Style Exchange Test Font', self.dialog.status_label.text())
                self.assertFalse(self.dialog.apply_button.isEnabled())
                self.dialog.import_button.click()
        self.assertIn('Skipped duplicates: 1.', self.dialog.library_status.text())
        self.assertEqual(len(self.dialog._library['presets']), 2)
        self.assertEqual(self.live_items[0].font_size, 16)
        self.dialog.cancel_button.click()
        self.assertIsNone(self.dialog.library_result)
        self.assertEqual(len(self.library['presets']), 1)

    def test_invalid_file_and_failed_export_preserve_draft_and_show_errors(self):
        before = copy.deepcopy(self.dialog._library)
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'unrelated.json'
            path.write_text('{}', encoding='utf-8')
            with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(path), '')):
                self.dialog.import_button.click()
            self.assertEqual(self.dialog.library_status.text(), 'This file is not a lettering preset library.')
            with patch.object(QtWidgets.QFileDialog, 'exec', return_value=QtWidgets.QDialog.Accepted), \
                 patch.object(QtWidgets.QFileDialog, 'selectedFiles', return_value=[temp]):
                self.dialog.export_button.click()
            self.assertEqual(self.dialog.library_status.text(), 'Could not write the preset file.')
            self.assertEqual(path.read_text(encoding='utf-8'), '{}')
        self.assertEqual(self.dialog._library, before)
        self.assertIsNone(self.dialog.library_result)

    def test_empty_duplicate_names_and_no_selected_properties(self):
        self.dialog.name_edit.setText(' ')
        self.dialog.capture_button.click()
        self.assertEqual(len(self.dialog._library['presets']), 1)
        self.dialog.name_edit.setText('dialogue')
        self.dialog.capture_button.click()
        self.assertEqual(len(self.dialog._library['presets']), 1)
        for check in self.dialog.field_checks.values():
            check.setChecked(False)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertIn('at least one property', self.dialog.status_label.text())

    def test_six_themes_keep_outlined_controls_and_paired_preview_scale(self):
        old = dayu_theme.theme_name
        def luminance(color):
            value = QtGui.QColor(color)
            channels = (value.redF(), value.greenF(), value.blueF())
            linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in channels]
            return sum(a * b for a, b in zip(linear, (0.2126, 0.7152, 0.0722)))
        try:
            self.dialog.show()
            self.app.processEvents()
            for theme in ('light', 'dark', 'midnight', 'parchment', 'lavender', 'mint'):
                dayu_theme.set_theme(theme)
                self.dialog._apply_theme()
                self.dialog.fit_preview()
                self.assertIn(dayu_theme.primary_text_color, self.dialog.styleSheet())
                self.assertIn(dayu_theme.background_in_color, self.dialog.styleSheet())
                self.assertIn('border: 1px solid', self.dialog.styleSheet())
                foreground = luminance(dayu_theme.primary_text_color)
                selected_background = luminance(dayu_theme.background_out_color)
                contrast = (max(foreground, selected_background) + 0.05) / (min(foreground, selected_background) + 0.05)
                self.assertGreaterEqual(contrast, 4.5, theme)
                self.assertAlmostEqual(self.dialog.current_view.transform().m11(), self.dialog.styled_view.transform().m11(), places=2)
        finally:
            dayu_theme.set_theme(old)


if __name__ == '__main__':
    unittest.main()
