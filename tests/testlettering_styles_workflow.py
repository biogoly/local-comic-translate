"""Project preset/application boundaries through actual text items and page undo."""
import copy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from testmanual_text_box import _ManualTextMain
from app.controllers.lettering_styles import LetteringStylesController
from app.controllers.lettering_style_format import styled_properties
from app.ui.dialogs.lettering_styles import LetteringStylesDialog
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.save_renderer import ImageSaveRenderer
from modules.utils.lettering_style_exchange import write_style_file
from modules.utils.lettering_styles import empty_style_library, normalize_style_library
from modules.utils.region_metadata import new_id
from modules.utils.textblock import TextBlock


class LetteringStyleWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        for button in self.main.alignment_tool_group.get_button_group().buttons():
            button.setCheckable(True)
        self.main.lettering_styles_button = QtWidgets.QPushButton()
        self.main._translation_project_epoch = 1
        self.main.image_states['page.png']['page_id'] = new_id()
        self.ctrl = LetteringStylesController(self.main)
        self.notices = []
        self.ctrl._notice = self.notices.append
        self.items = []
        for y, text in ((10, 'Bonjour mon ami!'), (50, 'Bonsoir mon amie!')):
            block = TextBlock(text_bbox=np.array([10, y, 130, y+40]), is_manual=True, translation=text)
            self.main.blk_list.append(block)
            item = self.main.text_ctrl.create_manual_text_item(block, text)
            item.region_id = block.region_id
            item._page_path = 'page.png'
            self.items.append(item)
        for item in self.items:
            item.selected = True
        cursor = QtGui.QTextCursor(self.items[0].document())
        cursor.setPosition(8)
        cursor.setPosition(11, QtGui.QTextCursor.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Bold)
        fmt.setFontItalic(True)
        fmt.setFontUnderline(True)
        cursor.mergeCharFormat(fmt)
        self.main.text_ctrl._commit_pending_text_command()
        self.stack = self.main.undo_stacks['page.png']
        self.library = {'schema_version': 1, 'presets': [{'id': new_id(), 'name': 'Dialogue',
            'properties': {'font_family': 'Arial', 'font_size': 18, 'text_color': '#123456', 'line_spacing': 1.4}}]}
        self.library = normalize_style_library(self.library)
        self.before = self.snapshots()

    def tearDown(self):
        self.ctrl.shutdown()
        self.main.deleteLater()
        self.app.processEvents()

    def snapshots(self):
        return [TextItemProperties.from_text_item(item).to_dict() for item in self.items]

    def run_editor(self, *, accept=True, during=None, library_only=False):
        def create(library, selections, parent=None):
            after = [styled_properties(props, self.library['presets'][0]['properties'],
                                       {'font_family', 'font_size', 'text_color', 'line_spacing', 'alignment'})
                     for props in selections]
            dialog = SimpleNamespace(library_result=copy.deepcopy(self.library),
                                     candidates=[] if library_only else after,
                                     deleteLater=lambda: None, reject=lambda: None)
            def execute():
                if during:
                    during()
                return QtWidgets.QDialog.Accepted if accept else QtWidgets.QDialog.Rejected
            dialog.exec = execute
            return dialog
        with patch('app.controllers.lettering_styles.LetteringStylesDialog', side_effect=create):
            self.ctrl.open_editor()

    def run_real_editor(self, modes=None):
        def create(library, selections, parent=None):
            dialog = LetteringStylesDialog(library, selections, parent=parent)
            def execute():
                for key, mode in (modes or {}).items():
                    combo = dialog.emphasis_combos[key]
                    combo.setCurrentIndex(combo.findData(mode))
                self.assertTrue(dialog.apply_button.isEnabled(), dialog.status_label.text())
                dialog.apply_selection()
                return dialog.result()
            dialog.exec = execute
            return dialog
        with patch('app.controllers.lettering_styles.LetteringStylesDialog', side_effect=create):
            self.ctrl.open_editor()

    @staticmethod
    def character_emphasis(item):
        values = []
        for position in range(len(item.toPlainText())):
            cursor = QtGui.QTextCursor(item.document())
            cursor.setPosition(position)
            cursor.setPosition(position + 1, QtGui.QTextCursor.KeepAnchor)
            font = cursor.charFormat().font()
            values.append({'bold': font.bold(), 'italic': font.italic(), 'underline': font.underline()})
        return values

    def test_real_dialog_emphasis_on_off_without_preset_and_exact_undo(self):
        original_emphasis = [self.character_emphasis(item) for item in self.items]
        baseline_index = self.stack.index()
        for field in ('bold', 'italic', 'underline'):
            for mode in ('on', 'off'):
                with self.subTest(field=field, mode=mode):
                    self.run_real_editor({field: mode})
                    self.assertEqual(self.stack.index(), baseline_index + 1)
                    self.assertEqual(self.main.lettering_styles, empty_style_library())
                    after = self.snapshots()
                    for item, original in zip(self.items, original_emphasis):
                        expected = [dict(values, **{field: mode == 'on'}) for values in original]
                        self.assertEqual(self.character_emphasis(item), expected)
                    for previous, current in zip(self.before, after):
                        for key in ('font_family', 'font_size', 'text_color', 'position', 'width'):
                            self.assertEqual(current[key], previous[key])
                    self.stack.undo()
                    self.assertEqual(self.snapshots(), self.before)
                    self.stack.redo()
                    self.assertEqual(self.snapshots(), after)
                    self.stack.undo()

    def test_word_toolbar_remains_selected_word_only_after_real_style_apply(self):
        self.main.lettering_styles = copy.deepcopy(self.library)
        original_emphasis = [self.character_emphasis(item) for item in self.items]
        self.run_real_editor()
        self.assertEqual([self.character_emphasis(item) for item in self.items], original_emphasis)
        styled = self.snapshots()
        item = self.items[0]
        self.items[1].selected = False
        self.main.text_ctrl.on_text_item_selected(item)
        item.enter_editing_mode()
        cursor = item.textCursor()
        cursor.setPosition(12)
        cursor.setPosition(15, QtGui.QTextCursor.KeepAnchor)
        item.setTextCursor(cursor)
        item.on_selection_changed()
        baseline_index = self.stack.index()
        self.main.bold_button.click()
        self.main.italic_button.click()
        self.main.underline_button.click()
        self.assertEqual(self.stack.index(), baseline_index + 3)
        expected = copy.deepcopy(original_emphasis[0])
        for position in range(12, 15):
            expected[position] = {'bold': True, 'italic': True, 'underline': True}
        self.assertEqual(self.character_emphasis(item), expected)
        self.assertEqual(self.character_emphasis(self.items[1]), original_emphasis[1])
        for _ in range(3):
            self.stack.undo()
        self.assertEqual(self.snapshots(), styled)
        self.stack.undo()
        self.assertEqual(self.snapshots(), self.before)

    def test_multi_selection_apply_and_undo_are_one_action(self):
        count = self.stack.count()
        self.run_editor()
        self.assertEqual(self.stack.count(), count+1)
        after = self.snapshots()
        for item, props in zip(self.items, self.before):
            self.assertEqual(item.font_size, 18)
            self.assertEqual(item.text_color.name(), '#123456')
            self.assertEqual((item.pos().x(), item.pos().y()), props['position'])
            self.assertEqual(item.textWidth(), props['width'])
        cursor = QtGui.QTextCursor(self.items[0].document())
        cursor.setPosition(9)
        self.assertTrue(cursor.charFormat().fontItalic())
        self.assertTrue(cursor.charFormat().fontUnderline())
        self.assertEqual(cursor.charFormat().fontWeight(), QtGui.QFont.Bold)
        self.stack.undo()
        self.assertEqual(self.snapshots(), self.before)
        self.assertEqual(self.main.lettering_styles, self.library)
        self.stack.redo()
        self.assertEqual(self.snapshots(), after)

    def test_cancel_never_changes_items_library_or_history(self):
        count = self.stack.count()
        self.run_editor(accept=False)
        self.assertEqual(self.snapshots(), self.before)
        self.assertEqual(self.main.lettering_styles, empty_style_library())
        self.assertEqual(self.stack.count(), count)

    def test_alignment_toolbar_tracks_style_apply_and_undo(self):
        buttons = self.main.alignment_tool_group.get_button_group().buttons()
        self.main.curr_tblock_item = self.items[0]
        for start, target, before_index, after_index in (
            (QtCore.Qt.AlignCenter, QtCore.Qt.AlignLeft, 1, 0),
            (QtCore.Qt.AlignLeft, QtCore.Qt.AlignHCenter, 0, 1),
            (QtCore.Qt.AlignCenter, QtCore.Qt.AlignRight, 1, 2),
        ):
            with self.subTest(start=start, target=target):
                self.items[0].set_alignment(start)
                self.main.text_ctrl.set_values_for_blk_item(self.items[0])
                self.assertTrue(buttons[before_index].isChecked())
                self.library['presets'][0]['properties']['alignment'] = int(target)
                self.run_editor()
                self.assertTrue(buttons[after_index].isChecked())
                self.stack.undo()
                self.assertTrue(buttons[before_index].isChecked())

    def test_highlight_toolbar_ignores_vertical_alignment_bits(self):
        buttons = self.main.alignment_tool_group.get_button_group().buttons()
        props = TextItemProperties.from_text_item(self.items[0]).to_dict()
        for alignment, index in ((QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, 0),
                                 (QtCore.Qt.AlignHCenter, 1),
                                 (QtCore.Qt.AlignRight | QtCore.Qt.AlignTop, 2)):
            props['alignment'] = alignment
            self.main.text_ctrl.set_values_from_highlight(props)
            self.assertTrue(buttons[index].isChecked())

    def test_library_only_save_leaves_lettering_history_intact(self):
        count = self.stack.count()
        self.run_editor(library_only=True)
        self.assertEqual(self.main.lettering_styles, self.library)
        self.assertEqual(self.snapshots(), self.before)
        self.assertEqual(self.stack.count(), count)
        self.assertTrue(self.main.dirty)

    def test_import_without_selected_text_only_persists_after_save(self):
        for item in self.items:
            item.selected = False
            item.setSelected(False)
        count = self.stack.count()
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'incoming.json'
            write_style_file(path, self.library)
            for save in (False, True):
                def execute(dialog):
                    self.assertFalse(dialog._selections)
                    self.assertFalse(dialog.export_button.isEnabled())
                    dialog.import_button.click()
                    self.assertTrue(dialog.export_button.isEnabled())
                    self.assertFalse(dialog.apply_button.isEnabled())
                    self.assertEqual(self.main.lettering_styles, empty_style_library())
                    button = dialog.save_button if save else dialog.cancel_button
                    button.click()
                    return dialog.result()
                with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(path), '')), \
                     patch.object(LetteringStylesDialog, 'exec', execute):
                    self.main.lettering_styles_button.click()
                self.assertEqual(len(self.main.lettering_styles['presets']), int(save))
                self.assertEqual(self.snapshots(), self.before)
                self.assertEqual(self.stack.count(), count)
        self.assertEqual(self.main.lettering_styles['presets'][0]['properties'],
                         self.library['presets'][0]['properties'])

    def test_project_change_during_preview_discards_everything(self):
        count = self.stack.count()
        self.run_editor(during=lambda: setattr(self.main, '_translation_project_epoch', 2))
        self.assertEqual(self.snapshots(), self.before)
        self.assertEqual(self.main.lettering_styles, empty_style_library())
        self.assertEqual(self.stack.count(), count)
        self.assertTrue(self.notices)

    def test_untracked_text_edit_during_preview_does_not_get_overwritten(self):
        self.run_editor(during=lambda: self.items[0].setHtml('Changed while preview open'))
        self.assertEqual(self.items[0].toPlainText(), 'Changed while preview open')
        self.assertEqual(TextItemProperties.from_text_item(self.items[1]).to_dict(), self.before[1])
        self.assertEqual(self.main.lettering_styles, empty_style_library())
        self.assertTrue(self.notices)

    def test_offpage_undo_updates_saved_state_without_touching_other_page(self):
        self.run_editor()
        self.main.image_files.append('second.png')
        self.main.image_states['second.png'] = {'page_id': new_id()}
        self.main.curr_img_idx = 1
        for item in self.items:
            self.main.image_viewer._scene.removeItem(item)
        self.main.image_viewer.text_items.clear()
        self.stack.undo()
        saved = self.main.image_states['page.png']['viewer_state']['text_items_state']
        by_id = {entry['region_id']: entry for entry in saved}
        for entry in self.before:
            self.assertEqual(by_id[entry['region_id']], entry)
        self.assertNotIn('viewer_state', self.main.image_states['second.png'])

    def test_reloaded_items_and_export_keep_style_and_undo(self):
        self.run_editor()
        after = self.snapshots()
        saved = copy.deepcopy(self.main.image_states['page.png']['viewer_state']['text_items_state'])
        for item in self.items:
            self.main.image_viewer._scene.removeItem(item)
        self.main.image_viewer.text_items.clear()
        self.items = []
        for props in saved:
            item = self.main.image_viewer.add_text_item(TextItemProperties.from_dict(props))
            item._page_path = 'page.png'
            self.items.append(item)
        self.assertEqual([item.toHtml() for item in self.items], [p['text'] for p in after])
        renderer = ImageSaveRenderer(np.full((160, 160, 3), 255, np.uint8))
        renderer.add_state_to_image({'text_items_state': saved})
        rendered = [item for item in renderer.scene.items() if hasattr(item, 'toHtml')]
        self.assertEqual(sorted(item.toHtml() for item in rendered), sorted(p['text'] for p in after))
        self.stack.undo()
        self.assertEqual(self.snapshots(), self.before)

    def test_busy_workflow_does_not_open_editor(self):
        self.main._batch_active = True
        with patch('app.controllers.lettering_styles.LetteringStylesDialog') as dialog:
            self.ctrl.open_editor()
            dialog.assert_not_called()
        self.assertTrue(self.notices)


if __name__ == '__main__':
    unittest.main()
