"""Context-menu Translate leaves artwork and lettering for Clean and Render."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from modules.utils.textblock import TextBlock
from testmanual_text_box import _ManualTextMain


class SelectedTranslationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.original = np.random.default_rng(7).integers(0, 256, (100, 140, 3), dtype=np.uint8)
        self.main.image_viewer.display_image_array(self.original.copy(), fit=False)
        self.main.loading = Mock()
        self.main.disable_hbutton_group = Mock()
        self.main.enable_hbutton_group = Mock()
        self.main.get_selected_page_paths = lambda: ['page.png']
        self.ctrl = self.main.manual_workflow_ctrl
        self.main.translate_image = self.ctrl.translate_image
        self.main.review_ctrl = SimpleNamespace(preview_replacement=Mock(return_value=False))
        self.block = TextBlock(text_bbox=np.array([20, 20, 115, 70]), text='Bonjour')
        self.main.blk_list.append(self.block)
        self.rect = self.main.image_viewer.add_rectangle(
            QtCore.QRectF(0, 0, 95, 50), QtCore.QPointF(20, 20))
        self.main.rect_item_ctrl.connect_rect_item_signals(self.rect)
        self.main.image_viewer.select_rectangle(self.rect)
        self.job = SimpleNamespace(path='page.png', run=Mock(), commit=Mock(side_effect=self.commit))
        self.job.run.return_value = self.job
        self.main.pipeline.translation_handler = SimpleNamespace(prepare_job=Mock(return_value=self.job))
        self.main.pipeline.inpainting = Mock()

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def commit(self):
        self.block.translation = 'Hello friend'
        return True

    def translate(self):
        with patch('app.controllers.manual_workflow.validate_translator', return_value=True):
            self.rect.signals.translate_block.emit()
        self.app.processEvents()

    def assert_artwork_unchanged(self):
        np.testing.assert_array_equal(self.main.image_viewer.get_image_array(), self.original)
        self.assertEqual(self.main.applied_patches, [])
        self.assertEqual(self.main.pipeline.inpainting.mock_calls, [])

    def leave_target_editor(self):
        QtWidgets.QApplication.sendEvent(self.main.t_text_edit,
            QtGui.QFocusEvent(QtCore.QEvent.Type.FocusOut))
        self.app.processEvents()

    def test_context_menu_translation_changes_only_target_for_all_region_types(self):
        for text_class, manual in [('text_bubble', False), ('text_free', False), (None, True)]:
            with self.subTest(text_class=text_class, manual=manual):
                self.block.text_class, self.block.is_manual = text_class, manual
                self.translate()
                self.assertEqual(self.block.translation, 'Hello friend')
                self.assertEqual(self.main.t_text_edit.toPlainText(), 'Hello friend')
                self.assertEqual(self.main.image_viewer.text_items, [])
                self.assertEqual(self.block.text_class, text_class)
                self.assert_artwork_unchanged()
        self.main.pipeline.translation_handler.prepare_job.assert_called_with(True, visible=False)
        self.main.review_ctrl.preview_replacement.assert_called_with('translation', update_lettering=False)
        self.assertTrue(self.main.dirty)

    def test_unchanged_target_focus_out_does_not_create_lettering(self):
        self.translate()
        self.leave_target_editor()
        self.assertEqual(self.main.image_viewer.text_items, [])
        self.assertEqual(self.main.undo_stacks['page.png'].count(), 0)
        self.assert_artwork_unchanged()

    def test_existing_rich_lettering_is_unchanged_including_target_focus_out(self):
        item = self.main.text_ctrl.create_manual_text_item(self.block, 'Old lettering')
        cursor = QtGui.QTextCursor(item.document())
        cursor.setPosition(0)
        cursor.setPosition(3, QtGui.QTextCursor.MoveMode.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Weight.Bold)
        cursor.mergeCharFormat(fmt)
        self.main.image_viewer.select_rectangle(self.rect)
        before = item.toHtml()
        undo_count = self.main.undo_stacks['page.png'].count()
        self.translate()
        self.leave_target_editor()
        self.assertEqual(item.toHtml(), before)
        self.assertEqual(self.main.image_viewer.text_items, [item])
        self.assertEqual(self.main.undo_stacks['page.png'].count(), undo_count)
        self.assertEqual(self.main.t_text_edit.toPlainText(), 'Hello friend')
        self.assert_artwork_unchanged()

    def test_webtoon_selected_translation_uses_same_non_rendering_completion(self):
        self.main.webtoon_mode = True
        self.translate()
        self.main.pipeline.translation_handler.prepare_job.assert_called_once_with(True, visible=True)
        self.assertEqual(self.main.t_text_edit.toPlainText(), 'Hello friend')
        self.assertEqual(self.main.image_viewer.text_items, [])
        self.assert_artwork_unchanged()

    def test_selecting_older_lettering_does_not_revert_new_translation_on_focus_out(self):
        item = self.main.text_ctrl.create_manual_text_item(self.block, 'Old lettering')
        self.main.image_viewer.select_rectangle(self.rect)
        self.translate()
        self.main.text_ctrl.on_text_item_selected(item)
        self.assertEqual(self.main.t_text_edit.toPlainText(), 'Old lettering')
        self.leave_target_editor()
        self.assertEqual(self.block.translation, 'Hello friend')
        self.assertEqual(item.toPlainText(), 'Old lettering')
        self.assert_artwork_unchanged()

    def test_missed_ime_edit_still_commits_after_selecting_existing_lettering(self):
        item = self.main.text_ctrl.create_manual_text_item(self.block, 'Old lettering')
        self.main.image_viewer.select_rectangle(self.rect)
        self.translate()
        self.main.text_ctrl.on_text_item_selected(item)
        with QtCore.QSignalBlocker(self.main.t_text_edit):
            self.main.t_text_edit.setPlainText('日本語の手入力')
        self.leave_target_editor()
        self.assertEqual(self.block.translation, '日本語の手入力')
        self.assertEqual(item.toPlainText(), '日本語の手入力')

    def test_failure_preserves_prior_target_and_artwork(self):
        self.block.translation = 'Prior target'
        self.main.image_viewer.select_rectangle(self.rect)
        self.job.run.side_effect = RuntimeError('Translation unavailable')
        self.main.default_error_handler = Mock()
        self.translate()
        self.job.commit.assert_not_called()
        self.assertEqual(self.block.translation, 'Prior target')
        self.assertEqual(self.main.t_text_edit.toPlainText(), 'Prior target')
        self.assertEqual(self.main.image_viewer.text_items, [])
        self.assert_artwork_unchanged()

    def test_explicit_segment_and_render_remain_available_after_translation(self):
        self.translate()
        segment = Mock(return_value=None)
        self.main.image_viewer.drawing_manager.make_segmentation_stroke_data = segment
        self.ctrl.load_segmentation_points()
        self.assertIs(segment.call_args.args[0], self.block)
        self.assertEqual(self.main.image_viewer.text_items, [])
        self.assert_artwork_unchanged()

        self.main.settings_page.get_max_font_size = lambda: 30
        self.main.settings_page.get_min_font_size = lambda: 4
        self.main.blk_rendered = SimpleNamespace(emit=self.main.text_ctrl.on_blk_rendered)
        self.main.text_ctrl.render_text()
        self.assertEqual(len(self.main.image_viewer.text_items), 1)
        self.assertEqual(self.main.image_viewer.text_items[0].toPlainText().replace('\n', ' '), 'Hello friend')


if __name__ == '__main__':
    unittest.main()
