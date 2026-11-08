"""Independent protections on the existing manual page workflow."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PySide6 import QtWidgets
from modules.utils.textblock import TextBlock
from testmanual_text_box import _ManualTextMain


class ReviewManualWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.main.loading = Mock()
        self.main.disable_hbutton_group = Mock()
        self.main.enable_hbutton_group = Mock()
        self.main.get_selected_page_paths = lambda: ['page.png']
        self.main.settings_page.get_min_font_size = lambda: 4
        self.main.settings_page.get_max_font_size = lambda: 30
        self.main.blk_rendered = SimpleNamespace(emit=self.main.text_ctrl.on_blk_rendered)
        self.block = TextBlock(text_bbox=np.array([10, 10, 130, 80]), text_class='text_bubble',
                               text='Bonjour', translation='Keep this Casing')
        self.main.blk_list = [self.block]
        self.main.image_states['page.png']['blk_list'] = self.main.blk_list

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def test_render_kept_translation_preserves_casing_with_uppercase_enabled(self):
        self.block.region_metadata['protections']['keep_translation'] = True
        self.main.settings_page.ui.uppercase_checkbox.setChecked(True)
        self.main.text_ctrl.render_text()
        self.assertEqual(self.block.translation, 'Keep this Casing')
        self.assertEqual(self.main.image_viewer.text_items[0].toPlainText().replace('\n', ' '), 'Keep this Casing')

    def test_keep_layout_also_preserves_intentionally_absent_lettering(self):
        self.block.region_metadata['protections']['keep_layout'] = True
        self.main.text_ctrl.render_text()
        self.assertEqual(self.main.image_viewer.text_items, [])
        self.assertEqual(self.block.translation, 'Keep this Casing')

    def test_segmentation_keeps_protected_automatic_lettering(self):
        item = self.main.text_ctrl.create_manual_text_item(self.block, self.block.translation)
        self.block.is_manual = False
        self.block.region_metadata['protections']['keep_layout'] = True
        item.setPos(41, 51)
        before = item.toHtml()
        self.main.manual_workflow_ctrl._clear_text_items_for_segmentation()
        self.assertIn(item, self.main.image_viewer.text_items)
        self.assertEqual(item.toHtml(), before)
        self.assertEqual((item.pos().x(), item.pos().y()), (41, 51))

    def test_segmentation_omits_skip_auto_cleanup_without_hiding_translation(self):
        self.block.region_metadata['protections']['skip_auto_cleanup'] = True
        build = Mock(return_value={'stroke': 'data'})
        self.main.image_viewer.drawing_manager.make_segmentation_stroke_data = build
        result = self.main.manual_workflow_ctrl._serialize_segmentation_strokes([self.block])
        self.assertEqual(result, [])
        build.assert_not_called()
        self.main.text_ctrl.render_text()
        self.assertEqual(len(self.main.image_viewer.text_items), 1)

    def test_page_translation_refresh_keeps_protected_layout_exact(self):
        item = self.main.text_ctrl.create_manual_text_item(self.block, self.block.translation)
        self.block.is_manual = False
        self.block.region_metadata['protections']['keep_layout'] = True
        item.setPos(35, 40)
        before = item.toHtml()
        self.block.translation = 'New target wording'
        self.main.manual_workflow_ctrl.update_translated_text_items(False)
        self.assertEqual(item.toHtml(), before)
        self.assertEqual(self.block.translation, 'New target wording')

    def test_page_translation_refresh_does_not_recase_kept_target(self):
        self.block.region_metadata['protections']['keep_translation'] = True
        self.main.settings_page.ui.uppercase_checkbox.setChecked(True)
        self.main.manual_workflow_ctrl.update_translated_text_items(False)
        self.assertEqual(self.block.translation, 'Keep this Casing')


if __name__ == '__main__':
    unittest.main()
