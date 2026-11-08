"""Real canvas integration for optional fitting; synthetic pixels, no models."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text.text_item_properties import TextItemProperties
from modules.rendering.auto_bubble_fit import fit_new_text_item
from modules.utils.textblock import TextBlock
from testmanual_text_box import _ManualTextMain


class AutomaticBubbleManualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.ctrl = self.main.text_ctrl
        self.main.loading = Mock()
        self.main.disable_hbutton_group = Mock()
        self.main.enable_hbutton_group = Mock()
        self.main.get_selected_page_paths = lambda: ['page.png']
        self.main.settings_page.get_max_font_size = lambda: 28
        self.main.settings_page.get_min_font_size = lambda: 10
        ui = self.main.settings_page.ui
        ui.max_font_spinbox.setValue(28)
        ui.min_font_spinbox.setValue(10)
        ui.auto_bubble_fit_checkbox = QtWidgets.QCheckBox()
        ui.auto_bubble_fit_checkbox.setChecked(True)
        ui.auto_bubble_padding_spinbox = QtWidgets.QDoubleSpinBox()
        ui.auto_bubble_padding_spinbox.setValue(8)
        self.main.blk_rendered = SimpleNamespace(emit=self.ctrl.on_blk_rendered)
        image = Image.new('RGB', (400, 260), (140, 95, 70))
        ImageDraw.Draw(image).ellipse((30, 30, 369, 229), fill=(215, 245, 250), outline='black', width=3)
        self.original = np.array(image)
        self.original.setflags(write=False)
        self.main.image_viewer.display_image_array(self.original.copy(), fit=False)
        self.main.image_data = {'page.png': self.original.copy()}
        self.block = TextBlock(text_bbox=np.array([130, 90, 270, 150]),
                               bubble_bbox=np.array([30, 30, 370, 230]),
                               text='Original OCR text', translation='Bonjour mon ami !')
        self.main.blk_list = [self.block]
        self.main.image_states['page.png']['blk_list'] = self.main.blk_list
        self.loader = patch('modules.rendering.auto_bubble_fit.load_original_for_fitting',
                            return_value=self.original).start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def test_render_fits_new_text_and_undo_redo_retains_fitted_layout(self):
        bounds = list(self.block.xyxy)
        before = self.main.image_data['page.png'].copy()
        self.ctrl.render_text()
        item = self.main.image_viewer.text_items[0]
        self.assertEqual(item.layout_preferences['mode'], 'bubble')
        self.assertEqual(item.toPlainText(), 'Bonjour mon ami !')
        self.assertEqual(list(self.block.xyxy), bounds)
        np.testing.assert_array_equal(self.main.image_data['page.png'], before)
        state = TextItemProperties.from_text_item(item).to_dict()
        stack = self.main.undo_group.activeStack()
        stack.undo()
        self.assertEqual(self.main.image_viewer.text_items, [])
        stack.redo()
        restored = self.main.image_viewer.text_items[0]
        self.assertEqual(TextItemProperties.from_text_item(restored).to_dict(), state)

    def test_repeat_render_preserves_moved_manual_item_and_rich_format(self):
        self.block.is_manual = True
        self.ctrl.render_text()
        item = self.main.image_viewer.text_items[0]
        item.setPos(11, 12)
        cursor = QtGui.QTextCursor(item.document())
        cursor.setPosition(7, QtGui.QTextCursor.MoveMode.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Weight.Bold)
        cursor.mergeCharFormat(fmt)
        before = TextItemProperties.from_text_item(item).to_dict()
        self.assertTrue(self.main.manual_workflow_ctrl._has_rendered_manual_translation(
            self.block, self.main.image_viewer.text_items))
        self.ctrl.render_text()
        self.assertEqual(len(self.main.image_viewer.text_items), 1)
        self.assertEqual(TextItemProperties.from_text_item(item).to_dict(), before)
        self.assertEqual(self.loader.call_count, 1)

    def test_disabled_does_not_read_original_or_add_bubble_preferences(self):
        self.main.settings_page.ui.auto_bubble_fit_checkbox.setChecked(False)
        self.ctrl.render_text()
        self.loader.assert_not_called()
        self.assertFalse(self.main.image_viewer.text_items[0].layout_preferences)

    def test_selected_pages_fit_only_missing_regions(self):
        existing = TextItemProperties(text='<b>Keep my lettering</b>', font_family='Arial',
                                      font_size=17, position=(5, 5), width=180,
                                      region_id=self.block.region_id).to_dict()
        self.main.image_states['page.png']['viewer_state'] = {'text_items_state': [copy.deepcopy(existing)]}
        second = self.block.deep_copy()
        self.main.image_files.append('second.png')
        self.main.image_states['second.png'] = {'blk_list': [second], 'target_lang': 'English'}
        self.main.get_selected_page_paths = lambda: self.main.image_files
        self.main.manual_workflow_ctrl._prepare_multi_page_context = Mock(return_value={
            'current_file': 'page.png', 'current_page_unloaded': False})
        self.ctrl.render_text()
        self.assertEqual(self.main.image_states['page.png']['viewer_state']['text_items_state'], [existing])
        rendered = self.main.image_states['second.png']['viewer_state']['text_items_state']
        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0]['layout_preferences']['mode'], 'bubble')
        self.loader.assert_called_once_with('second.png', None)

    def test_webtoon_prepared_item_matches_saved_layout_with_centered_page_origin(self):
        props = TextItemProperties(text=self.block.translation, font_family='Arial', font_size=18,
                                   text_color=QtGui.QColor('black'), position=(130, 90), width=140, height=60,
                                   region_id=self.block.region_id)
        prepared = fit_new_text_item(props, self.block, self.original, self.main.render_settings())
        self.main.webtoon_mode = True
        self.main.image_viewer.webtoon_manager.is_active = lambda: True
        self.main.image_viewer.hasPhoto = lambda: True
        self.ctrl._render_page_origin = Mock(return_value=QtCore.QPointF(50, 1000))
        emitted = self.block.deep_copy()
        emitted.xyxy = [130, 1090, 270, 1150]
        emitted.bubble_xyxy = [30, 1030, 370, 1230]
        emitted._render_text_properties = prepared.to_dict()
        emitted._render_page_origin = (0, 1000)
        self.ctrl.on_blk_rendered('legacy\nwrapping', 10, emitted, 'page.png')
        item = self.main.image_viewer.text_items[0]
        self.assertEqual(item.toPlainText(), self.block.translation)
        self.assertEqual(item.font_size, prepared.font_size)
        self.assertAlmostEqual(item.textWidth(), prepared.width)
        self.assertEqual((item.pos().x(), item.pos().y()),
                         (prepared.position[0] + 50, prepared.position[1] + 1000))
        self.assertEqual(emitted.xyxy, [180, 1090, 320, 1150])
        self.assertNotIn('_render_text_properties', emitted.__dict__)
        self.loader.assert_not_called()

    def test_manual_webtoon_render_uses_owner_page_and_both_scene_offsets(self):
        self.main.webtoon_mode = True
        self.main.image_viewer.webtoon_manager.is_active = lambda: True
        self.main.image_viewer.hasPhoto = lambda: True
        self.main.image_files.append('second.png')
        self.main.undo_stacks['second.png'] = QtGui.QUndoStack(self.main)
        self.main.image_viewer.scene_to_page_coordinates = Mock(return_value=(1, QtCore.QPointF(200, 120)))
        self.ctrl._render_page_origin = Mock(return_value=QtCore.QPointF(50, 1000))
        self.block.xyxy = [180, 1090, 320, 1150]
        self.block.bubble_xyxy = [80, 1030, 420, 1230]
        self.ctrl.render_text()
        item = self.main.image_viewer.text_items[0]
        self.assertEqual(item._page_path, 'second.png')
        self.assertEqual(item.layout_preferences['mode'], 'bubble')
        self.assertGreaterEqual(item.pos().x(), 80)
        self.assertGreaterEqual(item.pos().y(), 1030)
        self.loader.assert_called_once_with('second.png', None)


if __name__ == '__main__':
    unittest.main()
