"""A completed hand-drawn region survives later page-level processing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from modules.utils.textblock import TextBlock
from pipeline.block_detection import BlockDetectionHandler
from pipeline.cache_manager import CacheManager
from testmanual_text_box import _ManualTextMain


class ManualRegionRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.main.loading = Mock()
        self.main.disable_hbutton_group = Mock()
        self.main.enable_hbutton_group = Mock()
        self.main.get_selected_page_paths = lambda: ["page.png"]
        self.main.settings_page.get_max_font_size = lambda: 72
        self.main.settings_page.get_min_font_size = lambda: 4
        self.main.blk_rendered = SimpleNamespace(emit=self.main.text_ctrl.on_blk_rendered)
        self.main.image_viewer.command_emitted.connect(self.main.push_command)
        self.ctrl = self.main.manual_workflow_ctrl
        self.manual = TextBlock(
            text_bbox=np.array([10, 10, 65, 45]), text="Hello friend",
            translation="Bonjour mon ami", is_manual=True,
        )
        self.detected = TextBlock(
            text_bbox=np.array([75, 50, 135, 90]), text_class="text_bubble",
            text="Good night", translation="Bonne nuit",
        )
        self.main.blk_list = [self.manual, self.detected]
        self.main.image_states["page.png"]["blk_list"] = self.main.blk_list

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def styled_manual_item(self):
        item = self.main.text_ctrl.create_manual_text_item(self.manual, self.manual.translation)
        cursor = QtGui.QTextCursor(item.document())
        cursor.setPosition(0)
        cursor.setPosition(7, QtGui.QTextCursor.MoveMode.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Weight.Bold)
        fmt.setFontItalic(True)
        cursor.mergeCharFormat(fmt)
        return item

    def set_up_selected_pages(self):
        self.main.image_files.append("second.png")
        self.main.image_states["second.png"] = {
            "blk_list": [blk.deep_copy() for blk in self.main.blk_list],
        }
        self.main.get_selected_page_paths = lambda: list(self.main.image_files)
        self.main.image_data = {
            "page.png": np.full((100, 140, 3), 255, dtype=np.uint8),
            "second.png": np.full((100, 140, 3), 250, dtype=np.uint8),
        }
        self.main.s_combo = SimpleNamespace(currentText=lambda: "English")
        self.main.settings_page.get_tool_selection = lambda _tool: "Default"
        self.main.settings_page.is_gpu_enabled = lambda: False
        self.main.settings_page.get_llm_settings = lambda: {"extra_context": ""}
        self.main.pipeline.cache_manager = CacheManager()
        self.main.image_ctrl.save_current_image_state = Mock()

    def assert_selected_pages_preserve_manual_translation(self):
        for file_path in self.main.image_files:
            blocks = self.main.image_states[file_path]["blk_list"]
            self.assertEqual(len(blocks), 2)
            manual = next(blk for blk in blocks if blk.is_manual)
            self.assertEqual(manual.text, "Hello friend")
            self.assertEqual(manual.translation, "Bonjour mon ami")
            np.testing.assert_array_equal(manual.xyxy, [10, 10, 65, 45])

    def test_page_translation_keeps_manual_layer_style_and_case(self):
        item = self.styled_manual_item()
        before = item.document().toHtml()
        self.main.settings_page.ui.uppercase_checkbox.setChecked(True)

        self.ctrl.update_translated_text_items(False)

        self.assertEqual(item.document().toHtml(), before)
        self.assertEqual(self.manual.translation, "Bonjour mon ami")
        self.assertEqual(len(self.main.image_viewer.text_items), 2)
        self.assertIn(item, self.main.image_viewer._scene.items())
        self.assertEqual(self.detected.translation.replace("\n", " "), "BONNE NUIT")

    def test_missing_manual_layer_is_created_once_then_preserved(self):
        self.ctrl.update_translated_text_items(False)
        item = self.ctrl._find_text_item_for_block(self.manual, self.main.image_viewer.text_items)
        self.assertIsNotNone(item)
        before = item.document().toHtml()

        self.ctrl.update_translated_text_items(False)

        self.assertEqual(len(self.main.image_viewer.text_items), 2)
        self.assertEqual(item.document().toHtml(), before)
        self.assertIs(self.ctrl._find_text_item_for_block(self.manual, self.main.image_viewer.text_items), item)

    def test_explicit_single_region_translation_still_refreshes_manual_layer(self):
        item = self.styled_manual_item()
        self.main.curr_tblock = self.manual
        self.manual.translation = "Salut mon ami"

        self.ctrl.update_translated_text_items(True)

        self.assertIn("Salut mon ami", item.toPlainText().replace("\n", " "))
        self.assertEqual(len(self.main.image_viewer.text_items), 1)

    def test_segment_then_render_keeps_manual_item_and_does_not_duplicate(self):
        item = self.styled_manual_item()
        before = item.document().toHtml()
        self.main.text_ctrl.create_manual_text_item(self.detected, self.detected.translation)
        build = Mock(return_value=None)
        self.main.image_viewer.drawing_manager.make_segmentation_stroke_data = build

        self.ctrl.load_segmentation_points()

        self.assertEqual(self.main.image_viewer.text_items, [item])
        self.assertIs(build.call_args_list[0].args[0], self.manual)
        self.assertIs(build.call_args_list[1].args[0], self.detected)
        self.assertEqual(build.call_count, 2)
        self.assertIn(item, self.main.image_viewer._scene.items())
        self.main.settings_page.ui.uppercase_checkbox.setChecked(True)
        self.main.text_ctrl.render_text()
        self.main.text_ctrl.render_text()

        self.assertEqual(len(self.main.image_viewer.text_items), 2)
        self.assertEqual(item.document().toHtml(), before)
        self.assertEqual(self.manual.translation, "Bonjour mon ami")
        saved = self.main.image_viewer.save_state()["text_items_state"]
        self.assertEqual(len(saved), 2)

    def test_segmentation_does_not_skip_unrendered_manual_translation(self):
        build = Mock(return_value=None)
        self.main.image_viewer.drawing_manager.make_segmentation_stroke_data = build

        self.ctrl.load_segmentation_points()

        self.assertEqual(build.call_count, 2)
        self.assertIs(build.call_args_list[0].args[0], self.manual)

    def test_first_page_translation_layer_still_receives_initial_cleanup_mask(self):
        self.manual.translation = ""
        self.main.pipeline.translate_image = lambda _single: setattr(
            self.manual, "translation", "Bonjour mon ami",
        )

        with patch("app.controllers.manual_workflow.validate_translator", return_value=True):
            self.ctrl.translate_image()
        item = self.ctrl._find_text_item_for_block(self.manual, self.main.image_viewer.text_items)
        self.assertIsNotNone(item)
        build = Mock(return_value=None)
        self.main.image_viewer.drawing_manager.make_segmentation_stroke_data = build

        self.ctrl.load_segmentation_points()

        self.assertIs(build.call_args_list[0].args[0], self.manual)
        self.assertEqual(build.call_count, 2)
        self.assertIn(item, self.main.image_viewer.text_items)
        self.assertIn(item, self.main.image_viewer._scene.items())

    def test_render_uses_same_position_tolerance_as_manual_layer_matching(self):
        item = self.styled_manual_item()
        item.setPos(11.1, 10.2)
        item.setRotation(0.2)
        before = item.document().toHtml()

        self.main.text_ctrl.render_text()

        self.assertEqual(len(self.main.image_viewer.text_items), 2)
        self.assertEqual(item.document().toHtml(), before)
        self.assertEqual(item.pos(), QtCore.QPointF(11.1, 10.2))

    def test_detection_then_page_refresh_segmentation_render_preserves_manual(self):
        item = self.styled_manual_item()
        before = item.document().toHtml()
        self.main.s_combo = SimpleNamespace(currentText=lambda: "English")
        self.main.connect_rect_item_signals = self.main.rect_item_ctrl.connect_rect_item_signals
        duplicate = TextBlock(
            text_bbox=np.array([11, 10, 65, 46]), text_class="text_free",
        )
        detector = BlockDetectionHandler(self.main)

        detector.on_blk_detect_complete(([duplicate, self.detected], True, None))
        self.ctrl.update_translated_text_items(False)
        self.main.image_viewer.drawing_manager.make_segmentation_stroke_data = Mock(return_value=None)
        self.ctrl.load_segmentation_points()
        self.main.text_ctrl.render_text()

        self.assertEqual(len(self.main.blk_list), 2)
        self.assertEqual(len(self.main.image_viewer.text_items), 2)
        self.assertIn(item, self.main.image_viewer._scene.items())
        self.assertEqual(item.document().toHtml(), before)
        restored = next(blk for blk in self.main.blk_list if blk.is_manual)
        self.assertEqual(restored.text, "Hello friend")
        self.assertEqual(restored.translation, "Bonjour mon ami")

    def test_saved_page_segmentation_still_masks_rendered_manual_region(self):
        self.styled_manual_item()
        build = Mock(return_value={"stroke": "pending"})
        self.main.image_viewer.drawing_manager.make_segmentation_stroke_data = build

        strokes = self.ctrl._serialize_segmentation_strokes(self.main.blk_list)

        self.assertEqual(strokes, [{"stroke": "pending"}, {"stroke": "pending"}])
        self.assertIs(build.call_args_list[0].args[0], self.manual)
        self.assertIs(build.call_args_list[1].args[0], self.detected)

    def test_incomplete_manual_coordinates_do_not_break_segmentation_matching(self):
        saved = [{"position": (10, 10), "rotation": 0}]
        for coords in (None, [], [10, 10, 65]):
            self.manual.xyxy = coords
            self.assertFalse(self.ctrl._has_rendered_manual_translation(self.manual, saved))
            strokes = self.ctrl._serialize_segmentation_strokes([self.manual])
            self.assertEqual(strokes, [])

    def test_selected_page_detection_merges_manual_regions_on_each_page(self):
        self.set_up_selected_pages()
        duplicate = TextBlock(
            text_bbox=np.array([11, 10, 65, 46]), text_class="text_free",
        )
        detector = Mock()
        detector.detect.side_effect = lambda _image: [duplicate.deep_copy(), self.detected.deep_copy()]
        self.main.pipeline.block_detection = SimpleNamespace(
            block_detector_cache=detector, annotate_language_if_auto=Mock(),
        )
        self.main.pipeline.load_box_coords = Mock()

        with patch("app.controllers.manual_workflow.get_best_render_area"):
            self.ctrl.block_detect()

        self.assertEqual(detector.detect.call_count, 2)
        self.assert_selected_pages_preserve_manual_translation()
        self.main.pipeline.load_box_coords.assert_called_once()
        self.assertEqual(len(self.main.blk_list), 2)

    def test_selected_page_ocr_excludes_completed_manual_regions(self):
        self.set_up_selected_pages()
        ocr = Mock()

        def recognize(_image, blocks):
            self.assertEqual(len(blocks), 1)
            self.assertFalse(blocks[0].is_manual)
            blocks[0].text = "Recognized new region"

        ocr.process.side_effect = recognize
        with (
            patch("app.controllers.manual_workflow.validate_ocr", return_value=True),
            patch("app.controllers.manual_workflow.resolve_device", return_value="cpu"),
            patch("app.controllers.manual_workflow.OCRProcessor", return_value=ocr),
        ):
            self.ctrl.ocr()

        self.assertEqual(ocr.process.call_count, 2)
        self.assert_selected_pages_preserve_manual_translation()
        for state in self.main.image_states.values():
            detected = next(blk for blk in state["blk_list"] if not blk.is_manual)
            self.assertEqual(detected.text, "Recognized new region")

    def test_selected_page_translation_preserves_manual_and_its_rich_layer(self):
        self.set_up_selected_pages()
        item = self.styled_manual_item()
        before = item.document().toHtml()
        self.main.settings_page.ui.uppercase_checkbox.setChecked(True)
        translator = Mock()

        def translate(blocks, _image, _context):
            self.assertEqual(len(blocks), 1)
            self.assertFalse(blocks[0].is_manual)
            blocks[0].translation = "Nouvelle traduction"

        translator.translate.side_effect = translate
        with (
            patch("app.controllers.manual_workflow.validate_translator", return_value=True),
            patch("app.controllers.manual_workflow.Translator", return_value=translator),
        ):
            self.ctrl.translate_image()

        self.assertEqual(translator.translate.call_count, 2)
        self.assert_selected_pages_preserve_manual_translation()
        self.assertEqual(item.document().toHtml(), before)
        self.assertIn(item, self.main.image_viewer._scene.items())
        for state in self.main.image_states.values():
            detected = next(blk for blk in state["blk_list"] if not blk.is_manual)
            self.assertEqual(detected.translation.replace("\n", " "), "NOUVELLE TRADUCTION")

    def test_webtoon_segmentation_still_masks_rendered_manual_region(self):
        item = self.styled_manual_item()
        self.main.webtoon_mode = True
        build = Mock(return_value={"stroke": "pending"})
        self.main.image_viewer.drawing_manager.make_segmentation_stroke_data = build
        self.main.image_viewer.draw_segmentation_lines = Mock()
        self.main.undo_group.activeStack().beginMacro("segment")

        self.ctrl._on_segmentation_bboxes_ready([self.manual, self.detected])

        self.assertIs(build.call_args_list[0].args[0], self.manual)
        self.assertIs(build.call_args_list[1].args[0], self.detected)
        self.assertEqual(build.call_count, 2)
        self.assertIn(item, self.main.image_viewer._scene.items())


if __name__ == "__main__":
    unittest.main()
