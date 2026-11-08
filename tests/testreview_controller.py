"""Review integration with real Qt text items, synthetic pages and mocked models."""
import copy
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.review import ReviewController
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.review_panel import ReviewPanel
from modules.utils.region_metadata import new_id
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.ocr_handler import OCRHandler
from testmanual_text_box import _ManualTextMain


def utf16(text):
    return len(text.encode("utf-16-le")) // 2


class ReviewControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.main.s_combo = QtWidgets.QComboBox()
        self.main.s_combo.addItem("French")
        self.main.lang_mapping["French"] = "French"
        self.main.page_list = QtWidgets.QListWidget()
        self.main.page_list.addItem("page.png")
        self.main.page_list.setCurrentRow(0)
        self.main.image_states["page.png"] = {
            "page_id": new_id(), "source_lang": "French", "target_lang": "English",
            "blk_list": [], "viewer_state": {"text_items_state": []},
        }
        self.main.blk_list = []
        self.main.image_data = {"page.png": np.full((100, 140, 3), 255, dtype=np.uint8)}
        self.main.image_patches = {}
        self.main.review_panel = ReviewPanel(self.main)
        self.main.batch_report_ctrl.show_latest_batch_report = Mock()
        self.main.batch_report_ctrl._latest_batch_report = None
        self.main.show_main_page = Mock()
        self.main.settings_page.get_llm_settings = lambda: {"extra_context": ""}
        self.main.page_list.currentRowChanged.connect(self.load_page)
        self.ctrl = ReviewController(self.main)
        self.main.review_ctrl = self.ctrl
        self.main.review_panel.set_expanded(True)
        self.ctrl.timer.stop()

    def tearDown(self):
        self.ctrl.shutdown()
        self.main.text_ctrl._text_change_timer.stop()
        self.main.text_ctrl.fit_controller.timer.stop()
        self.main.deleteLater()
        self.app.processEvents()

    def add_page(self, path):
        self.main.image_files.append(path)
        self.main.page_list.addItem(path)
        self.main.image_states[path] = {
            "page_id": new_id(), "source_lang": "French", "target_lang": "English",
            "blk_list": [], "viewer_state": {"text_items_state": []},
        }
        self.main.image_data[path] = self.main.image_data["page.png"].copy()
        stack = QtGui.QUndoStack(self.main)
        self.main.undo_group.addStack(stack)
        self.main.undo_stacks[path] = stack

    def add_region(self, path="page.png", *, text="Bonjour", target="Hello", rendered=False, html=None):
        block = TextBlock(text_bbox=np.array([20, 20, 120, 75]), text=text,
                          translation=target, text_class="text_bubble", source_lang="fr", target_lang="en")
        self.main.image_states[path]["blk_list"].append(block.deep_copy())
        if path == self.main.image_files[self.main.curr_img_idx]:
            self.main.blk_list.append(block)
        item = None
        if rendered:
            props = TextItemProperties(text=html or target, region_id=block.region_id,
                font_family="Arial", font_size=16, text_color=QtGui.QColor("black"),
                position=(20, 20), width=100, height=55, line_spacing=1.0,
                layout_preferences={"bound_width": 100, "bound_height": 55, "min_font_size": 4})
            if path == self.main.image_files[self.main.curr_img_idx]:
                item = self.main.image_viewer.add_text_item(props)
                item._page_path = path
                data = TextItemProperties.from_text_item(item).to_dict()
            else:
                data = props.to_dict()
            self.main.image_states[path]["viewer_state"]["text_items_state"].append(data)
        return block, item

    def load_page(self, index):
        if not 0 <= index < len(self.main.image_files):
            return
        viewer = self.main.image_viewer
        for item in list(viewer.text_items):
            viewer._scene.removeItem(item)
        viewer.text_items.clear()
        for rect in list(viewer.rectangles):
            viewer._scene.removeItem(rect)
        viewer.rectangles.clear()
        self.main.curr_img_idx = index
        path = self.main.image_files[index]
        self.main.blk_list = [block.deep_copy() for block in self.main.image_states[path]["blk_list"]]
        self.main.curr_tblock = None
        self.main.curr_tblock_item = None
        self.main.undo_group.setActiveStack(self.main.undo_stacks[path])
        for props in self.main.image_states[path]["viewer_state"]["text_items_state"]:
            item = viewer.add_text_item(props)
            item._page_path = path

    def scan(self):
        self.ctrl.timer.stop()
        self.ctrl.refresh()
        self.ctrl.scan_timer.stop()
        while self.ctrl._jobs:
            self.ctrl._scan_slice()
            self.ctrl.scan_timer.stop()
        return self.ctrl._all_issues

    def key(self, block, path="page.png"):
        return self.main.image_states[path]["page_id"], block.region_id

    def select(self, block, item=None):
        self.main.curr_tblock = block
        self.main.curr_tblock_item = item
        if item is not None:
            self.main.text_ctrl.on_text_item_selected(item)

    def test_loaded_page_snapshot_uses_live_item_deletions(self):
        block, item = self.add_region(rendered=True)
        self.main.image_viewer._scene.removeItem(item)
        self.main.image_viewer.text_items.remove(item)
        snapshot = self.ctrl.snapshot()[self.key(block)]
        self.assertIsNone(snapshot["props"])
        self.assertTrue(self.main.image_states["page.png"]["viewer_state"]["text_items_state"])

    def test_uncertain_filter_updates_for_changed_score_and_source_correction(self):
        from modules.ocr.evidence import make_ocr_evidence
        block, _ = self.add_region()
        self.select(block)
        panel = self.main.review_panel
        panel.filter_combo.setCurrentIndex(panel.filter_combo.findData('uncertain_ocr'))
        self.scan()
        self.assertEqual(panel.issue_list.count(), 0)
        self.assertTrue(panel.ocr_score_label.isHidden())
        block.ocr_evidence = make_ocr_evidence(block.text, 'PP-OCR', 'ppocr-v5-rec-latin-mobile',
                                               line_scores=[0.61, 0.93])
        self.scan()
        self.assertEqual(panel.issue_list.count(), 1)
        self.assertIn('ppocr-v5-rec-latin-mobile', panel.issue_list.item(0).toolTip())
        self.assertFalse(panel.ocr_score_label.isHidden())
        self.assertTrue(panel.retry_ocr_button.isEnabled())
        block.text = 'Corrected source'
        self.scan()
        self.assertEqual(panel.issue_list.count(), 0)
        self.assertTrue(panel.ocr_score_label.isHidden())

    def test_loaded_page_snapshot_does_not_resurrect_deleted_regions(self):
        block, _ = self.add_region()
        self.add_page("other.png")
        other, _ = self.add_region("other.png")
        self.main.blk_list.remove(block)
        snapshot = self.ctrl.snapshot()
        self.assertNotIn(self.key(block), snapshot)
        self.assertIn(self.key(other, "other.png"), snapshot)

    def test_webtoon_snapshot_is_page_local_with_stable_ids(self):
        block, item = self.add_region(rendered=True)
        viewer = self.main.image_viewer
        self.main.webtoon_mode = True
        manager = viewer.webtoon_manager
        manager.layout_manager.image_positions = [1000]
        manager.layout_manager.image_heights = [100]
        manager.layout_manager.webtoon_width = 340
        manager.image_loader.image_data = {0: self.main.image_data["page.png"]}
        manager.image_loader.loaded_pages = {0}
        converter = manager.coordinate_converter
        viewer.scene_to_page_coordinates = lambda point: (0, converter.scene_to_page_local_position(point, 0))
        block.xyxy += np.array([100, 1000, 100, 1000])
        item.setPos(120, 1020)
        snapshot = self.ctrl.snapshot()[self.key(block)]
        np.testing.assert_array_equal(snapshot["block"].xyxy, [20, 20, 120, 75])
        self.assertEqual(snapshot["props"]["position"], (20, 20))
        self.assertEqual(snapshot["props"]["region_id"], block.region_id)

    def test_loaded_webtoon_snapshot_does_not_resurrect_deleted_regions(self):
        block, _ = self.add_region()
        viewer = self.main.image_viewer
        self.main.webtoon_mode = True
        viewer.webtoon_manager.layout_manager.image_positions = [1000]
        viewer.webtoon_manager.image_loader.loaded_pages = {0}
        self.main.blk_list.clear()
        self.assertNotIn(self.key(block), self.ctrl.snapshot())

    def test_missing_text_navigation_uses_page_identity_after_reorder(self):
        self.add_region()
        self.add_page("other.png")
        target, _ = self.add_region("other.png", text="", target="")
        issue = next(issue for issue in self.scan() if issue.kind == "missing_ocr")
        self.main.image_files.reverse()
        self.main.curr_img_idx = 1
        with QtCore.QSignalBlocker(self.main.page_list):
            self.main.page_list.clear()
            self.main.page_list.addItems(self.main.image_files)
            self.main.page_list.setCurrentRow(1)
        self.ctrl.open_issue(issue.key)
        self.ctrl.navigation_timer.stop()
        self.ctrl._finish_navigation()
        self.assertEqual(self.main.curr_img_idx, 0)
        self.assertEqual(self.main.curr_tblock.region_id, target.region_id)
        self.assertIsNotNone(self.main.image_viewer.selected_rect)
        self.assertEqual(self.main.image_viewer.selected_rect.pos(), QtCore.QPointF(20, 20))

    def test_protection_command_undo_redo_targets_original_page_after_navigation(self):
        original, _ = self.add_region()
        self.select(original)
        self.ctrl.set_protection("keep_source", True)
        self.assertTrue(original.region_metadata["protections"]["keep_source"])
        self.add_page("other.png")
        other, _ = self.add_region("other.png")
        self.load_page(1)
        self.main.undo_stacks["page.png"].undo()
        stored = self.main.image_states["page.png"]["blk_list"][0]
        self.assertFalse(stored.region_metadata["protections"]["keep_source"])
        self.assertFalse(self.main.blk_list[0].region_metadata["protections"]["keep_source"])
        self.assertEqual(self.main.blk_list[0].region_id, other.region_id)
        self.main.undo_stacks["page.png"].redo()
        self.assertTrue(stored.region_metadata["protections"]["keep_source"])
        self.assertFalse(self.main.blk_list[0].region_metadata["protections"]["keep_source"])

    def test_translation_approval_becomes_stale_after_direct_target_edit(self):
        block, item = self.add_region(rendered=True)
        self.select(block, item)
        self.ctrl.mark_current_reviewed("translation")
        self.assertFalse(any(i.stage == "translation" for i in self.scan()))
        self.main.t_text_edit.setPlainText("A new translation")
        self.main.text_ctrl._commit_pending_text_command()
        self.assertEqual(block.translation, "A new translation")
        self.assertTrue(any(i.stage == "translation" and i.kind == "awaiting_review" for i in self.scan()))

    def test_incremental_cache_measures_only_changed_layouts(self):
        block, item = self.add_region(rendered=True)
        with patch.object(self.ctrl, "_diagnostics", wraps=self.ctrl._diagnostics) as diagnostics:
            self.scan()
            self.assertEqual(diagnostics.call_count, 1)
            self.scan()
            self.assertEqual(diagnostics.call_count, 1)
            item.setPos(24, 25)
            self.scan()
            self.assertEqual(diagnostics.call_count, 2)

    def test_unopened_bubble_page_does_not_load_original_pixels(self):
        self.add_page("other.png")
        block, _ = self.add_region("other.png", rendered=True)
        saved = self.main.image_states["other.png"]["viewer_state"]["text_items_state"][0]
        saved["layout_preferences"] = {"mode": "bubble", "boundary_shape": "auto", "padding": 8}
        with patch("modules.rendering.auto_bubble_fit.load_original_for_fitting", side_effect=AssertionError("unexpected image read")) as load:
            issues = self.scan()
        load.assert_not_called()
        layout = next(issue for issue in issues if issue.region_id == block.region_id and issue.stage == "layout")
        self.assertEqual(layout.kind, "awaiting_review")
        self.assertEqual(layout.detail, "diagnostics_unknown")

    def test_cleanup_review_tracks_intersecting_inpaint_patches_only(self):
        block, _ = self.add_region()
        self.main.image_patches["page.png"] = [
            {"kind": "artistic_edit", "bbox": (20, 20, 20, 20), "hash": "art"},
            {"kind": "inpaint", "bbox": (500, 500, 20, 20), "hash": "outside"},
        ]
        self.assertFalse(any(issue.stage == "cleanup" for issue in self.scan()))
        patch_info = {"kind": "inpaint", "bbox": (30, 30, 30, 20), "patch_id": "a", "hash": "one"}
        self.main.image_patches["page.png"].append(patch_info)
        first = next(issue for issue in self.scan() if issue.stage == "cleanup")
        patch_info["hash"] = "two"
        second = next(issue for issue in self.scan() if issue.stage == "cleanup")
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.main.image_patches["page.png"].pop()
        self.assertFalse(any(issue.stage == "cleanup" for issue in self.scan()))

    def test_replacement_preserves_unicode_rich_styles_geometry_protection_and_undo(self):
        html = '<p><span style="font-weight:700">Hi</span> 🙂 old <span style="font-style:italic">世界</span></p>'
        block, item = self.add_region(rendered=True, html=html, target="Hi 🙂 old 世界")
        self.select(block, item)
        self.ctrl.set_protection("keep_translation", True)
        item.setPos(24.5, 26.5)
        item.setRotation(12)
        item.setScale(1.1)
        before = TextItemProperties.from_text_item(item).to_dict()
        key = self.key(block)
        snap = self.ctrl.snapshot()[key]
        replacement = "Hi 🙂 a longer new 世界"
        self.ctrl.apply_replacement(key, "translation", replacement, snap)
        self.assertEqual(item.toPlainText(), replacement)
        self.assertEqual(block.translation, replacement)
        self.assertTrue(block.region_metadata["protections"]["keep_translation"])
        for key_name in ("position", "rotation", "scale", "width", "font_family", "font_size"):
            self.assertEqual(TextItemProperties.from_text_item(item).to_dict()[key_name], before[key_name])
        cursor = QtGui.QTextCursor(item.document())
        cursor.setPosition(0)
        cursor.setPosition(2, QtGui.QTextCursor.MoveMode.KeepAnchor)
        self.assertGreaterEqual(cursor.charFormat().fontWeight(), QtGui.QFont.Weight.Bold)
        cursor.setPosition(utf16(replacement) - 2)
        cursor.setPosition(utf16(replacement), QtGui.QTextCursor.MoveMode.KeepAnchor)
        self.assertTrue(cursor.charFormat().fontItalic())
        self.main.undo_stacks["page.png"].undo()
        self.assertEqual(item.toPlainText(), "Hi 🙂 old 世界")
        self.assertEqual(item.toHtml(), before["text"])
        self.main.undo_stacks["page.png"].redo()
        self.assertEqual(item.toPlainText(), replacement)
        self.assertTrue(block.region_metadata["protections"]["keep_translation"])

    def test_replacement_full_document_outline_tracks_new_utf16_length(self):
        block, item = self.add_region(rendered=True, target="Hi 🙂 old")
        item.selection_outlines = [OutlineInfo(0, utf16(item.toPlainText()), QtGui.QColor("white"), 1.0, OutlineType.Full_Document)]
        self.select(block, item)
        replacement = "Hi 🙂 a considerably longer 日本語 replacement"
        self.ctrl.apply_replacement(self.key(block), "translation", replacement, self.ctrl.snapshot()[self.key(block)])
        self.assertEqual(item.selection_outlines[0].end, utf16(replacement))
        self.main.undo_stacks["page.png"].undo()
        self.assertEqual(item.selection_outlines[0].end, utf16("Hi 🙂 old"))

    def test_partial_outline_tracks_unicode_suffix_after_replacement_and_undo(self):
        original = "Hi \U0001f642 old \u4e16\u754c"
        replacement = "Hi \U0001f642 a longer new \u4e16\u754c"
        block, item = self.add_region(target=original, rendered=True)
        before_start = utf16(original[:-2])
        item.selection_outlines = [OutlineInfo(before_start, utf16(original), QtGui.QColor("white"),
                                               1.0, OutlineType.Selection)]
        self.ctrl.apply_replacement(self.key(block), "translation", replacement, self.ctrl.snapshot()[self.key(block)])
        self.assertEqual((item.selection_outlines[0].start, item.selection_outlines[0].end),
                         (utf16(replacement[:-2]), utf16(replacement)))
        self.main.undo_stacks["page.png"].undo()
        self.assertEqual((item.selection_outlines[0].start, item.selection_outlines[0].end),
                         (before_start, utf16(original)))

    def test_insertion_before_partial_outline_preserves_outlined_suffix_only(self):
        block, item = self.add_region(target="A tail", rendered=True)
        item.selection_outlines = [OutlineInfo(2, 6, QtGui.QColor("white"), 1.0, OutlineType.Selection)]
        self.ctrl.apply_replacement(self.key(block), "translation", "A new tail", self.ctrl.snapshot()[self.key(block)])
        self.assertEqual((item.selection_outlines[0].start, item.selection_outlines[0].end), (6, 10))

    def test_replacement_undo_redo_updates_unloaded_page_and_survives_reload(self):
        block, item = self.add_region(target="Original \U0001f642", rendered=True)
        self.select(block, item)
        self.ctrl.set_protection("keep_translation", True)
        before = TextItemProperties.from_text_item(item).to_dict()
        replacement = "Replacement \u65e5\u672c\u8a9e"
        self.ctrl.apply_replacement(self.key(block), "translation", replacement, self.ctrl.snapshot()[self.key(block)])
        self.add_page("second.png")
        second, _ = self.add_region("second.png", target="Unchanged", rendered=True)
        self.load_page(1)
        self.main.undo_stacks["page.png"].undo()
        saved = self.main.image_states["page.png"]
        self.assertEqual(saved["blk_list"][0].translation, "Original \U0001f642")
        self.assertEqual(saved["viewer_state"]["text_items_state"][0]["text"], before["text"])
        self.assertEqual(self.main.blk_list[0].translation, second.translation)
        self.main.undo_stacks["page.png"].redo()
        self.load_page(0)
        reloaded = self.main.image_viewer.text_items[0]
        self.assertEqual(reloaded.toPlainText(), replacement)
        self.assertEqual(reloaded.region_id, block.region_id)
        self.assertEqual(reloaded.pos(), QtCore.QPointF(*before["position"]))
        self.assertTrue(self.main.blk_list[0].region_metadata["protections"]["keep_translation"])

    def test_source_replacement_preserves_rendered_target_and_undo(self):
        block, item = self.add_region(rendered=True)
        before = TextItemProperties.from_text_item(item).to_dict()
        self.ctrl.apply_replacement(self.key(block), "source", "Corrected \u65e5\u672c\u8a9e", self.ctrl.snapshot()[self.key(block)])
        self.assertEqual(block.text, "Corrected \u65e5\u672c\u8a9e")
        self.assertEqual(block.translation, "Hello")
        self.assertEqual(TextItemProperties.from_text_item(item).to_dict(), before)
        self.main.undo_stacks["page.png"].undo()
        self.assertEqual(block.text, "Bonjour")
        self.assertEqual(TextItemProperties.from_text_item(item).to_dict(), before)

    def preview(self, block, dialog_result, *, update_lettering=True):
        self.select(block)
        self.ctrl.set_protection("keep_translation", True)
        def translate(blocks, _image, _context):
            blocks[0].translation = "Proposed words 日本語"
        with (
            patch("modules.utils.pipeline_config.validate_translator", return_value=True),
            patch("modules.rendering.auto_bubble_fit.load_original_for_fitting", return_value=self.main.image_data["page.png"]),
            patch("modules.translation.processor.Translator") as translator,
            patch("app.controllers.review.QtWidgets.QDialog.exec", autospec=True, side_effect=dialog_result),
        ):
            translator.return_value.translate.side_effect = translate
            self.assertTrue(self.ctrl.preview_replacement("translation", update_lettering=update_lettering))
        translator.return_value.translate.assert_called_once()
        translator.assert_called_once_with(self.main, "French", "English")

    def source_preview(self, block, dialog_result):
        self.select(block)
        self.ctrl.set_protection("keep_source", True)
        self.main.pipeline.cache_manager = CacheManager()
        handler = OCRHandler(self.main, self.main.pipeline.cache_manager, self.main.pipeline)
        self.main.pipeline.ocr_handler = handler
        handler.ocr = Mock()
        original = block.text
        def recognize(image, blocks):
            self.assertIs(image, self.main.image_data["page.png"])
            self.assertIsNot(blocks[0], block)
            self.assertEqual(blocks[0].text, "")
            blocks[0].text = "Recognized \u65e5\u672c\u8a9e"
            self.assertEqual(block.text, original)
        handler.ocr.process.side_effect = recognize
        with (
            patch("modules.rendering.auto_bubble_fit.load_original_for_fitting", return_value=self.main.image_data["page.png"]),
            patch("app.controllers.review.QtWidgets.QDialog.exec", autospec=True, side_effect=dialog_result),
            patch.object(handler, "_recognize_selected_block", wraps=handler._recognize_selected_block) as selected,
            patch("modules.translation.processor.Translator") as translator,
        ):
            self.assertTrue(self.ctrl.preview_replacement("source"))
        selected.assert_called_once()
        self.assertEqual(len(selected.call_args.kwargs["context_blocks"]), len(self.main.blk_list))
        handler.ocr.initialize.assert_called_once_with(self.main, "French")
        handler.ocr.process.assert_called_once()
        translator.assert_not_called()

    def test_cancelled_replacement_preview_does_not_change_text_or_undo_stack(self):
        block, item = self.add_region(rendered=True)
        self.select(block, item)
        self.ctrl.set_protection("keep_translation", True)
        before = self.main.undo_stacks["page.png"].count()
        self.preview(block, lambda *_: QtWidgets.QDialog.DialogCode.Rejected)
        self.assertEqual(block.translation, "Hello")
        self.assertEqual(item.toPlainText(), "Hello")
        self.assertEqual(self.main.undo_stacks["page.png"].count(), before)
        self.assertFalse(self.ctrl._preview_running)

    def test_translation_only_preview_preserves_live_and_saved_lettering_through_undo(self):
        block, item = self.add_region(rendered=True,
            html='<p><span style="font-weight:700">Hello</span></p>')
        before_live = TextItemProperties.from_text_item(item).to_dict()
        before_saved = copy.deepcopy(self.main.image_states["page.png"]["viewer_state"])
        before_pixels = self.main.image_data["page.png"].copy()
        self.main.image_patches["page.png"] = [{"kind": "inpaint", "hash": "existing"}]
        before_patches = copy.deepcopy(self.main.image_patches)
        self.preview(block, lambda *_: QtWidgets.QDialog.DialogCode.Accepted,
                     update_lettering=False)
        stack = self.main.undo_stacks["page.png"]
        for action, expected in ((lambda: None, "Proposed words 日本語"),
                                 (stack.undo, "Hello"),
                                 (stack.redo, "Proposed words 日本語")):
            action()
            self.assertEqual(block.translation, expected)
            self.assertEqual(self.main.image_states["page.png"]["blk_list"][0].translation, expected)
            self.assertEqual(self.main.t_text_edit.toPlainText(), expected)
            self.assertTrue(block.region_metadata["protections"]["keep_translation"])
            self.assertEqual(TextItemProperties.from_text_item(item).to_dict(), before_live)
            self.assertEqual(self.main.image_states["page.png"]["viewer_state"], before_saved)
            self.assertEqual(self.main.image_patches, before_patches)
            np.testing.assert_array_equal(self.main.image_data["page.png"], before_pixels)

    def test_translation_only_preview_does_not_create_missing_lettering(self):
        block, _ = self.add_region()
        self.preview(block, lambda *_: QtWidgets.QDialog.DialogCode.Accepted,
                     update_lettering=False)
        self.assertEqual(block.translation, "Proposed words 日本語")
        self.assertEqual(self.main.image_viewer.text_items, [])
        self.assertEqual(self.main.image_states["page.png"]["viewer_state"]["text_items_state"], [])
        self.assertEqual(self.main.image_patches, {})
        self.main.undo_stacks["page.png"].undo()
        self.assertEqual(block.translation, "Hello")
        self.main.undo_stacks["page.png"].redo()
        self.assertEqual(block.translation, "Proposed words 日本語")
        self.assertEqual(self.main.image_viewer.text_items, [])

    def test_translation_only_replacement_undo_on_unloaded_page_preserves_lettering(self):
        block, item = self.add_region(rendered=True)
        before = TextItemProperties.from_text_item(item).to_dict()
        self.preview(block, lambda *_: QtWidgets.QDialog.DialogCode.Accepted,
                     update_lettering=False)
        self.add_page("second.png")
        self.load_page(1)
        stack = self.main.undo_stacks["page.png"]
        stack.undo()
        self.assertEqual(self.main.image_states["page.png"]["blk_list"][0].translation, "Hello")
        stack.redo()
        self.assertEqual(self.main.image_states["page.png"]["blk_list"][0].translation, "Proposed words 日本語")
        self.load_page(0)
        self.assertEqual(self.main.blk_list[0].translation, "Proposed words 日本語")
        self.assertEqual(TextItemProperties.from_text_item(self.main.image_viewer.text_items[0]).to_dict(), before)

    def test_stale_translation_preview_cannot_overwrite_new_target(self):
        block, item = self.add_region(rendered=True)
        def change_then_accept(*_):
            block.translation = "User changed target while waiting"
            return QtWidgets.QDialog.DialogCode.Accepted
        self.preview(block, change_then_accept)
        self.assertEqual(block.translation, "User changed target while waiting")
        self.assertNotIn("Proposed", item.toPlainText())
        self.assertIn("changed", self.main.review_panel.status_label.text())

    def test_preview_accept_preserves_geometry_changed_during_dialog(self):
        block, item = self.add_region(rendered=True)
        def move_and_accept(*_):
            item.setPos(26.25, 33.75)
            item.setRotation(9)
            return QtWidgets.QDialog.DialogCode.Accepted
        self.preview(block, move_and_accept)
        self.assertEqual(item.toPlainText(), "Proposed words \u65e5\u672c\u8a9e")
        self.assertEqual(item.pos(), QtCore.QPointF(26.25, 33.75))
        self.assertEqual(item.rotation(), 9)

    def test_source_preview_cancel_leaves_original_and_target_untouched(self):
        block, item = self.add_region(rendered=True)
        before = item.toHtml()
        def cancel(*_):
            self.assertEqual(block.text, "Bonjour")
            return QtWidgets.QDialog.DialogCode.Rejected
        self.source_preview(block, cancel)
        self.assertEqual(block.text, "Bonjour")
        self.assertEqual(block.translation, "Hello")
        self.assertEqual(item.toHtml(), before)
        self.assertEqual(self.main.pipeline.cache_manager.ocr_cache, {})

    def test_source_preview_apply_changes_only_source_and_is_undoable(self):
        block, item = self.add_region(rendered=True)
        def accept(*_):
            self.assertEqual(block.text, "Bonjour")
            return QtWidgets.QDialog.DialogCode.Accepted
        self.source_preview(block, accept)
        self.assertEqual(block.text, "Recognized \u65e5\u672c\u8a9e")
        self.assertEqual(block.translation, "Hello")
        self.assertEqual(item.toPlainText(), "Hello")
        self.assertTrue(block.region_metadata["protections"]["keep_source"])
        self.main.undo_stacks["page.png"].undo()
        self.assertEqual(block.text, "Bonjour")

    def test_stale_source_preview_cannot_overwrite_new_source(self):
        block, _ = self.add_region()
        def edit_and_accept(*_):
            block.text = "User correction"
            return QtWidgets.QDialog.DialogCode.Accepted
        self.source_preview(block, edit_and_accept)
        self.assertEqual(block.text, "User correction")
        self.assertIn("changed", self.main.review_panel.status_label.text())

    def test_preview_apply_does_not_rewrite_caches_or_image_patches(self):
        block, _ = self.add_region(rendered=True)
        cache = self.main.pipeline.cache_manager = CacheManager()
        cache.ocr_cache = {("original", "ocr", "French"): {"old": "Source cached"}}
        cache.translation_cache = {("original", "translator", "French", "English"): {"old": "Target cached"}}
        self.main.image_patches["page.png"] = [{"kind": "artistic", "hash": "existing", "bbox": [1, 2, 3, 4]}]
        before = copy.deepcopy((cache.ocr_cache, cache.translation_cache, self.main.image_patches))
        self.preview(block, lambda *_: QtWidgets.QDialog.DialogCode.Accepted)
        self.assertEqual(block.translation, "Proposed words \u65e5\u672c\u8a9e")
        self.assertEqual((cache.ocr_cache, cache.translation_cache, self.main.image_patches), before)

    def test_context_protections_are_independent_and_target_clicked_item(self):
        block, item = self.add_region(rendered=True)
        other, other_item = self.add_region(rendered=True)
        self.select(other, other_item)
        menu = QtWidgets.QMenu()
        self.ctrl.add_context_actions(menu, item=item)
        actions = {action.text(): action for action in menu.actions() if action.isCheckable()}
        labels = {"Keep source": "keep_source", "Keep translation": "keep_translation",
                  "Keep layout": "keep_layout", "Skip automatic cleanup": "skip_auto_cleanup"}
        self.assertEqual(set(actions), set(labels))
        for label, key in labels.items():
            with self.subTest(label=label):
                actions[label].trigger()
                self.assertTrue(block.region_metadata["protections"][key])
                self.assertFalse(any(value for name, value in block.region_metadata["protections"].items() if name != key))
                self.assertFalse(any(other.region_metadata["protections"].values()))
                self.main.undo_stacks["page.png"].undo()
                self.assertFalse(any(block.region_metadata["protections"].values()))


if __name__ == "__main__":
    unittest.main()
