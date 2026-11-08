import os
import unittest
from copy import deepcopy
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets, QtTest

from app.controllers.local_repair import LocalRepairController
from app.controllers.pixel_protection import PixelProtectionController, ProtectionOverlay
from app.ui.canvas.image_viewer import ImageViewer
from app.ui.local_repair_panel import LocalRepairPanel
from modules.repair.protection import decode_protection, encode_protection


class PixelProtectionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.viewer = ImageViewer(None)
        self.image = np.full((80, 100, 3), (25, 115, 180), np.uint8)
        self.viewer.display_image_array(self.image, fit=False)
        self.panel = LocalRepairPanel()
        self.stack = QtGui.QUndoStack()
        self.errors = []
        self.main = SimpleNamespace(
            image_viewer=self.viewer, local_repair_panel=self.panel,
            image_files=["first.png", "second.png"], curr_img_idx=0,
            undo_stacks={"first.png": self.stack, "second.png": QtGui.QUndoStack()},
            image_states={"first.png": {"page_id": "page1"}, "second.png": {"page_id": "page2"}},
            _translation_project_epoch=1,
            image_ctrl=SimpleNamespace(get_composited_page_image=lambda _: self.image.copy()),
            set_tool=self.viewer.set_tool, set_slider_size=lambda _: None,
            default_error_handler=self.errors.append,
        )
        self.repair = LocalRepairController(self.main)
        self.controller = PixelProtectionController(self.main)
        self.main.pixel_protection_ctrl = self.controller

    def tearDown(self):
        self.controller.reset_page()
        self.repair.cancel()
        self.viewer.clear_scene()
        self.viewer.close()
        self.panel.close()
        self.assertEqual(self.errors, [])

    def draw(self, start=(30, 30), end=None, tool="protect_pixels", size=10):
        self.viewer.set_tool(tool)
        self.viewer.brush_size = size
        self.controller.press(QtCore.QPointF(*start))
        if end is not None:
            self.controller.move(QtCore.QPointF(*end))
        self.controller.release()

    def mask(self, path="first.png"):
        return decode_protection(self.main.image_states[path].get("pixel_protection"), self.image.shape)

    def test_stroke_changes_only_protection_and_undo_redo_is_exact(self):
        self.draw(end=(50, 30))
        after = deepcopy(self.main.image_states["first.png"]["pixel_protection"])
        mask = self.mask()
        self.assertEqual(mask[30, 40], 255)
        self.assertEqual(mask[45, 40], 0)
        self.assertEqual(self.stack.count(), 1)
        self.assertEqual(self.stack.undoText(), "Protect artwork")
        self.assertFalse(self.stack.isClean())
        np.testing.assert_array_equal(self.viewer.get_image_array(), self.image)
        self.assertEqual(self.viewer.save_brush_strokes(), [])
        self.stack.undo()
        self.assertNotIn("pixel_protection", self.main.image_states["first.png"])
        self.assertEqual(self.controller.overlays, {})
        self.assertTrue(self.stack.isClean())
        self.stack.redo()
        self.assertEqual(self.main.image_states["first.png"]["pixel_protection"], after)

    def test_unprotect_removes_only_painted_pixels_and_can_be_undone(self):
        self.draw(end=(60, 30), size=20)
        before = deepcopy(self.main.image_states["first.png"]["pixel_protection"])
        self.draw(start=(45, 30), tool="unprotect_pixels", size=6)
        self.assertEqual(self.mask()[30, 45], 0)
        self.assertEqual(self.mask()[30, 30], 255)
        self.assertEqual(self.stack.undoText(), "Unprotect artwork")
        self.stack.undo()
        self.assertEqual(self.main.image_states["first.png"]["pixel_protection"], before)
        self.stack.redo()
        self.assertEqual(self.mask()[30, 45], 0)

    def test_full_unprotect_and_undo_preserves_explicit_legacy_empty_key(self):
        self.main.image_states["first.png"]["pixel_protection"] = {}
        self.draw()
        self.stack.undo()
        self.assertEqual(self.main.image_states["first.png"]["pixel_protection"], {})
        self.stack.redo()
        self.draw(tool="unprotect_pixels", size=30)
        self.assertNotIn("pixel_protection", self.main.image_states["first.png"])
        self.stack.undo()
        self.assertEqual(self.mask()[30, 30], 255)

    def test_noop_and_tool_change_do_not_create_commands(self):
        self.draw(tool="unprotect_pixels")
        self.assertEqual(self.stack.count(), 0)
        self.viewer.set_tool("protect_pixels")
        self.controller.press(QtCore.QPointF(20, 20))
        self.viewer.set_tool("brush")
        self.controller.release()
        self.assertEqual(self.stack.count(), 0)
        self.assertIsNone(self.controller.preview)

    def test_overlay_and_live_stroke_are_not_exported_or_saved_as_cleanup_masks(self):
        before = self.viewer.get_image_array(paint_all=True)
        self.draw()
        self.assertIsInstance(self.controller.overlays["first.png"], ProtectionOverlay)
        self.controller.press(QtCore.QPointF(40, 40))
        self.controller.move(QtCore.QPointF(55, 50))
        np.testing.assert_array_equal(self.viewer.get_image_array(paint_all=True), before)
        np.testing.assert_array_equal(self.viewer.get_image_array(), self.image)
        self.assertEqual(self.viewer.save_brush_strokes(), [])
        self.assertFalse(self.viewer.has_drawn_elements())
        self.assertFalse(np.any(self.repair._current_mask(self.image.shape, QtCore.QPointF())))
        self.assertTrue(self.controller.overlays["first.png"].isVisible())
        self.controller.cancel()

    def test_overlay_toggle_does_not_change_protection_and_page_reload_restores_it(self):
        self.draw()
        state = deepcopy(self.main.image_states)
        self.panel.show_protected.setChecked(False)
        self.assertEqual(self.controller.overlays, {})
        self.assertEqual(self.main.image_states, state)
        self.panel.show_protected.setChecked(True)
        self.assertIn("first.png", self.controller.overlays)
        self.main.curr_img_idx = 1
        self.viewer.display_image_array(self.image, fit=False)
        self.assertEqual(self.controller.overlays, {})
        self.main.curr_img_idx = 0
        self.viewer.display_image_array(self.image, fit=False)
        self.assertIn("first.png", self.controller.overlays)

    def test_undo_while_another_page_is_visible_changes_original_page_only(self):
        self.draw()
        self.main.curr_img_idx = 1
        self.viewer.display_image_array(self.image, fit=False)
        self.stack.undo()
        self.assertIsNone(self.mask())
        self.assertNotIn("pixel_protection", self.main.image_states["second.png"])
        self.stack.redo()
        self.assertEqual(self.mask()[30, 30], 255)
        self.assertEqual(self.controller.overlays, {})

    def test_stale_page_identity_project_and_stack_changes_discard_stroke(self):
        for mutation in (lambda: self.main.image_states["first.png"].update(page_id="replaced"),
                         lambda: setattr(self.main, "_translation_project_epoch", 2),
                         lambda: setattr(self.main, "curr_img_idx", 1),
                         lambda: self.main.undo_stacks.update({"first.png": QtGui.QUndoStack()})):
            with self.subTest(mutation=mutation):
                self.main.curr_img_idx = 0
                self.main._translation_project_epoch = 1
                self.main.image_states["first.png"]["page_id"] = "page1"
                self.main.undo_stacks["first.png"] = self.stack
                self.viewer.set_tool("protect_pixels")
                self.controller.press(QtCore.QPointF(30, 30))
                mutation()
                self.controller.release()
                self.assertEqual(self.stack.count(), 0)
                self.assertIsNone(self.mask())

    def test_protection_mouse_event_overlays_and_indicator_match_image_pixels_at_zoom(self):
        self.viewer.resize(500, 400)
        self.viewer.show()
        self.app.processEvents()
        self.panel.protect.click()
        self.assertEqual(self.viewer.current_tool, "protect_pixels")
        self.viewer.set_br_er_size(12, 999)
        self.viewer.resetTransform()
        self.viewer.scale(3, 3)
        label = self.viewer._scene.addText("TEXT")
        label.setPos(20, 20)
        self.viewer.brush_cursor_overlay.refresh(scene_pos=QtCore.QPointF(30, 30))
        indicator = self.viewer.brush_cursor_overlay.item.rect()
        self.assertEqual(indicator.width(), 12)
        point = self.viewer.mapFromScene(QtCore.QPointF(30, 30))
        QtTest.QTest.mouseClick(self.viewer.viewport(), QtCore.Qt.LeftButton, pos=point)
        self.assertEqual(self.mask()[30, 30], 255)
        ys, xs = np.where(self.mask() != 0)
        self.assertLessEqual(int(xs.max() - xs.min() + 1), 14)
        self.assertLessEqual(int(ys.max() - ys.min() + 1), 14)
        self.assertEqual(self.stack.count(), 1)
        self.assertFalse(self.viewer.has_drawn_elements())
        self.viewer.set_tool(None)
        self.assertIs(self.viewer.itemAt(point), label)

    def test_brush_remains_on_start_page_with_webtoon_offsets(self):
        manager = self.viewer.webtoon_manager
        manager.layout_manager.image_positions = [200, 280]
        manager.layout_manager.image_heights = [80, 80]
        manager.layout_manager.page_bottoms = [280, 360]
        manager.layout_manager.webtoon_width = 160
        manager.image_loader.loaded_pages = {0, 1}
        manager.image_loader.image_data = {0: self.image.copy(), 1: self.image.copy()}
        self.assertEqual(self.viewer.scene_to_page_coordinates(QtCore.QPointF(60, 270)),
                         (0, QtCore.QPointF(30, 70)))
        self.draw(start=(60, 270), end=(60, 310))
        mask = self.mask()
        self.assertEqual(mask[70, 30], 255)
        self.assertEqual(mask[79, 30], 255)
        self.assertEqual(mask[70, 60], 0)
        self.assertNotIn("pixel_protection", self.main.image_states["second.png"])
        self.assertEqual(self.controller.overlays["first.png"].pos(), QtCore.QPointF(30, 200))
        self.assertEqual(self.main.undo_stacks["second.png"].count(), 0)
        self.stack.undo()
        self.assertEqual(self.controller.overlays, {})

    def test_invalid_mask_cannot_be_silently_replaced_by_brushing(self):
        self.main.image_states["first.png"]["pixel_protection"] = {"schema_version": 999}
        before = deepcopy(self.main.image_states)
        self.draw()
        self.assertEqual(len(self.errors), 1)
        self.errors.clear()
        self.assertEqual(self.main.image_states, before)
        self.assertEqual(self.stack.count(), 0)

    def test_webtoon_render_hides_protection_overlay_and_pending_stroke(self):
        manager = self.viewer.webtoon_manager
        manager.layout_manager.image_positions = [200]
        manager.layout_manager.image_heights = [80]
        manager.layout_manager.page_bottoms = [280]
        manager.layout_manager.webtoon_width = 100
        manager.image_loader.loaded_pages = {0}
        manager.image_loader.image_data = {0: self.image.copy()}
        self.viewer.photo.setPos(0, 200)
        manager.image_loader.image_items = {0: self.viewer.photo}
        render = lambda: manager.image_loader._render_page_with_scene_items(
            0, self.image, True, True, 0, 80)
        before = render()
        self.draw(start=(30, 230))
        self.controller.press(QtCore.QPointF(40, 240))
        self.controller.move(QtCore.QPointF(60, 245))
        np.testing.assert_array_equal(render(), before)
        self.assertTrue(self.controller.overlays["first.png"].isVisible())
        self.assertEqual(self.viewer.save_brush_strokes(), [])

    def test_heal_fill_choice_uses_sampled_color_gate_without_changing_tools(self):
        self.panel.fill_mode.setCurrentIndex(self.panel.fill_mode.findData("heal"))
        self.assertEqual(self.panel.fill.text(), "Preview Heal…")
        self.assertFalse(self.panel.fill.isEnabled())
        self.panel.set_color(QtGui.QColor(40, 80, 120))
        self.assertTrue(self.panel.fill.isEnabled())
        self.assertIsNone(self.viewer.current_tool)


if __name__ == "__main__":
    unittest.main()
