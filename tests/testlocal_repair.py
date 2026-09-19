import os
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets, QtTest

from app.controllers.local_repair import LocalRepairController, ClearPageMaskCommand, repair_patch, rasterize_paths
from app.ui.canvas.image_viewer import ImageViewer
from app.ui.commands.inpaint import PatchInsertCommand, PatchClearCommand
from app.ui.local_repair_panel import LocalRepairPanel


class LocalRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.viewer = ImageViewer(None)
        self.original = np.full((80, 100, 3), (30, 120, 200), np.uint8)
        self.viewer.display_image_array(self.original, fit=False)
        self.panel = LocalRepairPanel()
        self.stack = QtGui.QUndoStack()
        self.errors = []
        self.main = SimpleNamespace(
            image_viewer=self.viewer, local_repair_panel=self.panel,
            image_files=["page.png"], curr_img_idx=0, undo_stacks={"page.png": self.stack},
            temp_dir=self.tmp.name, image_patches={}, in_memory_patches={},
            image_ctrl=SimpleNamespace(load_image=lambda _: self.original.copy(),
                                       get_composited_page_image=lambda _: self.viewer.get_image_array()),
            set_tool=self.viewer.set_tool, set_slider_size=lambda _: None,
            default_error_handler=self.errors.append,
        )
        self.repair = LocalRepairController(self.main)
        self.notices = []
        self.repair._notice = self.notices.append

    def tearDown(self):
        self.repair.cancel()
        self.assertEqual(self.errors, [])
        self.viewer.close()
        self.panel.close()
        self.tmp.cleanup()

    def add_patch(self, color, bbox=(10, 10, 65, 55), kind="inpaint"):
        image = np.full((bbox[3], bbox[2], 3), color, np.uint8)
        self.stack.push(PatchInsertCommand(self.main, [{"bbox": bbox, "image": image}], "page.png", kind=kind))

    def add_mask(self):
        path = QtGui.QPainterPath()
        path.addRect(20, 20, 12, 15)
        return self.viewer._scene.addPath(path, QtGui.QPen(QtGui.QColor(255, 0, 0), 2), QtGui.QBrush(QtGui.QColor(255, 0, 0, 128)))

    def test_restore_only_brushed_pixels_and_undo_redo(self):
        self.add_patch((240, 20, 30))
        self.add_patch((1, 2, 3), (65, 60, 8, 8), kind="flux2_edit")
        before = self.viewer.get_image_array()
        self.viewer.brush_size = 8
        self.viewer.set_tool("restore")
        self.repair.press(QtCore.QPointF(25, 25))
        self.repair.move(QtCore.QPointF(35, 25))
        self.repair.release()
        after = self.viewer.get_image_array()
        np.testing.assert_array_equal(after[25, 30], self.original[25, 30])
        np.testing.assert_array_equal(after[45:, :], before[45:, :])
        self.assertEqual(self.stack.undoText(), "Restore area")
        self.stack.undo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), before)
        self.stack.redo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), after)
        self.assertIsNone(self.repair.preview)

    def test_single_click_restore_and_cancel_on_tool_change(self):
        self.add_patch((250, 0, 0))
        self.viewer.brush_size = 6
        self.viewer.set_tool("restore")
        self.repair.press(QtCore.QPointF(30, 30))
        self.repair.release()
        np.testing.assert_array_equal(self.viewer.get_image_array()[30, 30], self.original[30, 30])
        count = self.stack.count()
        self.repair.press(QtCore.QPointF(40, 40))
        self.viewer.set_tool("brush")
        self.repair.release()
        self.assertEqual(count, self.stack.count())

    def test_pick_color_ignores_mask_and_translated_text_overlays(self):
        self.add_mask()
        label = self.viewer._scene.addText("TEST")
        label.setPos(10, 10)
        self.viewer.set_tool("color_pick")
        self.repair.press(QtCore.QPointF(25, 25))
        self.assertEqual(self.repair.color, (30, 120, 200))
        self.assertTrue(self.panel.fill.isEnabled())
        self.assertEqual(self.viewer.current_tool, "brush")

    def test_fill_exact_mask_undo_restores_mask_and_previous_pixels(self):
        self.add_patch((255, 10, 20))
        before = self.viewer.get_image_array()
        self.add_mask()
        self.repair.color = (30, 120, 200)
        self.repair.fill_mask()
        after = self.viewer.get_image_array()
        np.testing.assert_array_equal(after[25, 25], (30, 120, 200))
        # Old inpainting dilation would have reached x=15; fill must not.
        np.testing.assert_array_equal(after[:, :18], before[:, :18])
        self.assertFalse(self.viewer.has_drawn_elements())
        self.assertEqual(self.stack.undoText(), "Sampled color fill")
        self.stack.undo()
        self.assertTrue(self.viewer.has_drawn_elements())
        np.testing.assert_array_equal(self.viewer.get_image_array(), before)
        self.stack.redo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), after)
        self.assertFalse(self.viewer.has_drawn_elements())

    def test_missing_mask_or_color_does_not_create_patch(self):
        self.repair.fill_mask()
        self.assertIn("Pick a background color first.", self.notices)
        self.repair.color = (10, 20, 30)
        self.repair.fill_mask()
        self.assertIn("Paint or segment the area to fill first.", self.notices)
        self.assertEqual(self.stack.count(), 0)

    def test_noop_restore_and_stale_undo_revision(self):
        self.viewer.set_tool("restore")
        self.repair.press(QtCore.QPointF(30, 30))
        self.repair.release()
        self.assertEqual(self.stack.count(), 0)
        self.add_patch((0, 0, 0))
        self.repair.press(QtCore.QPointF(30, 30))
        self.stack.undo()
        self.repair.release()
        self.assertEqual(self.stack.index(), 0)

    def test_masks_respect_offset_holes_and_exact_outside_pixels(self):
        path = QtGui.QPainterPath()
        path.addRect(120, 220, 20, 20)
        path.addRect(125, 225, 10, 10)
        mask = rasterize_paths(self.original.shape, [(path, QtCore.Qt.NoPen, QtGui.QBrush(QtCore.Qt.white))], QtCore.QPointF(100, 200))
        self.assertEqual(mask[30, 30], 0)
        self.assertEqual(mask[21, 21], 255)
        self.assertEqual(mask[15, 15], 0)
        patch = repair_patch(self.original, (255, 0, 0), mask)
        x, y, w, h = patch['bbox']
        np.testing.assert_array_equal(patch['image'][30-y, 30-x], self.original[30, 30])

    def test_existing_page_revert_includes_repairs_but_preserves_artistic_edits(self):
        self.add_patch((220, 0, 0))
        self.add_patch((0, 255, 0), (65, 65, 8, 8), kind="flux2_edit")
        self.add_mask()
        self.repair.color = (15, 15, 15)
        self.repair.fill_mask()
        self.stack.push(PatchClearCommand(self.main, "page.png", kind="inpaint"))
        np.testing.assert_array_equal(self.viewer.get_image_array()[25, 25], self.original[25, 25])
        np.testing.assert_array_equal(self.viewer.get_image_array()[67, 67], (0, 255, 0))

    def test_cross_page_mask_keeps_neighbour_and_undo_restores_original(self):
        path = QtGui.QPainterPath()
        path.addRect(10, 60, 20, 40)
        self.viewer._scene.addPath(path, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(QtGui.QColor(255, 0, 0, 128)))
        self.stack.push(ClearPageMaskCommand(self.viewer, QtCore.QRectF(0, 0, 100, 80)))
        paths = [i.path() for i in self.viewer._scene.items() if isinstance(i, QtWidgets.QGraphicsPathItem)]
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].contains(QtCore.QPointF(20, 70)))
        self.assertTrue(paths[0].contains(QtCore.QPointF(20, 90)))
        self.stack.undo()
        paths = [i.path() for i in self.viewer._scene.items() if isinstance(i, QtWidgets.QGraphicsPathItem)]
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], path)
        self.stack.redo()
        self.assertEqual(len([i for i in self.viewer._scene.items() if isinstance(i, QtWidgets.QGraphicsPathItem)]), 1)

    def test_restore_preview_is_not_saved_as_a_mask(self):
        self.add_patch((0, 0, 0))
        self.viewer.set_tool("restore")
        self.repair.press(QtCore.QPointF(30, 30))
        self.assertEqual(self.viewer.save_brush_strokes(), [])
        self.viewer.clear_scene()
        self.assertIsNone(self.repair.stroke)

    def test_mouse_events_restore_and_sample_over_existing_overlays(self):
        self.add_patch((255, 0, 0))
        self.viewer.resize(400, 300)
        self.viewer.show()
        self.app.processEvents()
        self.add_mask()
        label = self.viewer._scene.addText("TEST")
        label.setPos(20, 20)
        point = self.viewer.mapFromScene(QtCore.QPointF(27, 27))
        self.viewer.brush_size = 8
        self.panel.restore.click()
        QtTest.QTest.mouseClick(self.viewer.viewport(), QtCore.Qt.LeftButton, pos=point)
        np.testing.assert_array_equal(self.viewer.get_image_array()[27, 27], self.original[27, 27])
        self.assertEqual(self.stack.undoText(), "Restore area")
        self.panel.pick.click()
        QtTest.QTest.mouseClick(self.viewer.viewport(), QtCore.Qt.LeftButton, pos=point)
        self.assertEqual(self.repair.color, (30, 120, 200))

    def test_repair_patches_survive_scene_reload(self):
        from app.ui.commands.base import PatchCommandBase
        self.add_patch((255, 0, 0))
        self.add_mask()
        self.repair.color = (10, 20, 30)
        self.repair.fill_mask()
        expected = self.viewer.get_image_array()
        self.viewer.display_image_array(self.original, fit=False)
        for record in self.main.image_patches['page.png']:
            PatchCommandBase.create_patch_item(record, self.viewer)
        np.testing.assert_array_equal(self.viewer.get_image_array(), expected)

    def test_new_controls_are_translated_in_all_catalogs(self):
        from pathlib import Path
        folder = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            translator = QtCore.QTranslator()
            self.assertTrue(translator.load(str(folder / f'ct_{locale}.qm')))
            self.app.installTranslator(translator)
            try:
                panel = LocalRepairPanel()
                self.assertEqual(panel.restore.text(), translator.translate('LocalRepairPanel', 'Restore brush'))
                self.assertNotEqual(panel.restore.text(), 'Restore brush')
                self.assertEqual(panel.clone.text(), translator.translate('LocalRepairPanel', 'Clone brush'))
                self.assertNotEqual(panel.clone.text(), 'Clone brush')
                self.assertEqual(self.repair.clone.tr('Clone brush'), translator.translate('CloneBrushController', 'Clone brush'))
                self.assertEqual(self.repair.tr('Restore area'), translator.translate('LocalRepairController', 'Restore area'))
                panel.close()
            finally:
                self.app.removeTranslator(translator)

    def test_repair_icons_can_live_in_external_inpainting_toolbar(self):
        toolbar = QtWidgets.QHBoxLayout()
        panel = LocalRepairPanel(tool_layout=toolbar)
        self.assertEqual(toolbar.count(), 3)
        self.assertIs(toolbar.itemAt(0).widget(), panel.restore)
        self.assertIs(toolbar.itemAt(1).widget(), panel.pick)
        self.assertIs(toolbar.itemAt(2).widget(), panel.clone)
        self.assertEqual(panel.layout().count(), 2)
        self.assertIs(panel.layout().itemAt(0).widget(), panel.fill)
        for button in (panel.restore, panel.pick, panel.clone):
            self.assertEqual(button.toolButtonStyle(), QtCore.Qt.ToolButtonIconOnly)
            self.assertTrue(button.isCheckable())
            self.assertFalse(button.icon().isNull())
            self.assertTrue(button.accessibleName())
        panel.close()

    def test_brush_indicators_match_actual_diameter_at_different_zooms(self):
        for tool, size, diameter in [('brush', 14, 14), ('restore', 14, 14), ('eraser', 14, 28)]:
            self.viewer.set_tool(tool)
            self.viewer.set_br_er_size(size, 800)  # old page-based scaling is ignored
            for scale in (0.15, 1, 3.5):
                self.viewer.setTransform(QtGui.QTransform.fromScale(scale, scale))
                self.viewer.brush_cursor_overlay.refresh(scene_pos=QtCore.QPointF(30, 30))
                item = self.viewer.brush_cursor_overlay.item
                self.assertEqual(item.rect().center(), QtCore.QPointF(30, 30))
                self.assertEqual(item.rect().width(), diameter)
                self.assertAlmostEqual(self.viewer.transform().mapRect(item.rect()).width(), diameter * scale)

    def test_brush_indicators_do_not_enter_masks_or_exports(self):
        baseline = self.viewer.get_image_array(paint_all=True)
        self.viewer.set_tool('brush')
        self.viewer.brush_cursor_overlay.refresh(scene_pos=QtCore.QPointF(30, 30))
        self.assertEqual(self.viewer.save_brush_strokes(), [])
        np.testing.assert_array_equal(self.viewer.get_image_array(paint_all=True), baseline)
        self.viewer.clear_scene()
        self.assertIsNone(self.viewer.brush_cursor_overlay.item)

    def test_cleanup_brush_still_draws_and_eraser_still_erases(self):
        self.viewer.resize(400, 300)
        self.viewer.show()
        self.app.processEvents()
        self.viewer.command_emitted.connect(self.stack.push)
        start = self.viewer.mapFromScene(QtCore.QPointF(20, 20))
        end = self.viewer.mapFromScene(QtCore.QPointF(45, 20))
        self.viewer.set_tool('brush')
        QtTest.QTest.mousePress(self.viewer.viewport(), QtCore.Qt.LeftButton, pos=start)
        QtTest.QTest.mouseMove(self.viewer.viewport(), end)
        QtTest.QTest.mouseRelease(self.viewer.viewport(), QtCore.Qt.LeftButton, pos=end)
        self.assertTrue(self.viewer.has_drawn_elements())
        self.viewer.set_tool('eraser')
        self.viewer.eraser_size = 100
        QtTest.QTest.mouseClick(self.viewer.viewport(), QtCore.Qt.LeftButton, pos=start)
        self.assertFalse(self.viewer.has_drawn_elements())


if __name__ == "__main__":
    unittest.main()
