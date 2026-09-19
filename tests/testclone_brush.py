import unittest

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets, QtTest

from modules.utils.clone_brush import CloneStroke, capture_sample
from tests import testlocal_repair as repair_tests


class CloneEngineTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((60, 80))
        self.texture = np.stack((xx * 3, yy * 4, (xx + yy) * 1), axis=-1).astype(np.uint8)
        self.current = np.full_like(self.texture, 200)

    def test_reuses_fixed_texture_patch_without_touching_outside(self):
        sample = capture_sample(self.texture, 20, 20, 10)
        stroke = CloneStroke(self.current, sample, 0.25)
        stroke.move(40.5, 20.5)
        stroke.move(50.5, 20.5)
        patch = stroke.patch()
        x, y, w, h = patch['bbox']
        np.testing.assert_array_equal(patch['image'][20-y, 50-x], self.texture[20, 20])
        np.testing.assert_array_equal(patch['image'][20-y, 51-x], self.texture[20, 21])
        outside = stroke.mask[y:y+h, x:x+w] == 0
        np.testing.assert_array_equal(patch['image'][outside], self.current[y:y+h, x:x+w][outside])

    def test_soft_edges_are_blended_and_overlap_does_not_accumulate(self):
        stroke = CloneStroke(self.current, capture_sample(np.zeros_like(self.current), 20, 20, 20), 0.5)
        stroke.move(30, 30)
        original_mask = stroke.mask.copy()
        self.assertTrue(np.any((original_mask > 0) & (original_mask < 255)))
        self.assertEqual(original_mask[30, 30], 255)
        stroke.move(30, 30)
        np.testing.assert_array_equal(stroke.mask, original_mask)

    def test_source_boundaries_skip_invalid_pixels_without_wrapping(self):
        stroke = CloneStroke(self.current, capture_sample(self.texture, 0, 0, 16), 0)
        stroke.move(40, 20)
        self.assertFalse(np.any(stroke.mask[:20]))
        self.assertFalse(np.any(stroke.mask[:, :40]))
        self.assertTrue(np.any(stroke.mask[20:, 40:]))
        self.assertIsNotNone(stroke.patch())
        stroke = CloneStroke(self.current, capture_sample(self.texture, -40, -40, 20), 0)
        stroke.move(20, 20)
        self.assertIsNone(stroke.patch())

    def test_fast_drag_has_no_gaps(self):
        stroke = CloneStroke(self.current, capture_sample(self.texture, 20, 20, 6), 0)
        stroke.move(5, 30)
        stroke.move(70, 30)
        self.assertTrue(np.all(stroke.mask[30, 5:70] > 0))

    def test_long_stroke_never_reads_outside_selected_source(self):
        source = np.full_like(self.current, (255, 0, 0))
        source[15:26, 15:26] = (10, 20, 30)
        sample = capture_sample(source, 20, 20, 10)
        source[:] = (0, 255, 0)  # later page changes cannot alter the sample
        stroke = CloneStroke(self.current, sample, 0)
        stroke.move(10, 40)
        stroke.move(70, 40)
        self.assertTrue(np.all(stroke.result[40, 10:71] == (10, 20, 30)))
        self.assertFalse(sample.pixels.flags.writeable)


class CloneControllerTests(unittest.TestCase):
    setUpClass = classmethod(repair_tests.LocalRepairTests.setUpClass.__func__)
    tearDown = repair_tests.LocalRepairTests.tearDown
    add_patch = repair_tests.LocalRepairTests.add_patch

    def setUp(self):
        repair_tests.LocalRepairTests.setUp(self)
        yy, xx = np.indices(self.original.shape[:2])
        self.original[:] = np.stack((xx * 2, yy * 3, xx + yy), axis=-1).astype(np.uint8)
        self.viewer.display_image_array(self.original, fit=False)
        self.viewer.brush_size = 10
        self.viewer.set_tool('clone')

    def test_alt_click_copies_source_and_commits_one_undo_step(self):
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        self.assertEqual(self.stack.count(), 0)
        self.repair.press(QtCore.QPointF(50, 40))
        self.repair.move(QtCore.QPointF(60, 40))
        # Live preview isn't yet a persisted patch or cleanup mask.
        self.assertEqual(self.main.image_patches, {})
        self.assertEqual(self.viewer.save_brush_strokes(), [])
        np.testing.assert_array_equal(self.viewer.get_image_array(), self.original)
        self.repair.release()
        after = self.viewer.get_image_array()
        np.testing.assert_array_equal(after[40, 60], self.original[20, 20])
        self.assertEqual(self.stack.count(), 1)
        self.assertEqual(self.stack.undoText(), 'Clone brush')
        self.stack.undo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), self.original)
        self.stack.redo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), after)

    def test_one_pixel_brush_copies_the_clicked_pixel(self):
        self.viewer.brush_size = 1
        self.repair.press(QtCore.QPointF(20.8, 20.1), set_source=True)
        self.repair.press(QtCore.QPointF(55.1, 40.8))
        self.repair.release()
        expected = self.original.copy()
        expected[40, 55] = self.original[20, 20]
        np.testing.assert_array_equal(self.viewer.get_image_array(), expected)

    def test_original_and_current_source_modes_are_distinct_and_frozen(self):
        self.add_patch((255, 0, 0), (10, 10, 20, 20))
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        np.testing.assert_array_equal(self.repair.clone.source['sample'].pixels[5, 5], self.original[20, 20])
        self.panel.clone_original.setChecked(False)
        self.assertIsNone(self.repair.clone.source)
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        self.add_patch((0, 0, 255), (10, 10, 20, 20))
        self.repair.press(QtCore.QPointF(60, 40))
        self.repair.release()
        np.testing.assert_array_equal(self.viewer.get_image_array()[40, 60], (255, 0, 0))

    def test_sample_stays_fixed_across_strokes_until_resampled(self):
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        self.repair.press(QtCore.QPointF(50, 40))
        self.repair.release()
        self.repair.press(QtCore.QPointF(60, 50))
        self.repair.release()
        np.testing.assert_array_equal(self.viewer.get_image_array()[50, 60], self.original[20, 20])
        self.repair.press(QtCore.QPointF(15, 15), set_source=True)
        self.repair.press(QtCore.QPointF(70, 50))
        self.repair.release()
        np.testing.assert_array_equal(self.viewer.get_image_array()[50, 70], self.original[15, 15])

    def test_source_marker_is_stationary_and_destination_tracks_pointer_at_all_zooms(self):
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        for scale in (0.2, 1.0, 4.0):
            self.viewer.setTransform(QtGui.QTransform.fromScale(scale, scale))
            self.repair.clone.hover(QtCore.QPointF(60.5, 40.5))
            self.assertEqual(self.repair.clone.marker.rect().center(), QtCore.QPointF(20.5, 20.5))
            self.assertEqual(self.repair.clone.destination.rect().center(), QtCore.QPointF(60.5, 40.5))
            rect = self.repair.clone.destination.rect()
            self.assertEqual(rect.width(), self.viewer.brush_size)
            self.assertAlmostEqual(self.viewer.transform().mapRect(rect).width(), 10 * scale)

    def test_mouse_tracking_updates_destination_without_a_pressed_button(self):
        self.viewer.resize(400, 300)
        self.viewer.show()
        self.app.processEvents()
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        for scale in (1.0, 2.5):
            self.viewer.setTransform(QtGui.QTransform.fromScale(scale, scale))
            target = self.viewer.mapFromScene(QtCore.QPointF(60, 40))
            QtTest.QTest.mouseMove(self.viewer.viewport(), target)
            self.app.processEvents()
            mapped = self.viewer.mapToScene(target)
            center = self.repair.clone.destination.rect().center()
            self.assertLessEqual(abs(center.x() - mapped.x()), 0.5)
            self.assertLessEqual(abs(center.y() - mapped.y()), 0.5)
            self.assertEqual(self.repair.clone.marker.rect().center(), QtCore.QPointF(20.5, 20.5))

    def test_brush_size_change_requires_a_new_sample(self):
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        self.viewer.brush_size = 20
        self.repair.press(QtCore.QPointF(60, 40))
        self.assertIsNone(self.repair.clone.stroke)
        self.assertEqual(self.stack.count(), 0)
        self.assertTrue(self.notices)
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        self.assertEqual(self.repair.clone.source['sample'].size, 20)

    def test_missing_source_navigation_and_tool_change_are_safe(self):
        self.repair.press(QtCore.QPointF(30, 30))
        self.assertTrue(self.notices)
        self.assertEqual(self.stack.count(), 0)
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        self.repair.press(QtCore.QPointF(50, 40))
        self.viewer.set_tool('brush')
        self.assertFalse(self.repair.is_drawing)
        self.assertIsNone(self.repair.clone.preview)
        self.viewer.clear_scene()
        self.assertIsNone(self.repair.clone.source)

    def test_real_mouse_alt_click_then_paint(self):
        self.viewer.resize(400, 300)
        self.viewer.show()
        self.app.processEvents()
        source = self.viewer.mapFromScene(QtCore.QPointF(20, 20))
        target = self.viewer.mapFromScene(QtCore.QPointF(55, 40))
        QtTest.QTest.mouseClick(self.viewer.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.AltModifier, source)
        self.assertIsNotNone(self.repair.clone.source)
        QtTest.QTest.mouseClick(self.viewer.viewport(), QtCore.Qt.LeftButton, pos=target)
        np.testing.assert_array_equal(self.viewer.get_image_array()[40, 55], self.original[20, 20])

    def test_scene_offset_maps_source_and_destination_to_page_pixels(self):
        self.repair._page = lambda *args: ('page.png', QtCore.QPointF(100, 200))
        self.repair.press(QtCore.QPointF(120, 220), set_source=True)
        self.repair.press(QtCore.QPointF(155, 240))
        self.repair.release()
        np.testing.assert_array_equal(self.viewer.get_image_array()[40, 55], self.original[20, 20])

    def test_undo_during_stroke_discards_stale_preview(self):
        self.add_patch((255, 0, 0))
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        self.repair.press(QtCore.QPointF(55, 40))
        self.stack.undo()
        self.repair.release()
        self.assertEqual(self.stack.index(), 0)
        self.assertIsNone(self.repair.clone.preview)

    def test_export_hides_transient_source_marker_and_uncommitted_preview(self):
        baseline = self.viewer.get_image_array(paint_all=True)
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        self.repair.press(QtCore.QPointF(55, 40))
        self.assertIsNotNone(self.repair.clone.preview)
        np.testing.assert_array_equal(self.viewer.get_image_array(paint_all=True), baseline)

    def test_preview_pixels_match_committed_pixels_and_reload(self):
        from app.ui.commands.base import PatchCommandBase
        self.repair.press(QtCore.QPointF(20, 20), set_source=True)
        self.repair.press(QtCore.QPointF(55, 40))
        preview = self.repair.clone.stroke['engine'].patch()
        self.repair.release()
        x, y, w, h = preview['bbox']
        np.testing.assert_array_equal(self.viewer.get_image_array()[y:y+h, x:x+w], preview['image'])
        expected = self.viewer.get_image_array()
        self.viewer.display_image_array(self.original, fit=False)
        for record in self.main.image_patches['page.png']:
            PatchCommandBase.create_patch_item(record, self.viewer)
        np.testing.assert_array_equal(self.viewer.get_image_array(), expected)


if __name__ == '__main__':
    unittest.main()
