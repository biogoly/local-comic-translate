import unittest

import numpy as np
from PySide6 import QtWidgets

from app.ui.dialogs.heal_repair import HealRepairDialog
from modules.repair.heal import HealError, adjust_sample_color, preview_heal
from modules.utils.clone_brush import CloneSample, CloneStroke, capture_sample


class HealCoreTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((70, 90))
        texture = ((xx + 2 * yy) % 7).astype(np.uint8)
        self.source = np.stack((texture + 60, texture + 80, texture + 100), axis=-1)
        self.current = np.full_like(self.source, 198)
        self.sample = capture_sample(self.source, 20, 20, 12)
        self.mask = np.zeros(self.current.shape[:2], np.uint8)
        self.mask[36:44, 46:54] = 255
        self.target = (100, 130, 160)
        self.destination = (50, 40)

    def preview(self, **kwargs):
        return preview_heal(self.current, self.mask, self.sample, self.target, self.destination, **kwargs)

    def compose(self, result, name='heal_pixels'):
        x, y, w, h = result.bbox
        pixels = self.current.copy()
        pixels[y:y+h, x:x+w] = getattr(result, name)
        return pixels

    def assert_code(self, code, function):
        with self.assertRaises(HealError) as caught:
            function()
        self.assertEqual(caught.exception.code, code)

    def test_adjustment_preserves_texture_differences_and_uses_target_color(self):
        adjusted = adjust_sample_color(self.sample, self.target)
        values = adjusted.pixels[adjusted.valid].astype(int)
        original = self.sample.pixels[self.sample.valid].astype(int)
        np.testing.assert_array_equal(np.median(values, axis=0), self.target)
        np.testing.assert_array_equal(values - values[0], original - original[0])
        self.assertFalse(adjusted.pixels.flags.writeable)
        self.assertFalse(adjusted.valid.flags.writeable)

    def test_zero_strength_equals_ordinary_clone_and_midpoint_is_intermediate(self):
        unchanged = adjust_sample_color(self.sample, self.target, 0)
        np.testing.assert_array_equal(unchanged.pixels, self.sample.pixels)
        halfway = adjust_sample_color(self.sample, self.target, 0.5)
        full = adjust_sample_color(self.sample, self.target, 1)
        expected = (self.sample.pixels[self.sample.valid].astype(float) + full.pixels[full.valid]) / 2
        np.testing.assert_allclose(halfway.pixels[halfway.valid], expected, atol=0.5)
        preview = self.preview(strength=0)
        np.testing.assert_array_equal(preview.clone_pixels, preview.heal_pixels)

    def test_preview_clone_matches_existing_clone_stamp_inside_mask(self):
        stroke = CloneStroke(self.current, self.sample, 0.25)
        stroke.move(*self.destination)
        expected = self.current.copy()
        expected[self.mask != 0] = stroke.result[self.mask != 0]
        np.testing.assert_array_equal(self.compose(self.preview(), 'clone_pixels'), expected)

    def test_outside_mask_and_protection_remain_byte_identical(self):
        self.mask[39:42, 49:52] = 0
        protected = np.zeros(self.mask.shape, bool)
        protected[:, 51] = True
        result = self.preview(protection=protected, softness=1)
        after = self.compose(result)
        excluded = (self.mask == 0) | protected
        np.testing.assert_array_equal(after[excluded], self.current[excluded])
        self.assertTrue(np.any(after[~excluded] != self.current[~excluded]))

    def test_partial_mask_alpha_is_applied_after_soft_stamp_once(self):
        self.mask[:] = 0
        self.mask[40, 50] = 128
        full_mask = self.mask.copy()
        full_mask[40, 50] = 255
        full = preview_heal(self.current, full_mask, self.sample, self.target, self.destination)
        partial = self.preview()
        expected = np.rint(self.current[40, 50] * (1-128/255) + full.heal_pixels[0, 0] * (128/255))
        np.testing.assert_array_equal(partial.heal_pixels[0, 0], expected)

    def test_mask_larger_than_sample_rejected_even_if_protected(self):
        self.mask[0, 0] = 255
        protected = np.zeros(self.mask.shape, bool)
        protected[0, 0] = True
        self.assert_code('mask_outside_sample', lambda: self.preview(protection=protected))

    def test_source_image_edge_cannot_invent_missing_sample_pixels(self):
        self.sample = capture_sample(self.source, 0, 0, 12)
        self.assert_code('mask_outside_sample', self.preview)
        self.mask[:] = 0
        self.mask[40:43, 50:53] = 255
        self.assertTrue(self.preview().heal_pixels.size)

    def test_destination_near_edge_uses_valid_part_of_stamp_without_wrapping(self):
        self.destination = (0, 0)
        self.mask[:] = 0
        self.mask[:3, :3] = 255
        result = self.preview()
        self.assertEqual(result.bbox, (0, 0, 3, 3))
        np.testing.assert_array_equal(self.compose(result)[3:], self.current[3:])

    def test_frozen_sample_does_not_borrow_from_source_border_or_later_edits(self):
        expected = self.preview()
        self.source[:] = (255, 0, 255)
        self.current[:30] = (0, 255, 0)
        self.current[:, :40] = (255, 0, 0)
        actual = self.preview()
        np.testing.assert_array_equal(actual.heal_pixels, expected.heal_pixels)

    def test_inputs_never_mutated_and_outputs_read_only(self):
        originals = [array.copy() for array in (self.current, self.mask, self.sample.pixels, self.sample.valid)]
        result = self.preview()
        for original, actual in zip(originals, (self.current, self.mask, self.sample.pixels, self.sample.valid)):
            np.testing.assert_array_equal(actual, original)
        for array in (result.clone_pixels, result.heal_pixels, result.effective_mask):
            self.assertFalse(array.flags.writeable)

    def test_clipping_handles_extreme_target_without_uint8_wrap(self):
        adjusted = adjust_sample_color(self.sample, (255, 0, 255))
        values = adjusted.pixels[adjusted.valid]
        self.assertGreaterEqual(values[:, 0].min(), 249)
        self.assertLessEqual(values[:, 1].max(), 6)
        self.assertTrue(np.all(adjusted.pixels[~adjusted.valid] == 0))

    def test_empty_or_fully_protected_mask_cannot_commit(self):
        self.assert_code('fully_protected', lambda: self.preview(protection=np.ones(self.mask.shape, bool)))
        self.mask[:] = 0
        self.assert_code('invalid_mask', self.preview)

    def test_invalid_settings_fail_explicitly(self):
        for strength in (-0.1, 1.1, float('nan'), 'wrong'):
            self.assert_code('invalid_strength', lambda: self.preview(strength=strength))
        self.assert_code('invalid_softness', lambda: self.preview(softness=2))
        self.assert_code('invalid_color', lambda: adjust_sample_color(self.sample, (-1, 2, 3)))
        self.assert_code('invalid_color', lambda: adjust_sample_color(self.sample, (1, 2)))
        self.assert_code('invalid_protection', lambda: self.preview(protection=self.mask.astype(np.float32)))
        self.destination = (float('nan'), 0)
        self.assert_code('invalid_destination', self.preview)

    def test_invalid_sample_support_rejected(self):
        pixels = np.zeros((13, 13, 3), np.uint8)
        valid = np.ones((13, 13), bool)
        self.assert_code('invalid_sample', lambda: adjust_sample_color(CloneSample(pixels, valid, 12), self.target))

    def test_one_pixel_sample_has_valid_single_pixel_repair(self):
        self.sample = capture_sample(self.source, 20, 20, 1)
        self.mask[:] = 0
        self.mask[40, 50] = 255
        result = self.preview()
        self.assertEqual(result.bbox, (50, 40, 1, 1))
        np.testing.assert_array_equal(result.heal_pixels[0, 0], self.target)


class HealDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        fixture = HealCoreTests()
        fixture.setUp()
        self.fixture = fixture
        self.dialog = HealRepairDialog(fixture.current, fixture.mask, fixture.sample, fixture.target, fixture.destination)

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def test_preview_only_until_apply_and_independent_of_later_page_edits(self):
        self.assertIsNone(self.dialog.patch)
        expected = self.dialog._candidate['image'].copy()
        self.fixture.current[:] = 0
        self.fixture.mask[:] = 0
        self.dialog._generate()
        np.testing.assert_array_equal(self.dialog._candidate['image'], expected)
        self.assertTrue(self.dialog.apply_button.isEnabled())
        self.dialog.apply_button.click()
        self.assertEqual(self.dialog.result(), QtWidgets.QDialog.Accepted)
        np.testing.assert_array_equal(self.dialog.patch['image'], expected)
        self.assertEqual(self.dialog.metadata['target_rgb'], list(self.fixture.target))
        self.assertTrue(self.dialog.metadata['experimental'])

    def test_strength_change_regenerates_and_cancel_returns_no_patch(self):
        previous = self.dialog._candidate['image'].copy()
        self.dialog.strength_slider.setValue(0)
        self.assertFalse(np.array_equal(previous, self.dialog._candidate['image']))
        expected = self.fixture.preview(strength=0)
        np.testing.assert_array_equal(self.dialog._candidate['image'], expected.clone_pixels)
        self.dialog.cancel_button.click()
        self.assertIsNone(self.dialog.patch)

    def test_oversized_mask_shows_error_and_disables_apply(self):
        invalid = self.fixture.mask.copy()
        invalid[0, 0] = 255
        self.dialog.mask = invalid
        self.dialog._generate()
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertIn('larger clean sample', self.dialog.status_label.text())
        self.assertIsNone(self.dialog._candidate)

    def test_all_protected_shows_error_without_any_patch(self):
        self.dialog.protection = np.ones(self.fixture.mask.shape, bool)
        self.dialog._generate()
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertIn('protected', self.dialog.status_label.text())
        self.assertIsNone(self.dialog.patch)


if __name__ == '__main__':
    unittest.main()
