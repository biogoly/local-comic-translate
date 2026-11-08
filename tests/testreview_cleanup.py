"""Automatic cleanup preserves protected and outside-mask composed pixels."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from modules.utils.region_policy import subtract_cleanup_protections, composite_cleanup_result
from modules.utils.textblock import TextBlock
from pipeline.inpainting import InpaintingHandler, call_inpaint_image


def protected(bounds=(16.2, 12.5, 35.1, 32.7)):
    return TextBlock(text_bbox=np.asarray(bounds), text_class='text_free',
                     region_metadata={'protections': {'skip_auto_cleanup': True}})


class ReviewCleanupTests(unittest.TestCase):
    def setUp(self):
        self.before = np.random.default_rng(4).integers(0, 256, (50, 70, 3), dtype=np.uint8)
        self.mask = np.zeros((50, 70), dtype=np.uint8)
        self.mask[7:43, 9:61] = 255  # Already dilated neighbouring contribution.
        self.region = protected()
        self.effective = subtract_cleanup_protections(self.mask, [self.region])

    def assert_preserved(self, result):
        np.testing.assert_array_equal(result[self.effective == 0], self.before[self.effective == 0])
        self.assertTrue(np.any(result[self.effective > 0] != self.before[self.effective > 0]))

    def test_float_bounds_overlap_and_clipping_do_not_modify_original_mask(self):
        result = subtract_cleanup_protections(self.mask, [self.region, protected((-30, -20, 4, 6))])
        self.assertFalse(np.any(result[12:33, 16:36]))
        self.assertTrue(np.all(self.mask[12:33, 16:36] == 255))
        self.assertTrue(np.any(result > 0))
        self.assert_preserved(composite_cleanup_result(self.before, np.zeros_like(self.before), self.effective))

    def test_legacy_backend_cannot_change_protected_pixels_or_mutate_source(self):
        def legacy(image, mask, config):
            image[:] = 7
            mask[:] = 255  # An ill-behaved implementation cannot expand our final writable mask.
            return image
        handler = SimpleNamespace(inpaint_image=legacy)
        source = self.before.copy()
        result = call_inpaint_image(handler, source, self.mask, None, automatic=True, protected_blocks=[self.region])
        self.assert_preserved(result)
        np.testing.assert_array_equal(source, self.before)

    def test_fast_fill_only_restores_feather_spill_and_protected_pixels(self):
        handler = InpaintingHandler(None)
        handler._apply_fast_bubble_cleanup = Mock(return_value=(np.zeros_like(self.before), np.zeros_like(self.mask), 1))
        result = handler.inpaint_image(self.before, self.mask, None, automatic=True, protected_blocks=[self.region])
        self.assert_preserved(result)

    def test_fast_fill_then_neural_restores_pixels_before_and_after_neural_stage(self):
        handler = InpaintingHandler(None)
        handler._apply_fast_bubble_cleanup = Mock(return_value=(np.zeros_like(self.before), np.full_like(self.mask, 255), 1))
        def neural(image, mask, config):
            np.testing.assert_array_equal(image[self.effective == 0], self.before[self.effective == 0])
            self.assertFalse(np.any(mask[12:33, 16:36]))
            return np.full_like(image, 4)
        backend = Mock(side_effect=neural)
        backend.force_full_image_inpainting = True
        handler.inpainter_cache = backend
        result = handler.inpaint_image(self.before, self.mask, None, automatic=True, protected_blocks=[self.region])
        self.assert_preserved(result)

    def test_explicit_selected_or_brush_cleanup_can_edit_a_protected_region(self):
        handler = SimpleNamespace(inpaint_image=lambda image, mask, config, **kwargs: np.zeros_like(image))
        result = call_inpaint_image(handler, self.before, self.mask, None, blk_list=[self.region])
        self.assertTrue(np.all(result[12:33, 16:36] == 0))

    def test_only_cleanup_flag_protects_pixels(self):
        for key in ('keep_source', 'keep_translation', 'keep_layout'):
            region = TextBlock(text_bbox=np.asarray([10, 10, 30, 30]), text_class='text_bubble',
                               region_metadata={'protections': {key: True}})
            np.testing.assert_array_equal(subtract_cleanup_protections(self.mask, [region]), self.mask)

    def test_rotated_region_protects_corners_outside_the_unrotated_box(self):
        region = protected((20, 20, 50, 30))
        region.angle = 45
        effective = subtract_cleanup_protections(np.full_like(self.mask, 255), [region])
        # Rotated corners extend vertically beyond original y=20..30.
        self.assertEqual(effective[12, 28], 0)
        self.assertEqual(effective[37, 42], 0)
        self.assertEqual(effective[1, 1], 255)


if __name__ == '__main__':
    unittest.main()
