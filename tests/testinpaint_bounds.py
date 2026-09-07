import unittest
from types import SimpleNamespace

import numpy as np

from pipeline.inpainting import InpaintingHandler


class InpaintBoundsTests(unittest.TestCase):
    def setUp(self):
        self.handler = InpaintingHandler(SimpleNamespace())
        self.image = np.full((120, 140, 3), 255, dtype=np.uint8)
        self.block = SimpleNamespace(
            xyxy=np.array([42.25, 38.5, 78.75, 61.25], dtype=np.float32),
            bubble_xyxy=np.array(
                [30.25, 25.75, 90.5, 75.25], dtype=np.float32
            ),
            text_class="text_bubble",
        )

    def test_float_detector_bounds_are_safe_numpy_slice_indices(self):
        bounds = self.handler._get_fast_fill_bounds(self.block, self.image)

        self.assertEqual(bounds, (27, 23, 94, 78))
        self.assertTrue(all(isinstance(value, int) for value in bounds))

        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask[42:58, 46:76] = 255

        cleaned_image, residual_mask, cleaned_blocks = (
            self.handler._apply_fast_bubble_cleanup(
                self.image,
                mask,
                [self.block],
                preserve_unmatched_mask=True,
            )
        )

        self.assertEqual(cleaned_image.shape, self.image.shape)
        self.assertEqual(residual_mask.shape, mask.shape)
        self.assertGreaterEqual(cleaned_blocks, 0)


if __name__ == "__main__":
    unittest.main()
