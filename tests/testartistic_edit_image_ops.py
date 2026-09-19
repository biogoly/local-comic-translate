import unittest

import numpy as np
from PIL import Image

from modules.artistic_edit.image_ops import (
    ImageOperationError,
    blend_with_mask,
    compose_generated_patch,
    crop_padding,
    expand_bbox,
    make_rect_mask,
    mask_bbox,
    normalize_bbox,
    normalize_mask,
    normalize_rgb,
    pad_to_multiple,
    prepare_edit,
    resize_array,
    resize_to_fit,
    sha256_pixels,
)


class ArtisticEditImageOperationTests(unittest.TestCase):
    def test_bbox_normalization_handles_fractional_negative_and_edge_coordinates(self):
        self.assertEqual(normalize_bbox((1.8, 2.2, 4.4, 3.1), (10, 10)), (1, 2, 6, 4))
        self.assertEqual(normalize_bbox((8, 8, -5, -4), (10, 10)), (3, 4, 5, 4))
        self.assertEqual(normalize_bbox((-5, -2, 9, 5), (10, 10)), (0, 0, 4, 3))
        with self.assertRaisesRegex(ImageOperationError, "does not overlap"):
            normalize_bbox((20, 20, 4, 4), (10, 10))

    def test_context_expansion_obeys_minimum_maximum_and_page_edges(self):
        self.assertEqual(
            expand_bbox((10, 10, 20, 10), (100, 100), minimum_context=5),
            (5, 5, 30, 20),
        )
        self.assertEqual(
            expand_bbox((1, 1, 10, 10), (30, 30), minimum_context=20, maximum_context=6),
            (0, 0, 17, 17),
        )
        edge_cases = (
            ((0, 10, 5, 5), (0, 5, 10, 15)),
            ((25, 10, 5, 5), (20, 5, 10, 15)),
            ((10, 0, 5, 5), (5, 0, 15, 10)),
            ((10, 25, 5, 5), (5, 20, 15, 10)),
        )
        for selection, expected in edge_cases:
            with self.subTest(selection=selection):
                self.assertEqual(
                    expand_bbox(selection, (30, 30), minimum_context=5),
                    expected,
                )

    def test_normalization_accepts_grayscale_rgba_and_float_images(self):
        gray = np.array([[0, 255]], dtype=np.uint8)
        normalized_gray = normalize_rgb(gray)
        self.assertEqual(normalized_gray.shape, (1, 2, 3))
        self.assertTrue(np.array_equal(normalized_gray[0, 1], [255, 255, 255]))

        rgba = Image.fromarray(np.array([[[255, 0, 0, 128]]], dtype=np.uint8), "RGBA")
        pixel = normalize_rgb(rgba)[0, 0]
        self.assertEqual(int(pixel[0]), 255)
        self.assertIn(int(pixel[1]), (127, 128))
        self.assertIn(int(pixel[2]), (127, 128))

        float_image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        self.assertTrue(np.array_equal(normalize_rgb(float_image)[0, 0], [0, 128, 255]))

    def test_mask_normalization_uses_alpha_and_binary_output(self):
        rgba_mask = np.array([[[255, 255, 255, 0], [0, 0, 0, 4]]], dtype=np.uint8)
        self.assertTrue(np.array_equal(normalize_mask(rgba_mask), [[0, 255]]))
        self.assertIsNone(mask_bbox(np.zeros((3, 3), dtype=np.uint8)))
        self.assertEqual(mask_bbox(make_rect_mask((10, 8), (2, 3, 4, 2))), (2, 3, 4, 2))

    def test_padding_to_multiple_is_symmetric_and_exactly_reversible(self):
        source = np.arange(17 * 19, dtype=np.int32).reshape(17, 19)
        padded, crop_box = pad_to_multiple(source, multiple=16)
        self.assertEqual(padded.shape, (32, 32))
        self.assertEqual(crop_box, (6, 7, 25, 24))
        self.assertTrue(np.array_equal(crop_padding(padded, crop_box), source))

    def test_resize_to_fit_preserves_aspect_ratio_and_never_upscales(self):
        source = np.zeros((100, 200, 3), dtype=np.uint8)
        downscaled, scale = resize_to_fit(source, (80, 80))
        self.assertEqual(downscaled.shape, (40, 80, 3))
        self.assertEqual(scale, 0.4)
        unchanged, scale = resize_to_fit(source, (300, 300))
        self.assertEqual(unchanged.shape, source.shape)
        self.assertEqual(scale, 1.0)
        self.assertEqual(resize_array(source, (37, 29)).shape, (29, 37, 3))

    def test_blending_changes_only_masked_pixels_without_feathering(self):
        source = np.zeros((7, 7, 3), dtype=np.uint8)
        generated = np.full_like(source, 255)
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        blended = blend_with_mask(source, generated, mask)
        self.assertTrue(np.all(blended[2:5, 2:5] == 255))
        outside = blended.copy()
        outside[2:5, 2:5] = 0
        self.assertFalse(np.any(outside))

        feathered = blend_with_mask(source, generated, mask, feather_radius=1.5)
        self.assertGreater(int(feathered[1, 1, 0]), 0)
        self.assertLess(int(feathered[1, 1, 0]), 255)

    def test_prepare_and_compose_restore_geometry_and_smallest_patch(self):
        page = np.zeros((50, 60, 3), dtype=np.uint8)
        prepared = prepare_edit(
            page,
            (20, 15, 10, 8),
            minimum_context=5,
            maximum_model_size=(12, 12),
        )
        self.assertEqual(prepared.selection_bbox, (20, 15, 10, 8))
        self.assertEqual(prepared.context_bbox, (15, 10, 20, 18))
        self.assertLess(prepared.working_scale, 1.0)
        self.assertEqual(prepared.model_input.shape[0] % 16, 0)
        self.assertEqual(prepared.model_input.shape[1] % 16, 0)

        generated = np.full_like(prepared.model_input, 255)
        patch = compose_generated_patch(prepared, generated)
        self.assertEqual(patch.bbox, (20, 15, 10, 8))
        self.assertEqual(patch.image.shape, (8, 10, 3))
        self.assertTrue(np.all(patch.image == 255))
        self.assertTrue(np.all(patch.alpha == 255))

    def test_prepare_rejects_wrong_or_empty_masks_and_output_shape(self):
        page = np.zeros((20, 20, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ImageOperationError, "dimensions"):
            prepare_edit(page, (2, 2, 5, 5), edit_mask=np.zeros((10, 10), dtype=np.uint8))
        with self.assertRaisesRegex(ImageOperationError, "does not affect"):
            prepare_edit(page, (2, 2, 5, 5), edit_mask=np.zeros((20, 20), dtype=np.uint8))
        prepared = prepare_edit(page, (2, 2, 5, 5), minimum_context=0)
        with self.assertRaisesRegex(ImageOperationError, "dimensions"):
            compose_generated_patch(prepared, np.zeros((17, 17, 3), dtype=np.uint8))

    def test_edit_margin_preserves_overflow_but_not_context_pixels(self):
        page = np.zeros((100, 120, 3), dtype=np.uint8)
        prepared = prepare_edit(page, (40, 40, 20, 10), edit_margin=12)
        self.assertEqual(prepared.selection_bbox, (40, 40, 20, 10))
        patch = compose_generated_patch(prepared, np.full_like(prepared.model_input, 255))
        self.assertEqual(patch.bbox, (28, 28, 44, 34))
        result = page.copy()
        x, y, w, h = patch.bbox
        result[y:y+h, x:x+w] = patch.image
        self.assertTrue(np.all(result[40, 65] == 255))
        self.assertTrue(np.all(result[40, 75] == 0))
        np.testing.assert_array_equal(page, 0)

    def test_edit_margin_clamps_to_page_and_rejects_invalid_or_painted_expansion(self):
        page = np.zeros((30, 40, 3), dtype=np.uint8)
        for box, expected in [((1, 2, 5, 6), (0, 0, 16, 18)),
                              ((35, 24, 5, 6), (25, 14, 15, 16))]:
            prepared = prepare_edit(page, box, edit_margin=10)
            patch = compose_generated_patch(prepared, np.full_like(prepared.model_input, 255))
            self.assertEqual(patch.bbox, expected)
        for margin in (-1, 2.5, True):
            with self.assertRaises(ImageOperationError):
                prepare_edit(page, (1, 2, 5, 6), edit_margin=margin)
        with self.assertRaisesRegex(ImageOperationError, "rectangular"):
            prepare_edit(page, (1, 2, 5, 6), edit_margin=10,
                         edit_mask=make_rect_mask((40, 30), (1, 2, 5, 6)))

    def test_pixel_hash_is_stable_and_sensitive_to_content(self):
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = first.copy()
        self.assertEqual(sha256_pixels(first), sha256_pixels(second))
        second[0, 0] = 1
        self.assertNotEqual(sha256_pixels(first), sha256_pixels(second))


if __name__ == "__main__":
    unittest.main()
