"""Known color surfaces and explicit-sample failure cases; no model calls."""
import unittest

import numpy as np

from modules.repair.gradient import GradientFitError, GradientSampleLine, fit_gradient


class GradientRepairTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((90, 120))
        self.original = np.stack((30 + xx + yy, 60 + xx, 180 - yy), axis=-1).astype(np.uint8)
        self.mask = np.zeros((90, 120), np.uint8)
        self.mask[30:60, 40:80] = 255
        self.points = [(20, 15), (60, 15), (100, 15), (100, 45),
                       (100, 75), (60, 75), (20, 75), (20, 45)]

    def assert_error(self, code, points=None, source=None, mask=None):
        with self.assertRaises(GradientFitError) as error:
            fit_gradient(self.original if source is None else source,
                         self.mask if mask is None else mask,
                         self.points if points is None else points)
        self.assertEqual(error.exception.code, code)

    def test_removes_synthetic_lettering_without_sampling_nearby_border(self):
        damaged = self.original.copy()
        damaged[32:56:3, 42:77] = 0
        damaged[:, 81:87] = (240, 0, 20)  # Nearby frame, never selected.
        before = damaged.copy()
        result = fit_gradient(damaged, self.mask, self.points)
        np.testing.assert_array_equal(result.replacement[self.mask > 0], self.original[self.mask > 0])
        np.testing.assert_array_equal(result.replacement[self.mask == 0], damaged[self.mask == 0])
        np.testing.assert_array_equal(damaged, before)
        self.assertLess(result.rms_error, 1e-9)
        self.assertEqual(result.inlier_count, 8)
        self.assertFalse(result.replacement.flags.writeable)

    def test_robust_fit_ignores_one_user_sample_on_dark_ink(self):
        damaged = self.original.copy()
        x, y = self.points[1]
        damaged[y, x] = 0
        result = fit_gradient(damaged, self.mask, self.points)
        self.assertEqual(result.inlier_count, 7)
        np.testing.assert_array_equal(result.replacement[self.mask > 0], self.original[self.mask > 0])

    def test_four_consistent_corners_work_but_disagreement_needs_more_samples(self):
        corners = self.points[::2]
        result = fit_gradient(self.original, self.mask, corners)
        self.assertEqual(result.inlier_count, 4)
        damaged = self.original.copy()
        damaged[corners[0][1], corners[0][0]] = 0
        self.assert_error('inconsistent_samples', corners, damaged)

    def test_duplicates_do_not_count_as_independent_samples(self):
        self.assert_error('too_few_samples', [self.points[0]] * 4)
        result = fit_gradient(self.original, self.mask, self.points + self.points)
        self.assertEqual(result.sample_count, 8)

    def test_sample_lines_resist_print_specks_without_sampling_beside_the_trace(self):
        corners = self.points[::2]
        damaged = self.original.copy()
        for x, y in corners:
            damaged[y, x] = (12, 18, 24)
            damaged[y-1, x-5:x+6] = (255, 40, 0)
            damaged[y+1, x-5:x+6] = (255, 40, 0)
        lines = [GradientSampleLine((x-5, y), (x+5, y)) for x, y in corners]
        result = fit_gradient(damaged, self.mask, lines)
        self.assertEqual(result.sample_count, 4)
        np.testing.assert_allclose(result.replacement[self.mask > 0], self.original[self.mask > 0], atol=1)
        reversed_lines = [GradientSampleLine(line.end, line.start) for line in lines]
        repeated = fit_gradient(damaged, self.mask, reversed_lines + lines)
        np.testing.assert_array_equal(repeated.replacement, result.replacement)
        self.assertEqual(repeated.sample_count, 4)

    def test_one_line_is_one_observation_and_cannot_cross_a_masked_pixel(self):
        line = GradientSampleLine((8, 9), (89, 11))
        self.assert_error('too_few_samples', [line]*4)
        crossing = GradientSampleLine((5, 2), (115, 78))
        self.assert_error('sample_inside_mask', self.points + [crossing])
        invalid = GradientSampleLine((5, 2), (150, 78))
        self.assert_error('sample_outside_image', self.points + [invalid])

    def test_rank_and_coverage_failures_do_not_fallback_to_solid(self):
        self.assert_error('degenerate_samples', [(5, 5), (10, 10), (15, 15), (20, 20)])
        self.assert_error('insufficient_coverage', [(5, 5), (15, 5), (15, 15), (5, 15)])
        self.assert_error('too_few_samples', [])
        self.assert_error('too_few_samples', self.points[:3])

    def test_samples_surrounding_round_mask_need_not_cover_empty_box_corners(self):
        yy, xx = np.indices(self.mask.shape)
        mask = np.zeros_like(self.mask)
        mask[(xx - 60) ** 2 + (yy - 45) ** 2 <= 100] = 255
        points = [(60, 29), (76, 45), (60, 61), (44, 45)]
        result = fit_gradient(self.original, mask, points)
        np.testing.assert_array_equal(result.replacement[mask > 0], self.original[mask > 0])
        # A real painted pixel beyond the samples must still reject the fit,
        # even if its alpha is very small.
        mask[45, 85] = 1
        self.assert_error('insufficient_coverage', points, mask=mask)

    def test_irregular_mask_coverage_ignores_unpainted_bbox_corner(self):
        mask = np.zeros_like(self.mask)
        mask[30:60, 40:45] = 255
        mask[55:60, 40:80] = 255
        points = [(30, 20), (50, 20), (90, 55), (90, 70), (30, 70)]
        result = fit_gradient(self.original, mask, points)
        np.testing.assert_array_equal(result.replacement[mask > 0], self.original[mask > 0])

    def test_disjoint_masks_must_all_be_inside_sample_coverage(self):
        mask = np.zeros_like(self.mask)
        mask[30:40, 35:45] = 255
        mask[60:70, 75:85] = 255
        points = [(20, 20), (100, 20), (100, 80), (20, 80)]
        result = fit_gradient(self.original, mask, points)
        np.testing.assert_array_equal(result.replacement[mask > 0], self.original[mask > 0])
        self.assert_error('insufficient_coverage', [(25, 20), (55, 20), (55, 50), (25, 50)], mask=mask)

    def test_single_pixel_and_thin_masks_keep_valid_coverage(self):
        for rows, columns in ((slice(45, 46), slice(60, 61)),
                              (slice(45, 46), slice(40, 80)),
                              (slice(30, 60), slice(60, 61))):
            with self.subTest(rows=rows, columns=columns):
                mask = np.zeros_like(self.mask)
                mask[rows, columns] = 255
                result = fit_gradient(self.original, mask, self.points)
                np.testing.assert_array_equal(result.replacement[mask > 0], self.original[mask > 0])

    def test_samples_must_be_finite_integer_unmasked_image_coordinates(self):
        for bad in ((1.5, 2), (float('nan'), 3), (float('inf'), 4)):
            self.assert_error('invalid_samples', self.points + [bad])
        for bad in ((-1, 2), (120, 2), (2, 90)):
            self.assert_error('sample_outside_image', self.points + [bad])
        self.assert_error('sample_inside_mask', self.points + [(45, 35)])
        soft = self.mask.copy()
        soft[15, 20] = 1
        self.assert_error('sample_inside_mask', mask=soft)

    def test_image_mask_and_sample_limits(self):
        self.assert_error('invalid_image', source=self.original.astype(np.float32))
        self.assert_error('invalid_image', source=self.original[:, :, :2])
        self.assert_error('invalid_mask', mask=self.mask[:10])
        self.assert_error('invalid_mask', mask=self.mask.astype(bool))
        self.assert_error('empty_mask', mask=np.zeros_like(self.mask))
        self.assert_error('too_many_samples', [(x, 0) for x in range(65)])

    def test_too_many_conflicting_samples_fail(self):
        damaged = self.original.copy()
        for i, (x, y) in enumerate(self.points):
            damaged[y, x] = ((i * 73) % 256, (i * 107) % 256, (i * 149) % 256)
        self.assert_error('inconsistent_samples', source=damaged)

    def test_repeat_preview_uses_fixed_source_independent_of_click_order(self):
        source = np.full_like(self.original, 255)
        source[15, 20] = (235, 250, 255)
        first = fit_gradient(source, self.mask, self.points)
        second = fit_gradient(source, self.mask, list(reversed(self.points)))
        np.testing.assert_array_equal(first.replacement, second.replacement)
        self.assertEqual(first.replacement.dtype, np.uint8)
        self.assertGreaterEqual(first.replacement.min(), 0)
        self.assertLessEqual(first.replacement.max(), 255)

    def test_small_fit_overshoots_clamp_instead_of_wrapping(self):
        corners = self.points[::2]
        mask = np.zeros_like(self.mask)
        mask[68:74, 90:99] = 255
        for base, sample in ((255, (235, 250, 255)), (0, (20, 5, 0))):
            source = np.full_like(self.original, base)
            source[15, 20] = sample
            result = fit_gradient(source, mask, corners)
            # Least squares predicts slightly outside the RGB range near the
            # opposite corner even though every input color is valid.
            np.testing.assert_array_equal(result.replacement[mask > 0], base)

    def test_rejected_corner_cannot_leave_an_extrapolated_repair(self):
        points = self.points[::2] + [self.points[1], self.points[5]]
        source = self.original.copy()
        source[75, 100] = 0
        self.assert_error('insufficient_coverage', points, source)

    def test_local_coordinates_support_large_translated_page_offsets(self):
        large = np.full((900, 1200, 3), 0, np.uint8)
        large[700:790, 1000:1120] = self.original
        mask = np.zeros(large.shape[:2], np.uint8)
        mask[700:790, 1000:1120] = self.mask
        points = [(x + 1000, y + 700) for x, y in self.points]
        result = fit_gradient(large, mask, points)
        np.testing.assert_array_equal(result.replacement[mask > 0], large[mask > 0])

    def test_exact_mask_compositing_preserves_holes_and_soft_edges(self):
        # The caller's compositor must use current pixels even when fitting
        # from an original image. This exercises the existing repair boundary.
        from app.controllers.local_repair import repair_patch
        mask = self.mask.copy()
        mask[40:45, 50:60] = 0
        mask[30, 40:80] = 64
        current = np.full_like(self.original, (210, 15, 30))
        result = fit_gradient(self.original, mask, self.points)
        patch = repair_patch(current, result.replacement, mask)
        x, y, width, height = patch['bbox']
        after = current.copy()
        after[y:y + height, x:x + width] = patch['image']
        np.testing.assert_array_equal(after[mask == 0], current[mask == 0])
        np.testing.assert_array_equal(after[mask == 255], self.original[mask == 255])
        expected = np.rint(current[30, 50] * (1 - 64 / 255) + self.original[30, 50] * (64 / 255))
        np.testing.assert_array_equal(after[30, 50], expected.astype(np.uint8))


if __name__ == '__main__':
    unittest.main()
