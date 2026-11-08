"""Typography boundary geometry is independent of destructive cleanup masks."""
import unittest

import numpy as np
from PIL import Image, ImageDraw

from modules.rendering.bubble_boundary import estimate_bubble_boundary, MAX_ROI_SIDE


def fixture(shape="ellipse", color=(250, 250, 250), text=True):
    image = Image.new("RGB", (240, 200), (140, 95, 70))
    draw = ImageDraw.Draw(image)
    if shape == "ellipse":
        draw.ellipse((20, 20, 219, 179), fill=color, outline=(0, 0, 0), width=3)
    elif shape == "rectangle":
        draw.rectangle((20, 20, 219, 179), fill=color, outline=(0, 0, 0), width=3)
    elif shape == "tailed":
        # The narrow six-pixel protrusion must not become a writable text area.
        draw.rectangle((35, 25, 204, 144), fill=(0, 0, 0))
        draw.rectangle((111, 140, 122, 183), fill=(0, 0, 0))
        draw.rectangle((38, 28, 201, 141), fill=color)
        draw.rectangle((114, 138, 119, 180), fill=color)
    elif shape == "irregular":
        draw.polygon([(40, 25), (180, 20), (215, 65), (205, 160), (80, 180), (22, 125)], fill=(0, 0, 0))
        draw.polygon([(43, 29), (178, 24), (211, 66), (201, 156), (81, 176), (26, 124)], fill=color)
    if text:
        draw.rectangle((80, 80, 94, 87), fill=(0, 0, 0))
        draw.rectangle((102, 80, 119, 87), fill=(0, 0, 0))
        draw.rectangle((130, 100, 148, 107), fill=(0, 0, 0))
    return np.array(image)


def at(result, x, y):
    return bool(result.mask[y - result.origin[1], x - result.origin[0]])


class BubbleBoundaryTests(unittest.TestCase):
    def test_enclosed_round_bubble_fills_letter_holes_and_excludes_corners(self):
        result = estimate_bubble_boundary(fixture(), (65, 65, 165, 125), (20, 20, 220, 180), padding=8)
        self.assertTrue(result.available, result.reason)
        self.assertEqual(result.source, "estimated")
        self.assertTrue(at(result, 85, 83))
        self.assertFalse(at(result, 30, 30))
        self.assertTrue(at(result, 120, 100))

    def test_enclosed_rectangle_keeps_useful_corner_area(self):
        result = estimate_bubble_boundary(fixture("rectangle"), (65, 65, 165, 125), (20, 20, 220, 180), padding=8)
        self.assertTrue(result.available, result.reason)
        self.assertTrue(at(result, 35, 35))
        self.assertFalse(at(result, 25, 25))

    def test_flat_colored_bubble_uses_rgb_background(self):
        result = estimate_bubble_boundary(fixture(color=(65, 155, 215)), (65, 65, 165, 125), (20, 20, 220, 180))
        self.assertTrue(result.available, result.reason)
        self.assertTrue(at(result, 110, 90))

    def test_narrow_tail_is_removed_even_without_padding(self):
        result = estimate_bubble_boundary(fixture("tailed"), (65, 65, 165, 125), (32, 22, 208, 187), padding=0)
        self.assertTrue(result.available, result.reason)
        self.assertFalse(at(result, 116, 165))
        self.assertTrue(at(result, 116, 125))

    def test_clipped_closed_tail_on_each_edge_preserves_the_padded_body(self):
        # Rotate the same enclosed balloon through all four edges. A loose box
        # includes the whole tail and supplies an independent reference mask.
        image = fixture("tailed")
        text = (65, 65, 165, 125)
        tight = (32, 22, 208, 148)
        loose = (32, 22, 208, 187)
        for turns in range(4):
            with self.subTest(turns=turns):
                fitted = estimate_bubble_boundary(image, text, tight)
                reference = estimate_bubble_boundary(image, text, loose)
                self.assertTrue(fitted.available, fitted.reason)
                self.assertTrue(reference.available, reference.reason)
                page_masks = []
                for result in (fitted, reference):
                    canvas = np.zeros(image.shape[:2], dtype=bool)
                    x, y = result.origin
                    h, w = result.mask.shape
                    canvas[y:y+h, x:x+w] = result.mask
                    page_masks.append(canvas)
                np.testing.assert_array_equal(*page_masks)
            width = image.shape[1]
            def rotate(box):
                left, top, right, bottom = box
                return top, width - right, bottom, width - left
            text, tight, loose = map(rotate, (text, tight, loose))
            image = np.rot90(image)

    def test_clipped_tail_rejects_open_channel_and_adjacent_pocket(self):
        for kind in ("page edge", "wider pocket", "return into artwork"):
            with self.subTest(kind=kind):
                image = fixture("tailed")
                if kind == "page edge":
                    image[162:, 114:120] = 250
                elif kind == "wider pocket":
                    image[174:189, 90:142] = 250
                else:
                    image[162:172, 119:129] = 250
                    image[150:172, 127:130] = 250
                result = estimate_bubble_boundary(image, (65, 65, 165, 125), (32, 22, 208, 148))
                self.assertEqual(result.reason, "open_boundary")

    def test_irregular_enclosed_shape_retains_its_geometry(self):
        result = estimate_bubble_boundary(fixture("irregular"), (65, 65, 165, 125), (18, 16, 220, 185), padding=4)
        self.assertTrue(result.available, result.reason)
        self.assertTrue(at(result, 120, 100))
        self.assertFalse(at(result, 30, 30))

    def test_open_outline_on_same_background_is_rejected(self):
        image = fixture("rectangle", text=False)
        image[:] = 250
        image[20:180, 20:23] = 0
        image[20:180, 217:220] = 0
        image[20:23, 20:220] = 0
        image[177:180, 20:220] = 0
        image[70:95, 217:220] = 250
        result = estimate_bubble_boundary(image, (65, 65, 165, 125), (20, 20, 220, 180))
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "open_boundary")

    def test_borderless_background_is_not_assumed_to_be_an_ellipse(self):
        image = np.full((200, 240, 3), 250, dtype=np.uint8)
        result = estimate_bubble_boundary(image, (65, 65, 165, 125), (20, 20, 220, 180))
        self.assertFalse(result.available)
        self.assertEqual(result.source, "none")

    def test_gradient_or_texture_is_declined(self):
        image = fixture("rectangle", text=False)
        for y in range(23, 177):
            image[y, 23:217] = (80 + y // 2, 80 + y // 2, 80 + y // 2)
        result = estimate_bubble_boundary(image, (65, 65, 165, 125), (20, 20, 220, 180))
        self.assertFalse(result.available)
        self.assertIn(result.reason, {"ambiguous_background", "text_not_contained"})

    def test_padding_is_respected_and_monotonic(self):
        image = fixture("rectangle")
        args = (image, (65, 65, 165, 125), (20, 20, 220, 180))
        small = estimate_bubble_boundary(*args, padding=2)
        large = estimate_bubble_boundary(*args, padding=16)
        self.assertTrue(small.available and large.available)
        self.assertTrue(np.all(~large.mask | small.mask))
        self.assertGreater(small.mask.sum(), large.mask.sum())
        self.assertFalse(at(large, 30, 100))

    def test_estimation_does_not_modify_source(self):
        image = fixture()
        before = image.copy()
        image.setflags(write=False)
        estimate_bubble_boundary(image, (65, 65, 165, 125), (20, 20, 220, 180))
        np.testing.assert_array_equal(image, before)

    def test_explicit_overrides_work_without_image(self):
        rectangle = estimate_bubble_boundary(None, (10, 10, 90, 90), override="rectangle", padding=5)
        ellipse = estimate_bubble_boundary(None, (10, 10, 90, 90), override="ellipse", padding=5)
        self.assertTrue(rectangle.available and ellipse.available)
        self.assertEqual(rectangle.reliability, "explicit")
        self.assertTrue(at(rectangle, 16, 16))
        self.assertFalse(at(ellipse, 16, 16))
        self.assertFalse(at(rectangle, 13, 40))

    def test_override_bounds_take_precedence_and_clip_integer_roi(self):
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        result = estimate_bubble_boundary(image, (10, 10, 30, 30), (5, 5, 35, 35),
                                         override="rectangle", override_bounds=(-4.2, -2.5, 65.8, 55.1), padding=0)
        self.assertEqual(result.origin, (0, 0))
        self.assertEqual(result.mask.shape, (40, 50))
        self.assertTrue(result.mask.all())
        result = estimate_bubble_boundary(None, (1, 1, 5, 5), override="rectangle",
                                         override_bounds=(10.2, 20.8, 30.1, 40.2), padding=0)
        self.assertEqual(result.origin, (10, 20))
        self.assertEqual(result.mask.shape, (21, 21))

    def test_invalid_or_huge_requests_fail_without_allocating_large_masks(self):
        for bounds in ((1, 1, 0, 5), (0, 0, float("nan"), 10), None):
            self.assertEqual(estimate_bubble_boundary(None, bounds).reason, "invalid_bounds")
        self.assertEqual(estimate_bubble_boundary(None, (1, 1, 10, 10)).reason, "missing_image")
        self.assertEqual(estimate_bubble_boundary(None, (1, 1, 10, 10), padding=-1).reason, "invalid_padding")
        self.assertEqual(estimate_bubble_boundary(None, (1, 1, 10, 10), override="bad").reason, "invalid_override")
        huge = estimate_bubble_boundary(None, (0, 0, MAX_ROI_SIDE + 1, 2), override="rectangle")
        self.assertEqual(huge.reason, "region_too_large")
        self.assertEqual(huge.mask.size, 0)

    def test_excessive_padding_has_explicit_failure(self):
        result = estimate_bubble_boundary(None, (0, 0, 20, 20), override="ellipse", padding=50)
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "padding_empty")

    def test_missing_detector_box_can_search_for_local_closed_bubble(self):
        result = estimate_bubble_boundary(fixture("rectangle"), (50, 50, 190, 150), padding=8)
        self.assertTrue(result.available, result.reason)


if __name__ == "__main__":
    unittest.main()
