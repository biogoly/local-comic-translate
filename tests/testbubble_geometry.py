import importlib.util
from pathlib import Path
import unittest

import numpy as np


_MODULE_PATH = Path(__file__).parents[1] / "modules" / "utils" / "bubble_geometry.py"
_SPEC = importlib.util.spec_from_file_location("bubble_geometry", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_fallback_bubble_clip = _MODULE.build_fallback_bubble_clip


class BubbleFallbackGeometryTests(unittest.TestCase):
    def test_rectangular_caption_keeps_inset_corners(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:90, 10:90] = 240
        image[47:53, 47:53] = 0

        clip = build_fallback_bubble_clip(
            (100, 100),
            (0, 0, 100, 100),
            (10, 10, 90, 90),
            inset=3,
            image=image,
            seed_bbox=(40, 40, 60, 60),
        )

        self.assertTrue(clip[15, 15])
        self.assertTrue(clip[50, 50])
        self.assertFalse(clip[9, 9])

    def test_round_balloon_retains_ellipse_fallback(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:100, :100]
        circle = (xx - 50) ** 2 + (yy - 50) ** 2 <= 40 ** 2
        image[circle] = 240
        image[47:53, 47:53] = 0

        clip = build_fallback_bubble_clip(
            (100, 100),
            (0, 0, 100, 100),
            (10, 10, 90, 90),
            inset=3,
            image=image,
            seed_bbox=(40, 40, 60, 60),
        )

        self.assertTrue(clip[50, 50])
        self.assertFalse(clip[15, 15])

    def test_missing_pixels_uses_conservative_ellipse(self):
        clip = build_fallback_bubble_clip(
            (100, 100),
            (0, 0, 100, 100),
            (10, 10, 90, 90),
            inset=3,
        )

        self.assertTrue(clip[50, 50])
        self.assertFalse(clip[15, 15])


if __name__ == "__main__":
    unittest.main()
