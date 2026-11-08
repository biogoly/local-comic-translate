"""Measure source lettering against independently rotated synthetic type."""
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from modules.detection.heuristic_lines import annotate_blocks_with_heuristic_lines
from modules.detection.rtdetr_v2_onnx import RTDetrV2ONNXDetection
from modules.utils.textblock import TextBlock


def lettering(text, clockwise, *, light_ink=False, border=False):
    font = ImageFont.load_default(size=30)
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    x, y, right, bottom = measure.multiline_textbbox((0, 0), text, font=font, spacing=12)
    image = Image.new("L", (right - x + 24, bottom - y + 24), 255)
    ImageDraw.Draw(image).multiline_text((12 - x, 12 - y), text, font=font, fill=0, spacing=12)
    image = image.rotate(-clockwise, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    x, y, right, bottom = ImageOps.invert(image).getbbox()
    image = image.crop((x - 4, y - 4, right + 4, bottom + 4))
    if border:
        ImageDraw.Draw(image).line((0, 18, image.width - 1, 4), fill=0, width=3)
    if light_ink:
        image = ImageOps.invert(image)
    return np.array(image.convert("RGB"))


def detect(image):
    h, w = image.shape[:2]
    return RTDetrV2ONNXDetection().create_text_blocks(image, [[0, 0, w, h]], [])[0]


class LetteringAngleDetectionTests(unittest.TestCase):
    def test_detect_measures_clockwise_and_counterclockwise_lettering(self):
        for angle in (-28, -12, 5, 18, 31):
            with self.subTest(clockwise=angle):
                image = lettering("RAILWAY STATION", angle)
                original = image.copy()
                block = detect(image)
                self.assertIsNotNone(block.detected_lettering_angle)
                self.assertAlmostEqual(block.detected_lettering_angle, angle, delta=2.0)
                self.assertEqual(block.angle, 0)
                np.testing.assert_array_equal(block.xyxy, [0, 0, image.shape[1], image.shape[0]])
                np.testing.assert_array_equal(original, image)

    def test_measures_light_and_dark_ink_and_ignores_a_slanted_border(self):
        for light_ink in (True, False):
            with self.subTest(light_ink=light_ink):
                block = detect(lettering("Café des Roses", 25, light_ink=light_ink, border=True))
                self.assertIsNotNone(block.detected_lettering_angle)
                self.assertAlmostEqual(block.detected_lettering_angle, 25, delta=2)
                self.assertIsNone(detect(lettering("Café des Roses", 0, border=True)).detected_lettering_angle)

    def test_parallel_lines_share_a_confident_angle(self):
        block = detect(lettering("Danger ahead!\nStay behind me.", 25))
        self.assertGreaterEqual(len(block.lines), 2)
        self.assertAlmostEqual(block.detected_lettering_angle, 25, delta=2)

    def test_parallel_baselines_survive_a_merged_ocr_line(self):
        # The OCR line finder merges both rows at this tilt. Their separated
        # glyph bands still provide two independent baseline measurements.
        block = detect(lettering("Danger ahead!\nStay behind me.", -15))
        self.assertAlmostEqual(block.detected_lettering_angle, -15, delta=2)

    def test_horizontal_weak_steep_and_vertical_samples_stay_unset(self):
        for text, angle in (("RAILWAY STATION", 0), ("ABC xyz", 0), ("RAILWAY STATION", 1),
                            ("SOS", -18), ("RAILWAY STATION", 47), ("RAILWAY STATION", 90)):
            with self.subTest(text=text, angle=angle):
                self.assertIsNone(detect(lettering(text, angle)).detected_lettering_angle)

    def test_unlettered_border_and_disagreeing_lines_are_ambiguous(self):
        blank = Image.new('RGB', (250, 100), 'white')
        ImageDraw.Draw(blank).line((4, 70, 240, 17), fill='black', width=5)
        self.assertIsNone(detect(np.asarray(blank)).detected_lettering_angle)
        top, bottom = lettering('First notice', 15), lettering('Second notice', -15)
        combined = np.full((len(top) + len(bottom) + 8, max(top.shape[1], bottom.shape[1]), 3), 255, dtype=np.uint8)
        combined[:len(top), :top.shape[1]] = top
        combined[-len(bottom):, :bottom.shape[1]] = bottom
        self.assertIsNone(detect(combined).detected_lettering_angle)

        rng = np.random.default_rng(678)
        dots = Image.new('RGB', (260, 130), 'white')
        draw = ImageDraw.Draw(dots)
        for x, y in rng.integers([4, 4], [250, 120], size=(65, 2)):
            draw.ellipse((x, y, x + 5, y + 7), fill='black')
        self.assertIsNone(detect(np.asarray(dots)).detected_lettering_angle)

    def test_ocr_line_preparation_preserves_prior_estimate_and_identical_geometry(self):
        image = lettering('Café des Roses', -19)
        measured = detect(image)
        h, w = image.shape[:2]
        block = TextBlock(text_bbox=np.array([0, 0, w, h]))
        block.detected_lettering_angle = 7
        with patch('modules.detection.heuristic_lines.core.estimate_lettering_angle') as estimator:
            annotate_blocks_with_heuristic_lines(image, [block])
            estimator.assert_not_called()
        self.assertEqual(block.detected_lettering_angle, 7)
        self.assertEqual(block.lines, measured.lines)
        self.assertEqual(block.direction, measured.direction)
        np.testing.assert_array_equal(block.xyxy, measured.xyxy)
