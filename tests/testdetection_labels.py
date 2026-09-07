import unittest
from unittest.mock import Mock

import numpy as np

from modules.detection.base import DetectionEngine
from modules.detection.rtdetr_v2_onnx import RTDetrV2ONNXDetection
from modules.detection.utils.slicer import ImageSlicer


class _DetectionEngine(DetectionEngine):
    def initialize(self, **kwargs):
        pass

    def detect(self, image):
        return []


class DetectorRegionTests(unittest.TestCase):
    def test_text_without_a_bubble_outline_is_still_detected(self):
        image = np.full((120, 200, 3), 255, dtype=np.uint8)
        boxes = np.array([[10, 10, 80, 40], [100, 60, 180, 95]])
        blocks = _DetectionEngine().create_text_blocks(image, boxes, np.array([]))

        self.assertEqual(len(blocks), 2)
        self.assertEqual([block.text_class for block in blocks], ["text_free", "text_free"])
        np.testing.assert_array_equal([block.xyxy for block in blocks], boxes)

    def test_bubble_and_non_bubble_text_both_survive_block_creation(self):
        image = np.full((120, 200, 3), 255, dtype=np.uint8)
        boxes = np.array([[10, 10, 80, 40], [100, 60, 180, 95]])
        blocks = _DetectionEngine().create_text_blocks(
            image, boxes, np.array([[5, 5, 85, 45]])
        )

        self.assertEqual([block.text_class for block in blocks], ["text_bubble", "text_free"])
        np.testing.assert_array_equal([block.xyxy for block in blocks], boxes)

    def test_onnx_keeps_both_text_classes_and_original_merging_extent(self):
        engine = RTDetrV2ONNXDetection()
        engine.session = Mock()
        engine.session.run.return_value = [
            np.array([[1, 2]]),
            np.array([[[10, 10, 80, 40], [40, 10, 110, 40]]]),
            np.array([[0.9, 0.9]]),
        ]
        image = np.full((120, 200, 3), 255, dtype=np.uint8)
        bubbles, boxes = engine._detect_single_image(image)
        self.assertEqual(len(bubbles), 0)
        self.assertEqual(len(boxes), 2)
        blocks = engine.create_text_blocks(image, boxes, bubbles)
        self.assertEqual(len(blocks), 1)
        # Merge the full geometry instead of discarding the overlapping
        # text_free prediction in favor of the smaller text_bubble prediction.
        np.testing.assert_array_equal(blocks[0].xyxy, [10, 10, 110, 40])

    def test_slicer_keeps_text_regions_without_any_bubbles(self):
        slicer = ImageSlicer()
        image = np.full((700, 100, 3), 255, dtype=np.uint8)

        def detect(_slice):
            return np.array([]), np.array([[10, 10, 60, 40]])

        bubbles, boxes = slicer.process_slices_for_detection(image, detect)

        self.assertEqual(len(bubbles), 0)
        self.assertGreater(len(boxes), 1)


if __name__ == "__main__":
    unittest.main()
