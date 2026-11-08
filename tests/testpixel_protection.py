import base64
import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
import zlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from app.ui.canvas.image_viewer import ImageViewer
from app.ui.commands.inpaint import PatchInsertCommand
from modules.repair.protection import (
    MAX_PIXELS,
    PixelProtectionError,
    decode_protection,
    encode_protection,
    get_protection,
    normalize_protection,
    protect_image,
    protect_patches,
)


class ProtectionCodecTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((11, 13), np.uint8)
        self.mask[2:6, 3:8] = 1
        self.mask[8, 10] = 128
        self.record = encode_protection(self.mask)

    def test_json_roundtrip_is_binary_and_shape_checked(self):
        record = json.loads(json.dumps(self.record))
        decoded = decode_protection(record, (11, 13, 3))
        np.testing.assert_array_equal(decoded, (self.mask != 0).astype(np.uint8) * 255)
        self.assertEqual(decoded.dtype, np.uint8)
        with self.assertRaises(PixelProtectionError):
            decode_protection(record, (13, 11))

    def test_empty_masks_and_missing_page_state_are_absent(self):
        self.assertIsNone(encode_protection(np.zeros((2, 3), bool)))
        self.assertIsNone(decode_protection(None))
        self.assertIsNone(decode_protection({}))
        self.assertIsNone(get_protection(SimpleNamespace(), "page.png"))
        self.assertIsNone(get_protection(SimpleNamespace(image_states={}), "page.png", (3, 4)))

    def test_normalization_strips_unrecognized_fields(self):
        record = dict(self.record, unrelated="discard")
        self.assertEqual(normalize_protection(record), self.record)
        main = SimpleNamespace(image_states={"page.png": {"pixel_protection": record}})
        np.testing.assert_array_equal(get_protection(main, "page.png", self.mask.shape), decode_protection(self.record))

    def test_unknown_version_and_invalid_dimensions_fail_closed(self):
        for changes in (
            {"schema_version": 2}, {"schema_version": True}, {"schema_version": 1.0},
            {"width": 0}, {"width": -1}, {"height": True}, {"height": "11"},
            {"width": MAX_PIXELS}, {"width": 100_000, "height": 100_000},
        ):
            with self.subTest(changes=changes), self.assertRaises(PixelProtectionError):
                decode_protection(dict(self.record, **changes))

    def test_corrupt_truncated_trailing_and_oversized_payloads_fail_closed(self):
        expected_bytes = (self.mask.size + 7) // 8
        payloads = [
            "!base64", "", "a" * 10_000,
            base64.b64encode(b"not zlib").decode(),
            base64.b64encode(zlib.compress(b"\x00" * expected_bytes)[:-1]).decode(),
            base64.b64encode(zlib.compress(b"\x00" * expected_bytes) + b"trailing").decode(),
            base64.b64encode(zlib.compress(b"\x00" * (expected_bytes - 1))).decode(),
            base64.b64encode(zlib.compress(b"\x00" * 100_000)).decode(),
        ]
        for payload in payloads:
            with self.subTest(payload=payload[:30]), self.assertRaises(PixelProtectionError):
                decode_protection(dict(self.record, data=payload))

    def test_invalid_padding_and_noninteger_masks_are_rejected(self):
        record = {"schema_version": 1, "width": 1, "height": 1,
                  "data": base64.b64encode(zlib.compress(b"\x81")).decode()}
        with self.assertRaises(PixelProtectionError):
            decode_protection(record)
        for mask in (np.zeros((2, 2, 3), np.uint8), np.full((2, 2), np.nan), [[1]]):
            with self.assertRaises(PixelProtectionError):
                encode_protection(mask)


class ProtectionPixelTests(unittest.TestCase):
    def setUp(self):
        self.current = np.arange(12 * 14 * 3, dtype=np.uint16).reshape(12, 14, 3).astype(np.uint8)
        self.mask = np.zeros((12, 14), np.uint8)
        self.mask[4:9, 5:10] = 1

    def test_final_clip_preserves_exact_rgb_after_soft_blending(self):
        alpha = np.linspace(0, 1, 14)[None, :, None]
        generated = np.rint(self.current * (1 - alpha) + np.array([255, 3, 84]) * alpha).astype(np.uint8)
        expected = generated.copy()
        actual = protect_image(self.current, generated, self.mask)
        np.testing.assert_array_equal(actual[self.mask != 0], self.current[self.mask != 0])
        np.testing.assert_array_equal(actual[self.mask == 0], generated[self.mask == 0])
        np.testing.assert_array_equal(generated, expected)
        self.assertFalse(np.shares_memory(actual, generated))

    def test_partial_patch_preserves_metadata_and_input(self):
        patch = {"bbox": [3, 2, 7, 8], "image": np.full((8, 7, 3), 210, np.uint8),
                 "patch_id": "generated-id", "scene_pos": [50, 70], "page_index": 4}
        before = copy.deepcopy(patch)
        result = protect_patches(self.current, [patch], self.mask)[0]
        expected = np.full((8, 7, 3), 210, np.uint8)
        protected = self.mask[2:10, 3:10] != 0
        expected[protected] = self.current[2:10, 3:10][protected]
        np.testing.assert_array_equal(result["image"], expected)
        np.testing.assert_array_equal(patch["image"], before["image"])
        self.assertEqual({k: v for k, v in result.items() if k != "image"},
                         {k: v for k, v in patch.items() if k != "image"})

    def test_patches_overlapping_page_edges_keep_correct_source_coordinates(self):
        mask = np.ones((12, 14), np.uint8)
        for bbox in ((-3, -2, 8, 7), (10, 8, 8, 7), (20, 20, 2, 2)):
            x, y, width, height = bbox
            patch = {"bbox": bbox, "image": np.full((height, width, 3), 111, np.uint8)}
            result = protect_patches(self.current, [patch], mask)[0]["image"]
            for py in range(height):
                for px in range(width):
                    expected = self.current[py + y, px + x] if 0 <= py + y < 12 and 0 <= px + x < 14 else [111] * 3
                    np.testing.assert_array_equal(result[py, px], expected)

    def test_invalid_write_shapes_and_bounds_fail_closed(self):
        with self.assertRaises(PixelProtectionError):
            protect_image(self.current, self.current[:5], self.mask)
        with self.assertRaises(PixelProtectionError):
            protect_image(self.current, self.current, self.mask[:5])
        for bbox in ((0.5, 0, 3, 3), (0, 0, 3, 4), (0, 0, -3, 3)):
            with self.assertRaises(PixelProtectionError):
                protect_patches(self.current, [{"bbox": bbox, "image": np.zeros((3, 3, 3), np.uint8)}], self.mask)

    def test_no_protection_does_not_need_current_page(self):
        patches = [{"bbox": [1, 2, 3, 4]}]
        self.assertIs(protect_patches(None, patches, None), patches)


class ProtectionCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = "page.png"
        self.original = np.full((30, 40, 3), (20, 80, 140), np.uint8)
        self.viewer = ImageViewer(None)
        self.viewer.display_image_array(self.original, fit=False)
        self.main = SimpleNamespace(image_viewer=self.viewer, image_files=[self.path], curr_img_idx=0,
                                    temp_dir=self.tmp.name, image_patches={}, in_memory_patches={}, image_states={})
        self.stack = QUndoStack()

    def tearDown(self):
        self.stack.clear()
        self.viewer.close()
        self.tmp.cleanup()

    def patch(self, color):
        return {"bbox": [5, 5, 20, 20], "image": np.full((20, 20, 3), color, np.uint8)}

    def protect(self):
        mask = np.zeros(self.original.shape[:2], np.uint8)
        mask[10:15, 10:15] = 255
        self.main.image_states[self.path] = {"pixel_protection": encode_protection(mask)}
        return mask != 0

    def test_insertion_uses_existing_composed_patch_and_frozen_redo(self):
        self.stack.push(PatchInsertCommand(self.main, [self.patch((190, 30, 60))], self.path))
        before = self.viewer.get_image_array()
        protected = self.protect()
        calls = []
        self.main.image_ctrl = SimpleNamespace(get_composited_page_image=lambda path: calls.append(path) or self.viewer.get_image_array())
        command = PatchInsertCommand(self.main, [self.patch((5, 210, 40))], self.path)
        self.assertEqual(calls, [self.path])
        self.main.image_states.clear()
        self.stack.push(command)
        after = self.viewer.get_image_array()
        np.testing.assert_array_equal(after[protected], before[protected])
        np.testing.assert_array_equal(after[7, 7], (5, 210, 40))
        self.stack.undo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), before)
        self.stack.redo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), after)
        self.assertEqual(calls, [self.path])

    def test_active_protection_supports_normal_viewer_fallback(self):
        protected = self.protect()
        self.stack.push(PatchInsertCommand(self.main, [self.patch((5, 210, 40))], self.path))
        np.testing.assert_array_equal(self.viewer.get_image_array()[protected], self.original[protected])

    def test_no_mask_does_not_load_current_image(self):
        def unexpected_load(_):
            self.fail("No protection must not introduce a composed-page read.")
        self.main.image_ctrl = SimpleNamespace(get_composited_page_image=unexpected_load)
        self.stack.push(PatchInsertCommand(self.main, [self.patch((5, 210, 40))], self.path))
        np.testing.assert_array_equal(self.viewer.get_image_array()[12, 12], (5, 210, 40))

    def test_invalid_protection_or_missing_current_page_does_not_write_patch_files(self):
        self.protect()
        self.main.image_ctrl = SimpleNamespace(get_composited_page_image=lambda _: None)
        with self.assertRaises(PixelProtectionError):
            PatchInsertCommand(self.main, [self.patch((5, 210, 40))], self.path)
        self.assertEqual(os.listdir(self.tmp.name), [])
        self.main.image_states[self.path]["pixel_protection"]["schema_version"] = 999
        with self.assertRaises(PixelProtectionError):
            PatchInsertCommand(self.main, [self.patch((5, 210, 40))], self.path)
        self.assertEqual(os.listdir(self.tmp.name), [])


if __name__ == "__main__":
    unittest.main()
