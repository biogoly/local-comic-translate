"""Final pixel protection through real pipeline/export/project code.

Inference is replaced by intentionally mutating backends. PNG export and v2
project roundtrips use the installed image/SQLite implementation.
"""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import imkit as imk
import numpy as np
from PySide6.QtWidgets import QApplication

from app.projects.project_state_v2 import close_cached_connection, load_state_from_proj_file_v2, save_state_to_proj_file_v2
from modules.repair.protection import PixelProtectionError, encode_protection, get_protection
from pipeline.inpainting import call_inpaint_image
from pipeline.virtual_page import VirtualPage
from pipeline.webtoon_batch.chunk import ChunkMixin, _record_pixel_protection
import testartistic_edit_project_state as project_fixtures
import testreview_batch as batch_fixtures


class InpaintingProtectionBoundaryTests(unittest.TestCase):
    def test_backend_dilation_and_mutation_cannot_cross_protected_pixels(self):
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                image = np.random.default_rng(4).integers(0, 255, (24, 30, 3), dtype=np.uint8)
                original = image.copy()
                mask = np.zeros((24, 30), np.uint8)
                mask[4:20, 3:27] = 255
                original_mask = mask.copy()
                protected = np.zeros_like(mask)
                protected[8:15, 10:20] = 1
                def backend(pixels, effective_mask, _config, **_kwargs):
                    self.assertTrue(np.all(effective_mask[protected != 0] == 0))
                    # Simulate dilation/feather/fast-fill that writes much more
                    # than requested, even mutating the input mask and pixels.
                    effective_mask[:] = 255
                    pixels[:] = (8, 130, 250)
                    return pixels
                result = call_inpaint_image(SimpleNamespace(inpaint_image=backend), image, mask, None,
                                            automatic=automatic, pixel_protection=protected)
                np.testing.assert_array_equal(result[protected != 0], original[protected != 0])
                np.testing.assert_array_equal(result[5, 5], (8, 130, 250))
                if automatic:
                    np.testing.assert_array_equal(result[original_mask == 0], original[original_mask == 0])
                np.testing.assert_array_equal(image, original)
                np.testing.assert_array_equal(mask, original_mask)

    def test_invalid_shape_is_rejected_before_backend_execution(self):
        backend = Mock()
        with self.assertRaises(PixelProtectionError):
            call_inpaint_image(SimpleNamespace(inpaint_image=backend), np.zeros((20, 30, 3), np.uint8),
                               np.ones((20, 30), np.uint8), None, pixel_protection=np.ones((10, 30), np.uint8))
        backend.assert_not_called()

    def test_fully_protected_automatic_cleanup_skips_backend(self):
        image = np.full((20, 30, 3), (40, 60, 80), np.uint8)
        backend = Mock()
        result = call_inpaint_image(SimpleNamespace(inpaint_image=backend), image,
                                    np.full((20, 30), 255, np.uint8), None, automatic=True,
                                    pixel_protection=np.ones((20, 30), np.uint8))
        backend.assert_not_called()
        np.testing.assert_array_equal(result, image)


class _SeamProcessor(ChunkMixin):
    def __init__(self, states):
        self.main_page = SimpleNamespace(image_states=states, settings_page=None)
        self.seam_crop_pad_x = 5
        self.seam_crop_pad_y = 5
        self._get_page_scene_offset = lambda _index: 0
        self.backend_calls = []
        self.inpainting = SimpleNamespace(inpaint_image=self.backend)

    def backend(self, image, mask, _config, **kwargs):
        self.backend_calls.append(mask.copy())
        mask[:] = 255
        image[:] = 77
        return image


class WebtoonPixelProtectionTests(unittest.TestCase):
    def run_seam(self, virtual):
        width, height = 64, 100
        records, states, masks = [], {}, []
        for index in range(2):
            path = "long.png" if virtual else f"page-{index}.png"
            mask = np.zeros((300 if virtual else height, width), np.uint8)
            offset = 100 * (index + 1) if virtual else 0
            protected_y = 94 if index == 0 else 4
            mask[offset + protected_y:offset + protected_y + 3, 20:25] = 255
            if path in states:
                from modules.repair.protection import decode_protection
                mask |= decode_protection(states[path]["pixel_protection"])
            states[path] = {"pixel_protection": encode_protection(mask)}
            local_mask = np.zeros((height, width), bool)
            local_mask[protected_y:protected_y + 3, 20:25] = True
            masks.append(local_mask)
            record = {"path": path, "global_index": 0 if virtual else index, "y_offset": offset,
                      "image": np.zeros((height, width, 3), np.uint8),
                      "composed_image": np.full((height, width, 3), (index + 1) * 11, np.uint8)}
            if virtual:
                record["vpage"] = VirtualPage(physical_page_index=0, physical_page_path=path,
                    virtual_index=index + 1, crop_top=offset, crop_bottom=offset + height,
                    crop_height=height, physical_width=width, physical_height=300, virtual_id=str(index))
            records.append(record)
        processor = _SeamProcessor(states)
        seam = SimpleNamespace(top_page_index=0, bottom_page_index=1,
            matches=[SimpleNamespace(owner_block=batch_fixtures.block((10, 90, 42, 112)))])
        before = [record["composed_image"].copy() for record in records]
        with patch("pipeline.webtoon_batch.chunk.get_config", return_value=None), \
             patch("pipeline.webtoon_batch.chunk.generate_mask", side_effect=lambda image, _blocks:
                   np.full(image.shape[:2], 255, np.uint8)):
            patches = processor._process_seam_job_ocr_and_inpaint(seam, records)
        self.assertEqual(len(processor.backend_calls), 1)
        for index, record in enumerate(records):
            pixels = before[index].copy()
            self.assertTrue(patches[index])
            for piece in patches[index]:
                x, y, w, h = piece["bbox"]
                y -= record["y_offset"]
                pixels[y:y+h, x:x+w] = piece["image"]
            np.testing.assert_array_equal(pixels[masks[index]], before[index][masks[index]])
            np.testing.assert_array_equal(pixels, record["composed_image"])
            self.assertTrue(np.any(pixels == 77))
            self.assertTrue(np.all(record["image"] == 0))

    def test_both_physical_pages_are_protected_across_seam(self):
        self.run_seam(virtual=False)

    def test_both_virtual_crops_use_physical_page_mask_offsets(self):
        self.run_seam(virtual=True)

    def test_regular_virtual_chunk_crops_mask_and_rejects_invalid_source_shape(self):
        mask = np.zeros((300, 64), np.uint8)
        mask[150:155, 20:25] = 255
        main = SimpleNamespace(image_states={"page": {"pixel_protection": encode_protection(mask)}})
        vpage = VirtualPage(physical_page_index=0, physical_page_path="page", virtual_index=1,
            crop_top=100, crop_bottom=200, crop_height=100, physical_width=64, physical_height=300, virtual_id="v1")
        actual = _record_pixel_protection(main, {"path": "page", "vpage": vpage}, np.zeros((100, 64, 3), np.uint8))
        np.testing.assert_array_equal(actual, mask[100:200])
        with self.assertRaises(PixelProtectionError):
            _record_pixel_protection(main, {"path": "page", "vpage": vpage}, np.zeros((90, 64, 3), np.uint8))


class AutomaticProtectionExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_automatic_saved_cleaned_png_preserves_current_protected_pixels(self):
        fixture = batch_fixtures.ReviewBatchTests()
        original_setup = fixture.setup_page
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            def configure():
                original_setup()
                state = fixture.main.image_states.pop("page.png")
                fixture.main.image_states[path] = state
                fixture.main.image_files = [path]
                protected = np.zeros(fixture.original.shape[:2], np.uint8)
                protected[30:45, 285:305] = 255
                state["pixel_protection"] = encode_protection(protected)
                fixture.protected = protected != 0
                fixture.main.settings_page.get_export_settings = lambda: {
                    "export_raw_text": False, "export_translated_text": False, "export_inpainted_image": True}
            fixture.setup_page = configure
            fixture.run_page()
            outputs = list(Path(tmp).glob("comic_translate_*/cleaned_images/page_cleaned.png"))
            self.assertEqual(len(outputs), 1)
            saved = imk.read_image(str(outputs[0]))
            np.testing.assert_array_equal(saved[fixture.protected], fixture.composed[fixture.protected])
            self.assertTrue(np.all(saved[50:60, 285:305] == 5))
            emitted = fixture.main.patches_processed.emit.call_args.args[0][0]["image"]
            np.testing.assert_array_equal(saved, emitted)
            fixture.main.image_skipped.emit.assert_not_called()


class PixelProtectionProjectTests(unittest.TestCase):
    def roundtrip(self, record):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir, load_dir = os.path.join(tmp, "save"), os.path.join(tmp, "load")
            os.makedirs(save_dir)
            os.makedirs(load_dir)
            project = os.path.join(tmp, "protected.ctpr")
            page = project_fixtures._write_png(os.path.join(save_dir, "page.png"), (40, 120, 180))
            main = project_fixtures._make_controller(save_dir, [page])
            main.image_states[page] = {"pixel_protection": copy.deepcopy(record)} if record is not None else {}
            try:
                save_state_to_proj_file_v2(main, project)
                loaded = project_fixtures._make_controller(load_dir)
                load_state_from_proj_file_v2(loaded, project)
                loaded_page = loaded.image_files[0]
                state = copy.deepcopy(loaded.image_states[loaded_page])
                return state
            finally:
                close_cached_connection(project)

    def test_v2_save_reopen_preserves_portable_protection(self):
        mask = np.zeros((8, 8), np.uint8)
        mask[2:5, 3:7] = 255
        record = encode_protection(mask)
        state = self.roundtrip(record)
        self.assertEqual(state["pixel_protection"], record)
        actual = get_protection(SimpleNamespace(image_states={"reopened": state}), "reopened", (8, 8, 3))
        np.testing.assert_array_equal(actual, mask)

    def test_legacy_project_without_protection_stays_unprotected(self):
        state = self.roundtrip(None)
        self.assertNotIn("pixel_protection", state)
        self.assertIsNone(get_protection(SimpleNamespace(image_states={"reopened": state}), "reopened", (8, 8)))

    def test_unknown_version_survives_reload_and_cannot_silently_disable_protection(self):
        record = encode_protection(np.ones((8, 8), np.uint8))
        record["schema_version"] = 999
        state = self.roundtrip(record)
        with self.assertRaises(PixelProtectionError):
            get_protection(SimpleNamespace(image_states={"reopened": state}), "reopened", (8, 8))


if __name__ == "__main__":
    unittest.main()
