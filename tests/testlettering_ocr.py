"""Isolated OCR routing and writable preview extraction; no model/network calls."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.artistic_edit.image_ops import compose_generated_patch, prepare_edit
from modules.artistic_edit.lettering_ocr import (
    capture_ocr_settings, prepare_lettering_crop, run_lettering_ocr,
)
from modules.ocr.factory import OCRFactory


def settings(key='Default', credentials=None):
    return SimpleNamespace(
        get_tool_selection=lambda _tool: key,
        get_credentials=lambda _service: credentials or {},
        is_gpu_enabled=lambda: False,
        ui=SimpleNamespace(tr=lambda text: text),
    )


class LetteringOCRTests(unittest.TestCase):
    def test_snapshot_canonicalizes_localized_selection_and_freezes_credentials(self):
        credentials = {'api_key': 'private-secret', 'save_key': False}
        live = settings('Localized GPT', credentials)
        live.ui.tr = lambda text: 'Localized GPT' if text == 'GPT-4.1-mini' else text
        snapshot = capture_ocr_settings(live)
        self.assertEqual(snapshot.ocr_key, 'GPT-4.1-mini')
        self.assertTrue(snapshot.hosted)
        self.assertFalse(snapshot.use_gpu)
        self.assertNotIn('private-secret', repr(snapshot))
        credentials['api_key'] = 'changed'
        live.get_tool_selection = lambda _tool: 'Default'
        live.is_gpu_enabled = lambda: True
        self.assertEqual(snapshot.get_tool_selection('ocr'), 'GPT-4.1-mini')
        self.assertEqual(snapshot.get_credentials('Open AI GPT')['api_key'], 'private-secret')
        self.assertEqual(snapshot.get_credentials('GPT-4.1-mini')['api_key'], 'private-secret')
        returned = snapshot.get_credentials('Open AI GPT')
        returned['api_key'] = 'changed again'
        self.assertEqual(snapshot.get_credentials('Open AI GPT')['api_key'], 'private-secret')
        with self.assertRaises(FrozenInstanceError):
            snapshot.ocr_key = 'Default'

    def test_all_hosted_service_aliases_are_available_without_live_settings(self):
        for key, service in (('Microsoft OCR', 'Microsoft Azure'),
                             ('Google Cloud Vision', 'Google Cloud'),
                             ('GPT-4.1-mini', 'Open AI GPT'),
                             ('Gemini-2.5-Flash-Lite', 'Google Gemini')):
            with self.subTest(key=key):
                snapshot = capture_ocr_settings(settings(key, {'api_key': 'key'}))
                self.assertEqual(snapshot.get_credentials(key), {'api_key': 'key'})
                self.assertEqual(snapshot.get_credentials(service), {'api_key': 'key'})
                self.assertTrue(snapshot.hosted)
                self.assertEqual(snapshot.get_credentials('unrelated'), {})
        self.assertFalse(capture_ocr_settings(settings()).hosted)
        with self.assertRaises(ValueError):
            capture_ocr_settings(settings('unrecognized engine'))

    def test_target_language_routes_fresh_region_without_live_data_or_pixel_mutation(self):
        pixels = np.full((28, 90, 3), 255, np.uint8)
        pixels[8:15, 20:70] = 0
        original = pixels.copy()
        snapshot = capture_ocr_settings(settings())
        seen = []

        def recognize(image, regions):
            region = regions[0]
            seen.append(region)
            self.assertEqual(region.text, '')
            self.assertEqual(region.translation, '')
            self.assertEqual(region.source_lang, 'fr')
            self.assertTrue(region.lines)
            self.assertEqual(region.xyxy.dtype.kind, 'i')
            np.testing.assert_array_equal(region.xyxy, [0, 0, 90, 28])
            image[:] = 123  # Even a mutating recognizer cannot alter preview pixels.
            region.text = 'MA CHÉRIE!'
            return regions

        engine = SimpleNamespace(process_image=recognize)
        with patch.object(OCRFactory, '_create_new_engine', return_value=engine) as create:
            actual = run_lettering_ocr(pixels, 'French', snapshot)
        self.assertEqual(actual, 'MA CHÉRIE!')
        self.assertEqual(create.call_args.args[1:3], ('French', 'Default'))
        np.testing.assert_array_equal(pixels, original)
        self.assertEqual(len(seen), 1)

    def test_private_factory_does_not_reuse_global_or_prior_job_engine(self):
        snapshot = capture_ocr_settings(settings())
        pixels = np.zeros((20, 50, 3), np.uint8)
        key = OCRFactory._create_cache_key('Default', 'English', snapshot)
        previous = OCRFactory._engines.copy()
        global_engine = Mock()
        OCRFactory._engines[key] = global_engine
        engines, region_ids = [], []

        def fresh(*_args):
            def process(_image, regions):
                region_ids.append(regions[0].region_id)
                regions[0].text = 'fresh'
                return regions
            engine = SimpleNamespace(process_image=process)
            engines.append(engine)
            return engine

        try:
            with patch.object(OCRFactory, '_create_new_engine', side_effect=fresh) as create:
                self.assertEqual(run_lettering_ocr(pixels, 'English', snapshot), 'fresh')
                self.assertEqual(run_lettering_ocr(pixels, 'English', snapshot), 'fresh')
                self.assertEqual(create.call_count, 2)
            self.assertIs(OCRFactory._engines[key], global_engine)
            global_engine.process_image.assert_not_called()
            self.assertNotEqual(region_ids[0], region_ids[1])
        finally:
            OCRFactory._engines.clear()
            OCRFactory._engines.update(previous)

    def test_blank_and_failure_do_not_fallback_or_seed_expected_words(self):
        pixels = np.zeros((15, 40, 3), np.uint8)
        snapshot = capture_ocr_settings(settings('Google Cloud Vision', {'api_key': 'fixture'}))
        empty = Mock()
        with patch.object(OCRFactory, '_create_new_engine', return_value=empty) as create:
            self.assertEqual(run_lettering_ocr(pixels, 'Vietnamese', snapshot), '')
            create.assert_called_once()
            self.assertIsNone(empty.process_image.call_args.args[1][0].lines)
        failure = Mock()
        failure.process_image.side_effect = RuntimeError('OCR unavailable')
        with patch.object(OCRFactory, '_create_new_engine', return_value=failure) as create:
            with self.assertRaisesRegex(RuntimeError, 'OCR unavailable'):
                run_lettering_ocr(pixels, 'English', snapshot)
            create.assert_called_once()

    def test_unsupported_target_or_route_reports_error_before_silent_fallback(self):
        snapshot = capture_ocr_settings(settings())
        pixels = np.zeros((15, 40, 3), np.uint8)
        with patch.object(OCRFactory, '_create_new_engine', return_value=None) as create:
            for language in ('Auto', 'invented'):
                with self.assertRaises(ValueError):
                    run_lettering_ocr(pixels, language, snapshot)
            create.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'does not support'):
                run_lettering_ocr(pixels, 'Thai', snapshot)
            create.assert_called_once()

    def test_japanese_crop_does_not_trigger_page_size_override_to_second_model(self):
        def recognize(_self, _image, regions):
            self.assertEqual(regions[0].source_lang, 'ja')
            self.assertEqual(regions[0].text_class, '')
            regions[0].text = 'こんにちは'
            return regions
        engine = type('MangaOCRMobileONNXEngine', (), {'process_image': recognize})()
        with (patch.object(OCRFactory, '_create_new_engine', return_value=engine),
              patch.object(OCRFactory, '_create_ppocr') as override):
            result = run_lettering_ocr(np.zeros((50, 80, 3), np.uint8), 'Japanese',
                                      capture_ocr_settings(settings()))
        self.assertEqual(result, 'こんにちは')
        override.assert_not_called()


class LetteringCropTests(unittest.TestCase):
    def test_crop_uses_final_composed_patch_not_raw_model_or_context(self):
        source = np.zeros((60, 100, 3), np.uint8)
        prepared = prepare_edit(source, (20, 10, 40, 30), minimum_context=5,
                                maximum_model_size=(100, 100))
        generated = np.full_like(prepared.model_input, 180)
        patch = compose_generated_patch(prepared, generated)
        # Final protection/composition may replace model pixels before preview.
        patch.image[10:12, 10:13] = 73
        before = patch.image.copy()
        crop = prepare_lettering_crop(prepared, patch)
        np.testing.assert_array_equal(crop.image[8:-8, 8:-8], before)
        self.assertTrue(np.all(crop.image[:8] == 255))
        self.assertFalse(crop.partial_mask)
        self.assertFalse(crop.image.flags.writeable)
        np.testing.assert_array_equal(patch.image, before)

    def test_mask_holes_and_full_page_protection_are_whitened_without_mutation(self):
        source = np.full((60, 100, 3), 40, np.uint8)
        mask = np.zeros((60, 100), np.uint8)
        mask[10:40, 20:60] = 255
        mask[20:25, 30:35] = 0
        prepared = prepare_edit(source, (20, 10, 40, 30), edit_mask=mask, minimum_context=5)
        patch = compose_generated_patch(prepared, np.full_like(prepared.model_input, 180))
        protection = np.zeros((60, 100), np.uint8)
        protection[25:30, 40:45] = 255
        old_mask, old_protection, old_pixels = mask.copy(), protection.copy(), patch.image.copy()
        crop = prepare_lettering_crop(prepared, patch, protection)
        self.assertTrue(crop.partial_mask)
        self.assertTrue(np.all(crop.image[18:23, 18:23] == 255))
        self.assertTrue(np.all(crop.image[23:28, 28:33] == 255))
        np.testing.assert_array_equal(mask, old_mask)
        np.testing.assert_array_equal(protection, old_protection)
        np.testing.assert_array_equal(patch.image, old_pixels)
        with self.assertRaisesRegex(ValueError, 'no writable'):
            prepare_lettering_crop(prepared, patch, np.full_like(protection, 255))
        with self.assertRaises(ValueError):
            prepare_lettering_crop(prepared, patch, np.zeros((4, 4), np.uint8))


if __name__ == '__main__':
    unittest.main()
