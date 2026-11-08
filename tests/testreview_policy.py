"""Independent per-region protection, identity and whole-page text policies."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.utils.manual_regions import merge_manual_regions
from modules.utils.region_policy import (
    is_protected, permits_region_operation, retained_layout_states, shifted_region_copy,
)
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.ocr_handler import OCRHandler
from pipeline.translation_handler import TranslationHandler
from modules.translation.llm.local import LocalLLMTranslation
from modules.translation.request import build_request


def region(bounds=(10, 10, 50, 50), **protections):
    return TextBlock(text_bbox=np.array(bounds), text_class='text_bubble',
                     text='User source', translation='User target',
                     region_metadata={'protections': protections})


class ReviewPolicyTests(unittest.TestCase):
    def test_each_flag_controls_its_own_stage_and_explicit_actions_remain_available(self):
        flags = {'source': 'keep_source', 'translation': 'keep_translation',
                 'layout': 'keep_layout', 'cleanup': 'skip_auto_cleanup'}
        for stage, flag in flags.items():
            block = region(**{flag: True})
            for tested in flags:
                self.assertEqual(permits_region_operation(block, tested), tested != stage)
                self.assertTrue(permits_region_operation(block, tested, explicit=True))
        self.assertFalse(is_protected(region(keep_source='yes'), 'keep_source'))

    def test_detection_retains_protected_automatic_regions_and_ids_without_transfer(self):
        protected = region(keep_source=True)
        duplicate = region((12, 12, 52, 52))
        neighbour = region((49, 10, 100, 50))
        old_unprotected = region((120, 10, 170, 50))
        original_id = protected.region_id
        merged = merge_manual_regions([protected, old_unprotected], [duplicate, neighbour])
        self.assertEqual(merged, [protected, neighbour])
        self.assertEqual(merged[0].region_id, original_id)
        self.assertNotEqual(neighbour.region_id, original_id)
        self.assertFalse(is_protected(neighbour, 'keep_source'))

    def test_layout_preservation_is_id_based_independent_and_deep_copied(self):
        layout = region(keep_layout=True)
        wording = region((70, 10, 110, 50), keep_translation=True)
        states = [{'region_id': layout.region_id, 'text': '<b>custom</b>', 'position': (7, 8),
                   'selection_outlines': [{'start': 0, 'end': 3}]},
                  {'region_id': wording.region_id, 'text': 'translation only'}]
        retained = retained_layout_states([layout, wording], states)
        self.assertEqual(list(retained), [layout.region_id])
        self.assertEqual(retained[layout.region_id], states[0])
        retained[layout.region_id]['selection_outlines'][0]['end'] = 9
        self.assertEqual(states[0]['selection_outlines'][0]['end'], 3)
        self.assertEqual(retained_layout_states([layout], states + [copy.deepcopy(states[0])]), {})

    def test_crop_copy_preserves_metadata_and_moves_all_geometry(self):
        block = region((10, 90, 70, 130), keep_translation=True)
        block.lines = [[11, 91, 69, 129]]
        block.segm_pts = np.array([[12, 92], [65, 125]])
        shifted = shifted_region_copy(block, dx=5, dy=-100)
        self.assertEqual(shifted.region_id, block.region_id)
        self.assertEqual(shifted.region_metadata, block.region_metadata)
        np.testing.assert_array_equal(shifted.xyxy, [15, -10, 75, 30])
        np.testing.assert_array_equal(shifted.lines, [[16, -9, 74, 29]])
        np.testing.assert_array_equal(shifted.segm_pts, [[17, -8], [70, 25]])
        np.testing.assert_array_equal(block.xyxy, [10, 90, 70, 130])


class ReviewPageTextTests(unittest.TestCase):
    def setUp(self):
        self.source = region(keep_source=True)
        self.target = region((60, 10, 110, 50), keep_translation=True)
        self.cleanup = region((120, 10, 170, 50), skip_auto_cleanup=True)
        self.cleanup.text_class = 'text_free'
        self.blocks = [self.source, self.target, self.cleanup]
        self.image = np.full((100, 200, 3), 211, dtype=np.uint8)
        self.main = SimpleNamespace(
            s_combo=SimpleNamespace(currentText=lambda: 'English'),
            t_combo=SimpleNamespace(currentText=lambda: 'French'), lang_mapping={},
            blk_list=self.blocks, webtoon_mode=False,
            image_viewer=SimpleNamespace(hasPhoto=lambda: True, rectangles=[1],
                                         get_image_array=lambda **_: self.image),
            settings_page=SimpleNamespace(get_tool_selection=lambda _: 'Local LLM',
                is_gpu_enabled=lambda: False, get_llm_settings=lambda: {'extra_context': 'context'},
                ui=SimpleNamespace(uppercase_checkbox=SimpleNamespace(isChecked=lambda: True))),
        )
        self.cache = CacheManager()
        self.pipeline = SimpleNamespace(get_selected_block=lambda: self.source)

    def test_ocr_filters_only_source_and_updates_protected_cache(self):
        handler = OCRHandler(self.main, self.cache, self.pipeline)
        handler.ocr = Mock()
        def recognize(image, blocks):
            self.assertIs(image, self.image)
            for block in blocks:
                block.text = 'New OCR'
        handler.ocr.process.side_effect = recognize
        handler.OCR_image()
        self.assertEqual(handler.ocr.process.call_args.args[1], [self.target, self.cleanup])
        self.assertEqual(self.source.text, 'User source')
        self.assertEqual(self.target.text, 'New OCR')
        self.assertEqual(self.target.translation, 'User target')
        key = self.cache._get_ocr_cache_key(self.image, 'English', 'Local LLM', 'cpu')
        self.assertEqual(self.cache._get_cached_text_for_block(key, self.source), 'User source')

    def test_translation_keeps_target_case_and_full_image_context_but_updates_other_regions(self):
        handler = TranslationHandler(self.main, self.cache, self.pipeline)
        with patch('pipeline.translation_handler.Translator') as factory:
            translator = factory.return_value
            translator.source_lang, translator.target_lang, translator.translator_key = 'English', 'French', 'Local LLM'
            engine = LocalLLMTranslation()
            engine.source_lang, engine.target_lang = 'English', 'French'
            engine.img_as_llm_input = True
            translator.prepare_request.side_effect = lambda blocks, image, context: build_request(engine, blocks, image, context)
            def translate(blocks, image, context, *, request):
                np.testing.assert_array_equal(image, self.image)
                self.assertIsNot(image, self.image)
                self.assertEqual(context, 'context')
                self.assertEqual(request.region_ids, (self.source.region_id, self.cleanup.region_id))
                for block in blocks:
                    block.translation = 'nouvelle traduction'
            factory.return_value.translate.side_effect = translate
            handler.translate_image()
        translated = factory.return_value.translate.call_args.args[0]
        self.assertEqual([b.region_id for b in translated], [self.source.region_id, self.cleanup.region_id])
        self.assertTrue(all(a is not b for a, b in zip(translated, [self.source, self.cleanup])))
        self.assertEqual(self.target.translation, 'User target')
        self.assertEqual(self.source.translation, 'NOUVELLE TRADUCTION')
        self.assertEqual(self.cleanup.translation, 'NOUVELLE TRADUCTION')
        key = next(iter(self.cache.translation_cache))
        self.assertEqual(key[0], 'request-v1')
        self.assertIsNone(self.cache._get_cached_translation_for_block(key, self.target))
        self.assertEqual(self.cache._get_cached_translation_for_block(key, self.source), 'nouvelle traduction')

    def test_selected_ocr_can_produce_replacement_for_protected_source(self):
        handler = OCRHandler(self.main, self.cache, self.pipeline)
        handler.ocr = Mock()
        handler.ocr.process.side_effect = lambda image, blocks: setattr(blocks[0], 'text', 'Preview source')
        handler.OCR_image(single_block=True)
        self.assertEqual(self.source.text, 'Preview source')
        self.assertTrue(is_protected(self.source, 'keep_source'))

    def test_empty_kept_target_evicts_old_cache_without_initializing_translator(self):
        handler = TranslationHandler(self.main, self.cache, self.pipeline)
        key = self.cache._get_translation_cache_key(self.image, 'English', 'French', 'Local LLM', 'context')
        self.cache._cache_translation_results(key, self.blocks)
        self.target.translation = ''
        for block in self.blocks:
            block.region_metadata['protections']['keep_translation'] = True
        with patch('pipeline.translation_handler.Translator') as translator:
            handler.translate_image()
        translator.assert_not_called()
        self.assertEqual(self.target.translation, '')
        self.assertIsNone(self.cache._get_cached_translation_for_block(key, self.target))


if __name__ == '__main__':
    unittest.main()
