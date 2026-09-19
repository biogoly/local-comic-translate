"""Page-wide actions must retain completed, user-drawn text regions."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.projects.parsers import ProjectDecoder, ProjectEncoder
from modules.utils.manual_regions import (
    has_manual_translation, is_manual_region, merge_manual_regions,
)
from modules.utils.textblock import TextBlock
from pipeline.block_detection import BlockDetectionHandler
from pipeline.cache_manager import CacheManager
from pipeline.ocr_handler import OCRHandler
from pipeline.translation_handler import TranslationHandler


def block(coords=(10, 10, 90, 70), manual=False, **kwargs):
    return TextBlock(text_bbox=np.array(coords), is_manual=manual,
                     text_class='text_bubble', **kwargs)


class ManualRegionIdentityTests(unittest.TestCase):
    def test_provenance_survives_deep_copy_and_project_roundtrip(self):
        original = block(manual=True, text='Hello', translation='Bonjour')
        for restored in (
            original.deep_copy(),
            ProjectDecoder.decode_textblock(ProjectEncoder.encode_textblock(original)),
        ):
            self.assertTrue(is_manual_region(restored))
            self.assertTrue(has_manual_translation(restored))
            self.assertEqual(restored.text, 'Hello')
            self.assertEqual(restored.translation, 'Bonjour')
            np.testing.assert_array_equal(restored.xyxy, original.xyxy)

    def test_old_projects_without_provenance_remain_supported(self):
        original = TextBlock(text_bbox=np.array([10, 10, 90, 70]), translation='Bonjour')
        del original.is_manual
        restored = ProjectDecoder.decode_textblock(ProjectEncoder.encode_textblock(original))
        self.assertTrue(has_manual_translation(restored))
        self.assertTrue(has_manual_translation(original.deep_copy()))
        self.assertFalse(has_manual_translation(block(translation='Bonjour')))

    def test_only_renderable_manual_translations_are_protected(self):
        for translation in ('', '   ', '...'):
            self.assertFalse(has_manual_translation(block(manual=True, translation=translation)))
        self.assertFalse(has_manual_translation(block(manual=True)))
        self.assertTrue(has_manual_translation(block(manual=True, translation='Oui')))

    def test_detection_keeps_manual_geometry_and_avoids_duplicates(self):
        manual = block(manual=True, text='Hello', translation='Bonjour')
        old_auto = block((100, 100, 150, 150))
        near_duplicate = block((12, 12, 92, 72))
        inner_line = block((20, 20, 70, 35))
        neighbour = block((85, 10, 150, 70))
        merged = merge_manual_regions([manual, old_auto], [near_duplicate, inner_line, neighbour])
        self.assertEqual(merged, [manual, neighbour])
        np.testing.assert_array_equal(manual.xyxy, [10, 10, 90, 70])

    def test_detection_retains_unfinished_manual_and_large_distinct_paragraph(self):
        manual = block(manual=True)
        large = block((0, 0, 400, 400))
        self.assertEqual(merge_manual_regions([manual], [large]), [manual, large])
        self.assertEqual(merge_manual_regions([manual, block()], []), [manual])


class ManualRegionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((200, 300, 3), 255, dtype=np.uint8)
        self.manual = block(manual=True, text='Hello friend', translation='Bonjour mon ami')
        self.pending = block((100, 100, 180, 160), text='Good night')
        self.main = SimpleNamespace(
            s_combo=SimpleNamespace(currentText=lambda: 'English'),
            t_combo=SimpleNamespace(currentText=lambda: 'French'),
            lang_mapping={}, webtoon_mode=False,
            blk_list=[self.manual, self.pending],
            image_files=['page.png'], curr_img_idx=0,
            image_states={'page.png': {}},
            settings_page=SimpleNamespace(
                get_tool_selection=lambda name: 'Default' if name == 'ocr' else 'Local LLM',
                is_gpu_enabled=lambda: False,
                get_llm_settings=lambda: {'extra_context': ''},
                ui=SimpleNamespace(uppercase_checkbox=SimpleNamespace(isChecked=lambda: True)),
            ),
            image_viewer=SimpleNamespace(
                hasPhoto=lambda: True, rectangles=[object(), object()],
                get_image_array=lambda **_: self.image,
            ),
        )
        self.cache = CacheManager()
        self.pipeline = SimpleNamespace(get_selected_block=lambda: self.manual)
        self.ocr = OCRHandler(self.main, self.cache, self.pipeline)
        self.ocr.ocr = Mock()
        self.translation = TranslationHandler(self.main, self.cache, self.pipeline)

    def test_detection_replaces_automatic_boxes_but_retains_manual_region_and_state(self):
        new = block((200, 100, 280, 180))
        handler = BlockDetectionHandler(self.main)
        handler.on_blk_detect_complete(([block(), new], False, None))
        self.assertEqual(self.main.blk_list, [self.manual, new])
        saved = self.main.image_states['page.png']['blk_list']
        self.assertTrue(saved[0].is_manual)
        self.assertEqual(saved[0].translation, 'Bonjour mon ami')
        handler.on_blk_detect_complete(([], False, None))
        self.assertEqual(self.main.blk_list, [self.manual])

    def test_whole_page_ocr_does_not_reread_cleaned_manual_region(self):
        def recognize(image, blocks):
            self.assertEqual(blocks, [self.pending])
            blocks[0].text = 'Good evening'

        self.ocr.ocr.process.side_effect = recognize
        with patch('pipeline.ocr_handler.resolve_device', return_value='cpu'):
            self.ocr.OCR_image()
        self.ocr.ocr.process.assert_called_once()
        self.assertEqual(self.manual.text, 'Hello friend')
        self.assertEqual(self.manual.translation, 'Bonjour mon ami')
        self.assertEqual(self.pending.text, 'Good evening')

    def test_cached_ocr_does_not_overwrite_completed_manual_text(self):
        old = self.manual.deep_copy()
        old.text = 'bad old reading'
        key = self.cache._get_ocr_cache_key(self.image, 'English', 'Default', 'cpu')
        self.cache._cache_ocr_results(key, [old, self.pending])
        with patch('pipeline.ocr_handler.resolve_device', return_value='cpu'):
            self.ocr.OCR_image()
        self.ocr.ocr.process.assert_not_called()
        self.assertEqual(self.manual.text, 'Hello friend')

    def test_whole_page_translation_does_not_change_manual_text_or_case(self):
        with patch('pipeline.translation_handler.Translator') as factory:
            def translate(blocks, image, context):
                self.assertEqual(blocks, [self.pending])
                blocks[0].translation = 'Bonne nuit'
            factory.return_value.translate.side_effect = translate
            self.translation.translate_image()
            factory.return_value.translate.assert_called_once()
        self.assertEqual(self.manual.translation, 'Bonjour mon ami')
        self.assertEqual(self.pending.translation, 'BONNE NUIT')

    def test_cached_translation_does_not_overwrite_manual_target(self):
        old = self.manual.deep_copy()
        old.translation = 'An obsolete translation'
        pending = self.pending.deep_copy()
        pending.translation = 'Bonne nuit'
        key = self.cache._get_translation_cache_key(self.image, 'English', 'French', 'Local LLM', '')
        self.cache._cache_translation_results(key, [old, pending])
        with patch('pipeline.translation_handler.Translator') as factory:
            self.translation.translate_image()
            factory.return_value.translate.assert_not_called()
        self.assertEqual(self.manual.translation, 'Bonjour mon ami')
        self.assertEqual(self.pending.translation, 'BONNE NUIT')

    def test_untranslated_manual_box_still_participates_in_page_processing(self):
        self.manual.translation = ''
        with patch('pipeline.ocr_handler.resolve_device', return_value='cpu'):
            self.ocr.OCR_image()
        self.assertEqual(self.ocr.ocr.process.call_args.args[1], self.main.blk_list)
        with patch('pipeline.translation_handler.Translator') as factory:
            self.translation.translate_image()
            self.assertEqual(factory.return_value.translate.call_args.args[0], self.main.blk_list)

    def test_page_with_only_completed_manual_regions_does_not_call_models(self):
        self.main.blk_list = [self.manual]
        with patch('pipeline.ocr_handler.resolve_device', return_value='cpu'):
            self.ocr.OCR_image()
        self.ocr.ocr.process.assert_not_called()
        with patch('pipeline.translation_handler.Translator') as factory:
            self.translation.translate_image()
            factory.return_value.translate.assert_not_called()

    def setup_webtoon(self):
        self.main.webtoon_mode = True
        for blk in self.main.blk_list:
            blk.xyxy += np.array([0, 1000, 0, 1000])
        self.manual.lines = [[15, 1015, 80, 1035]]
        self.main.image_viewer.webtoon_manager = SimpleNamespace(
            image_positions=[1000], image_heights=[200],
            image_data={0: self.image}, webtoon_width=300,
        )
        self.mappings = [{
            'page_index': 0, 'page_crop_top': 0, 'page_crop_bottom': 200,
            'combined_y_start': 0, 'combined_y_end': 200,
            'scene_y_start': 1000, 'scene_y_end': 1200,
        }]
        self.main.image_viewer.get_visible_area_image = lambda **_: (self.image, self.mappings)

    def test_webtoon_detection_keeps_manual_regions_inside_and_outside_view(self):
        self.setup_webtoon()
        outside = block((10, 1500, 90, 1600), manual=True)
        partial = block((100, 990, 180, 1070), manual=True)
        self.main.blk_list.extend([outside, partial])
        BlockDetectionHandler(self.main).on_blk_detect_complete(([
            block((12, 1012, 88, 1068)), block((102, 1000, 178, 1068)),
        ], False, self.mappings))
        self.assertCountEqual(self.main.blk_list, [self.manual, outside, partial])

    def test_webtoon_page_ocr_and_translation_preserve_manual_region_and_coordinates(self):
        self.setup_webtoon()
        before = self.manual.xyxy.copy()
        self.ocr.OCR_webtoon_visible_area()
        self.assertEqual(self.ocr.ocr.process.call_args.args[1], [self.pending])
        with patch('pipeline.translation_handler.Translator') as factory:
            self.translation.translate_webtoon_visible_area()
            self.assertEqual(factory.return_value.translate.call_args.args[0], [self.pending])
        np.testing.assert_array_equal(self.manual.xyxy, before)
        self.assertEqual(self.manual.lines, [[15, 1015, 80, 1035]])
        self.assertEqual(self.manual.translation, 'Bonjour mon ami')
        self.assertFalse(hasattr(self.manual, '_original_xyxy'))


if __name__ == '__main__':
    unittest.main()
