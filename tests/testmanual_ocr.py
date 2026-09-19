"""Manual regions must receive the same line preparation as detected regions."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.detection.heuristic_lines import annotate_blocks_with_heuristic_lines
from modules.ocr.ppocr.engine import PPOCRv5Engine
from modules.ocr.processor import OCRProcessor
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.ocr_handler import OCRHandler


def sample_page():
    image = Image.new('RGB', (400, 200), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    for y, text in zip((20, 55, 90), ('HELLO MY FRIEND', 'HOW ARE YOU', 'DOING TODAY?')):
        draw.text((16, y), text, fill='black', font=font)
    return np.array(image)


def manual_block(**kwargs):
    return TextBlock(text_bbox=np.array([10, 10, 340, 140]), **kwargs)


def main_page(image, blocks, source='English', ocr_model='Default'):
    return SimpleNamespace(
        s_combo=SimpleNamespace(currentText=lambda: source),
        t_combo=SimpleNamespace(currentText=lambda: 'French'),
        lang_mapping={},
        blk_list=blocks,
        webtoon_mode=False,
        settings_page=SimpleNamespace(
            get_tool_selection=lambda _: ocr_model,
            is_gpu_enabled=lambda: False,
            ui=SimpleNamespace(tr=lambda text: text),
        ),
        image_viewer=SimpleNamespace(
            hasPhoto=lambda: True,
            rectangles=[object() for _ in blocks],
            get_image_array=lambda: image,
        ),
    )


class ManualOCRPreparationTests(unittest.TestCase):
    def setUp(self):
        self.image = sample_page()
        self.main = main_page(self.image, [])
        self.processor = OCRProcessor()
        self.processor.initialize(self.main, 'English')

    def test_drawn_english_paragraph_matches_detected_line_preparation(self):
        drawn = manual_block()
        detected = manual_block(text_class='text_bubble', source_lang='en')
        annotate_blocks_with_heuristic_lines(self.image, [detected])
        engine = Mock()
        with patch('modules.ocr.processor.OCRFactory.create_engine', return_value=engine) as factory:
            self.processor.process(self.image, [drawn])
        factory.assert_called_once_with(self.main.settings_page, 'English', 'Default')
        self.assertEqual(drawn.source_lang, 'en')
        self.assertEqual(len(drawn.lines), 3)
        self.assertEqual(drawn.lines, detected.lines)
        self.assertEqual(drawn.direction, 'horizontal')
        np.testing.assert_array_equal(drawn.xyxy, [10, 10, 340, 140])

    def test_mixed_page_does_not_replace_existing_detected_lines(self):
        drawn = manual_block()
        detected = TextBlock(text_bbox=np.array([10, 150, 150, 190]),
                             lines=[[12, 152, 148, 182]], direction='horizontal')
        old_lines = detected.lines
        blocks = [detected, drawn]
        engine = Mock()
        with patch('modules.ocr.processor.OCRFactory.create_engine', return_value=engine):
            self.processor.process(self.image, blocks)
        self.assertIs(engine.process_image.call_args.args[1], blocks)
        self.assertIs(detected.lines, old_lines)
        self.assertEqual(len(drawn.lines), 3)

    def test_ppocr_receives_three_line_crops_not_a_whole_paragraph(self):
        engine = PPOCRv5Engine()
        engine.rec_sess = object()
        engine.decoder = object()
        block = manual_block()
        with (
            patch('modules.ocr.processor.OCRFactory.create_engine', return_value=engine),
            patch.object(engine, '_det_infer') as detect,
            patch.object(engine, '_rec_infer', return_value=(['HELLO', 'MY', 'FRIEND'], [1.] * 3)) as recognize,
        ):
            self.processor.process(self.image, [block])
        detect.assert_not_called()
        crops = recognize.call_args.args[0]
        self.assertEqual(len(crops), 3)
        self.assertTrue(all(crop.shape[0] < 40 for crop in crops))
        self.assertEqual(block.text, 'HELLO MY FRIEND')

    def test_remote_ocr_preparation_is_unchanged(self):
        block = manual_block()
        self.processor.ocr_key = 'Microsoft OCR'
        with (
            patch('modules.ocr.processor.OCRFactory.create_engine'),
            patch('modules.ocr.processor.annotate_blocks_with_heuristic_lines') as prepare,
        ):
            self.processor.process(self.image, [block])
        prepare.assert_not_called()
        self.assertIsNone(block.lines)


class SelectedOCRRetryTests(unittest.TestCase):
    def setUp(self):
        self.image = sample_page()
        self.selected = manual_block(text='old gibberish', translation='Bonjour')
        self.other = TextBlock(text_bbox=np.array([10, 150, 150, 190]), text='Keep me')
        self.main = main_page(self.image, [self.selected, self.other])
        self.cache = CacheManager()
        self.pipeline = SimpleNamespace(get_selected_block=lambda: self.selected)
        self.handler = OCRHandler(self.main, self.cache, self.pipeline)
        self.handler.ocr = Mock()
        self.key = self.cache._get_ocr_cache_key(self.image, 'English', 'Default', 'cpu')

    @staticmethod
    def recognize(_image, blocks):
        blocks[0].text = 'HELLO MY FRIEND'
        blocks[0].texts = ['HELLO MY FRIEND']
        blocks[0].source_lang = 'en'

    def test_existing_text_and_cache_do_not_prevent_explicit_retry(self):
        self.cache._cache_ocr_results(self.key, self.main.blk_list)
        self.handler.ocr.process.side_effect = self.recognize
        with patch('pipeline.ocr_handler.resolve_device', return_value='cpu'):
            self.handler.OCR_image(single_block=True)
        self.handler.ocr.initialize.assert_called_once_with(self.main, 'English')
        processed = self.handler.ocr.process.call_args.args[1]
        self.assertEqual(len(processed), 1)
        self.assertIsNot(processed[0], self.selected)
        self.assertEqual(self.selected.text, 'HELLO MY FRIEND')
        self.assertEqual(self.selected.translation, 'Bonjour')
        self.assertEqual(self.other.text, 'Keep me')
        self.assertEqual(self.cache._get_cached_text_for_block(self.key, self.selected), 'HELLO MY FRIEND')
        self.assertEqual(self.cache._get_cached_text_for_block(self.key, self.other), 'Keep me')

    def test_uncached_retry_does_not_recognize_the_whole_page(self):
        self.handler.ocr.process.side_effect = self.recognize
        with patch('pipeline.ocr_handler.resolve_device', return_value='cpu'):
            self.handler.OCR_image(single_block=True)
        self.assertEqual(len(self.handler.ocr.process.call_args.args[1]), 1)
        self.assertIsNone(self.cache._get_cached_text_for_block(self.key, self.other))

    def test_retry_refreshes_manual_lines_and_clears_old_recognition(self):
        self.selected.lines = [[0, 0, 1, 1]]
        self.selected.direction = 'vertical'
        self.selected.texts = ['old']
        self.selected.skipped_small_texts = [{'text': 'old'}]

        def recognize(image, blocks):
            working = blocks[0]
            self.assertIsNone(working.lines)
            self.assertEqual(working.direction, '')
            self.assertEqual(working.text, '')
            self.assertEqual(working.texts, [])
            self.assertEqual(working.skipped_small_texts, [])
            self.recognize(image, blocks)

        self.handler.ocr.process.side_effect = recognize
        self.handler.OCR_image(single_block=True)

    def test_retry_preserves_detected_line_boundaries(self):
        self.selected.text_class = 'text_bubble'
        self.selected.lines = [[10, 20, 150, 40]]
        self.handler.ocr.process.side_effect = self.recognize
        self.handler.OCR_image(single_block=True)
        self.assertEqual(self.handler.ocr.process.call_args.args[1][0].lines, self.selected.lines)

    def test_auto_retry_retains_known_page_script_without_processing_other_boxes(self):
        self.main.s_combo.currentText = lambda: 'Auto'
        self.other.script = 'Latin'
        self.handler.ocr.process.side_effect = self.recognize
        self.handler.OCR_image(single_block=True)
        processed = self.handler.ocr.process.call_args.args[1]
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].script, 'Latin')
        self.assertEqual(self.selected.script, '')

    def test_auto_retry_does_not_override_selected_box_script(self):
        self.main.s_combo.currentText = lambda: 'Auto'
        self.selected.script = 'Japanese'
        self.other.script = 'Latin'
        self.handler.ocr.process.side_effect = self.recognize
        self.handler.OCR_image(single_block=True)
        self.assertEqual(self.handler.ocr.process.call_args.args[1][0].script, 'Japanese')

    def test_failed_retry_keeps_existing_text_and_cache(self):
        self.cache._cache_ocr_results(self.key, self.main.blk_list)

        def fail(_image, blocks):
            blocks[0].text = 'partial result'
            raise RuntimeError('recognizer failed')

        self.handler.ocr.process.side_effect = fail
        with patch('pipeline.ocr_handler.resolve_device', return_value='cpu'):
            with self.assertRaisesRegex(RuntimeError, 'recognizer failed'):
                self.handler.OCR_image(single_block=True)
        self.assertEqual(self.selected.text, 'old gibberish')
        self.assertEqual(self.selected.translation, 'Bonjour')
        self.assertEqual(self.cache._get_cached_text_for_block(self.key, self.selected), 'old gibberish')

    def test_empty_retry_evicts_stale_exact_and_fuzzy_cached_results(self):
        self.cache._cache_ocr_results(self.key, self.main.blk_list)
        nearby = self.selected.deep_copy()
        nearby.xyxy += 1
        self.cache.ocr_cache[self.key][self.cache._get_block_id(nearby)] = 'stale near match'
        # No recognition: the clean working copy remains empty.
        with patch('pipeline.ocr_handler.resolve_device', return_value='cpu'):
            self.handler.OCR_image(single_block=True)
        self.assertEqual(self.selected.text, '')
        self.assertIsNone(self.cache._get_cached_text_for_block(self.key, self.selected))
        self.assertEqual(self.cache._get_cached_text_for_block(self.key, self.other), 'Keep me')

    def test_whole_page_ocr_still_uses_cache(self):
        self.cache._cache_ocr_results(self.key, self.main.blk_list)
        with patch('pipeline.ocr_handler.resolve_device', return_value='cpu'):
            self.handler.OCR_image(single_block=False)
        self.handler.ocr.process.assert_not_called()

    def test_no_selection_does_not_process_other_boxes(self):
        self.pipeline.get_selected_block = lambda: None
        self.handler.OCR_image(single_block=True)
        self.handler.ocr.process.assert_not_called()

    def setup_webtoon(self):
        self.main.webtoon_mode = True
        self.selected.xyxy += np.array([0, 1000, 0, 1000])
        self.selected.lines = [[10, 1020, 150, 1040]]
        self.selected.direction = 'horizontal'
        self.main.image_viewer.webtoon_manager = SimpleNamespace(
            image_positions=[1000], image_heights=[200],
            image_data={0: self.image}, webtoon_width=400,
        )
        self.main.image_viewer.get_visible_area_image = lambda: (self.image, [{
            'page_index': 0, 'page_crop_top': 0, 'page_crop_bottom': 200,
            'combined_y_start': 0, 'combined_y_end': 200,
        }])

    def test_webtoon_retry_restores_scene_coordinates_and_line_metadata(self):
        self.setup_webtoon()

        def recognize(image, blocks):
            np.testing.assert_array_equal(blocks[0].xyxy, [10, 10, 340, 140])
            self.assertIsNone(blocks[0].lines)
            blocks[0].lines = [[10, 20, 150, 40]]
            self.recognize(image, blocks)

        self.handler.ocr.process.side_effect = recognize
        self.handler.OCR_webtoon_visible_area(single_block=True)
        np.testing.assert_array_equal(self.selected.xyxy, [10, 1010, 340, 1140])
        self.assertEqual(self.selected.lines, [[10, 1020, 150, 1040]])
        self.assertEqual(self.selected.text, 'HELLO MY FRIEND')
        self.assertFalse(hasattr(self.selected, '_original_xyxy'))

    def test_webtoon_retry_failure_restores_coordinates_and_old_text(self):
        self.setup_webtoon()
        self.handler.ocr.process.side_effect = RuntimeError('recognizer failed')
        with self.assertRaisesRegex(RuntimeError, 'recognizer failed'):
            self.handler.OCR_webtoon_visible_area(single_block=True)
        np.testing.assert_array_equal(self.selected.xyxy, [10, 1010, 340, 1140])
        self.assertEqual(self.selected.lines, [[10, 1020, 150, 1040]])
        self.assertEqual(self.selected.direction, 'horizontal')
        self.assertEqual(self.selected.text, 'old gibberish')
        self.assertFalse(hasattr(self.selected, '_original_xyxy'))


if __name__ == '__main__':
    unittest.main()
