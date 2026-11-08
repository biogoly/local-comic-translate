"""OCR review evidence follows the actual reading and recognizer through the workflow."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import msgpack
import numpy as np

from app.projects.parsers import ProjectDecoder, ProjectEncoder
from modules.ocr.evidence import current_ocr_evidence, make_ocr_evidence, uncertain_ocr_evidence
from modules.ocr.manga_ocr.mobile.onnx_engine import MangaOCRMobileONNXEngine
from modules.ocr.microsoft_ocr import MicrosoftOCR
from modules.ocr.pororo.onnx_engine import PororoOCREngineONNX
from modules.ocr.ppocr.engine import PPOCRv5Engine
from modules.ocr.processor import OCRProcessor
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.ocr_handler import OCRHandler
from testmanual_ocr import main_page, sample_page


MODEL = 'ppocr-v5-rec-en-mobile'


def region(text='Hello', **kwargs):
    return TextBlock(text_bbox=np.array([0, 0, 60, 50]), text=text, **kwargs)


def ppocr():
    engine = PPOCRv5Engine()
    engine.recognition_model = MODEL
    engine.rec_sess, engine.decoder = object(), object()
    return engine


class OcrEvidenceTests(unittest.TestCase):
    def test_source_edit_and_detached_copies(self):
        block = region()
        block.ocr_evidence = make_ocr_evidence(block.text, 'PP-OCR', MODEL, line_scores=[0.3, 0.96])
        self.assertEqual(uncertain_ocr_evidence(block)['score'], 0.3)
        self.assertEqual(uncertain_ocr_evidence(vars(block))['line_count'], 2)
        block.deep_copy().ocr_evidence['score'] = 0.9
        current_ocr_evidence(block)['score'] = 0.99
        self.assertEqual(block.ocr_evidence['score'], 0.3)
        block.text = 'HELLO'
        self.assertIsNone(current_ocr_evidence(block))

    def test_missing_or_invalid_scores_are_unknown_and_zero_is_real(self):
        for scores in (None, [], [None], [True], [0.8, np.nan], [-0.1], [1.2], ['0.6']):
            block = region()
            block.ocr_evidence = make_ocr_evidence(block.text, 'PP-OCR', MODEL, line_scores=scores)
            self.assertEqual(current_ocr_evidence(block)['scope'], 'unscored')
            self.assertIsNone(uncertain_ocr_evidence(block))
        for score, should_flag in ((0, True), (0.749, True), (0.75, False), (1.0, False)):
            block.ocr_evidence = make_ocr_evidence(block.text, 'PP-OCR', MODEL, line_scores=[score])
            self.assertEqual(uncertain_ocr_evidence(block) is not None, should_flag)
        for engine, scope in (('Microsoft OCR', 'lowest_word'), ('Pororo OCR', 'lowest_sequence_product')):
            block.ocr_evidence = make_ocr_evidence(block.text, engine, 'fixture', line_scores=[0.2])
            self.assertEqual(current_ocr_evidence(block)['scope'], scope)
            self.assertIsNone(uncertain_ocr_evidence(block))

    def test_malformed_and_old_projects_are_safe(self):
        block = region()
        block.ocr_evidence = make_ocr_evidence(block.text, 'Manga OCR', 'manga-ocr-mobile-onnx')
        damaged = dict(block.ocr_evidence)
        del damaged['score']
        for record in (None, [], damaged, dict(block.ocr_evidence, line_count=False),
                       dict(block.ocr_evidence, engine='PP-OCR', score=0.2),
                       dict(block.ocr_evidence, scope='lowest_line', score=0.1, line_count=1)):
            self.assertIsNone(current_ocr_evidence({'text': block.text, 'ocr_evidence': record}))
        self.assertIsNone(current_ocr_evidence(ProjectDecoder.decode_textblock({'data': {'text': 'old'}})))
        block.ocr_evidence = make_ocr_evidence(block.text, 'PP-OCR', MODEL, line_scores=[0.7])
        restored = msgpack.unpackb(msgpack.packb(block, default=ProjectEncoder().encode, use_bin_type=True),
                                   object_hook=ProjectDecoder().decode)
        self.assertEqual(uncertain_ocr_evidence(restored), block.ocr_evidence)

    def test_batched_recognition_ignores_empty_and_skipped_ruby_lines(self):
        block = region(lines=[[2, 2, 45, 14], [2, 16, 45, 28], [5, 30, 18, 32]], source_lang='ja')
        other = region(lines=[[2, 30, 45, 42], [3, 43, 30, 49]], source_lang='en')
        engine = ppocr()
        with patch.object(engine, '_rec_infer', return_value=(['本', '文', 'よみ', '', 'Good'],
                                                              [.86, .92, .08, 0., .71])):
            engine.process_image(np.zeros((60, 60, 3), dtype=np.uint8), [block, other])
        self.assertEqual(block.text, '本文')
        self.assertEqual(current_ocr_evidence(block)['score'], .86)
        self.assertEqual(current_ocr_evidence(block)['line_count'], 2)
        self.assertEqual(uncertain_ocr_evidence(other)['score'], .71)
        self.assertEqual(uncertain_ocr_evidence(other)['line_count'], 1)

    def test_fallback_uses_recognition_score_for_contained_lines(self):
        engine = ppocr()
        boxes = np.array([[[4, 3], [42, 3], [42, 12], [4, 12]],
                          [[4, 20], [42, 20], [42, 32], [4, 32]],
                          [[1, 51], [56, 51], [56, 58], [1, 58]]])
        block = region(source_lang='en')
        with patch.object(engine, '_det_infer', return_value=(boxes, [.02, .03, .02])), \
             patch.object(engine, '_rec_infer', return_value=(['First', 'line', 'outside'], [.92, .84, .05])):
            engine.process_image(np.zeros((60, 60, 3), np.uint8), [block])
        self.assertEqual(block.text, 'First line')
        self.assertEqual(current_ocr_evidence(block)['score'], .84)
        self.assertEqual(current_ocr_evidence(block)['line_count'], 2)

    def test_unscored_recognizer_supersedes_same_text_ppocr_score(self):
        block = region()
        block.ocr_evidence = make_ocr_evidence(block.text, 'PP-OCR', MODEL, line_scores=[.2])
        manga = MangaOCRMobileONNXEngine()
        manga.model = Mock()
        manga.model.process_batch.return_value = [block.text]
        manga.process_image(np.zeros((60, 60, 3), np.uint8), [block])
        self.assertEqual(current_ocr_evidence(block)['engine'], 'Manga OCR')
        self.assertIsNone(uncertain_ocr_evidence(block))

    def test_skipped_crop_or_older_adapter_cannot_keep_stale_score(self):
        block = region()
        block.ocr_evidence = make_ocr_evidence(block.text, 'PP-OCR', MODEL, line_scores=[.3])
        image = np.zeros((60, 60, 3), np.uint8)
        processor = OCRProcessor()
        processor.initialize(main_page(image, [block]), 'English')
        engine = Mock()
        engine.process_image.return_value = [block]
        with patch.object(processor.engine_factory, 'create_engine', return_value=engine):
            processor.process(image, [block])
        self.assertEqual(block.text, 'Hello')
        self.assertIsNone(block.ocr_evidence)

    def test_pororo_sequence_products_remain_separate_from_ppocr(self):
        block = region(lines=[[1, 1, 55, 12], [1, 14, 55, 26]], source_lang='ko')
        engine = object.__new__(PororoOCREngineONNX)
        engine.opt2val, engine.use_text_lines = {}, True
        engine._recognize = Mock(return_value=[(None, '오늘', .05), (None, '날씨', .1)])
        engine.process_image(np.zeros((60, 60, 3), np.uint8), [block])
        evidence = current_ocr_evidence(block)
        self.assertEqual((evidence['engine'], evidence['model']), ('Pororo OCR', 'pororo-onnx'))
        self.assertEqual((evidence['scope'], evidence['score']), ('lowest_sequence_product', .05))
        self.assertIsNone(uncertain_ocr_evidence(block))

    def test_azure_words_aggregate_only_when_supplied(self):
        block = region()
        line = SimpleNamespace(bounding_polygon=[{'x': x, 'y': y} for x, y in
                               [(2, 2), (50, 2), (50, 14), (2, 14)]], text='Agua fresca',
                               words=[SimpleNamespace(confidence=.3), SimpleNamespace(confidence=.91)])
        read = SimpleNamespace(blocks=[SimpleNamespace(lines=[line])])
        engine = MicrosoftOCR()
        engine.client = Mock()
        engine.client.analyze.return_value = SimpleNamespace(read=read, model_version='test-read')
        sdk = SimpleNamespace(VisualFeatures=SimpleNamespace(READ='read'))
        with patch.dict('sys.modules', {'azure.ai.vision.imageanalysis.models': sdk}):
            engine.process_image(np.zeros((60, 60, 3), np.uint8), [block])
            evidence = current_ocr_evidence(block)
            self.assertEqual((evidence['scope'], evidence['score'], evidence['line_count']), ('lowest_word', .3, 1))
            self.assertIsNone(uncertain_ocr_evidence(block))
            del line.words[1].confidence
            engine.process_image(np.zeros((60, 60, 3), np.uint8), [block])
            self.assertIsNone(current_ocr_evidence(block)['score'])

    def test_cache_caches_only_evidence_for_the_stored_reading(self):
        block = region()
        reading = block.deep_copy()
        reading.ocr_evidence = make_ocr_evidence(reading.text, 'PP-OCR', MODEL, line_scores=[.43])
        cache = CacheManager()
        cache._cache_ocr_results('key', [block], [reading])
        reading.ocr_evidence['score'] = .9
        cache._apply_cached_ocr_to_blocks('key', [block])
        self.assertEqual(current_ocr_evidence(block)['score'], .43)
        block.text = 'hand corrected'
        cache.update_ocr_cache_for_block('key', block)
        cache._apply_cached_ocr_to_blocks('key', [block])
        self.assertIsNone(block.ocr_evidence)
        cache.ocr_cache['key'][cache._get_block_id(block)] = 'legacy reading'
        cache._apply_cached_ocr_to_blocks('key', [block])
        self.assertEqual(block.text, 'legacy reading')
        self.assertIsNone(block.ocr_evidence)

    def test_selected_retry_copies_back_only_successful_reading(self):
        block = region(text_class='text_bubble')
        block.ocr_evidence = make_ocr_evidence(block.text, 'PP-OCR', MODEL, line_scores=[.38])
        before = copy.deepcopy(block.ocr_evidence)
        image = sample_page()
        handler = OCRHandler(main_page(image, [block]), CacheManager(), Mock())
        handler.ocr = Mock()
        handler.ocr.process.side_effect = RuntimeError('failed')
        with self.assertRaises(RuntimeError):
            handler._recognize_selected_block(image, block, 'English')
        self.assertEqual(block.ocr_evidence, before)
        def recognize(_image, blocks):
            self.assertIsNone(blocks[0].ocr_evidence)
            blocks[0].text = 'Fixed'
            blocks[0].ocr_evidence = make_ocr_evidence('Fixed', 'PP-OCR', MODEL, line_scores=[.95])
        handler.ocr.process.side_effect = recognize
        handler._recognize_selected_block(image, block, 'English')
        self.assertEqual(current_ocr_evidence(block)['score'], .95)
