"""Language routing and Vietnamese recognition contracts (no model downloads)."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from modules.ocr.factory import OCRFactory
from modules.ocr.processor import OCRProcessor
from modules.ocr.vietnamese import VietnameseOCREngine, vietnamese_rec_width
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.textblock import TextBlock


class LanguageOCRRoutingTests(unittest.TestCase):
    def test_portuguese_uses_unchanged_latin_engine(self):
        for name in ('Portuguese', 'Brazilian Portuguese', 'French', 'Spanish', 'latin'):
            with self.subTest(language=name), patch.object(OCRFactory, '_create_ppocr') as create:
                settings = object()
                result = OCRFactory._create_new_engine(settings, name, 'Default')
                create.assert_called_once_with(settings, 'latin', 'onnx')
                self.assertIs(result, create.return_value)

    def test_vietnamese_gets_its_own_default_engine_not_latin(self):
        with patch.object(OCRFactory, '_create_vietnamese_ocr') as create:
            settings = object()
            result = OCRFactory._create_new_engine(settings, 'Vietnamese', 'Default')
            create.assert_called_once_with(settings)
            self.assertIs(result, create.return_value)

    def test_vietnamese_still_honors_provider_ocr_selection(self):
        with patch.object(OCRFactory, '_create_google_ocr') as create, \
                patch.object(OCRFactory, '_create_vietnamese_ocr') as local:
            settings = object()
            OCRFactory._create_new_engine(settings, 'Vietnamese', 'Google Cloud Vision')
            create.assert_called_once_with(settings)
            local.assert_not_called()

    def test_manual_regions_preserve_diacritics_spaces_and_source_code(self):
        cases = [('Vietnamese', 'vi', 'Bạn khỏe không?'),
                 ('Portuguese', 'pt', 'Olá, coração!'),
                 ('Brazilian Portuguese', 'pt-br', 'Você está bem?')]
        for language, code, content in cases:
            with self.subTest(language=language):
                main = SimpleNamespace(lang_mapping={}, settings_page=SimpleNamespace(
                    get_tool_selection=lambda _: 'Default',
                    ui=SimpleNamespace(tr=lambda text: text)))
                processor = OCRProcessor()
                processor.initialize(main, language)
                block = TextBlock(text_bbox=np.array([0, 0, 140, 50]),
                                  lines=[[0, 0, 140, 50]], direction='horizontal')
                engine = VietnameseOCREngine()
                engine.rec_sess = Mock()
                engine.decoder = Mock()
                with patch.object(OCRFactory, 'create_engine', return_value=engine), \
                        patch.object(engine, '_rec_infer', return_value=([content], [0.99])):
                    processor.process(np.zeros((60, 150, 3), dtype=np.uint8), [block])
                self.assertEqual(block.source_lang, code)
                self.assertEqual(block.text, content)

    def test_vietnamese_assets_are_pinned_and_separate_from_latin(self):
        vietnamese = ModelDownloader.registry[ModelID.PPOCR_V6_REC_VI]
        latin = ModelDownloader.registry[ModelID.PPOCR_V5_REC_LATIN_MOBILE]
        self.assertNotEqual(Path(vietnamese.save_dir), Path(latin.save_dir))
        self.assertIn('9b5346432fb84a447fdc4d780a7741126b446b0f', vietnamese.url)
        self.assertEqual(vietnamese.files, ['recognition.onnx', 'charset.txt'])
        self.assertEqual(vietnamese.sha256, [
            '767305d37d06728fe9578ff8132595ce23f66139df5e2937fd4131bd1dc34f12',
            '5d801bf60472c4c1b184b6b6843aa2c754d8daf1429247e07658ee0bd79ab448',
        ])
        self.assertEqual(latin.sha256[0],
                         'b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a')


class VietnameseRecognitionTests(unittest.TestCase):
    def test_widths_use_only_qualified_buckets(self):
        for shape, expected in [((48, 100, 3), 320), ((48, 320, 3), 320),
                                ((48, 321, 3), 640), ((48, 2000, 3), 640),
                                ((96, 640, 3), 320)]:
            with self.subTest(shape=shape):
                self.assertEqual(vietnamese_rec_width(np.zeros(shape)), expected)

    def test_initialization_checks_dictionary_and_output_classes(self):
        for dictionary_size, output_classes, ok in [(228, 230, True), (502, 504, False),
                                                     (228, 229, False)]:
            engine = VietnameseOCREngine()
            session = Mock()
            session.get_outputs.return_value = [SimpleNamespace(shape=[None, None, output_classes])]
            decoder = SimpleNamespace(dict_chars=['a'] * dictionary_size)
            with self.subTest(size=dictionary_size, classes=output_classes), \
                    patch('modules.ocr.vietnamese.ModelDownloader.file_path_map',
                          return_value={'recognition.onnx': 'model.onnx', 'charset.txt': 'chars.txt'}), \
                    patch('modules.ocr.vietnamese.CTCLabelDecoder', return_value=decoder), \
                    patch('modules.ocr.vietnamese.make_session', return_value=session):
                if ok:
                    engine.initialize()
                    self.assertIs(engine.rec_sess, session)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'do not match'):
                        engine.initialize()
                    self.assertIsNone(engine.rec_sess)

    def test_batches_are_bounded_and_original_order_is_restored(self):
        engine = VietnameseOCREngine()
        engine.rec_batch_size = 4
        engine.rec_sess = Mock()
        engine.rec_sess.get_inputs.return_value = [SimpleNamespace(name='x')]
        engine.rec_sess.get_outputs.return_value = [SimpleNamespace(name='out')]
        seen = []

        def run(_outputs, inputs):
            batch = inputs['x']
            seen.append(batch.shape)
            logits = np.zeros((len(batch), 2, 230), dtype=np.float32)
            logits[:, 0, 0] = batch[:, 0, 0, 0]
            return [logits]

        engine.rec_sess.run.side_effect = run
        engine.decoder = lambda logits, **_: ([str(int(round((v + 1) * 127.5)))
                                               for v in logits[:, 0, 0]], [1.0] * len(logits))
        crops = [np.full((48, 100 if i % 3 else 500, 3), i, dtype=np.uint8) for i in range(13)]
        texts, scores = engine._rec_infer(crops)
        self.assertEqual(texts, [str(i) for i in range(13)])
        self.assertEqual(scores, [1.0] * 13)
        self.assertTrue(all(shape[0] <= 4 and shape[1:3] == (3, 48) for shape in seen))
        self.assertEqual({shape[3] for shape in seen}, {320, 640})
        self.assertEqual(engine._rec_infer([]), ([], []))

    def test_unexpected_runtime_output_is_not_silently_decoded(self):
        engine = VietnameseOCREngine()
        engine.rec_sess = Mock()
        engine.decoder = Mock()
        engine.rec_sess.get_inputs.return_value = [SimpleNamespace(name='x')]
        engine.rec_sess.get_outputs.return_value = [SimpleNamespace(name='out')]
        for shape in [(1, 10, 504), (0, 10, 230), (2, 10, 230)]:
            engine.rec_sess.run.return_value = [np.zeros(shape, dtype=np.float32)]
            with self.subTest(shape=shape), self.assertRaisesRegex(RuntimeError, 'output shape'):
                engine._rec_infer([np.zeros((48, 120, 3), dtype=np.uint8)])
        engine.decoder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
