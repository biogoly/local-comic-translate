"""Explicit one-region local OCR with protected source preview, apply and undo."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6 import QtWidgets

import testreview_controller as fixtures
from modules.ocr.evidence import current_ocr_evidence, make_ocr_evidence
from modules.ocr.retry import retry_recognizers, recognize_alternative, LocalRecognizer
from modules.ocr.ppocr.engine import LANG_TO_REC_MODEL
from modules.utils.download import ModelID
from modules.utils.textblock import TextBlock


class AlternativeRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.block = TextBlock(text_bbox=np.array([10.0, 6.0, 60.0, 35.0]),
                               text_class='text_bubble', text='Hola')

    def routes(self, language, default=True, **kwargs):
        return retry_recognizers(self.block, language, default_selected=default, **kwargs)

    def test_only_distinct_supported_model_weights_are_offered(self):
        routes = self.routes('Spanish')
        self.assertEqual([r.model for r in routes], [ModelID.PPOCR_V6_REC_SMALL])
        self.assertEqual(routes[0].source_code, 'es')
        self.assertEqual(LANG_TO_REC_MODEL[routes[0].pp_language], routes[0].model)
        self.assertEqual(self.routes('Vietnamese'), [])
        self.assertEqual(self.routes('Vietnamese', default=False)[0].model, ModelID.PPOCR_V6_REC_VI)
        self.assertEqual(self.routes('Russian'), [])
        self.assertEqual(self.routes('Korean')[0].model, ModelID.PORORO_ONNX)
        self.assertEqual(self.routes('Auto'), [])
        self.block.script = 'Latin'
        self.assertEqual(self.routes('Auto')[0].model, ModelID.PPOCR_V6_REC_SMALL)

    def test_actual_previous_model_takes_precedence_over_current_setting(self):
        self.block.ocr_evidence = make_ocr_evidence('Hola', 'PP-OCR', ModelID.PPOCR_V6_REC_SMALL.value,
                                                    line_scores=[.52])
        for selected in (False, True):
            self.assertEqual([r.model for r in self.routes('Spanish', selected)], [ModelID.PPOCR_V5_REC_LATIN_MOBILE])

    def test_japanese_large_title_uses_actual_default_override(self):
        self.block.text_class = 'text_free'
        self.assertFalse(self.routes('Japanese', page_size=None))
        self.assertEqual(self.routes('Japanese', page_size=(80, 80))[0].model, ModelID.MANGA_OCR_MOBILE_ONNX)
        self.assertEqual(self.routes('Japanese', page_size=(1000, 1000))[0].model, ModelID.PPOCR_V6_REC_SMALL)

    def test_recognizer_receives_only_copied_selected_pixels_and_no_target_text(self):
        page = np.zeros((75, 100, 3), np.uint8)
        page[6:35, 10:60] = 230
        self.block.translation = 'must not be sent'
        def read(crop, blocks):
            np.testing.assert_array_equal(crop, np.full((29, 50, 3), 230, np.uint8))
            self.assertEqual(len(blocks), 1)
            region = blocks[0]
            self.assertEqual((region.text, region.translation, region.source_lang), ('', '', 'es'))
            region.text = '¡Hola!'
            region.ocr_evidence = make_ocr_evidence(region.text, 'PP-OCR', ModelID.PPOCR_V6_REC_SMALL.value,
                                                    line_scores=[.91])
            crop[:] = 0
        with patch.object(LocalRecognizer, 'load', return_value=SimpleNamespace(process_image=read)):
            text, evidence = recognize_alternative(page, self.block, self.routes('Spanish')[0], False)
        self.assertEqual(text, '¡Hola!')
        self.assertEqual(evidence['score'], .91)
        self.assertEqual(self.block.text, 'Hola')
        self.assertTrue((page[6:35, 10:60] == 230).all())


class ReviewRetryTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.ReviewControllerTests.setUpClass.__func__)
    tearDown = fixtures.ReviewControllerTests.tearDown
    load_page = fixtures.ReviewControllerTests.load_page
    add_region = fixtures.ReviewControllerTests.add_region

    def setUp(self):
        fixtures.ReviewControllerTests.setUp(self)
        self.main.s_combo.addItem('Spanish')
        self.main.s_combo.setCurrentText('Spanish')
        self.main.image_states['page.png']['source_lang'] = 'Spanish'
        self.main.settings_page.get_tool_selection = lambda _: 'Default'
        self.main.settings_page.is_gpu_enabled = lambda: False
        self.block, self.item = self.add_region(text='H0la', target='Hello', rendered=True)
        self.main.curr_tblock = self.block
        self.block.region_metadata['protections']['keep_source'] = True
        self.old = make_ocr_evidence(self.block.text, 'PP-OCR', ModelID.PPOCR_V5_REC_LATIN_MOBILE.value,
                                     line_scores=[.4])
        self.block.ocr_evidence = self.old
        self.new = make_ocr_evidence('Hola', 'PP-OCR', ModelID.PPOCR_V6_REC_SMALL.value, line_scores=[.97])

    def run_retry(self, answer, *, chooser_ok=True, read=True):
        dialogs = []
        def respond(dialog):
            dialogs.append(dialog)
            return answer(dialog)
        def recognize(image, block, route, gpu):
            self.assertIsNot(block, self.block)
            self.assertIs(image, self.main.image_data['page.png'])
            self.assertEqual(route.model, ModelID.PPOCR_V6_REC_SMALL)
            self.assertFalse(gpu)
            return 'Hola', self.new
        with (
            patch('app.controllers.review.QtWidgets.QInputDialog.getItem',
                  return_value=('PP-OCRv6 (Multilingual)', chooser_ok)) as choose,
            patch('modules.rendering.auto_bubble_fit.load_original_for_fitting',
                  return_value=self.main.image_data['page.png']),
            patch('modules.ocr.retry.recognize_alternative', side_effect=recognize) as run,
            patch('app.controllers.review.QtWidgets.QDialog.exec', new=respond),
        ):
            self.main.review_panel.retry_ocr_requested.emit()
        choose.assert_called_once()
        self.assertIn('downloaded', choose.call_args.args[2])
        self.assertEqual(run.call_count, int(read))
        self.assertEqual(len(dialogs), int(read))

    def test_apply_replaces_only_source_and_preserves_evidence_through_undo(self):
        before_item = self.item.toHtml()
        def accept(dialog):
            current, proposed = dialog.findChildren(QtWidgets.QPlainTextEdit)
            self.assertEqual([current.toPlainText(), proposed.toPlainText()], ['H0la', 'Hola'])
            self.assertTrue(current.isReadOnly())
            self.assertEqual(self.block.text, 'H0la')
            return QtWidgets.QDialog.Accepted
        self.run_retry(accept)
        stack = self.main.undo_stacks['page.png']
        for action, text, evidence in ((lambda: None, 'Hola', self.new), (stack.undo, 'H0la', self.old),
                                       (stack.redo, 'Hola', self.new)):
            action()
            self.assertEqual(self.block.text, text)
            self.assertEqual(self.main.s_text_edit.toPlainText(), text)
            self.assertEqual(current_ocr_evidence(self.block), evidence)
            saved = self.main.image_states['page.png']['blk_list'][0]
            self.assertEqual(current_ocr_evidence(saved), evidence)
            self.assertEqual(self.block.translation, 'Hello')
            self.assertTrue(self.block.region_metadata['protections']['keep_source'])
            self.assertEqual(self.item.toHtml(), before_item)
            self.assertFalse(self.main.image_patches)

    def test_preview_correction_drops_machine_score(self):
        def edit(dialog):
            dialog.findChildren(QtWidgets.QPlainTextEdit)[1].setPlainText('¡Hola!')
            return QtWidgets.QDialog.Accepted
        self.run_retry(edit)
        self.assertEqual(self.block.text, '¡Hola!')
        self.assertIsNone(self.block.ocr_evidence)
        self.main.undo_stacks['page.png'].undo()
        self.assertEqual(current_ocr_evidence(self.block), self.old)

    def test_cancel_and_stale_preview_do_not_commit(self):
        stack = self.main.undo_stacks['page.png']
        before = stack.count()
        self.run_retry(lambda _: QtWidgets.QDialog.Rejected, chooser_ok=False, read=False)
        self.run_retry(lambda _: QtWidgets.QDialog.Rejected)
        def stale(_):
            self.block.text = 'a new edit'
            return QtWidgets.QDialog.Accepted
        self.run_retry(stale)
        self.assertEqual(self.block.text, 'a new edit')
        self.assertEqual(stack.count(), before)
        self.assertIsNone(current_ocr_evidence(self.block))

    def test_unsupported_and_unreadable_regions_never_replace_the_source(self):
        self.main.image_states['page.png']['source_lang'] = 'Russian'
        self.block.ocr_evidence = None
        with (
            patch('app.controllers.review.QtWidgets.QMessageBox.information') as notice,
            patch('app.controllers.review.QtWidgets.QInputDialog.getItem') as choose,
        ):
            self.ctrl.retry_ocr()
        self.assertIn('No alternative', notice.call_args.args[2])
        choose.assert_not_called()
        self.main.image_states['page.png']['source_lang'] = 'Spanish'
        with (
            patch('app.controllers.review.QtWidgets.QMessageBox.information') as notice,
            patch('app.controllers.review.QtWidgets.QInputDialog.getItem', return_value=('PP-OCRv6 (Multilingual)', True)),
            patch('modules.rendering.auto_bubble_fit.load_original_for_fitting', return_value=self.main.image_data['page.png']),
            patch('modules.ocr.retry.recognize_alternative', return_value=('', None)),
            patch('app.controllers.review.QtWidgets.QDialog.exec') as compare,
        ):
            self.ctrl.retry_ocr()
        self.assertIn('no text', notice.call_args.args[2])
        compare.assert_not_called()
        self.assertEqual(self.block.text, 'H0la')

    def test_geometry_change_while_running_discards_reading_before_comparison(self):
        def stale(*_):
            self.block.xyxy += [0, 1, 0, 1]
            return 'Hola', self.new
        with (
            patch('app.controllers.review.QtWidgets.QInputDialog.getItem', return_value=('PP-OCRv6 (Multilingual)', True)),
            patch('modules.rendering.auto_bubble_fit.load_original_for_fitting', return_value=self.main.image_data['page.png']),
            patch('modules.ocr.retry.recognize_alternative', side_effect=stale),
            patch('app.controllers.review.QtWidgets.QDialog.exec') as compare,
        ):
            self.ctrl.retry_ocr()
        compare.assert_not_called()
        self.assertEqual(self.block.text, 'H0la')
        self.assertEqual(current_ocr_evidence(self.block), self.old)
