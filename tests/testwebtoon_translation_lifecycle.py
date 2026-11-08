"""Incremental webtoon publication must not invalidate its own request state."""
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from modules.translation.context import empty_translation_profile
from modules.translation.llm.base import BaseLLMTranslation
from modules.translation.processor import Translator
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.translation_job import BatchTranslationSession, StaleTranslationError
from pipeline.webtoon_batch.flow import FlowMixin
from pipeline.webtoon_batch.render import RenderMixin


class _Engine(BaseLLMTranslation):
    def __init__(self):
        super().__init__()
        self.source_lang, self.target_lang = 'English', 'French'
        self.model, self.max_tokens = 'fixture-model', 512
        self.calls = []

    def _perform_translation(self, prompt, _system, _image):
        data = json.loads(prompt.split('Translate this JSON:\n')[-1])
        self.calls.append(data)
        return json.dumps({key: 'translated ' + text for key, text in data.items()})


class _Processor(FlowMixin, RenderMixin):
    def __init__(self, main):
        self.main_page = main
        self.final_patches_for_save = {}
        self._get_page_scene_offset = lambda _index: 1000
        self._emit_progress = Mock()
        self._save_final_rendered_page = Mock()


class WebtoonTranslationLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        render = SimpleNamespace(upper_case=True, outline=False, font_family='Arial', color='#000000',
            max_font_size=30, min_font_size=8, line_spacing=1.0, outline_width=0,
            bold=False, italic=False, underline=False, alignment_id=0,
            direction=Qt.LayoutDirection.LeftToRight, auto_bubble_fit=False)
        settings = SimpleNamespace(ui=SimpleNamespace(tr=lambda value: value),
            get_tool_selection=lambda _tool: 'OpenAI', get_llm_settings=lambda: {'extra_context': ''})
        self.main = SimpleNamespace(settings_page=settings, lang_mapping={}, webtoon_mode=True,
            translation_profile=empty_translation_profile(), image_files=['page.png'], curr_img_idx=0,
            blk_list=[], image_states={'page.png': {'page_id': 'page1', 'source_lang': 'English',
            'target_lang': 'French', 'blk_list': [], 'viewer_state': {'text_items_state': []}}},
            image_viewer=SimpleNamespace(webtoon_manager=SimpleNamespace(loaded_pages={0})),
            render_settings=lambda: render, button_to_alignment={0: Qt.AlignmentFlag.AlignCenter},
            s_combo=SimpleNamespace(currentText=lambda: 'English'),
            t_combo=SimpleNamespace(currentText=lambda: 'French'),
            _translation_project_epoch=0, _translation_cancel_epoch=0)
        for signal in ('batch_page_edit_started', 'batch_page_edit_finished', 'patches_processed',
                       'render_state_ready', 'blk_rendered'):
            setattr(self.main, signal, Mock())
        self.engine = _Engine()
        factory = patch('modules.translation.processor.TranslationFactory.create_engine', return_value=self.engine)
        factory.start()
        self.addCleanup(factory.stop)
        self.processor = _Processor(self.main)
        self.processor._translation_session = BatchTranslationSession(self.main, ['page.png'], Translator)
        self.session = self.processor._translation_session
        self.cache = CacheManager()
        self.image = np.full((300, 200, 3), 200, np.uint8)
        self.info = {'path': 'page.png', 'global_index': 0, 'selected_index': 0, 'height': 300, 'width': 200}
        self.accum = {'page.png': {'blocks': [], 'patches': [], 'emitted_block_count': 0, 'emitted_patch_count': 0}}

    def make_chunk(self, source, top):
        return TextBlock(text_bbox=np.array([10, top, 170, top + 60]), text=source, text_class='text_bubble')

    def publish(self, blocks):
        self.accum['page.png']['blocks'].extend(blocks)
        self.processor._publish_physical_page_progress(self.info, self.accum)

    def test_two_chunks_wrapped_render_and_finalize_keep_session_valid(self):
        first = self.make_chunk('first phrase', 20)
        second = self.make_chunk('second phrase', 160)
        def wrap(text, *_args, **_kwargs):
            return text.replace(' ', '\n'), 12, 80, 50
        with patch('pipeline.webtoon_batch.render.get_best_render_area'), \
             patch('pipeline.webtoon_batch.render.pyside_word_wrap', side_effect=wrap) as wrapper:
            for block in (first, second):
                self.session.translate([block], self.image, 'English', 'French', 'page.png', self.cache)
                self.publish([block])
                self.session.validate('page.png')
            self.processor._finalize_physical_page(self.info, self.accum, 1, 'fixture', emit_live=False, emit_patches=False)
            self.session.validate('page.png')
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(wrapper.call_count, 5)
        state = self.main.image_states['page.png']
        self.assertEqual([block.translation for block in state['blk_list']],
                         ['TRANSLATED FIRST PHRASE', 'TRANSLATED SECOND PHRASE'])
        self.assertTrue(all('\n' in item['text'] for item in state['viewer_state']['text_items_state']))
        self.assertEqual(len(state['viewer_state']['text_items_state']), 2)
        self.processor._save_final_rendered_page.assert_called_once()

    def test_intervening_edits_cannot_be_acknowledged_as_renderer_publication(self):
        for changed in ('source', 'target', 'profile', 'epoch'):
            with self.subTest(changed=changed):
                self.setUp()
                first = self.make_chunk('first phrase', 20)
                self.session.translate([first], self.image, 'English', 'French', 'page.png', self.cache)
                def wrap(text, *_args, **_kwargs):
                    if changed == 'source':
                        first.text = 'User corrected source'
                    elif changed == 'target':
                        first.translation = 'User corrected target'
                    elif changed == 'profile':
                        self.main.translation_profile['notes'] = 'User changed context'
                    else:
                        self.main._translation_cancel_epoch += 1
                    return text.replace(' ', '\n'), 12, 80, 50
                with patch('pipeline.webtoon_batch.render.get_best_render_area'), \
                     patch('pipeline.webtoon_batch.render.pyside_word_wrap', side_effect=wrap):
                    self.publish([first])
                second = self.make_chunk('second phrase', 160)
                with self.assertRaises(StaleTranslationError):
                    self.session.translate([second], self.image, 'English', 'French', 'page.png', self.cache)
                self.assertEqual(len(self.engine.calls), 1)
                self.assertEqual(second.translation, '')
                if changed == 'target':
                    self.assertEqual(first.translation, 'User corrected target')
                elif changed == 'source':
                    self.assertEqual(first.text, 'User corrected source')


if __name__ == '__main__':
    unittest.main()
