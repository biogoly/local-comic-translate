"""Page and virtual-page protection integration, with model calls replaced."""
import os
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from modules.utils.region_policy import subtract_cleanup_protections, shifted_region_copy
from modules.utils.textblock import TextBlock
from pipeline.batch_processor import BatchProcessor
from pipeline.cache_manager import CacheManager
from pipeline.virtual_page import VirtualPage
from pipeline.webtoon_batch.chunk import ChunkMixin
from pipeline.webtoon_batch.flow import FlowMixin
from pipeline.webtoon_batch.render import RenderMixin


def block(bounds, flag=None):
    return TextBlock(text_bbox=np.asarray(bounds), text_class='text_bubble',
                     text='Retained source', translation='Retained Target',
                     region_metadata={'protections': {flag: True} if flag else {}})


def render_settings():
    return SimpleNamespace(upper_case=True, outline=False, font_family='Arial', color='#000000',
                           max_font_size=30, min_font_size=8, line_spacing=1.2, outline_width=0,
                           bold=False, italic=False, underline=False, alignment_id=0,
                           direction=Qt.LayoutDirection.LeftToRight, auto_bubble_fit=False)


class ReviewBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setup_page(self):
        self.blocks = [block((10 + 65*i, 20, 60 + 65*i, 80), flag) for i, flag in enumerate(
            ('keep_source', 'keep_translation', 'keep_layout', 'skip_auto_cleanup'))]
        self.blocks[3].text_class = 'text_free'
        self.original = np.full((120, 350, 3), 252, dtype=np.uint8)
        self.composed = np.random.default_rng(7).integers(0, 240, self.original.shape, dtype=np.uint8)
        self.kept = {'region_id': self.blocks[2].region_id, 'text': '<p><b>Custom</b> lettering</p>',
                     'position': (150.5, 33.5), 'width': 71, 'height': 44, 'font_size': 19,
                     'layout_preferences': {'mode': 'box'}}
        self.main = SimpleNamespace(
            image_files=['page.png'], curr_img_idx=0, current_worker=None, blk_list=[],
            image_states={'page.png': {'source_lang': 'English', 'target_lang': 'French',
                          'blk_list': deepcopy(self.blocks), 'viewer_state': {'text_items_state': [deepcopy(self.kept)]}}},
            image_ctrl=SimpleNamespace(get_composited_page_image=lambda _: self.composed.copy()),
            settings_page=SimpleNamespace(get_tool_selection=lambda _: 'Local LLM', is_gpu_enabled=lambda: False,
                get_llm_settings=lambda: {'extra_context': ''}, get_export_settings=lambda: {
                    'export_raw_text': False, 'export_translated_text': False, 'export_inpainted_image': False}),
            file_handler=SimpleNamespace(should_pre_materialize=lambda _: False, archive_info=[]),
            render_settings=render_settings, button_to_alignment={0: Qt.AlignmentFlag.AlignCenter},
            image_viewer=SimpleNamespace(webtoon_manager=None), webtoon_mode=False, lang_mapping={},
        )
        for name in ('progress_update', 'image_skipped', 'batch_page_edit_started', 'patches_processed',
                     'render_state_ready', 'batch_page_edit_finished', 'blk_rendered'):
            setattr(self.main, name, Mock())

    def run_page(self, absent_layout=False):
        self.setup_page()
        if absent_layout:
            self.main.image_states['page.png']['viewer_state']['text_items_state'] = []
        detected = [block(tuple(region.xyxy)) for region in self.blocks]
        detected.append(block((280, 20, 330, 80)))
        detected[-1].text_class = ''  # Unknown class still receives every normal stage.
        detector = SimpleNamespace(block_detector_cache=Mock(), annotate_language_if_auto=Mock())
        detector.block_detector_cache.detect.return_value = detected
        ocr = SimpleNamespace(ocr=Mock())
        def recognize(image, blocks):
            self.assertIs(image, self.original)
            for region in blocks:
                region.text = 'New source'
        ocr.ocr.process.side_effect = recognize
        def translate(blocks, image, context, *, request=None):
            self.assertIs(image, self.original)
            for region in blocks:
                region.translation = 'New target'
        def inpaint(image, mask, config, **kwargs):
            image[:] = 5
            return image
        handler = SimpleNamespace(inpaint_image=inpaint, get_inpainted_patches=lambda mask, image:
                                  [{'bbox': [0, 0, 350, 120], 'image': image.copy()}])
        processor = BatchProcessor(self.main, CacheManager(), detector, handler, ocr)
        mask = np.zeros(self.original.shape[:2], dtype=np.uint8)
        mask[8:99, 8:342] = 255
        with ExitStack() as stack:
            for name, kwargs in {
                'ensure_path_materialized': {}, 'imk.read_image': {'return_value': self.original},
                'Translator': {}, 'get_config': {'return_value': None},
                'generate_mask': {'return_value': mask}, 'get_best_render_area': {},
                'pyside_word_wrap': {'return_value': ('New target', 12, 50, 50)},
            }.items():
                mocked = stack.enter_context(patch('pipeline.batch_processor.' + name, **kwargs))
                if name == 'Translator':
                    mocked.return_value.translate.side_effect = translate
            processor.batch_process()
        return mask, ocr

    def test_whole_page_processes_each_stage_independently_and_preserves_composed_pixels(self):
        mask, ocr = self.run_page()
        result = self.main.image_states['page.png']
        by_id = {region.region_id: region for region in result['blk_list']}
        self.assertEqual(len(by_id), 5)
        self.assertEqual(by_id[self.blocks[0].region_id].text, 'Retained source')
        self.assertEqual(by_id[self.blocks[0].region_id].translation, 'NEW TARGET')
        self.assertEqual(by_id[self.blocks[1].region_id].text, 'New source')
        self.assertEqual(by_id[self.blocks[1].region_id].translation, 'Retained Target')
        self.assertEqual(by_id[self.blocks[2].region_id].translation, 'NEW TARGET')
        self.assertEqual(by_id[self.blocks[3].region_id].translation, 'NEW TARGET')
        states = result['viewer_state']['text_items_state']
        self.assertEqual(next(state for state in states if state['region_id'] == self.kept['region_id']), self.kept)
        self.assertEqual(len(ocr.ocr.process.call_args.args[1]), 4)
        self.main.image_skipped.emit.assert_not_called()
        pixels = self.main.patches_processed.emit.call_args.args[0][0]['image']
        effective = subtract_cleanup_protections(mask, self.blocks)
        np.testing.assert_array_equal(pixels[effective == 0], self.composed[effective == 0])
        self.assertTrue(np.all(pixels[effective > 0] == 5))
        self.assertTrue(np.all(self.original == 252))

    def test_keep_layout_preserves_absent_text_layer_without_skipping_other_regions(self):
        self.run_page(absent_layout=True)
        states = self.main.image_states['page.png']['viewer_state']['text_items_state']
        self.assertNotIn(self.blocks[2].region_id, {state['region_id'] for state in states})
        self.assertEqual(len(states), 4)

    def test_webtoon_renderer_preserves_kept_item_when_optional_auto_fit_is_disabled(self):
        self.setup_page()
        proxy = SimpleNamespace(main_page=self.main, _get_page_scene_offset=lambda _: 0)
        regions = [self.blocks[2], block((280, 20, 330, 80))]
        with patch('pipeline.webtoon_batch.render.get_best_render_area') as adjust, \
             patch('pipeline.webtoon_batch.render.pyside_word_wrap', return_value=('fresh', 12, 40, 30)) as wrap:
            RenderMixin._store_page_text_items(proxy, 0, 'page.png', regions, self.original.shape, live_blocks=[])
        states = self.main.image_states['page.png']['viewer_state']['text_items_state']
        self.assertEqual(states[0], self.kept)
        self.assertEqual(len(states), 2)
        self.assertEqual(adjust.call_args.args[0], [regions[1]])
        self.assertEqual(wrap.call_count, 1)


class ReviewVirtualRegionTests(unittest.TestCase):
    def proxy(self, previous, *, previous_path=None):
        main = SimpleNamespace(image_states={}, image_files=['page'], image_ctrl=None)
        return SimpleNamespace(main_page=main, _review_previous_blocks={'page': previous},
            _review_retained_ids={'page': set()}, _review_previous_path=previous_path or {},
            _review_page_heights={}, _read_virtual_image=lambda _: np.full((100, 160, 3), 200, dtype=np.uint8),
            _detect_blocks_for_page=lambda image: [block((12, 0, 68, 30)), block((90, 40, 140, 80))],
            _emit_progress=Mock())

    def record(self, top=0):
        vpage = VirtualPage(physical_page_index=0, physical_page_path='page', virtual_index=top//100,
                            crop_top=top, crop_bottom=top+100, crop_height=100,
                            physical_width=160, physical_height=200, virtual_id=str(top))
        return {'path': 'page', 'vpage': vpage, 'selected_index': 0}

    def test_region_crossing_virtual_seam_has_one_owner_and_protects_both_chunks(self):
        retained = block((10, 80, 70, 130), 'skip_auto_cleanup')
        proxy = self.proxy([retained])
        first = FlowMixin._load_virtual_record(proxy, self.record(), 1, False)
        second = FlowMixin._load_virtual_record(proxy, self.record(100), 1, False)
        self.assertIn(retained.region_id, [region.region_id for region in first['detected_blocks']])
        self.assertNotIn(retained.region_id, [region.region_id for region in second['detected_blocks']])
        self.assertEqual(len(second['detected_blocks']), 1)  # Duplicate below seam suppressed, neighbour retained.
        clipped = subtract_cleanup_protections(np.full((100, 160), 255, dtype=np.uint8), second['protection_blocks'])
        self.assertFalse(np.any(clipped[:30, 10:70]))
        self.assertTrue(np.all(clipped[40:80, 90:140] == 255))
        np.testing.assert_array_equal(retained.xyxy, [10, 80, 70, 130])

    def test_previous_physical_page_cleanup_protection_carries_over_to_next_page(self):
        retained = block((10, 90, 70, 125), 'skip_auto_cleanup')
        proxy = self.proxy([], previous_path={'page': 'previous'})
        proxy._review_previous_blocks['previous'] = [retained]
        proxy._review_page_heights['previous'] = 100
        loaded = FlowMixin._load_virtual_record(proxy, self.record(), 1, False)
        effective = subtract_cleanup_protections(np.full((100, 160), 255, dtype=np.uint8), loaded['protection_blocks'])
        self.assertFalse(np.any(effective[:25, 10:70]))
        self.assertTrue(np.all(effective[40:80, 90:140] == 255))

    def test_webtoon_translation_failure_does_not_clear_kept_target(self):
        kept, fresh = block((10, 10, 50, 50), 'keep_translation'), block((60, 10, 100, 50))
        main = SimpleNamespace(settings_page=SimpleNamespace(get_llm_settings=lambda: {'extra_context': ''}),
                               image_skipped=Mock(), image_files=['page'],
                               image_states={'page': {'source_lang': 'English', 'target_lang': 'French',
                                                      'blk_list': [kept, fresh]}})
        proxy = SimpleNamespace(main_page=main, cache_manager=CacheManager(),
                                _extract_error_message=lambda error, **kwargs: str(error))
        with patch('pipeline.webtoon_batch.chunk.Translator') as translator:
            translator.return_value.translate.side_effect = RuntimeError('fixture failed')
            with self.assertRaisesRegex(RuntimeError, 'fixture failed'):
                ChunkMixin._run_translation_on_blocks(proxy, np.zeros((80, 120, 3), dtype=np.uint8),
                                                      [kept, fresh], 'English', 'French', 'page')
        self.assertEqual(kept.translation, 'Retained Target')
        self.assertEqual(fresh.translation, 'Retained Target')

    def test_seam_cleanup_uses_and_updates_inflight_composed_records_without_touching_originals(self):
        original = np.zeros((100, 80, 3), dtype=np.uint8)
        records = [{'image': original.copy(), 'composed_image': np.full_like(original, color),
                    'global_index': index, 'path': str(index), 'y_offset': 0}
                   for index, color in enumerate((11, 22))]
        owner = block((20, 85, 50, 115))
        def clean(image, blocks, **kwargs):
            self.assertTrue(np.all(image[:30] == 11))
            self.assertTrue(np.all(image[30:] == 22))
            image = image.copy()
            image[15:45, 20:50] = 33
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[15:45, 20:50] = 255
            return mask, image
        proxy = SimpleNamespace(
            seam_crop_pad_x=0, seam_crop_pad_y=0,
            _build_stitched_pair=lambda top, bottom: np.vstack((top, bottom)),
            _compute_union_crop=lambda *args, **kwargs: [0, 70, 80, 130],
            _localize_blocks_to_crop=lambda blocks, bounds: [shifted_region_copy(region, -bounds[0], -bounds[1]) for region in blocks],
            _inpaint_image_with_blocks=clean,
            _extract_seam_patches_from_mask=lambda **kwargs: kwargs,
        )
        seam = SimpleNamespace(top_page_index=0, bottom_page_index=1,
                               matches=[SimpleNamespace(owner_block=owner)])
        ChunkMixin._process_seam_job_ocr_and_inpaint(proxy, seam, records)
        self.assertTrue(np.all(records[0]['composed_image'][85:100, 20:50] == 33))
        self.assertTrue(np.all(records[1]['composed_image'][:15, 20:50] == 33))
        self.assertTrue(np.all(records[0]['composed_image'][:85] == 11))
        self.assertTrue(np.all(records[1]['composed_image'][15:] == 22))
        for record in records:
            np.testing.assert_array_equal(record['image'], original)


if __name__ == '__main__':
    unittest.main()
