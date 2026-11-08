"""Spatial requests use delivered pixels and reject stale live geometry."""
import copy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtCore, QtWidgets

from modules.translation.context import empty_translation_profile
from modules.translation.processor import Translator
from modules.translation.spatial import build_visual_reference
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.translation_geometry import capture_visible_translation, compose_visible_translation
from pipeline.translation_handler import TranslationHandler
from pipeline.translation_job import BatchTranslationSession, StaleTranslationError, TranslationJob
from testtranslation_jobs import FixtureLLM


class SpatialWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.blocks = [TextBlock(text_bbox=[2., 3., 10., 8.], text='same words', translation='old one'),
                       TextBlock(text_bbox=[15., 11., 28., 18.], text='same words', translation='old two')]
        self.image = np.full((20, 30, 3), 128, np.uint8)
        self.engine = FixtureLLM()
        self.engine.spatial_context_enabled = True
        settings = SimpleNamespace(ui=SimpleNamespace(tr=lambda text: text,
            uppercase_checkbox=SimpleNamespace(isChecked=lambda: False)),
            get_tool_selection=lambda _tool: 'OpenAI', get_llm_settings=lambda: {'extra_context': ''})
        self.main = SimpleNamespace(settings_page=settings, lang_mapping={},
            translation_profile=empty_translation_profile(), image_files=['one.png', 'two.png'],
            curr_img_idx=0, blk_list=self.blocks, webtoon_mode=False,
            image_states={'one.png': {'page_id': 'one', 'source_lang': 'English', 'target_lang': 'French',
                                     'blk_list': self.blocks},
                          'two.png': {'page_id': 'two', 'source_lang': 'English', 'target_lang': 'French',
                                     'blk_list': []}},
            _translation_project_epoch=0, _translation_cancel_epoch=0)
        self.cache = CacheManager()
        factory = patch('modules.translation.processor.TranslationFactory.create_engine', return_value=self.engine)
        factory.start()
        self.addCleanup(factory.stop)

    def job(self, blocks=None):
        return TranslationJob(self.main, Translator(self.main, 'English', 'French'),
            self.blocks if blocks is None else blocks, self.image, '', self.cache,
            resolve_blocks=lambda: self.blocks, path='one.png')

    def test_move_rotate_and_pivot_changes_reject_whole_commit(self):
        for change in ('move', 'angle', 'pivot', 'invalid'):
            with self.subTest(change=change):
                block = self.blocks[0]
                block.xyxy, block.angle, block.tr_origin_point = [2., 3., 10., 8.], 0, ()
                job = self.job().run()
                if change == 'move':
                    block.xyxy[0] += 1
                elif change == 'angle':
                    block.angle = 25
                elif change == 'pivot':
                    block.tr_origin_point = (0, 0)
                else:
                    block.xyxy[0] = float('nan')
                with self.assertRaises(StaleTranslationError):
                    job.commit()
                self.assertEqual([b.translation for b in self.blocks], ['old one', 'old two'])

    def test_image_relative_copies_compare_against_original_live_geometry(self):
        copies = [b.deep_copy() for b in self.blocks]
        for block in self.blocks:
            block.xyxy = np.asarray(block.xyxy) + [100, 200, 100, 200]
        job = self.job(copies).run()
        self.assertTrue(job.commit())
        self.assertEqual(self.blocks[0].xyxy[0], 102)
        visual = json.loads(job.request.spatial_reference_json)
        self.assertEqual(visual['regions'][0]['bbox_normalized'][0], .06666)

    def test_disabled_and_text_only_modes_ignore_geometry_changes(self):
        for mode in ('off', 'no_image', 'unsupported'):
            with self.subTest(mode=mode):
                self.engine.spatial_context_enabled = mode != 'off'
                self.engine.img_as_llm_input = mode != 'no_image'
                self.engine.supports_images = mode != 'unsupported'
                job = self.job().run()
                self.blocks[0].angle += 10
                self.assertEqual(job.request.spatial_reference_json, '')
                self.assertTrue(job.commit())

    def test_batch_guards_queued_and_in_flight_geometry(self):
        for when in ('before', 'during'):
            with self.subTest(when=when):
                self.engine.transport['calls'].clear()
                self.engine.transport['callback'] = None
                self.cache = CacheManager()
                session = BatchTranslationSession(self.main, self.main.image_files, Translator)
                change = lambda: setattr(self.blocks[0], 'angle', self.blocks[0].angle + 20)
                if when == 'before':
                    change()
                else:
                    self.engine.transport['callback'] = change
                with self.assertRaises(StaleTranslationError):
                    session.translate(self.blocks, self.image, 'English', 'French', 'one.png', self.cache)
                self.assertEqual(len(self.engine.transport['calls']), 0 if when == 'before' else 1)
                self.assertEqual([b.translation for b in self.blocks], ['old one', 'old two'])

    def webtoon_batch_fixture(self):
        self.main.webtoon_mode = True
        self.main.blk_list = [block.deep_copy() for block in self.blocks]
        for block in self.main.blk_list:
            block.xyxy = np.asarray(block.xyxy) + [1.5, 100.5, 1.5, 100.5]
        manager = SimpleNamespace(loaded_pages={0}, image_positions=[100.5, 200.5], webtoon_width=33,
            image_loader=SimpleNamespace(get_image_data=lambda _index: self.image))
        self.main.image_viewer = SimpleNamespace(webtoon_manager=manager)
        return manager

    def test_loaded_webtoon_moves_before_and_during_batch_preserve_all_targets(self):
        for when in ('before', 'during'):
            with self.subTest(when=when):
                self.webtoon_batch_fixture()
                self.engine.transport['calls'].clear()
                self.engine.transport['callback'] = None
                session = BatchTranslationSession(self.main, iter(self.main.image_files), Translator)
                change = lambda: self.main.blk_list[0].xyxy.__setitem__(0,
                    self.main.blk_list[0].xyxy[0] + 1)
                if when == 'before':
                    change()
                else:
                    self.engine.transport['callback'] = change
                with self.assertRaises(StaleTranslationError):
                    session.translate(self.blocks, self.image, 'English', 'French', 'one.png', CacheManager())
                self.assertEqual(len(self.engine.transport['calls']), 0 if when == 'before' else 1)
                self.assertEqual([b.translation for b in self.blocks], ['old one', 'old two'])
                self.assertEqual([b.translation for b in self.main.blk_list], ['old one', 'old two'])

    def test_webtoon_unload_reload_reposition_and_own_publication_are_not_edits(self):
        manager = self.webtoon_batch_fixture()
        session = BatchTranslationSession(self.main, iter(self.main.image_files), Translator)
        live = self.main.blk_list
        manager.loaded_pages.clear()
        self.main.blk_list = []
        session.validate('one.png')
        manager.loaded_pages.add(0)
        manager.image_positions[0] += 50.25
        for block in live:
            block.xyxy += [0, 50.25, 0, 50.25]
        self.main.blk_list = live
        session.validate('one.png')
        # Incremental rendering updates the saved page before its scene signal.
        added = TextBlock(text_bbox=[1, 2, 4, 8], text='new chunk')
        self.main.image_states['one.png']['blk_list'] = self.blocks + [added]
        session.mark_published('one.png')
        session.validate('one.png')
        scene_added = added.deep_copy()
        scene_added.xyxy = np.asarray(added.xyxy) + [1.5, 150.75, 1.5, 150.75]
        self.main.blk_list.append(scene_added)
        session.validate('one.png')
        scene_added.angle = 10
        with self.assertRaises(StaleTranslationError):
            session.validate('one.png')

    def test_nonspatial_batch_does_not_inspect_live_webtoon_geometry(self):
        self.webtoon_batch_fixture()
        self.engine.spatial_context_enabled = False
        session = BatchTranslationSession(self.main, iter(self.main.image_files), Translator)
        self.main.blk_list[0].xyxy[0] += 1
        request = session.translate(self.blocks, self.image, 'English', 'French', 'one.png', self.cache)
        self.assertEqual(request.spatial_reference_json, '')
        self.assertEqual(self.blocks[0].translation, 'Translated same words')

    def test_spatial_webtoon_render_keeps_canonical_geometry_across_publications(self):
        from testauto_bubble_batch import settings as render_settings
        from testmanual_text_box import _ManualTextMain
        from testwebtoon_translation_lifecycle import _Processor
        for fitting in (False, True):
            with self.subTest(fitting=fitting):
                main = _ManualTextMain()
                self.addCleanup(main.deleteLater)
                main.webtoon_mode = True
                main.translation_profile = empty_translation_profile()
                main.settings_page.get_tool_selection = lambda _tool: 'OpenAI'
                main.settings_page.get_llm_settings = lambda: {'extra_context': ''}
                main.settings_page.ui.tr = lambda text: text
                main.render_settings = lambda: render_settings(fitting)
                main.s_combo = SimpleNamespace(currentText=lambda: 'English')
                main.image_states['page.png'] = {'page_id': 'one', 'source_lang': 'English',
                    'target_lang': 'English', 'blk_list': [], 'viewer_state': {'text_items_state': []}}
                pixels = np.full((300, 200, 3), 255, np.uint8)
                manager = main.image_viewer.webtoon_manager
                manager.is_active = lambda: True
                manager.layout_manager.image_positions = [1000]
                manager.layout_manager.webtoon_width = 301
                manager.image_loader.image_data = {0: pixels}
                manager.image_loader.loaded_pages = {0}
                main.image_viewer.hasPhoto = lambda: True
                for name in ('batch_page_edit_started', 'batch_page_edit_finished', 'patches_processed'):
                    setattr(main, name, Mock())
                main.blk_rendered = SimpleNamespace(emit=main.text_ctrl.on_blk_rendered)
                processor = _Processor(main)
                session = BatchTranslationSession(main, ['page.png'], Translator)
                processor._translation_session = session
                info = {'path': 'page.png', 'global_index': 0, 'height': 300, 'width': 200}
                accum = {'page.png': {'blocks': [], 'patches': [], 'emitted_patch_count': 0}}
                blocks = [TextBlock(text_bbox=np.array([10., top, 170., top + 60.]),
                    text='first phrase' if top == 20 else 'second phrase', text_class='text_bubble')
                    for top in (20., 160.)]
                expected = [list(block.xyxy) for block in blocks]
                with patch('pipeline.webtoon_batch.render.load_original_for_fitting', return_value=pixels):
                    for block in blocks:
                        session.translate([block], pixels, 'English', 'English', 'page.png', CacheManager())
                        accum['page.png']['blocks'].append(block)
                        processor._publish_physical_page_progress(info, accum)
                        session.validate('page.png')
                self.assertEqual([list(block.xyxy) for block in blocks], expected)
                self.assertEqual(len(main.image_viewer.text_items), 2)
                for local, live in zip(blocks, main.blk_list):
                    np.testing.assert_allclose(live.xyxy, np.asarray(local.xyxy) + [50.5, 1000, 50.5, 1000])
                main.blk_list[0].angle += 15
                with self.assertRaises(StaleTranslationError):
                    session.validate('page.png')

    def test_selected_page_handler_uses_original_full_page_and_region_identity(self):
        self.main.s_combo = SimpleNamespace(currentText=lambda: 'English')
        self.main.t_combo = SimpleNamespace(currentText=lambda: 'French')
        self.main.image_viewer = SimpleNamespace(hasPhoto=lambda: True,
            get_image_array=lambda **_kwargs: self.image)
        pipeline = SimpleNamespace(get_selected_block=lambda: self.blocks[1])
        job = TranslationHandler(self.main, self.cache, pipeline).prepare_job(single_block=True).run()
        visual = json.loads(job.request.spatial_reference_json)
        self.assertEqual(job.request.region_ids, (self.blocks[1].region_id,))
        self.assertEqual((visual['image_width'], visual['image_height']), (30, 20))
        self.assertEqual(visual['regions'][0]['bbox_normalized'], [0.5, .55, .93334, .9])

    def test_batch_virtual_seam_keeps_full_text_with_partial_visual_reference(self):
        self.blocks[0].xyxy = [2, 15, 10, 25]
        session = BatchTranslationSession(self.main, self.main.image_files, Translator)
        request = session.translate(self.blocks, self.image, 'English', 'French', 'one.png', self.cache)
        region = json.loads(request.spatial_reference_json)['regions'][0]
        self.assertTrue(region['clipped'])
        self.assertEqual(region['bbox_normalized'][3], 1.)
        self.assertEqual(request.source_texts, ('same words', 'same words'))
        self.assertEqual(request.region_ids, tuple(b.region_id for b in self.blocks))

    def test_visible_handler_commits_against_scene_geometry_not_cropped_coordinates(self):
        visible, pipeline = VisibleSpatialGeometryTests().fixture()
        self.main.image_viewer = visible.image_viewer
        self.main.image_files, self.main.image_states = visible.image_files, visible.image_states
        self.main.blk_list, self.main.webtoon_mode = visible.blk_list, True
        self.main.s_combo = SimpleNamespace(currentText=lambda: 'English')
        self.main.t_combo = SimpleNamespace(currentText=lambda: 'French')
        before = [list(b.xyxy) for b in self.main.blk_list]
        job = TranslationHandler(self.main, self.cache, pipeline).prepare_job(visible=True).run()
        visual = json.loads(job.request.spatial_reference_json)
        self.assertEqual((visual['image_width'], visual['image_height']), (8, 12))
        self.assertTrue(visual['regions'][0]['clipped'])
        self.assertTrue(job.commit())
        self.assertEqual([b.xyxy for b in self.main.blk_list], before)
        self.assertEqual([b.translation for b in self.main.blk_list], ['Translated first', 'Translated second'])


class VisibleSpatialGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_exact_crop_rows_centering_gaps_and_seam_coordinates(self):
        pages = [dict(image=np.full((8, 8, 3), [255, 0, 0], np.uint8), scene_x=2., scene_y=100.,
                      crop_top=1, crop_bottom=8),
                 dict(image=np.full((6, 12, 3), [0, 0, 255], np.uint8), scene_x=0., scene_y=112.,
                      crop_top=0, crop_bottom=4)]
        blocks = [TextBlock(text_bbox=[3, 100.5, 7, 105], text='first'),
                  TextBlock(text_bbox=[3, 106, 7, 114], text='seam')]
        original = copy.deepcopy([b.xyxy for b in blocks])
        pixels, copies = compose_visible_translation(pages, blocks)
        self.assertEqual(pixels.shape, (15, 12, 3))
        np.testing.assert_array_equal(pixels[0, 3], [255, 0, 0])
        np.testing.assert_array_equal(pixels[8, 3], [255, 255, 255])
        np.testing.assert_array_equal(pixels[12, 3], [0, 0, 255])
        np.testing.assert_allclose(copies[0].xyxy, [3, -.5, 7, 4])
        np.testing.assert_allclose(copies[1].xyxy, [3, 5, 7, 13])
        self.assertEqual([b.xyxy for b in blocks], original)
        self.assertEqual([b.region_id for b in copies], [b.region_id for b in blocks])
        visual = build_visual_reference(copies, pixels.shape, ('block_0', 'block_1'))
        self.assertTrue(visual['regions'][0]['clipped'])
        self.assertFalse(visual['regions'][1]['clipped'])

    def fixture(self):
        images = [np.full((8, 5, 3), [255, 0, 0], np.uint8),
                  np.full((7, 8, 3), [0, 0, 255], np.uint8)]
        blocks = [TextBlock(text_bbox=[2.5, 104.5, 5.5, 107], text='first'),
                  TextBlock(text_bbox=[1, 113, 5, 116], text='second')]
        bounds = QtCore.QRectF(0, 105.8, 8, 12.1)
        manager = SimpleNamespace(image_positions=[100., 112.], webtoon_width=8,
            layout_manager=SimpleNamespace(get_pages_for_scene_bounds=lambda _bounds: {0, 1}),
            image_loader=SimpleNamespace(get_image_data=lambda index: images[index]))
        viewer = SimpleNamespace(webtoon_manager=manager, viewport=lambda: SimpleNamespace(rect=lambda: bounds),
            mapToScene=lambda _rect: SimpleNamespace(boundingRect=lambda: bounds), hasPhoto=lambda: True)
        states = {'first': {'page_id': 'page-first', 'blk_list': [blocks[0].deep_copy()]},
                  'second': {'page_id': 'page-second', 'blk_list': [blocks[1].deep_copy()]}}
        main = SimpleNamespace(image_viewer=viewer, image_files=['first', 'second'], image_states=states, blk_list=blocks)
        return main, SimpleNamespace(get_selected_block=lambda: blocks[0])

    def test_fractional_viewport_and_odd_page_width_keep_float_geometry(self):
        main, pipeline = self.fixture()
        pixels, blocks = capture_visible_translation(main, pipeline)
        self.assertEqual(pixels.shape, (12, 8, 3))
        np.testing.assert_allclose(blocks[0].xyxy, [2.5, -.5, 5.5, 2])
        np.testing.assert_array_equal(pixels[0, 3], [255, 0, 0])
        np.testing.assert_array_equal(pixels[8, 3], [0, 0, 255])
        self.assertEqual([b._page_id for b in blocks], ['page-first', 'page-second'])

    def test_page_reorder_retains_stable_owners_and_request_order(self):
        main, pipeline = self.fixture()
        main.image_files.reverse()
        main.blk_list.reverse()
        # Geometry is deliberately the current live scene; ownership comes
        # from stable page IDs, never the text or request-local ordinal.
        _, blocks = capture_visible_translation(main, pipeline)
        self.assertEqual([b._page_id for b in blocks], ['page-second', 'page-first'])
        self.assertEqual([b.region_id for b in blocks], [b.region_id for b in main.blk_list])

    def test_rotated_visible_region_and_explicit_off_image_selection(self):
        main, pipeline = self.fixture()
        selected = main.blk_list[0]
        selected.xyxy, selected.angle, selected.tr_origin_point = [1, 102, 7, 104], 90, (3, 1)
        pixels, blocks = capture_visible_translation(main, pipeline)
        self.assertIn(selected.region_id, [b.region_id for b in blocks])
        selected.xyxy, selected.angle = [50, 200, 60, 210], 0
        pixels, blocks = capture_visible_translation(main, pipeline, single_block=True)
        self.assertEqual(len(blocks), 1)
        visual = build_visual_reference(blocks, pixels.shape, ('block_0',))
        self.assertFalse(visual['regions'][0]['visible'])
        self.assertEqual(blocks[0].text, selected.text)


if __name__ == '__main__':
    unittest.main()
