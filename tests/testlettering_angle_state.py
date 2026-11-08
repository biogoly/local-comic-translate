"""Persist the typesetter's angle without changing region geometry or ownership."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from app.controllers.search_replace import BlockKey, SearchReplaceController
from app.projects.parsers import ProjectDecoder
from app.projects.project_metadata import duplicate_page_state
from app.projects.project_state import load_state_from_proj_file
from app.projects.project_state_v2 import save_state_to_proj_file_v2
from modules.utils.lettering_angle import render_lettering_angle
from modules.utils.manual_regions import merge_manual_regions
from modules.utils.region_policy import permits_region_operation
from modules.utils.textblock import TextBlock
from testtranslation_profile_persistence import main_fixture, prepared_main, project_directory


class LetteringAngleStateTests(unittest.TestCase):
    def test_old_project_uses_stored_box_angle_and_ignores_nonfinite_estimate(self):
        old = ProjectDecoder.decode_textblock({'data': {'angle': -16.0}})
        self.assertIsNone(old.manual_lettering_angle)
        self.assertIsNone(old.detected_lettering_angle)
        self.assertEqual(render_lettering_angle(old), -16.0)
        old.detected_lettering_angle = 11
        self.assertEqual(render_lettering_angle(old), -16.0)
        invalid = ProjectDecoder.decode_textblock({'data': {
            'angle': 0, 'manual_lettering_angle': float('nan'), 'detected_lettering_angle': float('inf')}})
        self.assertIsNone(invalid.manual_lettering_angle)
        self.assertIsNone(invalid.detected_lettering_angle)
        self.assertEqual(render_lettering_angle(invalid), 0)

    def test_project_reopen_and_redetection_respect_a_manual_zero(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            block = main.image_states[main.image_files[0]]['blk_list'][0]
            block.text_class = 'text_bubble'
            block.detected_lettering_angle = 16.0
            block.manual_lettering_angle = 0.0
            original_box = block.xyxy.copy()
            project = str(Path(directory) / 'angled-credits.ctpr')
            save_state_to_proj_file_v2(main, project)
            reopened = main_fixture(Path(directory) / 'reopened')
            load_state_from_proj_file(reopened, project)
            saved = reopened.image_states[reopened.image_files[0]]['blk_list'][0]
            self.assertEqual(saved.deep_copy().detected_lettering_angle, 16.0)
            self.assertEqual(saved.deep_copy().manual_lettering_angle, 0.0)
            self.assertEqual(saved.angle, 0.0)
            self.assertEqual(render_lettering_angle(saved), 0.0)
            detected = TextBlock(text_bbox=original_box.copy(), text_class='text_bubble',
                                 detected_lettering_angle=17.0)
            self.assertEqual(merge_manual_regions([saved], [detected]), [saved])
            np.testing.assert_array_equal(saved.xyxy, original_box)
            for stage in ('ocr', 'translation', 'cleanup', 'layout'):
                self.assertTrue(permits_region_operation(saved, stage))
            saved.manual_lettering_angle = None
            self.assertEqual(render_lettering_angle(saved), 16.0)
            self.assertEqual(merge_manual_regions([saved], [detected]), [detected])

    def test_search_replace_finds_rotated_lettering_by_region(self):
        block = TextBlock(text_bbox=np.array([10, 20, 80, 50]), detected_lettering_angle=-18)
        key = BlockKey('page.png', (10, 20, 80, 50), 0.0)
        neighbour = {'position': (10, 20), 'rotation': 0, 'region_id': 'another-region'}
        lettering = {'position': (12, 8), 'rotation': -18, 'region_id': block.region_id}
        state = {'blk_list': [block], 'viewer_state': {'text_items_state': [neighbour, lettering]}}
        self.assertIs(SearchReplaceController._find_matching_text_item_state(None, state, key), lettering)
        items = [SimpleNamespace(region_id=data['region_id']) for data in [neighbour, lettering]]
        controller = SimpleNamespace(main=SimpleNamespace(image_viewer=SimpleNamespace(text_items=items)))
        self.assertIs(SearchReplaceController._find_text_item_for_block(controller, block), items[1])

    def test_page_copy_relinks_the_angle_owning_rectangle_and_lettering(self):
        block = TextBlock(text_bbox=np.array([10, 20, 80, 50]), manual_lettering_angle=0)
        linked = {'region_id': block.region_id}
        state = {'blk_list': [block], 'viewer_state': {
            'rectangles': [dict(linked)], 'text_items_state': [dict(linked)]}}
        copy = duplicate_page_state(state)
        new = copy['blk_list'][0]
        self.assertNotEqual(new.region_id, block.region_id)
        self.assertEqual(new.manual_lettering_angle, 0)
        self.assertEqual(copy['viewer_state']['rectangles'][0]['region_id'], new.region_id)
        self.assertEqual(copy['viewer_state']['text_items_state'][0]['region_id'], new.region_id)
