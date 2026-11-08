"""Real Qt lettering and flattened-image checks for detected baseline angles."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PySide6 import QtCore, QtWidgets

from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.canvas.text.text_item_properties import TextItemProperties
from modules.rendering.render import orient_new_lettering
from modules.utils.textblock import TextBlock
from testmanual_text_box import _ManualTextMain
import testauto_bubble_batch as batch_fixtures


class LetteringAngleRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.addCleanup(self.main.deleteLater)
        self.main.loading = Mock()
        self.main.disable_hbutton_group = Mock()
        self.main.enable_hbutton_group = Mock()
        self.main.get_selected_page_paths = lambda: ['page.png']
        self.main.settings_page.get_min_font_size = lambda: 16
        self.main.settings_page.get_max_font_size = lambda: 16
        self.main.blk_rendered = SimpleNamespace(emit=self.main.text_ctrl.on_blk_rendered)
        self.block = TextBlock(text_bbox=[50, 50, 260, 90], text='aaaa', translation='HHHHHHHHHH')
        self.block.detected_lettering_angle = -18.0
        self.main.blk_list = [self.block]
        self.main.image_states['page.png']['blk_list'] = self.main.blk_list
        self.page = np.full((150, 300, 3), 255, dtype=np.uint8)
        self.main.image_viewer.display_image_array(self.page.copy(), fit=False)

    def test_render_and_export_follow_tilt_and_share_region_center(self):
        self.main.text_ctrl.render_text()
        item, = self.main.image_viewer.text_items
        self.assertEqual(item.rotation(), -18)
        scene_center = item.mapToScene(item.boundingRect().center())
        self.assertLess((scene_center - QtCore.QPointF(155, 70)).manhattanLength(), .01)
        np.testing.assert_array_equal(self.block.xyxy, [50, 50, 260, 90])
        self.assertEqual(self.block.angle, 0)
        self.assertIs(self.main.manual_workflow_ctrl._find_text_item_for_block(self.block, [item]), item)
        saved = TextItemProperties.from_text_item(item).to_dict()
        renderer = ImageSaveRenderer(self.page)
        renderer.add_state_to_image({'text_items_state': [saved]})
        pixels = renderer.render_to_image()
        y, x = np.nonzero(np.min(pixels, axis=2) < 100)
        self.assertGreater(x.size, 100)
        # The principal direction of the exported row of identical glyphs
        # detects sign errors, duplicate rotation and a flat raster export.
        _, axes = np.linalg.eigh(np.cov(x, y))
        vx, vy = axes[:, -1]
        angle = (np.degrees(np.arctan2(vy, vx)) + 90) % 180 - 90
        self.assertAlmostEqual(angle, -18, delta=1.5)
        self.assertLess(abs(x.mean() - 155), 5)
        self.assertLess(abs(y.mean() - 70), 5)
        self.main.text_ctrl.render_text()
        self.assertEqual(self.main.image_viewer.text_items, [item])

    def test_multipage_render_persists_positive_angle_over_correct_box(self):
        self.main.image_files.append('second.png')
        second = self.block.deep_copy()
        second.detected_lettering_angle = 13
        self.main.image_states['second.png'] = {'blk_list': [second], 'target_lang': 'English'}
        self.main.get_selected_page_paths = lambda: self.main.image_files
        self.main.manual_workflow_ctrl._prepare_multi_page_context = Mock(return_value={
            'current_file': None, 'current_page_unloaded': False})
        self.main.text_ctrl.render_text()
        saved, = self.main.image_states['second.png']['viewer_state']['text_items_state']
        self.assertEqual(saved['rotation'], 13)
        np.testing.assert_allclose(np.asarray(saved['position']) + saved['transform_origin'], [155, 70])

    def test_legacy_pivot_and_explicit_zero_are_preserved(self):
        self.block.angle = 24
        props = TextItemProperties(position=(20, 30), width=50, height=20, transform_origin=(3, 4))
        self.assertFalse(orient_new_lettering(props, self.block))
        self.assertEqual((props.rotation, props.position, props.transform_origin), (24, (20, 30), (3, 4)))
        self.block.manual_lettering_angle = 0
        self.assertFalse(orient_new_lettering(props, self.block))
        self.assertEqual(props.rotation, 0)

    def test_manual_typing_centers_tilted_lettering_and_preserves_document(self):
        item = self.main.text_ctrl.create_manual_text_item(self.block, 'Hand lettered text')
        self.assertEqual(item.rotation(), -18)
        self.assertEqual(item.toPlainText(), 'Hand lettered text')
        self.assertLess((item.mapToScene(item.boundingRect().center()) - QtCore.QPointF(155, 70)).manhattanLength(), .01)

    def test_batch_state_and_loaded_webtoon_share_the_same_visual_pivot(self):
        fixture = batch_fixtures.AutomaticBubbleBatchTests()
        main = batch_fixtures.main_page(False)
        region = batch_fixtures.block()
        region.detected_lettering_angle = 14.0
        bounds = region.xyxy.copy()
        fixture.run_regular(main, [region])
        np.testing.assert_array_equal(region.xyxy, bounds)
        saved, = main.image_states['page.png']['viewer_state']['text_items_state']
        self.assertEqual(saved['rotation'], 14)
        np.testing.assert_allclose(np.asarray(saved['position']) + saved['transform_origin'], [35, 35])
        handler = fixture.webtoon_handler(enabled=False)
        fixture.store_webtoon(handler, [region])
        webtoon_saved, = handler.main_page.image_states['page.png']['viewer_state']['text_items_state']
        emitted = handler.main_page.blk_rendered.emit.call_args.args[2]
        self.assertEqual(webtoon_saved['rotation'], 14)
        self.assertEqual(emitted._render_text_properties, webtoon_saved)
        self.assertEqual(emitted._render_page_origin, (0, 1000))
        np.testing.assert_allclose(np.asarray(webtoon_saved['position']) + webtoon_saved['transform_origin'], [35, 35])


if __name__ == '__main__':
    unittest.main()
