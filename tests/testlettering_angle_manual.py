"""Manual canvas angle overrides survive editing, navigation and undo/redo."""
import unittest
import math
from types import SimpleNamespace

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets

from app.ui.canvas.rectangle import RectState
from app.ui.canvas.text_item import TextBlockState
from modules.utils.textblock import TextBlock
from pipeline.main_pipeline import ComicTranslatePipeline
from testmanual_text_box import _ManualTextMain


class ManualLetteringAngleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.viewer = self.main.image_viewer
        self.stack = self.main.undo_stacks['page.png']
        self.block = TextBlock(text_bbox=np.array([12, 20, 108, 70]), translation='Hello!')
        self.block.detected_lettering_angle = 17
        self.main.blk_list = [self.block]
        self.saved = self.block.deep_copy()
        self.main.image_states['page.png']['blk_list'] = [self.saved]

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def assert_override(self, angle):
        for block in (self.block, self.saved):
            self.assertEqual(getattr(block, 'manual_lettering_angle', None), angle)

    def test_rotating_detected_lettering_to_horizontal_is_a_single_undoable_edit(self):
        item = self.main.text_ctrl.create_manual_text_item(self.block, 'Hello!')
        item._page_path = 'page.png'
        item.setRotation(17)
        item.setPos(18, 30)  # Deliberately differs from the detected region.
        item.selected = True
        before = TextBlockState.from_item(item)
        count = self.stack.count()
        item.old_state = before
        item.rotating = True
        item.setRotation(0.2)
        self.viewer.event_handler._release_handle_item_interaction()
        self.assertEqual(self.stack.count(), count + 1)
        self.assertEqual(item.rotation(), 0)
        self.assert_override(0)
        self.stack.undo()
        self.assertEqual(item.rotation(), 17)
        self.assert_override(None)
        self.stack.redo()
        self.assertEqual(item.rotation(), 0)
        self.assert_override(0)
        np.testing.assert_array_equal(self.block.xyxy, [12, 20, 108, 70])
        self.assertEqual(self.block.angle, 0)
        self.assertEqual(self.block.detected_lettering_angle, 17)

    def test_moving_detected_lettering_does_not_turn_estimate_into_an_override(self):
        item = self.main.text_ctrl.create_manual_text_item(self.block, 'Hello!')
        item._page_path = 'page.png'
        item.setRotation(17)
        before = TextBlockState.from_item(item)
        item.setPos(45, 46)
        item.change_undo.emit(before, TextBlockState.from_item(item))
        self.assert_override(None)
        self.stack.undo()
        self.assertEqual(item.pos(), QtCore.QPointF(*before.properties['position']))
        self.stack.redo()
        self.assertEqual(item.pos(), QtCore.QPointF(45, 46))
        self.assert_override(None)
        np.testing.assert_array_equal(self.block.xyxy, [12, 20, 108, 70])

    def test_viewport_ring_drag_rotates_new_tilted_item_and_undo_restores_estimate(self):
        layout = QtWidgets.QVBoxLayout(self.main)
        layout.addWidget(self.viewer)
        self.main.resize(700, 650)
        self.viewer.display_image_array(np.full((500, 500, 3), 245, dtype=np.uint8), fit=False)
        self.block.xyxy[:] = [120, 150, 350, 235]
        self.main.show()
        self.app.processEvents()
        self.viewer.fitInView()
        item = self.main.text_ctrl.create_manual_text_item(self.block, self.block.translation)
        self.assertAlmostEqual(item.rotation(), 17)
        center = item.mapToScene(item.boundingRect().center())
        start = item.mapToScene(item.boundingRect().topRight() + QtCore.QPointF(20, -20))
        offset = start - center
        theta = math.radians(20)
        finish = center + QtCore.QPointF(offset.x()*math.cos(theta) - offset.y()*math.sin(theta),
                                         offset.x()*math.sin(theta) + offset.y()*math.cos(theta))
        count = self.stack.count()
        QtTest.QTest.mousePress(self.viewer.viewport(), QtCore.Qt.LeftButton, pos=self.viewer.mapFromScene(start))
        self.assertTrue(item.rotating)
        self.assertFalse(item.resizing)
        QtTest.QTest.mouseMove(self.viewer.viewport(), self.viewer.mapFromScene(finish))
        QtTest.QTest.mouseRelease(self.viewer.viewport(), QtCore.Qt.LeftButton, pos=self.viewer.mapFromScene(finish))
        self.app.processEvents()
        self.assertAlmostEqual(item.rotation(), 37, delta=1)
        self.assert_override(item.rotation())
        self.assertEqual(self.stack.count(), count + 1)
        self.stack.undo()
        self.assertAlmostEqual(item.rotation(), 17)
        self.assert_override(None)
        self.main.close()

    def test_selection_and_ocr_use_rotated_region_identity_after_reload(self):
        self.block.xyxy[:] = [12, 20, 128, 31]
        rect = self.viewer.add_rectangle(QtCore.QRectF(0, 0, 116, 11), QtCore.QPointF(12, 20), 70)
        rect.region_id = self.block.region_id
        rect.setTransformOriginPoint(rect.rect().center())
        other = TextBlock(text_bbox=np.array(rect.mapRectToScene(rect.rect()).getCoords()), text='Wrong region')
        self.main.blk_list.insert(0, other)
        state = self.viewer.save_state()
        self.assertEqual(state['rectangles'][0]['region_id'], self.block.region_id)
        self.viewer.clear_rectangles(page_switch=True)
        self.viewer.load_state(state)
        restored = self.viewer.rectangles[0]
        self.viewer.select_rectangle(restored)
        self.assertIs(self.main.curr_tblock, self.block)
        self.assertIs(ComicTranslatePipeline.get_selected_block(SimpleNamespace(main_page=self.main)), self.block)
        self.assertIs(self.main.rect_item_ctrl.find_corresponding_rect(self.block, 0.5), restored)
        # Legacy project with no rectangle ID can use the unrotated geometry.
        restored.region_id = None
        self.assertIs(self.main.rect_item_ctrl.find_corresponding_text_block(restored), self.block)
        self.assertEqual(restored.region_id, self.block.region_id)

    def test_ocr_box_rotation_writes_override_and_undo_restores_estimate(self):
        rect = self.viewer.add_rectangle(QtCore.QRectF(0, 0, 96, 50), QtCore.QPointF(12, 20))
        saved_rect = {'rect': (12, 20, 96, 50), 'rotation': 0, 'transform_origin': (0, 0)}
        self.main.image_states['page.png']['viewer_state'] = {'rectangles': [saved_rect]}
        before = RectState.from_item(rect)
        rect.setTransformOriginPoint(rect.boundingRect().center())
        rect.setRotation(-8)
        rect.signals.change_undo.emit(before, RectState.from_item(rect))
        for angle, override, action in ((-8, -8, lambda: None), (0, None, self.stack.undo),
                                         (-8, -8, self.stack.redo)):
            action()
            self.assertEqual(rect.rotation(), angle)
            self.assertEqual(saved_rect['rotation'], angle)
            self.assertEqual(self.block.angle, angle)
            self.assertEqual(self.saved.angle, angle)
            self.assert_override(override)
            self.assertEqual(self.block.detected_lettering_angle, 17)
            np.testing.assert_array_equal(self.block.xyxy, [12, 20, 108, 70])

    def test_undo_after_page_change_updates_only_the_owning_page(self):
        item = self.main.text_ctrl.create_manual_text_item(self.block, 'Hello!')
        item._page_path = 'page.png'
        item.setRotation(17)
        before = TextBlockState.from_item(item)
        item.setRotation(5)
        item.change_undo.emit(before, TextBlockState.from_item(item))
        self.assert_override(5)
        other = self.block.deep_copy()
        self.main.blk_list = [other]
        self.main.image_files.append('second.png')
        self.main.image_states['second.png'] = {'blk_list': [other]}
        self.main.curr_img_idx = 1
        self.stack.undo()
        self.assertIsNone(self.saved.manual_lettering_angle)
        self.assertEqual(other.manual_lettering_angle, 5)
        self.stack.redo()
        self.assertEqual(self.saved.manual_lettering_angle, 5)

    def test_webtoon_ocr_angle_is_scoped_to_page_when_fragments_share_an_id(self):
        other = self.block.deep_copy()
        other.xyxy = other.xyxy + np.array([0, 100, 0, 100])
        self.main.blk_list.append(other)
        self.main.image_files.append('second.png')
        self.main.image_states['second.png'] = {'blk_list': [other.deep_copy()]}
        first = self.viewer.add_rectangle(QtCore.QRectF(0, 0, 96, 50), QtCore.QPointF(12, 20))
        second = self.viewer.add_rectangle(QtCore.QRectF(0, 0, 96, 50), QtCore.QPointF(12, 120))
        first.region_id = second.region_id = self.block.region_id
        self.viewer.webtoon_manager.layout_manager.image_positions = [0, 100]
        self.viewer.webtoon_manager.layout_manager.image_heights = [100, 100]
        before = RectState.from_item(first)
        first.setRotation(11)
        first.signals.change_undo.emit(before, RectState.from_item(first))
        self.assert_override(11)
        self.assertEqual(first.rotation(), 11)
        self.assertEqual(second.rotation(), 0)
        self.assertIsNone(getattr(other, 'manual_lettering_angle', None))
        self.stack.undo()
        self.assert_override(None)
        self.assertEqual(first.rotation(), 0)


if __name__ == '__main__':
    unittest.main()
