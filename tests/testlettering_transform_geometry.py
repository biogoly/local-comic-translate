"""Real Qt editable-projective geometry, interaction and page-local undo."""
import copy
import math
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text_item import TextBlockItem, TextBlockState
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text.lettering_transform import lettering_qtransform
from modules.utils.lettering_transform import normalize_lettering_transform
from modules.utils.textblock import TextBlock
from testmanual_text_box import _ManualTextMain


class LetteringTransformGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.viewer = self.main.image_viewer
        self.viewer._scene.setSceneRect(-1000, -1000, 4000, 4000)
        self.block = TextBlock(text_bbox=np.array([300, 300, 500, 400]),
                               is_manual=True, translation='Bonjour mon ami!')
        self.main.blk_list.append(self.block)
        self.item = self.main.text_ctrl.create_manual_text_item(self.block, self.block.translation)
        self.item._page_path = 'page.png'
        self.item.setTextWidth(220)
        self.item.setPos(400, 400)
        self.item.set_format_selection(8, 11)
        self.item.set_bold(True)
        self.item.clear_format_selection()
        self.main.text_ctrl._commit_pending_text_command()
        self.settings = {'scale_x': 1.3, 'scale_y': .8, 'skew_x': 17,
                         'perspective_x': .35, 'perspective_y': -.25}
        self.item.set_lettering_transform(self.settings)
        self.item.setRotation(27)
        self.item.setScale(1.15)
        self.item.selected = True
        self.stack = self.main.undo_stacks['page.png']

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def assertPointClose(self, a, b, places=6):
        self.assertAlmostEqual(a.x(), b.x(), places=places)
        self.assertAlmostEqual(a.y(), b.y(), places=places)

    def test_identity_is_canonical_and_bad_values_fail_closed(self):
        self.assertEqual(normalize_lettering_transform(None), {})
        self.assertEqual(normalize_lettering_transform({'scale_x':1, 'perspective_y':0}), {})
        for value in ([], {'bogus':1}, {'scale_x':0}, {'scale_y':5}, {'skew_x':46},
                      {'perspective_x':-.7}, {'scale_y':True}, {'scale_x':float('nan')},
                      {'perspective_y':float('inf')}, {'scale_x':'1'}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_lettering_transform(value)

    def test_qtransform_matches_the_normalized_equations_at_corners_and_interior(self):
        rect = QtCore.QRectF(7, 11, 220, 85)
        matrix = lettering_qtransform(rect, self.settings)
        for u, v in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5),(0,0),(.23,-.31)):
            point = QtCore.QPointF(rect.center().x()+rect.width()*u, rect.center().y()+rect.height()*v)
            denominator = 1+self.settings['perspective_x']*u+self.settings['perspective_y']*v
            expected = QtCore.QPointF(
                rect.center().x()+(rect.width()*1.3*u+rect.height()*math.tan(math.radians(17))*v)/denominator,
                rect.center().y()+rect.height()*.8*v/denominator)
            self.assertPointClose(matrix.map(point), expected)
            self.assertPointClose(matrix.inverted()[0].map(expected), point)

    def test_extreme_valid_settings_remain_finite_and_invertible(self):
        rect = QtCore.QRectF(0, 0, 300, 90)
        for px in (-.6,.6):
            for py in (-.6,.6):
                matrix = lettering_qtransform(rect, dict(scale_x=.25, scale_y=4, skew_x=45,
                                                         perspective_x=px, perspective_y=py))
                self.assertTrue(matrix.isInvertible())
                for point in (rect.topLeft(),rect.topRight(),rect.bottomLeft(),rect.bottomRight()):
                    mapped = matrix.map(point)
                    self.assertTrue(math.isfinite(mapped.x()) and math.isfinite(mapped.y()))
                    self.assertPointClose(matrix.inverted()[0].map(mapped), point)

    def test_transformed_hit_testing_and_text_edits_keep_native_document(self):
        local = self.item.boundingRect().center()
        scene = self.item.mapToScene(local)
        self.assertPointClose(self.item.mapFromScene(scene), local)
        self.assertTrue(self.item.contains(self.item.mapFromScene(scene)))
        before = self.item.lettering_transform.copy()
        self.item.enter_editing_mode()
        self.item.set_format_selection(8,11)
        self.item.set_italic(True)
        cursor = self.item.textCursor()
        cursor.setPosition(0)
        cursor.insertText('Salut! ')
        self.assertTrue(self.item.toPlainText().startswith('Salut!'))
        self.assertEqual(self.item.lettering_transform, before)
        self.assertEqual(self.item.textInteractionFlags(), QtCore.Qt.TextEditorInteraction)
        self.assertEqual(self.item.transform(), lettering_qtransform(self.item.boundingRect(), before))

    def test_reflow_recomputes_transform_and_reset_restores_identity(self):
        self.item.setTextWidth(80)
        self.assertEqual(self.item.transform(), lettering_qtransform(self.item.boundingRect(), self.settings))
        self.item.set_lettering_transform({})
        self.assertTrue(self.item.transform().isIdentity())
        self.assertEqual(self.item.toPlainText(), 'Bonjour mon ami!')

    def test_each_resize_handle_preserves_opposite_anchor_under_full_transform(self):
        original = TextItemProperties.from_text_item(self.item).to_dict()
        from app.controllers.lettering_style_format import apply_style_properties
        for handle in ('left','right','top','bottom','top_left','top_right','bottom_left','bottom_right'):
            with self.subTest(handle=handle):
                apply_style_properties(self.item, original)
                rect = self.item.boundingRect()
                anchor = self.item.mapToScene(self.item._resize_anchor(rect, handle))
                local = QtCore.QPointF(rect.left() if 'left' in handle else rect.right() if 'right' in handle else rect.center().x(),
                                      rect.top() if 'top' in handle else rect.bottom() if 'bottom' in handle else rect.center().y())
                start = self.item.mapToScene(local)
                self.item.resize_handle = handle
                self.item.init_resize(start)
                target = self.item.mapToScene(local + QtCore.QPointF(-8 if 'left' in handle else 8 if 'right' in handle else 0,
                                                                    -5 if 'top' in handle else 5 if 'bottom' in handle else 0))
                self.item.resize_item(target)
                after_anchor = self.item.mapToScene(self.item._resize_anchor(self.item.boundingRect(), handle))
                self.assertPointClose(anchor, after_anchor)
                self.assertNotEqual(self.item.boundingRect().size(), rect.size())
                self.assertEqual(self.item.lettering_transform, normalize_lettering_transform(self.settings))
                cursor = QtGui.QTextCursor(self.item.document())
                cursor.setPosition(9)
                self.assertEqual(cursor.charFormat().fontWeight(), QtGui.QFont.Bold)

    def test_rejected_out_of_bounds_resize_restores_exact_snapshot(self):
        self.viewer._scene.setSceneRect(self.item.sceneBoundingRect().adjusted(-1,-1,1,1))
        before = TextItemProperties.from_text_item(self.item).to_dict()
        self.item.resize_handle = 'right'
        start = self.item.mapToScene(self.item.boundingRect().topRight())
        self.item.init_resize(start)
        self.item.resize_item(start+QtCore.QPointF(800,0))
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), before)

    def test_rotation_keeps_visible_center_and_uses_unwarped_pointer(self):
        center = self.item.boundingRect().center()
        visible_center = self.item.mapToScene(center)
        radius = 100
        start = self.item.mapToScene(center+QtCore.QPointF(radius,0))
        target = self.item.mapToScene(center+QtCore.QPointF(radius*math.cos(math.radians(30)),radius*math.sin(math.radians(30))))
        rotation = self.item.rotation()
        self.item.init_rotation(start)
        self.item.rotate_item(target)
        self.assertAlmostEqual(self.item.rotation(), rotation+30, places=6)
        self.assertPointClose(self.item.mapToScene(center), visible_center)

    def test_resize_grab_distance_is_screen_based_at_multiple_zooms(self):
        manager = self.viewer.interaction_manager
        for zoom in (.5,1,3):
            with self.subTest(zoom=zoom):
                self.viewer.resetTransform()
                self.viewer.scale(zoom,zoom)
                quad = manager._transformed_quad(self.item)
                a,b = quad[1],quad[2]
                middle = (a+b)*.5
                delta = b-a
                norm = math.hypot(delta.x(),delta.y())
                point = middle+QtCore.QPointF(delta.y()/norm,-delta.x()/norm)*5
                scene = self.viewer.viewportTransform().inverted()[0].map(point)
                self.assertTrue(manager._in_resize_area(self.item,scene))
                self.assertEqual(manager.get_resize_handle(self.item,self.item.mapFromScene(scene)), 'right')
                far = middle+QtCore.QPointF(delta.y()/norm,-delta.x()/norm)*30
                scene_far = self.viewer.viewportTransform().inverted()[0].map(far)
                self.assertFalse(manager._in_resize_area(self.item,scene_far))
                self.assertTrue(manager._in_rotate_ring(self.item,scene_far))

    def test_geometry_undo_restores_document_without_changing_detection_bounds(self):
        before = TextBlockState.from_item(self.item)
        detection = self.block.xyxy.copy()
        self.item.resize_handle = 'bottom_right'
        self.item.init_resize(self.item.mapToScene(self.item.boundingRect().bottomRight()))
        self.item.resize_item(self.item.resize_start+QtCore.QPointF(12,8))
        after = TextBlockState.from_item(self.item)
        count = self.stack.count()
        self.main.rect_item_ctrl.rect_change_undo(before,after)
        self.assertEqual(self.stack.count(),count+1)
        np.testing.assert_array_equal(self.block.xyxy,detection)
        self.stack.undo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(),before.properties)
        self.stack.redo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(),after.properties)
        np.testing.assert_array_equal(self.block.xyxy,detection)

    def test_copy_preserves_transform_with_independent_parameters(self):
        duplicate = copy.copy(self.item)
        self.assertEqual(duplicate.lettering_transform,self.item.lettering_transform)
        self.assertEqual(duplicate.transform(),self.item.transform())
        duplicate.lettering_transform['scale_x']=2
        self.assertEqual(self.item.lettering_transform['scale_x'],1.3)

    def test_empty_text_and_retyping_keep_transform_settings(self):
        settings = copy.deepcopy(self.item.lettering_transform)
        self.item.setPlainText('')
        self.item.textCursor().insertText('Retyped text')
        self.assertEqual(self.item.lettering_transform,settings)
        self.assertEqual(self.item.transform(),lettering_qtransform(self.item.boundingRect(),settings))

    def test_format_command_undo_refreshes_geometry_after_document_reflow(self):
        from app.ui.commands.textformat import TextFormatCommand
        before = self.item.transform()
        before_html = self.item.toHtml()
        command = TextFormatCommand(self.viewer,self.item)
        self.item.set_font_size(28)
        command.finalize_new_state()
        self.stack.push(command)
        after = self.item.transform()
        self.assertNotEqual(before,after)
        self.stack.undo()
        self.assertEqual(self.item.toHtml(),before_html)
        self.assertEqual(self.item.transform(),before)
        self.stack.redo()
        self.assertEqual(self.item.transform(),after)

    def test_event_resize_records_one_full_snapshot_command(self):
        manager = self.viewer.interaction_manager
        quad = manager._transformed_quad(self.item)
        a,b = quad[1],quad[2]
        delta = b-a
        norm = math.hypot(delta.x(),delta.y())
        point = (a+b)*.5+QtCore.QPointF(delta.y()/norm,-delta.x()/norm)*5
        scene = self.viewer.viewportTransform().inverted()[0].map(point)
        event = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,point,point,
                                 QtCore.Qt.LeftButton,QtCore.Qt.LeftButton,QtCore.Qt.NoModifier)
        before = TextItemProperties.from_text_item(self.item).to_dict()
        count = self.stack.count()
        handler = self.viewer.event_handler
        self.assertTrue(handler._press_handle_resize(event,scene))
        self.assertTrue(handler._move_handle_item_interaction(scene+QtCore.QPointF(10,5)))
        self.assertTrue(handler._release_handle_item_interaction())
        self.assertEqual(self.stack.count(),count+1)
        self.assertIsNone(self.item.old_state)
        self.assertFalse(self.item.resizing)
        self.stack.undo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(),before)

    def test_drag_tracks_incremental_pointer_motion_without_accumulation(self):
        handler = self.viewer.event_handler
        start = self.item.mapToScene(self.item.boundingRect().center())
        point = self.viewer.viewportTransform().map(start)
        event = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,point,point,
                                 QtCore.Qt.LeftButton,QtCore.Qt.LeftButton,QtCore.Qt.NoModifier)
        before = self.item.pos()
        self.assertTrue(handler._press_handle_drag(event,start))
        handler._move_handle_drag(event,start+QtCore.QPointF(10,5))
        handler._move_handle_drag(event,start+QtCore.QPointF(20,10))
        self.assertPointClose(self.item.pos(),before+QtCore.QPointF(20,10))
        handler._release_handle_item_interaction()
        self.stack.undo()
        self.assertPointClose(self.item.pos(),before)


if __name__ == '__main__':
    unittest.main()
