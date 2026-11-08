"""Curved text's real Qt caret, selection, formatting and canvas handle paths."""
import copy
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text.text_item_properties import TextItemProperties
from modules.utils.textblock import TextBlock
from testmanual_text_box import _ManualTextMain


class CurvedLetteringInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.viewer = self.main.image_viewer
        self.viewer._scene.setSceneRect(-1000, -1000, 4000, 4000)
        self.block = TextBlock(text_bbox=np.array([300, 300, 700, 500]),
                               is_manual=True, translation='plain styled word')
        self.main.blk_list.append(self.block)
        self.item = self.main.text_ctrl.create_manual_text_item(self.block, self.block.translation)
        self.item._page_path = 'page.png'
        self.item.set_font('Arial', 26)
        self.item.setTextWidth(400)
        self.item.setPos(400, 400)
        self.item.set_lettering_transform({'arc_angle': 95})
        self.item.selected = True
        self.item.setSelected(True)
        self.main.text_ctrl.on_text_item_selected(self.item)
        self.main.text_ctrl._commit_pending_text_command()
        self.stack = self.main.undo_stacks['page.png']

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def point_at(self, position):
        line = self.item.layout.caret_line(position)
        return (line.p1() + line.p2()) * .5

    def event(self, kind, point, *, modifiers=QtCore.Qt.NoModifier):
        event = QtWidgets.QGraphicsSceneMouseEvent(kind)
        event.setPos(point)
        event.setScenePos(self.item.mapToScene(point))
        event.setLastPos(point)
        event.setLastScenePos(self.item.mapToScene(point))
        event.setButton(QtCore.Qt.LeftButton)
        event.setButtons(QtCore.Qt.LeftButton)
        event.setModifiers(modifiers)
        return event

    def click_at(self, position, *, shift=False):
        point = self.point_at(position)
        modifiers = QtCore.Qt.ShiftModifier if shift else QtCore.Qt.NoModifier
        self.item.mousePressEvent(self.event(QtCore.QEvent.GraphicsSceneMousePress, point,
                                            modifiers=modifiers))
        self.item.mouseReleaseEvent(self.event(QtCore.QEvent.GraphicsSceneMouseRelease, point,
                                              modifiers=modifiers))

    def format_at(self, position):
        cursor = QtGui.QTextCursor(self.item.document())
        cursor.setPosition(position)
        cursor.setPosition(position + 1, QtGui.QTextCursor.KeepAnchor)
        return cursor.charFormat()

    def test_curved_click_and_shift_click_format_only_the_selected_word(self):
        self.item.enter_editing_mode()
        self.click_at(6)
        self.assertEqual(self.item.textCursor().position(), 6)
        self.click_at(12, shift=True)
        self.assertEqual(self.item.textCursor().selectedText(), 'styled')
        self.main.bold_button.click()
        self.main.italic_button.click()
        self.main.underline_button.click()
        plain, styled = self.format_at(1), self.format_at(8)
        self.assertFalse(plain.font().bold())
        self.assertFalse(plain.fontItalic())
        self.assertFalse(plain.fontUnderline())
        self.assertTrue(styled.font().bold())
        self.assertTrue(styled.fontItalic())
        self.assertTrue(styled.fontUnderline())
        self.assertEqual(self.item.lettering_transform['arc_angle'], 95)
        self.stack.undo()
        self.assertFalse(self.format_at(8).fontUnderline())
        self.stack.redo()
        self.assertTrue(self.format_at(8).fontUnderline())

    def test_curved_drag_selection_uses_visual_points_in_both_directions(self):
        self.item.enter_editing_mode()
        for start, end in ((6, 12), (12, 6)):
            with self.subTest(start=start):
                self.item.mousePressEvent(self.event(QtCore.QEvent.GraphicsSceneMousePress, self.point_at(start)))
                self.item.mouseMoveEvent(self.event(QtCore.QEvent.GraphicsSceneMouseMove, self.point_at(end)))
                self.item.mouseReleaseEvent(self.event(QtCore.QEvent.GraphicsSceneMouseRelease, self.point_at(end)))
                self.assertEqual(self.item.textCursor().selectedText(), 'styled')
                self.assertEqual(self.item.textCursor().anchor(), start)

    def test_format_undo_redo_refreshes_toolbar_for_the_selected_word(self):
        self.item.enter_editing_mode()
        self.click_at(6)
        self.click_at(12, shift=True)
        for button in (self.main.bold_button, self.main.italic_button, self.main.underline_button):
            with self.subTest(button=button):
                self.assertFalse(button.isChecked())
                button.click()
                self.assertTrue(button.isChecked())
                self.stack.undo()
                self.assertEqual(self.item.textCursor().selectedText(), 'styled')
                self.assertFalse(button.isChecked())
                self.stack.redo()
                self.assertTrue(button.isChecked())
                self.stack.undo()
                self.assertFalse(button.isChecked())

    def test_double_click_selects_curved_word_through_qt(self):
        point = self.point_at(8)
        self.item.mouseDoubleClickEvent(self.event(QtCore.QEvent.GraphicsSceneMouseDoubleClick, point))
        self.assertTrue(self.item.editing_mode)
        self.assertEqual(self.item.textCursor().selectedText(), 'styled')

    def test_keyboard_arrows_do_not_split_surrogate_pairs_or_combining_marks(self):
        self.item.setPlainText('A😀e\u0301B')
        self.item.setTextWidth(300)
        self.item.enter_editing_mode()
        self.click_at(1)
        key = lambda code: QtGui.QKeyEvent(QtCore.QEvent.KeyPress, code, QtCore.Qt.NoModifier)
        self.item.keyPressEvent(key(QtCore.Qt.Key_Right))
        self.assertEqual(self.item.textCursor().position(), 3)
        self.item.keyPressEvent(key(QtCore.Qt.Key_Right))
        self.assertEqual(self.item.textCursor().position(), 5)
        self.item.keyPressEvent(key(QtCore.Qt.Key_Left))
        self.assertEqual(self.item.textCursor().position(), 3)

    def test_japanese_ime_commit_and_candidate_anchor_preserve_curve(self):
        self.item.enter_editing_mode()
        self.click_at(6)
        caret = self.item.layout.caret_line(6)
        candidate_rect = self.item.inputMethodQuery(QtCore.Qt.ImCursorRectangle)
        self.assertTrue(candidate_rect.contains((caret.p1() + caret.p2()) * .5))
        preedit = QtGui.QInputMethodEvent('日本', [])
        self.item.inputMethodEvent(preedit)
        self.assertEqual(self.item.toPlainText(), 'plain styled word')
        commit = QtGui.QInputMethodEvent()
        commit.setCommitString('日本語')
        self.item.inputMethodEvent(commit)
        self.assertEqual(self.item.toPlainText(), 'plain 日本語styled word')
        self.assertEqual(self.item.textCursor().position(), 9)
        self.assertEqual(self.item.lettering_transform['arc_angle'], 95)

    def test_arrow_navigation_across_curved_lines_preserves_line_order(self):
        self.item.setPlainText('first line\nsecond line')
        self.item.setTextWidth(400)
        self.item.enter_editing_mode()
        self.click_at(2)
        self.item.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Down, QtCore.Qt.NoModifier))
        self.assertGreaterEqual(self.item.textCursor().position(), 11)
        self.assertLessEqual(self.item.textCursor().position(), self.item.document().characterCount() - 1)
        self.item.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Up, QtCore.Qt.NoModifier))
        self.assertLess(self.item.textCursor().position(), 11)

    def test_formatted_word_still_allows_resize_then_exact_undo(self):
        self.item.enter_editing_mode()
        self.click_at(6)
        self.click_at(12, shift=True)
        self.main.bold_button.click()
        self.main.text_ctrl._commit_pending_text_command()
        before = TextItemProperties.from_text_item(self.item).to_dict()
        cleanup = self.block.xyxy.copy()
        bounds = self.item.boundingRect()
        start = self.item.mapToScene(bounds.topLeft() - QtCore.QPointF(4, 4))
        handler = self.viewer.event_handler
        count = self.stack.count()
        self.assertTrue(handler._press_handle_resize(SimpleNamespace(accept=lambda: None), start))
        self.assertFalse(self.item.editing_mode)
        self.assertTrue(handler._move_handle_item_interaction(start - QtCore.QPointF(12, 12)))
        self.assertTrue(handler._release_handle_item_interaction())
        self.assertEqual(self.stack.count(), count + 1)
        self.assertTrue(self.format_at(8).font().bold())
        self.assertFalse(self.format_at(1).font().bold())
        self.stack.undo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), before)
        np.testing.assert_array_equal(self.block.xyxy, cleanup)

    def test_resize_grab_distance_stays_screen_sized_when_curved_and_rotated(self):
        self.item.setRotation(23)
        manager = self.viewer.interaction_manager
        for zoom in (.5, 1, 3):
            with self.subTest(zoom=zoom):
                self.viewer.resetTransform()
                self.viewer.scale(zoom, zoom)
                quad = manager._transformed_quad(self.item)
                a, b = quad[1:3]
                delta = b - a
                length = (delta.x() ** 2 + delta.y() ** 2) ** .5
                point = (a + b) * .5 + QtCore.QPointF(delta.y(), -delta.x()) * (5 / length)
                scene_point = self.viewer.viewportTransform().inverted()[0].map(point)
                self.assertEqual(manager.get_resize_handle(self.item, self.item.mapFromScene(scene_point)), 'right')

    def test_copy_has_independent_layout_and_stable_wrap_width(self):
        original_width = self.item.textWidth()
        original_html = self.item.toHtml()
        clone = copy.copy(self.item)
        self.assertIsNot(clone.layout, self.item.layout)
        self.assertEqual(clone.textWidth(), original_width)
        clone.set_font_size(34)
        clone.set_lettering_transform({'arc_angle': -75})
        self.assertEqual(self.item.toHtml(), original_html)
        self.assertEqual(self.item.textWidth(), original_width)
        self.assertEqual(self.item.lettering_transform['arc_angle'], 95)


if __name__ == '__main__':
    unittest.main()
