"""Exercise the full viewport event path for curved resize handles."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from app.ui.canvas.text.text_item_properties import TextItemProperties
from modules.utils.textblock import TextBlock
from testmanual_text_box import _ManualTextMain


class CurvedLetteringViewportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_visible_viewport_resize_outside_signed_curves_keeps_exact_undo(self):
        for angle in (-60, 60):
            for corner in (False, True):
                with self.subTest(angle=angle, corner=corner):
                    main = _ManualTextMain()
                    viewer = main.image_viewer
                    layout = QtWidgets.QVBoxLayout(main)
                    layout.addWidget(viewer)
                    main.resize(700, 800)
                    viewer.display_image_array(np.full((700, 500, 3), 255, dtype=np.uint8), fit=False)
                    main.show()
                    self.app.processEvents()
                    viewer.fitInView()
                    block = TextBlock(text_bbox=np.array([90, 115, 410, 265]),
                                      is_manual=True, translation='Curve it your way!')
                    main.blk_list.append(block)
                    item = main.text_ctrl.create_manual_text_item(block, block.translation)
                    item._page_path = 'page.png'
                    item.set_font('Arial', 26)
                    item.setTextWidth(320)
                    item.set_lettering_transform({'arc_angle': angle})
                    item.selected = True
                    item.setSelected(True)
                    main.text_ctrl.on_text_item_selected(item)
                    item.enter_editing_mode()
                    cursor = item.textCursor()
                    cursor.setPosition(9)
                    cursor.setPosition(13, QtGui.QTextCursor.KeepAnchor)
                    item.setTextCursor(cursor)
                    main.text_ctrl._commit_pending_text_command()
                    self.app.processEvents()
                    before = TextItemProperties.from_text_item(item).to_dict()
                    stack = main.undo_stacks['page.png']
                    count = stack.count()
                    quad = viewer.interaction_manager._transformed_quad(item)
                    point = quad[2] + QtCore.QPointF(5, 5) if corner else (quad[1] + quad[2]) / 2 + QtCore.QPointF(5, 0)
                    start = point.toPoint()
                    finish = start + QtCore.QPoint(20, 15 if corner else 0)
                    QtTest.QTest.mousePress(viewer.viewport(), QtCore.Qt.LeftButton, pos=start)
                    self.assertTrue(item.resizing)
                    self.assertFalse(item.editing_mode)
                    QtTest.QTest.mouseMove(viewer.viewport(), finish)
                    QtTest.QTest.mouseRelease(viewer.viewport(), QtCore.Qt.LeftButton, pos=finish)
                    self.app.processEvents()
                    self.assertGreater(item.textWidth(), 320)
                    self.assertEqual(stack.count(), count + 1)
                    stack.undo()
                    self.assertEqual(TextItemProperties.from_text_item(item).to_dict(), before)
                    main.close()
                    main.deleteLater()
                    self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
