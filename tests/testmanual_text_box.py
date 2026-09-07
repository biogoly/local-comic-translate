import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from app.controllers.rect_item import RectItemController
from app.controllers.manual_workflow import ManualWorkflowController
from app.controllers.text import TextController
from app.ui.canvas.image_viewer import ImageViewer
from modules.utils.textblock import TextBlock


class _AlignmentToolGroup:
    def __init__(self):
        self._button_group = QtWidgets.QButtonGroup()
        for _ in range(3):
            self._button_group.addButton(QtWidgets.QToolButton())

    def get_dayu_checked(self):
        return 1

    def get_button_group(self):
        return self._button_group


class _ManualTextMain(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.image_viewer = ImageViewer(self)
        self.image_viewer.display_image_array(
            np.full((100, 140, 3), 255, dtype=np.uint8), fit=False
        )

        self.font_dropdown = QtWidgets.QComboBox()
        self.font_dropdown.addItem("Arial")
        self.font_size_dropdown = QtWidgets.QComboBox()
        self.font_size_dropdown.addItem("16")
        self.line_spacing_dropdown = QtWidgets.QComboBox()
        self.line_spacing_dropdown.addItem("1.2")
        self.outline_width_dropdown = QtWidgets.QComboBox()
        self.outline_width_dropdown.addItem("1.0")

        self.block_font_color_button = QtWidgets.QPushButton()
        self.block_font_color_button.setProperty("selected_color", "#000000")
        self.outline_font_color_button = QtWidgets.QPushButton()
        self.outline_font_color_button.setProperty("selected_color", "#ffffff")
        self.outline_checkbox = QtWidgets.QCheckBox()
        self.outline_checkbox.setChecked(False)
        self.bold_button = QtWidgets.QToolButton()
        self.italic_button = QtWidgets.QToolButton()
        self.underline_button = QtWidgets.QToolButton()
        self.bold_button.setCheckable(True)
        self.italic_button.setCheckable(True)
        self.underline_button.setCheckable(True)
        self.bold_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.italic_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.underline_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        min_font = QtWidgets.QSpinBox()
        min_font.setValue(4)
        max_font = QtWidgets.QSpinBox()
        max_font.setValue(72)
        uppercase = QtWidgets.QCheckBox()
        self.settings_page = SimpleNamespace(
            ui=SimpleNamespace(
                min_font_spinbox=min_font,
                max_font_spinbox=max_font,
                uppercase_checkbox=uppercase,
            )
        )

        self.alignment_tool_group = _AlignmentToolGroup()
        self.button_to_alignment = {
            0: QtCore.Qt.AlignmentFlag.AlignLeft,
            1: QtCore.Qt.AlignmentFlag.AlignCenter,
            2: QtCore.Qt.AlignmentFlag.AlignRight,
        }
        self.t_combo = QtWidgets.QComboBox()
        self.t_combo.addItem("English")
        self.lang_mapping = {"English": "English"}
        self.s_text_edit = QtWidgets.QTextEdit()
        self.t_text_edit = QtWidgets.QTextEdit()

        self.image_files = ["page.png"]
        self.curr_img_idx = 0
        self.blk_list = []
        self.image_states = {"page.png": {"blk_list": self.blk_list}}
        self.curr_tblock = None
        self.curr_tblock_item = None
        self.webtoon_mode = False
        self.dirty = False
        self.translation_requests = []
        self.finished_count = 0
        self.applied_patches = []

        self.undo_group = QtGui.QUndoGroup(self)
        stack = QtGui.QUndoStack(self)
        self.undo_group.addStack(stack)
        self.undo_group.setActiveStack(stack)
        self.undo_stacks = {self.image_files[0]: stack}

        self.ocr = lambda *_args, **_kwargs: None
        self.translate_image = (
            lambda *args, **kwargs: self.translation_requests.append((args, kwargs))
        )
        self.rect_item_ctrl = RectItemController(self)
        self.text_ctrl = TextController(self)
        self.pipeline = SimpleNamespace(
            get_selected_block=lambda: self.curr_tblock,
        )
        self.batch_report_ctrl = SimpleNamespace(
            resolve_translated_pages=lambda _paths: None,
        )
        self.image_ctrl = SimpleNamespace(
            on_inpaint_patches_processed=lambda patches, path: self.applied_patches.append(
                (path, patches)
            ),
        )
        self.manual_workflow_ctrl = ManualWorkflowController(self)
        self.bold_button.clicked.connect(self.text_ctrl.bold)
        self.italic_button.clicked.connect(self.text_ctrl.italic)
        self.underline_button.clicked.connect(self.text_ctrl.underline)
        alignment_buttons = self.alignment_tool_group.get_button_group().buttons()
        alignment_buttons[0].clicked.connect(self.text_ctrl.left_align)
        alignment_buttons[1].clicked.connect(self.text_ctrl.center_align)
        alignment_buttons[2].clicked.connect(self.text_ctrl.right_align)

        self.image_viewer.rectangle_selected.connect(
            self.rect_item_ctrl.handle_rectangle_selection
        )
        self.image_viewer.connect_rect_item.connect(
            self.rect_item_ctrl.connect_rect_item_signals
        )
        self.image_viewer.connect_text_item.connect(
            self.text_ctrl.connect_text_item_signals
        )
        self.t_text_edit.textChanged.connect(
            self.text_ctrl.update_text_block_from_edit
        )

    def mark_project_dirty(self):
        self.dirty = True

    def render_settings(self):
        return self.text_ctrl.render_settings()

    def default_error_handler(self, error_tuple):
        raise error_tuple[1]

    def run_threaded(
        self,
        callback,
        result_callback=None,
        error_callback=None,
        finished_callback=None,
        *args,
        **kwargs,
    ):
        try:
            result = callback(*args, **kwargs)
            if result_callback is not None:
                result_callback(result)
        except Exception as exc:
            if error_callback is not None:
                error_callback((type(exc), exc, ""))
            else:
                raise
        finally:
            if finished_callback is not None:
                finished_callback()

    def run_finish_only(self, finished_callback, error_callback=None):
        del error_callback
        finished_callback()

    def on_manual_finished(self):
        self.finished_count += 1

    def set_tool(self, _tool):
        pass

    def set_font(self, family):
        self.font_dropdown.setCurrentText(family)

    def push_command(self, command):
        self.undo_group.activeStack().push(command)


class ManualTextBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def test_typing_in_new_box_creates_visible_borderless_text(self):
        rect = self.main.image_viewer.add_rectangle(
            QtCore.QRectF(0, 0, 80, 45), QtCore.QPointF(20, 20)
        )
        self.main.rect_item_ctrl.handle_rectangle_creation(rect)

        self.assertIsNotNone(self.main.curr_tblock)
        self.assertEqual(len(self.main.image_viewer.text_items), 0)

        # Editor guides must not appear in a flattened page.
        before_text = self.main.image_viewer.get_image_array(paint_all=True)
        self.assertTrue(np.all(before_text == 255))

        self.main.t_text_edit.setPlainText("Hello bubble")
        self.app.processEvents()

        self.assertEqual(len(self.main.image_viewer.text_items), 1)
        text_item = self.main.image_viewer.text_items[0]
        self.assertEqual(text_item.toPlainText(), "Hello bubble")
        self.assertAlmostEqual(text_item.textWidth(), 80.0, delta=0.5)
        self.assertEqual(self.main.curr_tblock.translation, "Hello bubble")
        self.assertTrue(self.main.dirty)

        flattened = self.main.image_viewer.get_image_array(paint_all=True)
        self.assertTrue(np.any(flattened < 100))

        # Undoing the text layer and typing again should recreate a live item,
        # rather than leaving the editor bound to a deleted graphics object.
        self.main.undo_stacks["page.png"].undo()
        self.assertEqual(len(self.main.image_viewer.text_items), 0)
        self.main.t_text_edit.setPlainText("Replacement text")
        self.app.processEvents()
        self.assertEqual(len(self.main.image_viewer.text_items), 1)
        self.assertEqual(
            self.main.image_viewer.text_items[0].toPlainText(), "Replacement text"
        )

    def test_retyping_after_deleting_all_text_preserves_item_font(self):
        self.main.t_text_edit.setPlainText("Initial")
        self.app.processEvents()
        item = self.main.image_viewer.text_items[0] if self.main.image_viewer.text_items else None
        if item is None:
            rect = self.main.image_viewer.add_rectangle(
                QtCore.QRectF(0, 0, 80, 45), QtCore.QPointF(20, 20)
            )
            self.main.rect_item_ctrl.handle_rectangle_creation(rect)
            self.main.t_text_edit.setPlainText("Initial")
            self.app.processEvents()
            item = self.main.image_viewer.text_items[0]

        item.set_font("Courier New", 16)
        cursor = item.textCursor()
        cursor.select(QtGui.QTextCursor.SelectionType.Document)
        cursor.removeSelectedText()
        item.setTextCursor(cursor)
        self.app.processEvents()

        cursor = item.textCursor()
        cursor.insertText("Replacement")
        item.setTextCursor(cursor)
        self.app.processEvents()

        cursor.setPosition(0)
        cursor.setPosition(1, QtGui.QTextCursor.MoveMode.KeepAnchor)
        self.assertEqual(cursor.charFormat().font().family(), "Courier New")

    def test_bold_italic_and_underline_apply_only_to_selected_word(self):
        rect = self.main.image_viewer.add_rectangle(
            QtCore.QRectF(0, 0, 100, 45), QtCore.QPointF(20, 20)
        )
        self.main.rect_item_ctrl.handle_rectangle_creation(rect)
        self.main.t_text_edit.setPlainText("plain styled")
        self.app.processEvents()
        item = self.main.image_viewer.text_items[0]
        item.enter_editing_mode()

        cursor = item.textCursor()
        cursor.setPosition(6)
        cursor.setPosition(12, QtGui.QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.on_selection_changed()

        self.main.bold_button.click()
        self.main.italic_button.click()
        self.main.underline_button.click()

        plain = QtGui.QTextCursor(item.document())
        plain.setPosition(0)
        plain.setPosition(1, QtGui.QTextCursor.MoveMode.KeepAnchor)
        styled = QtGui.QTextCursor(item.document())
        styled.setPosition(6)
        styled.setPosition(7, QtGui.QTextCursor.MoveMode.KeepAnchor)

        self.assertFalse(plain.charFormat().font().bold())
        self.assertFalse(plain.charFormat().font().italic())
        self.assertFalse(plain.charFormat().font().underline())
        self.assertTrue(styled.charFormat().font().bold())
        self.assertTrue(styled.charFormat().font().italic())
        self.assertTrue(styled.charFormat().font().underline())

    def test_inline_formatting_uses_target_text_editor_selection(self):
        rect = self.main.image_viewer.add_rectangle(
            QtCore.QRectF(0, 0, 100, 45), QtCore.QPointF(20, 20)
        )
        self.main.rect_item_ctrl.handle_rectangle_creation(rect)
        self.main.t_text_edit.setPlainText("plain styled")
        self.app.processEvents()
        item = self.main.image_viewer.text_items[0]

        for widget in (
            self.main.t_text_edit,
            self.main.bold_button,
            self.main.italic_button,
            self.main.underline_button,
        ):
            widget.setParent(self.main)
            widget.show()
        self.main.show()
        self.main.t_text_edit.setFocus()
        editor_cursor = self.main.t_text_edit.textCursor()
        editor_cursor.setPosition(6)
        editor_cursor.setPosition(12, QtGui.QTextCursor.MoveMode.KeepAnchor)
        self.main.t_text_edit.setTextCursor(editor_cursor)
        self.app.processEvents()
        self.assertTrue(self.main.t_text_edit.hasFocus())

        QtTest.QTest.mouseClick(self.main.bold_button, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(self.main.italic_button, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(self.main.underline_button, QtCore.Qt.MouseButton.LeftButton)
        self.assertTrue(self.main.t_text_edit.hasFocus())

        plain = QtGui.QTextCursor(item.document())
        plain.setPosition(0)
        plain.setPosition(1, QtGui.QTextCursor.MoveMode.KeepAnchor)
        styled = QtGui.QTextCursor(item.document())
        styled.setPosition(6)
        styled.setPosition(7, QtGui.QTextCursor.MoveMode.KeepAnchor)

        self.assertFalse(plain.charFormat().font().bold())
        self.assertFalse(plain.charFormat().font().italic())
        self.assertFalse(plain.charFormat().font().underline())
        self.assertTrue(styled.charFormat().font().bold())
        self.assertTrue(styled.charFormat().font().italic())
        self.assertTrue(styled.charFormat().font().underline())

    def test_full_page_translation_creates_missing_text_layers(self):
        existing_blk = TextBlock(
            text_bbox=np.array([10, 10, 65, 45]),
            text_class="text_bubble",
            text="old one",
            translation="Updated one",
        )
        missing_blk = TextBlock(
            text_bbox=np.array([75, 45, 135, 90]),
            text="old two",
            translation="Newly visible two",
        )
        self.main.blk_list = [existing_blk, missing_blk]
        self.main.image_states["page.png"]["blk_list"] = self.main.blk_list

        existing_item = self.main.text_ctrl.create_manual_text_item(
            existing_blk, "Stale translation"
        )
        self.main.manual_workflow_ctrl.update_translated_text_items(False)
        self.app.processEvents()

        self.assertEqual(len(self.main.image_viewer.text_items), 2)
        self.assertIn("Updated one", existing_item.toPlainText().replace("\n", " "))
        created = next(
            item
            for item in self.main.image_viewer.text_items
            if item is not existing_item
        )
        self.assertIn("Newly visible two", created.toPlainText().replace("\n", " "))

    def test_full_page_translation_renders_free_text_without_reclassification(self):
        bubble = TextBlock(
            text_bbox=np.array([10, 10, 65, 45]),
            text_class="text_bubble",
            text="dialogue",
            translation="Visible translation",
        )
        artwork = TextBlock(
            text_bbox=np.array([75, 45, 135, 90]),
            text_class="text_free",
            text="sign",
            translation="Visible free text",
        )
        self.main.blk_list = [bubble, artwork]
        self.main.image_states["page.png"]["blk_list"] = self.main.blk_list

        self.main.manual_workflow_ctrl.update_translated_text_items(False)
        self.app.processEvents()

        self.assertEqual(len(self.main.image_viewer.text_items), 2)
        self.assertIn(
            "Visible translation",
            self.main.image_viewer.text_items[0].toPlainText().replace("\n", " "),
        )
        self.assertEqual(artwork.translation.replace("\n", " "), "Visible free text")
        self.assertEqual(artwork.text_class, "text_free")
        rendered_texts = [
            item.toPlainText().replace("\n", " ")
            for item in self.main.image_viewer.text_items
        ]
        self.assertIn("Visible free text", rendered_texts)

    def test_selected_translation_inpaints_before_creating_text_layer(self):
        blk = TextBlock(
            text_bbox=np.array([20, 20, 115, 70]),
            text="texto original",
            translation="Translated text",
        )
        self.main.blk_list = [blk]
        self.main.image_states["page.png"]["blk_list"] = self.main.blk_list
        self.main.curr_tblock = blk
        events = []
        patch = {
            "bbox": [20, 20, 95, 50],
            "image": np.full((50, 95, 3), 255, dtype=np.uint8),
        }

        self.main.manual_workflow_ctrl._inpaint_translated_block = (
            lambda _blk: {"page.png": [patch]}
        )
        original_apply = self.main.image_ctrl.on_inpaint_patches_processed
        self.main.image_ctrl.on_inpaint_patches_processed = (
            lambda patches, path: (
                events.append("inpaint"),
                original_apply(patches, path),
            )
        )
        original_render = self.main.text_ctrl.on_blk_rendered
        self.main.text_ctrl.on_blk_rendered = (
            lambda *args: (events.append("render"), original_render(*args))
        )

        self.main.manual_workflow_ctrl._finish_translation(True)
        self.app.processEvents()

        self.assertEqual(events[:2], ["inpaint", "render"])
        self.assertEqual(len(self.main.applied_patches), 1)
        self.assertEqual(len(self.main.image_viewer.text_items), 1)
        self.assertIn(
            "Translated text",
            self.main.image_viewer.text_items[0].toPlainText().replace("\n", " "),
        )

    def test_selected_translate_still_works_without_reclassifying_free_text(self):
        block = TextBlock(
            text_bbox=np.array([20, 20, 115, 70]),
            text_class="text_free",
            text="texto original",
            translation="Cached translation",
        )
        self.main.blk_list = [block]
        self.main.image_states["page.png"]["blk_list"] = self.main.blk_list
        rect = self.main.image_viewer.add_rectangle(
            QtCore.QRectF(0, 0, 95, 50), QtCore.QPointF(20, 20)
        )
        self.main.rect_item_ctrl.connect_rect_item_signals(rect)
        self.main.image_viewer.select_rectangle(rect)

        rect.signals.translate_block.emit()

        self.assertEqual(block.text_class, "text_free")
        self.assertIs(self.main.curr_tblock, block)
        self.assertEqual(self.main.translation_requests, [((True,), {})])


if __name__ == "__main__":
    unittest.main()
