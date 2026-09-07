import os
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from app.ui.canvas.image_viewer import ImageViewer
from app.ui.commands.inpaint import PatchClearCommand, PatchInsertCommand


class InpaintUndoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "page.png")
        self.viewer = ImageViewer(None)
        self.viewer.display_image_array(np.zeros((50, 50, 3), dtype=np.uint8), fit=False)
        self.controller = SimpleNamespace(
            image_viewer=self.viewer,
            image_files=[self.file_path],
            curr_img_idx=0,
            temp_dir=self.temp_dir.name,
            image_patches={},
            in_memory_patches={},
        )
        self.stack = QUndoStack()

    def tearDown(self):
        self.viewer.deleteLater()
        self.temp_dir.cleanup()

    @staticmethod
    def _patch(color, bbox):
        _x, _y, width, height = bbox
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :] = color
        return {"bbox": bbox, "image": image}

    def _pixel(self, x, y):
        return tuple(int(value) for value in self.viewer.get_image_array()[y, x])

    def test_newest_patch_is_composited_last_and_each_pass_is_undoable(self):
        self.stack.push(
            PatchInsertCommand(
                self.controller,
                [self._patch((255, 0, 0), (10, 10, 20, 20))],
                self.file_path,
            )
        )
        self.stack.push(
            PatchInsertCommand(
                self.controller,
                [self._patch((0, 0, 255), (15, 15, 20, 20))],
                self.file_path,
            )
        )

        self.assertEqual(self._pixel(16, 16), (0, 0, 255))
        self.stack.undo()
        self.assertEqual(self._pixel(16, 16), (255, 0, 0))
        self.stack.undo()
        self.assertEqual(self._pixel(16, 16), (0, 0, 0))
        self.stack.redo()
        self.stack.redo()
        self.assertEqual(self._pixel(16, 16), (0, 0, 255))

    def test_revert_all_is_itself_undoable(self):
        self.stack.push(
            PatchInsertCommand(
                self.controller,
                [self._patch((255, 0, 0), (10, 10, 20, 20))],
                self.file_path,
            )
        )
        self.stack.push(
            PatchInsertCommand(
                self.controller,
                [self._patch((0, 0, 255), (15, 15, 20, 20))],
                self.file_path,
            )
        )

        self.stack.push(PatchClearCommand(self.controller, self.file_path))
        self.assertEqual(self._pixel(16, 16), (0, 0, 0))
        self.assertEqual(self.controller.image_patches[self.file_path], [])

        self.stack.undo()
        self.assertEqual(self._pixel(16, 16), (0, 0, 255))
        self.assertEqual(len(self.controller.image_patches[self.file_path]), 2)


if __name__ == "__main__":
    unittest.main()
