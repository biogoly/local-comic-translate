import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.controllers.image import ImageStateController, sort_image_paths
from app.ui.list_view import PageListView


class ImageSortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_name_sort_uses_natural_numeric_order(self):
        paths = [
            os.path.join("comic", "page10.png"),
            os.path.join("comic", "Page2.png"),
            os.path.join("comic", "page1.png"),
        ]

        ascending = sort_image_paths(paths, "name_asc")
        descending = sort_image_paths(paths, "name_desc")

        self.assertEqual(
            [os.path.basename(path) for path in ascending],
            ["page1.png", "Page2.png", "page10.png"],
        )
        self.assertEqual(descending, list(reversed(ascending)))

    def test_modified_sort_keeps_unavailable_files_last(self):
        paths = ["new.png", "missing.png", "old.png"]
        mtimes = {"new.png": 300.0, "old.png": 100.0}

        def getmtime(path):
            if path not in mtimes:
                raise OSError("not materialized")
            return mtimes[path]

        with patch("app.controllers.image.os.path.getmtime", side_effect=getmtime):
            oldest = sort_image_paths(paths, "modified_asc")
            newest = sort_image_paths(paths, "modified_desc")

        self.assertEqual(oldest, ["old.png", "new.png", "missing.png"])
        self.assertEqual(newest, ["new.png", "old.png", "missing.png"])

    def test_controller_uses_existing_safe_reorder_path(self):
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = SimpleNamespace(
            image_files=["page10.png", "page2.png", "page1.png"]
        )
        controller.handle_image_reorder = Mock()

        controller.sort_images("name_asc")

        controller.handle_image_reorder.assert_called_once_with(
            ["page1.png", "page2.png", "page10.png"]
        )

    def test_page_sort_menu_emits_the_selected_mode(self):
        page_list = PageListView()
        menu = QtWidgets.QMenu()
        received = []
        page_list.sort_requested.connect(received.append)

        page_list.populate_sort_menu(menu)

        self.assertEqual(len(menu.actions()), 4)
        menu.actions()[3].trigger()
        self.assertEqual(received, ["modified_desc"])
        page_list.deleteLater()

    def test_async_image_eviction_tolerates_an_already_removed_buffer(self):
        controller = ImageStateController.__new__(ImageStateController)
        main = SimpleNamespace(
            image_files=["current.png"],
            image_data={},
            image_history={"current.png": ["current.png"]},
            in_memory_history={"stale.png": []},
            current_history_index={"current.png": 0},
            loaded_images=["stale.png"],
            max_images_in_memory=1,
            in_memory_patches={"stale.png": []},
        )
        controller.main = main
        controller.display_image = Mock()
        pixels = object()

        controller.display_image_from_loaded(pixels, 0)

        self.assertIs(main.image_data["current.png"], pixels)
        self.assertEqual(main.loaded_images, ["current.png"])
        self.assertNotIn("stale.png", main.in_memory_patches)


if __name__ == "__main__":
    unittest.main()
