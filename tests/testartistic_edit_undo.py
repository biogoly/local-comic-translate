import os
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from app.projects.patch_metadata import (
    MAX_METADATA_STRING_LENGTH,
    PATCH_KIND_FLUX2_EDIT,
    PATCH_KIND_INPAINT,
    normalize_patch_kind,
    patch_matches_kind,
)
from app.ui.canvas.image_viewer import ImageViewer
from app.ui.commands.inpaint import PatchClearCommand, PatchInsertCommand


class ArtisticEditUndoTests(unittest.TestCase):
    """Typed-patch undo behavior required by the artistic edit milestone."""

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
    def _patch(color, bbox, patch_id=None):
        _x, _y, width, height = bbox
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :] = color
        patch = {"bbox": bbox, "image": image}
        if patch_id is not None:
            patch["patch_id"] = patch_id
        return patch

    def _pixel(self, x, y):
        return tuple(int(value) for value in self.viewer.get_image_array()[y, x])

    def _record_ids(self):
        return [patch.get("patch_id") for patch in self.controller.image_patches[self.file_path]]

    def _push_inpaint(self, color, bbox, patch_id=None):
        self.stack.push(
            PatchInsertCommand(
                self.controller,
                [self._patch(color, bbox, patch_id=patch_id)],
                self.file_path,
            )
        )

    def _push_artistic(self, color, bbox, patch_id=None, metadata=None):
        self.stack.push(
            PatchInsertCommand(
                self.controller,
                [self._patch(color, bbox, patch_id=patch_id)],
                self.file_path,
                kind=PATCH_KIND_FLUX2_EDIT,
                metadata=metadata,
            )
        )

    def test_default_insert_stays_untyped_inpaint(self):
        self._push_inpaint((255, 0, 0), (10, 10, 20, 20))
        self.assertEqual(self.stack.undoText(), "Inpaint")
        record = self.controller.image_patches[self.file_path][0]
        self.assertNotIn("kind", record)
        self.assertNotIn("metadata", record)

    def test_artistic_insert_is_one_undo_entry_with_kind_and_metadata(self):
        metadata = {"prompt": "Replace the title", "seed": 7, "steps": 4}
        self._push_artistic(
            (0, 255, 0),
            (15, 15, 20, 20),
            patch_id="flux-1",
            metadata=metadata,
        )

        self.assertEqual(self.stack.undoText(), "Artistic edit")
        self.assertEqual(self.stack.count(), 1)
        record = self.controller.image_patches[self.file_path][0]
        self.assertEqual(record["kind"], PATCH_KIND_FLUX2_EDIT)
        self.assertEqual(record["metadata"], metadata)
        self.assertEqual(record["patch_id"], "flux-1")
        self.assertEqual(self._pixel(16, 16), (0, 255, 0))

        self.stack.undo()
        self.assertEqual(self._pixel(16, 16), (0, 0, 0))
        self.assertEqual(self.controller.image_patches[self.file_path], [])

        self.stack.redo()
        self.assertEqual(self._pixel(16, 16), (0, 255, 0))
        self.assertEqual(self._record_ids(), ["flux-1"])
        self.assertEqual(
            self.controller.image_patches[self.file_path][0]["metadata"], metadata
        )

    def test_insert_metadata_is_sanitized(self):
        self._push_artistic(
            (0, 255, 0),
            (15, 15, 10, 10),
            metadata={
                "nested": {"a": 1},
                "nan": float("nan"),
                "infinity": float("inf"),
                "huge": "x" * (MAX_METADATA_STRING_LENGTH + 3000),
                "flag": False,
                "words": ["a", "b", 3],
                "none": None,
            },
        )
        metadata = self.controller.image_patches[self.file_path][0]["metadata"]
        self.assertNotIn("nested", metadata)
        self.assertNotIn("nan", metadata)
        self.assertNotIn("infinity", metadata)
        self.assertEqual(len(metadata["huge"]), MAX_METADATA_STRING_LENGTH)
        self.assertIs(metadata["flag"], False)
        self.assertEqual(metadata["words"], ["a", "b", 3])
        self.assertIsNone(metadata["none"])

    def test_revert_inpainting_leaves_artistic_edit_intact(self):
        self._push_inpaint((255, 0, 0), (10, 10, 20, 20), patch_id="inp-1")
        self._push_artistic((0, 255, 0), (25, 25, 10, 10), patch_id="flux-1")

        self.stack.push(
            PatchClearCommand(self.controller, self.file_path, kind=PATCH_KIND_INPAINT)
        )
        self.assertEqual(self.stack.undoText(), "Revert inpainting")

        # Inpaint pixels are gone, artistic pixels untouched.
        self.assertEqual(self._pixel(11, 11), (0, 0, 0))
        self.assertEqual(self._pixel(26, 26), (0, 255, 0))
        self.assertEqual(self._record_ids(), ["flux-1"])

        self.stack.undo()
        self.assertEqual(self._pixel(11, 11), (255, 0, 0))
        self.assertEqual(self._pixel(26, 26), (0, 255, 0))
        # Original compositing order is restored, not append-at-end.
        self.assertEqual(self._record_ids(), ["inp-1", "flux-1"])

    def test_filtered_revert_restores_order_around_untouched_kinds(self):
        self._push_inpaint((255, 0, 0), (5, 5, 10, 10), patch_id="a")
        self._push_artistic((0, 255, 0), (20, 20, 10, 10), patch_id="b")
        self._push_inpaint((0, 0, 255), (30, 30, 10, 10), patch_id="c")

        self.stack.push(
            PatchClearCommand(self.controller, self.file_path, kind=PATCH_KIND_INPAINT)
        )
        self.assertEqual(self._record_ids(), ["b"])

        self.stack.undo()
        self.assertEqual(self._record_ids(), ["a", "b", "c"])

    def test_revert_artistic_edits_leaves_inpaint_intact(self):
        self._push_inpaint((255, 0, 0), (10, 10, 20, 20), patch_id="inp-1")
        self._push_artistic((0, 255, 0), (25, 25, 10, 10), patch_id="flux-1")

        self.stack.push(
            PatchClearCommand(self.controller, self.file_path, kind=PATCH_KIND_FLUX2_EDIT)
        )
        self.assertEqual(self.stack.undoText(), "Revert artistic edits")

        self.assertEqual(self._pixel(11, 11), (255, 0, 0))
        # The surviving inpaint patch is underneath the removed artistic edit
        # in their overlapping area, so it becomes visible again.
        self.assertEqual(self._pixel(26, 26), (255, 0, 0))
        self.assertEqual(self._record_ids(), ["inp-1"])

        self.stack.undo()
        self.assertEqual(self._record_ids(), ["inp-1", "flux-1"])
        self.assertEqual(self._pixel(26, 26), (0, 255, 0))

    def test_clear_without_kind_still_clears_everything(self):
        self._push_inpaint((255, 0, 0), (10, 10, 20, 20), patch_id="inp-1")
        self._push_artistic((0, 255, 0), (25, 25, 10, 10), patch_id="flux-1")

        self.stack.push(PatchClearCommand(self.controller, self.file_path))
        self.assertEqual(self.controller.image_patches[self.file_path], [])
        self.assertEqual(self._pixel(11, 11), (0, 0, 0))
        self.assertEqual(self._pixel(26, 26), (0, 0, 0))

        self.stack.undo()
        self.assertEqual(self._record_ids(), ["inp-1", "flux-1"])

    def test_visually_identical_patches_remain_distinct_by_patch_id(self):
        bbox = (10, 10, 10, 10)
        self._push_artistic((0, 255, 0), bbox, patch_id="dup-1")
        self._push_artistic((0, 255, 0), bbox, patch_id="dup-2")
        self.assertEqual(self._record_ids(), ["dup-1", "dup-2"])

        self.stack.undo()
        self.assertEqual(self._record_ids(), ["dup-1"])
        # The first patch is still drawn even though the undone one was
        # pixel-identical.
        self.assertEqual(self._pixel(12, 12), (0, 255, 0))

        self.stack.redo()
        self.assertEqual(self._record_ids(), ["dup-1", "dup-2"])

    def test_legacy_untyped_patches_match_inpaint_kind(self):
        self._push_inpaint((255, 0, 0), (10, 10, 10, 10))
        record = self.controller.image_patches[self.file_path][0]
        self.assertNotIn("kind", record)
        self.assertTrue(patch_matches_kind(record, PATCH_KIND_INPAINT))
        self.assertFalse(patch_matches_kind(record, PATCH_KIND_FLUX2_EDIT))


class PatchKindHelperTests(unittest.TestCase):
    def test_normalize_patch_kind_defaults(self):
        self.assertEqual(normalize_patch_kind(None), PATCH_KIND_INPAINT)
        self.assertEqual(normalize_patch_kind(""), PATCH_KIND_INPAINT)
        self.assertEqual(normalize_patch_kind(5), PATCH_KIND_INPAINT)
        self.assertEqual(normalize_patch_kind("flux2_edit"), "flux2_edit")

    def test_unknown_kinds_match_nothing_but_themselves(self):
        record = {"kind": "zzz_something"}
        self.assertFalse(patch_matches_kind(record, PATCH_KIND_INPAINT))
        self.assertFalse(patch_matches_kind(record, PATCH_KIND_FLUX2_EDIT))
        self.assertTrue(patch_matches_kind(record, "zzz_something"))


if __name__ == "__main__":
    unittest.main()
