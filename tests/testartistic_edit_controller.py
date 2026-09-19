import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QUndoStack

from app.controllers.artistic_edit import (
    ArtisticEditController,
    ArtisticEditSession,
    build_translation_prompt,
    preview_after_image,
)
from app.projects.patch_metadata import PATCH_KIND_FLUX2_EDIT
from app.ui.dayu_widgets import dayu_theme
from app.ui.artistic_edit_panel import ArtisticEditPanel
from app.ui.canvas.image_viewer import ImageViewer
from app.ui.dialogs.artistic_edit_preview import ArtisticEditPreviewDialog
from modules.artistic_edit.contracts import (
    ArtisticEditRequest,
    DEFAULT_MODEL_SPECS,
    DevicePolicy,
    ModelKey,
    SelectionMode,
)
from modules.artistic_edit.image_ops import compose_generated_patch, prepare_edit


class _ImageController:
    def __init__(self, page):
        self.page = page

    def get_composited_page_image(self, _path):
        return self.page.copy()


class _PreviewDecision:
    def __init__(self, decision):
        self.decision = decision

    def exec(self):
        return self.decision


class _FakeWorker(QtCore.QObject):
    ready = QtCore.Signal(object)
    loading = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    result = QtCore.Signal(object)
    cancelled = QtCore.Signal(str)
    error = QtCore.Signal(str, str, str)
    diagnostics = QtCore.Signal(str)
    worker_crashed = QtCore.Signal(str)

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.running = False
        self.request = None
        self.cuda_device_index = _kwargs.get("cuda_device_index")

    def is_running(self):
        return self.running

    def start(self, _policy):
        self.running = True
        self.ready.emit({"protocol_version": 1})

    def generate(self, request):
        self.request = request

    def cancel(self):
        return False

    def unload(self):
        return "unload"

    def close(self):
        self.running = False


class ArtisticEditControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.page_path = str(Path(self.temporary.name) / "page.png")
        self.page = np.zeros((64, 80, 3), dtype=np.uint8)
        self.viewer = ImageViewer(None)
        self.viewer.display_image_array(self.page, fit=False)
        self.addCleanup(self.viewer.deleteLater)
        self.panel = ArtisticEditPanel()
        self.addCleanup(self.panel.deleteLater)
        self.stack = QUndoStack()
        self.window_modified_updates = 0
        self.main = SimpleNamespace(
            artistic_edit_panel=self.panel,
            image_viewer=self.viewer,
            image_ctrl=_ImageController(self.page),
            image_files=[self.page_path],
            curr_img_idx=0,
            webtoon_mode=False,
            temp_dir=self.temporary.name,
            image_patches={},
            in_memory_patches={},
            undo_stacks={self.page_path: self.stack},
            _update_window_modified=self._mark_window_update,
        )
        self.controller = ArtisticEditController(self.main)
        self.addCleanup(self.controller.shutdown)

    def _mark_window_update(self):
        self.window_modified_updates += 1

    def _session(self):
        prepared = prepare_edit(
            self.page,
            (20, 20, 20, 16),
            minimum_context=8,
            maximum_model_size=(128, 128),
        )
        request_id = str(uuid4())
        request_dir = Path(self.temporary.name) / "artistic_edit" / request_id
        request_dir.mkdir(parents=True)
        input_path = request_dir / "input.png"
        output_path = request_dir / "output.png"
        input_path.touch()
        generated = np.full_like(prepared.model_input, 180)
        patch = compose_generated_patch(prepared, generated, feather_radius=0)
        request = ArtisticEditRequest(
            request_id=request_id,
            input_path=str(input_path.resolve()),
            output_path=str(output_path.resolve()),
            prompt="Replace the lettering",
            width=prepared.model_input.shape[1],
            height=prepared.model_input.shape[0],
            steps=4,
            guidance=1.0,
            seed=42,
            model=DEFAULT_MODEL_SPECS[ModelKey.KLEIN_4B],
            lora=None,
            device_policy=DevicePolicy.SINGLE_GPU_0,
        )
        return ArtisticEditSession(
            page_path=self.page_path,
            page_index=0,
            selection_mode=SelectionMode.SELECTED_BOX,
            prepared=prepared,
            request=request,
            request_dir=request_dir,
            lora_name=None,
            feather_radius=0,
            patch=patch,
        )

    def test_selected_translation_prompt_uses_exactly_the_supplied_pair(self):
        prompt = build_translation_prompt(
            'UNA REVOLUCIÓN "LLAMADA"',
            "A REVOLUTION CALLED",
            "lettering",
        )
        self.assertIn('"UNA REVOLUCIÓN \\"LLAMADA\\""', prompt)
        self.assertIn('exactly "A REVOLUTION CALLED"', prompt)
        self.assertIn("Leave every other part", prompt)

        bubble_prompt = build_translation_prompt("Hola", "Hello", "empty_bubble")
        self.assertIn("Do not put any text inside the bubble", bubble_prompt)
        with self.assertRaisesRegex(ValueError, "no translation"):
            build_translation_prompt("Hola", "", "lettering")

    def test_panel_accepts_full_nonnegative_64_bit_seed_range(self):
        self.panel.seed_edit.setText(str((2**63) - 1))
        self.assertEqual(self.panel.options().seed, (2**63) - 1)
        self.panel.seed_edit.setText(str(2**63))
        with self.assertRaisesRegex(ValueError, "at most"):
            self.panel.options()

    def test_edit_margin_is_only_used_for_selected_boxes(self):
        self.panel.edit_margin_spin.setValue(48)
        self.assertEqual(self.panel.options().edit_margin, 48)
        for mode in (SelectionMode.PAINTED_AREA, SelectionMode.WHOLE_PAGE):
            self.panel.area_combo.setCurrentIndex(self.panel.area_combo.findData(mode.value))
            self.assertFalse(self.panel.edit_margin_spin.isEnabled())
            self.assertEqual(self.panel.options().edit_margin, 0)
        self.panel.area_combo.setCurrentIndex(
            self.panel.area_combo.findData(SelectionMode.SELECTED_BOX.value)
        )
        self.assertTrue(self.panel.edit_margin_spin.isEnabled())
        self.assertEqual(self.panel.options().edit_margin, 48)

    def test_offload_gpu_index_is_enabled_only_for_offload_policies(self):
        self.assertEqual(
            self.panel.device_combo.currentData(),
            DevicePolicy.MODEL_CPU_OFFLOAD.value,
        )
        self.assertTrue(self.panel.offload_gpu_spin.isEnabled())
        single_gpu_index = self.panel.device_combo.findData(DevicePolicy.SINGLE_GPU_0.value)
        self.panel.device_combo.setCurrentIndex(single_gpu_index)
        self.assertFalse(self.panel.offload_gpu_spin.isEnabled())
        index = self.panel.device_combo.findData(DevicePolicy.MODEL_CPU_OFFLOAD.value)
        self.panel.device_combo.setCurrentIndex(index)
        self.assertTrue(self.panel.offload_gpu_spin.isEnabled())
        self.panel.offload_gpu_spin.setValue(1)
        self.assertEqual(self.panel.options().cuda_device_index, 1)

    def test_controller_passes_physical_gpu_index_to_offload_worker(self):
        self.controller.shutdown()
        self.controller.worker_factory = _FakeWorker
        worker = self.controller._new_worker(
            os.path.abspath(__file__),
            DevicePolicy.MODEL_CPU_OFFLOAD,
            1,
        )
        self.assertEqual(worker.cuda_device_index, 1)

    def test_disabled_generate_button_explains_missing_runtime(self):
        self.panel.set_prompt("Replace this lettering")
        reason = "Configure a FLUX Python executable."
        self.panel.set_context_available(
            False,
            reason,
            runtime_configuration_needed=True,
        )
        self.assertFalse(self.panel.generate_button.isEnabled())
        self.assertEqual(self.panel.generate_button.toolTip(), reason)
        self.assertFalse(self.panel.configure_runtime_button.isHidden())
        self.assertEqual(self.panel.status_label.text(), reason)

    def test_generate_action_is_before_model_options_in_panel(self):
        layout = self.panel.layout()
        self.assertLess(
            layout.indexOf(self.panel.generate_button),
            layout.indexOf(self.panel.advanced_button),
        )

    def test_artistic_edit_actions_have_scoped_button_outlines(self):
        buttons = (
            self.panel.use_translation_button,
            self.panel.generate_button,
            self.panel.configure_runtime_button,
            self.panel.refresh_lora_button,
            self.panel.open_lora_button,
            self.panel.advanced_button,
            self.panel.cancel_button,
            self.panel.unload_button,
            self.panel.revert_button,
        )
        self.assertTrue(all(button.property("artisticEditAction") for button in buttons))
        style = self.panel.styleSheet()
        self.assertIn(f"background-color: {dayu_theme.background_in_color}", style)
        self.assertIn(f"color: {dayu_theme.primary_text_color}", style)
        self.assertNotIn("palette(", style)

    def test_artistic_edit_action_colors_follow_dayu_theme(self):
        was_dark = QtGui.QColor(dayu_theme.background_color).lightness() < 128
        try:
            dayu_theme.set_theme("dark")
            self.panel.apply_theme()
            self.assertIn("background-color: #3a3a3a", self.panel.styleSheet())
            self.assertIn("color: #d9d9d9", self.panel.styleSheet())

            dayu_theme.set_theme("light")
            self.panel.apply_theme()
            self.assertIn("background-color: #ffffff", self.panel.styleSheet())
            self.assertIn("color: #595959", self.panel.styleSheet())
        finally:
            dayu_theme.set_theme("dark" if was_dark else "light")
            self.panel.apply_theme()

    def test_preview_image_changes_only_the_composed_patch_area(self):
        session = self._session()
        preview = preview_after_image(session.prepared, session.patch)
        self.assertEqual(preview.shape, session.prepared.source_crop.shape)
        changed = np.any(preview != session.prepared.source_crop, axis=2)
        self.assertEqual(int(changed.sum()), session.patch.bbox[2] * session.patch.bbox[3])

    def test_stale_pixels_and_wrong_page_both_block_apply(self):
        session = self._session()
        current, reason = self.controller._source_is_current(session)
        self.assertTrue(current)
        self.assertEqual(reason, "")

        self.main.image_ctrl.page[session.prepared.context_bbox[1], session.prepared.context_bbox[0]] = 1
        current, reason = self.controller._source_is_current(session)
        self.assertFalse(current)
        self.assertIn("source pixels changed", reason.lower())

        self.main.image_ctrl.page = self.page.copy()
        self.main.image_files = [str(Path(self.temporary.name) / "other.png")]
        current, reason = self.controller._source_is_current(session)
        self.assertFalse(current)
        self.assertIn("another page", reason.lower())

    def test_discard_preview_never_pushes_undo_or_marks_dirty(self):
        session = self._session()
        self.controller._session = session
        self.controller.preview_factory = lambda *args, **kwargs: _PreviewDecision(
            QtWidgets.QDialog.DialogCode.Rejected
        )
        result = SimpleNamespace(seed=42, elapsed_seconds=1.25)
        self.controller._show_preview(result)
        self.assertEqual(self.stack.count(), 0)
        self.assertTrue(self.stack.isClean())
        self.assertEqual(self.window_modified_updates, 0)
        self.assertIsNone(self.controller._session)

    def test_apply_creates_one_typed_undoable_patch_and_marks_dirty(self):
        session = self._session()
        self.controller._session = session
        self.controller._apply_session(session)
        self.assertEqual(self.stack.count(), 1)
        self.assertEqual(self.stack.undoText(), "Artistic edit")
        self.assertFalse(self.stack.isClean())
        self.assertEqual(self.window_modified_updates, 1)
        record = self.main.image_patches[self.page_path][0]
        self.assertEqual(record["kind"], PATCH_KIND_FLUX2_EDIT)
        self.assertEqual(record["metadata"]["seed"], 42)
        self.assertEqual(record["metadata"]["cuda_device_index"], 0)
        self.assertEqual(record["metadata"]["selection_mode"], "selected_box")

        self.stack.undo()
        self.assertEqual(self.main.image_patches[self.page_path], [])
        self.stack.redo()
        self.assertEqual(
            self.main.image_patches[self.page_path][0]["kind"], PATCH_KIND_FLUX2_EDIT
        )

    def test_generation_captures_composited_page_without_mutating_project(self):
        # Make the displayed canvas visibly different. Generation must still use
        # ImageStateController's overlay-free composited pixels.
        self.viewer.display_image_array(np.full_like(self.page, 255), fit=False)
        rect = self.viewer.add_rectangle(
            QtCore.QRectF(0, 0, 20, 16), QtCore.QPointF(20, 20)
        )
        self.viewer.select_rectangle(rect)
        self.panel.set_prompt("Repair only this lettering")
        self.panel.edit_margin_spin.setValue(8)

        self.controller.shutdown()
        self.controller.worker_factory = _FakeWorker
        self.controller.configured_worker_python = lambda: os.path.abspath(__file__)
        self.controller.update_availability()
        self.controller.generate_preview()

        worker = self.controller.worker
        self.assertIsNotNone(worker)
        self.assertIsNotNone(worker.request)
        with Image.open(worker.request.input_path) as image:
            model_input = np.asarray(image.convert("RGB"))
        self.assertFalse(np.any(model_input))
        prepared = self.controller._session.prepared
        patch = compose_generated_patch(prepared, np.full_like(prepared.model_input, 255))
        self.assertEqual(patch.bbox, (12, 12, 36, 32))
        self.assertEqual(rect.rect().getRect(), (0, 0, 20, 16))
        self.assertEqual(prepared.selection_bbox, (20, 20, 20, 16))
        self.assertEqual(self.stack.count(), 0)
        self.assertTrue(self.stack.isClean())


if __name__ == "__main__":
    unittest.main()
