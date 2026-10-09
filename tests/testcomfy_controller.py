import ast
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6 import QtCore, QtWidgets

from app.ui.dialogs.artistic_edit_preview import ArtisticEditPreviewDialog
from modules.artistic_edit.backend_contracts import ComfyCompletion, ComfyJobEvent
from modules.artistic_edit.comfy_config import ComfyLoraSelection, ComfySettings, save_comfy_settings
from modules.artistic_edit.contracts import ArtisticEditResult
from tests import testartistic_edit_controller as harness


class FakeComfyJob(QtCore.QObject):
    event = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, spec, parent=None, *, runtime=None):
        super().__init__(parent)
        self.spec = spec
        self.cancel_requested = False
        self.running = False

    def start(self):
        self.running = True

    def cancel(self):
        self.cancel_requested = True

    def isRunning(self):
        return self.running

    def wait(self, *_args):
        self.running = False
        return True

    def finish(self, kind, payload):
        self.event.emit(ComfyJobEvent(self.spec.raster.request_id, self.spec.job_token, kind, payload))
        self.running = False
        self.finished.emit()


class ComfyControllerTests(unittest.TestCase):
    setUpClass = classmethod(harness.ArtisticEditControllerTests.setUpClass.__func__)
    _mark_window_update = harness.ArtisticEditControllerTests._mark_window_update

    def setUp(self):
        harness.ArtisticEditControllerTests.setUp(self)
        self.settings = QtCore.QSettings(str(Path(self.temporary.name) / "comfy.ini"), QtCore.QSettings.IniFormat)
        save_comfy_settings(self.settings, ComfySettings())
        self.addCleanup(patch.stopall)
        patch("app.controllers.artistic_edit.app_settings", return_value=self.settings).start()
        self.controller.comfy_job_factory = FakeComfyJob
        self.controller._comfy_runtime = SimpleNamespace(shutdown=lambda: None, is_running=lambda: False)
        self.controller._warning = Mock()
        self.controller.configured_worker_python = Mock(side_effect=AssertionError("Comfy does not use FLUX Python"))
        self.panel.backend_combo.setCurrentIndex(self.panel.backend_combo.findData("comfyui"))
        self.panel.area_combo.setCurrentIndex(2)
        self.panel.set_prompt("Replace title with HELLO")
        self.controller.update_availability()

    def pump(self):
        for _ in range(8):
            self.app.processEvents()

    def generate(self):
        self.controller.generate_preview(confirm_whole_page=False)
        self.assertIsNotNone(self.controller._comfy_job)
        return self.controller._comfy_job

    def completion(self, job, *, output_path=None):
        raster = job.spec.raster
        output_path = output_path or raster.output_path
        Image.new("RGB", (raster.width, raster.height), "blue").save(output_path)
        result = ArtisticEditResult(raster.request_id, output_path, raster.seed, 0.1, raster.width, raster.height)
        return ComfyCompletion(result, {"workflow_id": "klein-distilled-v1", "model_source": job.spec.profile.diffusion_model})

    def test_backend_index_and_controls_preserve_existing_backends(self):
        self.assertEqual(self.panel.backend_combo.itemData(1), "bfl")
        self.assertTrue(self.panel.generate_button.isEnabled())
        for widget in (self.panel.device_combo, self.panel.offload_gpu_spin,
                       self.panel.guidance_spin, self.panel.lora_combo, self.panel.model_combo):
            self.assertFalse(widget.isEnabled())
        self.assertTrue(self.panel.steps_spin.isEnabled())
        self.assertEqual(self.panel.options().lora_index, -1)
        self.controller.configured_worker_python.assert_not_called()

    def test_unconfigured_comfy_disables_generation_without_probing(self):
        self.settings.setValue("artistic_edit/comfyui/configured", False)
        self.controller.update_availability()
        self.assertFalse(self.panel.generate_button.isEnabled())
        self.assertIn("Configure ComfyUI", self.panel.generate_button.toolTip())
        self.assertIsNone(self.controller._comfy_job)

    def test_result_dispatches_preview_on_gui_thread_and_discard_is_clean(self):
        seen = []
        def preview(*_args, **_kwargs):
            seen.append(QtCore.QThread.currentThread())
            return harness._PreviewDecision(QtWidgets.QDialog.Rejected)
        self.controller.preview_factory = preview
        job = self.generate()
        directory = Path(job.spec.raster.input_path).parent
        job.finish("completed", self.completion(job))
        self.pump()
        self.assertEqual(seen, [self.app.thread()])
        self.assertIsNone(self.controller._session)
        self.assertIsNone(self.controller._comfy_job)
        self.assertFalse(directory.exists())
        self.assertTrue(self.stack.isClean())
        self.assertEqual(self.stack.count(), 0)

    def test_real_background_job_delivers_preview_on_gui_thread(self):
        from modules.artistic_edit.comfy_job import ComfyEditJob
        seen = []
        generator_threads = []
        def generate(spec, _cancelled, **_kwargs):
            generator_threads.append(QtCore.QThread.currentThread())
            return self.completion(SimpleNamespace(spec=spec))
        self.controller.comfy_job_factory = lambda spec, parent, **_kwargs: ComfyEditJob(
            spec, parent, generator=generate
        )
        def preview(*_args, **_kwargs):
            seen.append(QtCore.QThread.currentThread())
            return harness._PreviewDecision(QtWidgets.QDialog.Rejected)
        self.controller.preview_factory = preview
        self.generate()
        timer = QtCore.QElapsedTimer()
        timer.start()
        while self.panel.is_busy() and timer.elapsed() < 3000:
            self.app.processEvents()
        self.assertFalse(self.panel.is_busy())
        self.assertEqual(seen, [self.app.thread()])
        self.assertEqual(len(generator_threads), 1)
        self.assertIsNot(generator_threads[0], self.app.thread())
        self.assertIsNone(self.controller._comfy_job)

    def test_late_local_and_wrong_job_events_do_not_change_comfy_session(self):
        job = self.generate()
        session = self.controller._session
        self.panel.set_status("current job")
        self.controller._on_worker_error("old", "late error", "")
        self.controller._on_worker_crashed("late crash")
        self.controller._on_worker_loading("late loading")
        self.controller._on_worker_progress(99, 100)
        self.controller._on_comfy_event(ComfyJobEvent(str(uuid4()), job.spec.job_token, "failed", {"message": "other"}))
        self.controller._on_comfy_event(ComfyJobEvent(job.spec.raster.request_id, str(uuid4()), "status", "other"))
        self.assertIs(self.controller._session, session)
        self.assertEqual(self.panel.status_label.text(), "current job")
        self.controller._warning.assert_not_called()

    def test_cancel_rejects_completed_race_but_releases_job_and_files(self):
        job = self.generate()
        completion = self.completion(job)
        self.controller.preview_factory = Mock(side_effect=AssertionError("cancelled result cannot preview"))
        self.controller.cancel()
        self.assertTrue(job.cancel_requested)
        self.assertFalse(self.controller._session.accept_result)
        self.assertTrue(self.panel.is_busy())
        job.finish("completed", completion)
        self.pump()
        self.assertIsNone(self.controller._session)
        self.assertIsNone(self.controller._comfy_job)
        self.assertFalse(self.panel.is_busy())
        self.assertFalse(Path(completion.result.output_path).exists())
        self.assertEqual(self.stack.count(), 0)

    def test_cancel_notice_keeps_external_computation_uncertainty(self):
        job = self.generate()
        self.controller.cancel()
        job.finish("cancelled", {"server_may_continue": True})
        self.pump()
        self.assertIn("may still be computing", self.panel.status_label.text())

    def test_project_replacement_with_same_pixels_rejects_old_result(self):
        job = self.generate()
        session = self.controller._session
        self.controller.invalidate_project()
        self.assertFalse(self.controller._source_is_current(session)[0])
        self.controller.preview_factory = Mock(side_effect=AssertionError("old project result"))
        job.finish("completed", self.completion(job))
        self.pump()
        self.assertEqual(self.stack.count(), 0)
        self.assertFalse(self.panel.is_busy())

    def test_result_path_must_match_owned_request_output(self):
        job = self.generate()
        foreign = str(Path(self.temporary.name) / "foreign.png")
        job.finish("completed", self.completion(job, output_path=foreign))
        self.pump()
        self.assertEqual(self.stack.count(), 0)
        self.controller._warning.assert_called_once()
        self.assertIsNone(self.controller._session)

    def test_same_seed_regeneration_uses_frozen_configuration_and_prompt(self):
        self.controller.preview_factory = lambda *_args, **_kwargs: harness._PreviewDecision(
            ArtisticEditPreviewDialog.REGENERATE_SAME_SEED
        )
        job = self.generate()
        original = job.spec
        self.panel.set_prompt("Changed UI prompt")
        changed = ComfySettings(profile=replace(original.profile, diffusion_model="different.safetensors"))
        save_comfy_settings(self.settings, changed)
        job.finish("completed", self.completion(job))
        self.pump()
        regenerated = self.controller._comfy_job.spec
        self.assertEqual(regenerated.raster.prompt, original.raster.prompt)
        self.assertEqual(regenerated.raster.seed, original.raster.seed)
        self.assertEqual(regenerated.profile, original.profile)
        self.assertNotEqual(regenerated.job_token, original.job_token)
        self.assertNotEqual(regenerated.raster.request_id, original.raster.request_id)
        self.assertFalse(Path(original.raster.input_path).exists())

    def test_apply_is_one_typed_patch_with_comfy_provenance(self):
        self.controller.preview_factory = lambda *_args, **_kwargs: harness._PreviewDecision(QtWidgets.QDialog.Accepted)
        job = self.generate()
        job.finish("completed", self.completion(job))
        self.pump()
        self.assertEqual(self.stack.count(), 1)
        record = self.main.image_patches[self.page_path][0]
        self.assertEqual(record["kind"], "flux2_edit")
        self.assertEqual(record["metadata"]["backend"], "comfyui")
        self.assertEqual(record["metadata"]["steps"], 4)
        self.assertEqual(record["metadata"]["workflow_id"], "klein-distilled-v1")
        self.assertNotIn("https://api.bfl.ai", repr(record["metadata"]))
        self.stack.undo()
        self.assertEqual(self.main.image_patches[self.page_path], [])
        self.stack.redo()
        self.assertEqual(len(self.main.image_patches[self.page_path]), 1)

    def test_page_mutation_inside_preview_blocks_apply(self):
        def preview(*_args, **_kwargs):
            self.main.image_ctrl.page[0, 0] = 222
            return harness._PreviewDecision(QtWidgets.QDialog.Accepted)
        self.controller.preview_factory = preview
        job = self.generate()
        job.finish("completed", self.completion(job))
        self.pump()
        self.assertEqual(self.stack.count(), 0)
        self.controller._warning.assert_called_once()

    def test_settings_requires_new_lora_verification_after_server_change(self):
        from app.ui.dialogs.comfyui_settings import ComfyUISettingsDialog
        config = ComfySettings(lora=ComfyLoraSelection("ink.safetensors", 0.8, "klein-9b", "a" * 64))
        dialog = ComfyUISettingsDialog(config, self.controller._comfy_runtime)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog._read_config().lora.sha256, "a" * 64)
        dialog.url.setText("http://127.0.0.1:8190")
        with self.assertRaisesRegex(ValueError, "matching local LoRA"):
            dialog._read_config()

    def test_connection_probe_does_not_require_model_selection(self):
        from app.ui.dialogs.comfyui_settings import ComfyUISettingsDialog
        dialog = ComfyUISettingsDialog(ComfySettings(), self.controller._comfy_runtime)
        self.addCleanup(dialog.deleteLater)
        dialog.diffusion.setCurrentText("")
        dialog.encoder.setCurrentText("")
        info = SimpleNamespace(version="test", catalogs={"diffusion_models": ("available.safetensors",)})
        with patch("app.ui.dialogs.comfyui_settings.probe_comfy", return_value=info) as probe:
            dialog._probe()
            self.assertIsNotNone(dialog._job)
            self.assertTrue(dialog._job.wait(3000))
            self.pump()
        probe.assert_called_once()
        self.assertIsNone(dialog._job)
        self.assertIn("Connected to ComfyUI test", dialog.status.text())
        self.assertGreaterEqual(dialog.diffusion.findText("available.safetensors"), 0)
        self.assertEqual(dialog.diffusion.currentText(), "")

    def test_close_waits_for_cancelled_job_before_retrying_window_close(self):
        self.main.close = Mock()
        job = self.generate()
        directory = Path(job.spec.raster.input_path).parent
        self.assertFalse(self.controller.prepare_close())
        self.assertTrue(job.cancel_requested)
        self.assertTrue(directory.exists())
        self.main.close.assert_not_called()
        job.finish("cancelled", {"server_may_continue": False})
        self.pump()
        self.assertFalse(directory.exists())
        self.main.close.assert_called_once()
        self.assertTrue(self.controller.prepare_close())

    def test_close_refuses_to_stop_runtime_with_foreign_work(self):
        self.main.close = Mock()
        self.controller._comfy_runtime = SimpleNamespace(
            is_running=lambda: True,
            shutdown=Mock(side_effect=RuntimeError("Foreign job is running")),
        )
        self.assertFalse(self.controller.prepare_close())
        self.assertTrue(self.controller._comfy_stop_job.wait(3000))
        self.pump()
        self.main.close.assert_not_called()
        self.controller._warning.assert_called_once_with("Foreign job is running")
        self.assertFalse(self.controller._close_pending)
        self.assertFalse(self.panel.is_busy())
        self.controller._comfy_runtime = None

    def test_connected_disconnect_does_not_stop_a_retained_owned_runtime(self):
        stop = Mock()
        disconnect = Mock()
        self.controller._comfy_runtime = SimpleNamespace(
            is_running=lambda: True, shutdown=stop, disconnect=disconnect,
        )
        self.controller.unload()
        stop.assert_not_called()
        disconnect.assert_called_once()
        self.assertIsNone(self.controller._comfy_stop_job)
        self.controller._comfy_runtime = None

    def test_cancelling_before_local_worker_ready_releases_pending_session(self):
        session = harness.ArtisticEditControllerTests._session(self)
        worker = Mock()
        self.controller.worker = worker
        self.controller._session = session
        self.controller._awaiting_worker_ready = True
        self.panel.set_busy(True)
        self.controller.cancel()
        self.assertFalse(self.controller._awaiting_worker_ready)
        self.assertIsNone(self.controller._session)
        self.assertFalse(self.panel.is_busy())
        worker.cancel.assert_not_called()
        self.controller._on_worker_ready({"loaded_model": None})
        worker.generate.assert_not_called()
        self.controller.worker = None

    def test_retiring_local_worker_error_cannot_discard_new_request(self):
        class RetiringWorker(harness._FakeWorker):
            def close(worker):
                worker.worker_crashed.emit("late retirement failure")
        session = harness.ArtisticEditControllerTests._session(self)
        old = RetiringWorker()
        old.worker_crashed.connect(self.controller._on_worker_crashed)
        self.controller.worker = old
        self.controller.worker_factory = harness._FakeWorker
        self.controller._session = session
        self.controller._worker_python = "previous-python"
        self.controller._dispatch_request("new-python", session.request.device_policy, 0)
        self.assertIs(self.controller._session, session)
        self.assertIs(self.controller.worker.request, session.request)
        self.controller._warning.assert_not_called()

    def test_backend_switch_revokes_comfy_result_without_losing_cleanup(self):
        self.controller.connect_signals()
        self.controller.preview_factory = Mock(side_effect=AssertionError("retired backend preview"))
        job = self.generate()
        self.panel.backend_combo.setCurrentIndex(1)
        self.assertTrue(job.cancel_requested)
        self.assertFalse(self.controller._session.accept_result)
        job.finish("completed", self.completion(job))
        self.pump()
        self.assertIsNone(self.controller._session)
        self.assertFalse(self.panel.is_busy())
        self.assertEqual(self.stack.count(), 0)


class WindowCloseOrderingTests(unittest.TestCase):
    """Exercise the actual close handler without loading optional OCR runtimes."""

    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "controller.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ComicTranslate")
        method = next(node for node in main_class.body if isinstance(node, ast.FunctionDef) and node.name == "closeEvent")
        cls.close_code = compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec")

    def setUp(self):
        self.choice = "Cancel"
        self.prompt_count = 0
        test = self
        class MessageBox:
            Question = 1
            ButtonRole = SimpleNamespace(AcceptRole=1, DestructiveRole=2, RejectRole=3)
            def __init__(self, _parent):
                test.prompt_count += 1
            def setIcon(self, _value): pass
            def setWindowTitle(self, _value): pass
            def setText(self, _value): pass
            def setDefaultButton(self, _value): pass
            def addButton(self, text, _role): return text
            def exec(self): pass
            def clickedButton(self): return test.choice
        scope = {"QtWidgets": SimpleNamespace(QMessageBox=MessageBox)}
        exec(self.close_code, scope)
        self.close_event = scope["closeEvent"]
        self.artistic = SimpleNamespace(_project_epoch=0, prepare_close=Mock(return_value=False), cancel_close=Mock())
        self.window = SimpleNamespace(
            artistic_edit_ctrl=self.artistic,
            text_ctrl=SimpleNamespace(_commit_pending_text_command=Mock()),
            _dirty_revision=1, has_unsaved_changes=lambda: True, tr=lambda text: text,
            project_ctrl=SimpleNamespace(thread_save_project=Mock(), shutdown_autosave=Mock()),
            _finish_close_after_save=Mock(),
        )
        self.event = Mock()

    def test_cancelling_unsaved_prompt_does_not_close_any_resource(self):
        self.close_event(self.window, self.event)
        self.artistic.prepare_close.assert_not_called()
        self.window.project_ctrl.shutdown_autosave.assert_not_called()
        self.event.ignore.assert_called_once()

    def test_async_close_reuses_approval_only_for_unchanged_document(self):
        self.choice = "Don't Save"
        self.close_event(self.window, self.event)
        self.close_event(self.window, self.event)
        self.assertEqual(self.prompt_count, 1)
        self.assertEqual(self.artistic.prepare_close.call_count, 2)
        self.window._dirty_revision += 1
        self.close_event(self.window, self.event)
        self.assertEqual(self.prompt_count, 2)
        self.artistic._project_epoch += 1
        self.close_event(self.window, self.event)
        self.assertEqual(self.prompt_count, 3)

    def test_save_request_finishes_before_resource_close(self):
        self.choice = "Save"
        self.close_event(self.window, self.event)
        self.window.project_ctrl.thread_save_project.assert_called_once_with(
            post_save_callback=self.window._finish_close_after_save
        )
        self.artistic.prepare_close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
