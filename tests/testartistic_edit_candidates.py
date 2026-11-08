"""Two completed previews remain independent of the next asynchronous request."""
import os
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtTest, QtWidgets
from shiboken6 import isValid

from app.ui.dialogs.artistic_edit_preview import ArtisticEditPreviewDialog
from modules.artistic_edit.backend_contracts import ComfyCompletion
from modules.artistic_edit.comfy_config import ComfySettings, save_comfy_settings
from modules.artistic_edit.contracts import ArtisticEditResult
from tests import testartistic_edit_controller as fixtures
from tests import testcomfy_controller as comfy_fixtures


class _CandidateDecision(QtCore.QObject):
    candidate_selected = QtCore.Signal(int)

    def __init__(self, decision, selected_index):
        super().__init__()
        self.decision = decision
        self.chosen_index = selected_index
        self.selected_candidate_index = 0
        self.labels = []
        self.updates = []

    def configure_candidates(self, labels, selected_index):
        self.labels = list(labels)
        self.selected_candidate_index = selected_index

    def set_candidate(self, *args, **kwargs):
        self.updates.append((args, kwargs))

    def exec(self):
        self.selected_candidate_index = self.chosen_index
        self.candidate_selected.emit(self.chosen_index)
        return self.decision


class _FakeBflJob(QtCore.QObject):
    result = QtCore.Signal(object)
    error = QtCore.Signal(str)
    cancelled = QtCore.Signal(str)

    def __init__(self, request, _api_key, parent=None):
        super().__init__(parent)
        self.request = request
        self.running = False

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def cancel(self):
        self.running = False

    def wait(self, *_args):
        self.running = False
        return True


class ArtisticEditCandidateTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.ArtisticEditControllerTests.setUpClass.__func__)
    _mark_window_update = fixtures.ArtisticEditControllerTests._mark_window_update
    _session = fixtures.ArtisticEditControllerTests._session
    setUp = comfy_fixtures.ComfyControllerTests.setUp
    generate = comfy_fixtures.ComfyControllerTests.generate
    pump = comfy_fixtures.ComfyControllerTests.pump

    def complete(self, job, color):
        raster = job.spec.raster
        Image.new("RGB", (raster.width, raster.height), color).save(raster.output_path)
        result = ArtisticEditResult(raster.request_id, raster.output_path, raster.seed,
                                    .1, raster.width, raster.height)
        completion = ComfyCompletion(result, {
            "workflow_id": "klein-distilled-v1",
            "model_source": job.spec.profile.diffusion_model,
        })
        # Leave the preview open in controller state without entering a modal UI.
        with patch.object(self.controller, "_show_preview"):
            job.finish("completed", completion)
            self.pump()
        return self.controller._session, result

    def two_candidates(self):
        first = self.complete(self.generate(), "red")
        dialog = _CandidateDecision(ArtisticEditPreviewDialog.GENERATE_ALTERNATIVE, 0)
        self.controller.preview_factory = lambda *_args, **_kwargs: dialog
        with patch("app.controllers.artistic_edit.secrets.randbelow", return_value=202):
            self.controller._show_preview(first[1])
        self.assertEqual(len(dialog.labels), 1)
        self.assertEqual(len(self.controller._candidates), 1)
        second = self.complete(self.controller._comfy_job, "blue")
        self.assertEqual(len(self.controller._candidates), 2)
        self.assertIs(self.controller._candidates[0][0], first[0])
        self.assertIs(self.controller._candidates[1][0], second[0])
        self.assertFalse(self.controller._generate_candidate(second[0], 404))
        self.assertIsNone(self.controller._comfy_job)
        return first, second

    def choose(self, index, decision=QtWidgets.QDialog.Accepted):
        dialog = _CandidateDecision(decision, index)
        self.controller.preview_factory = lambda *_args, **_kwargs: dialog
        self.controller._show_preview(self.controller._candidates[-1][1])
        self.assertEqual(len(dialog.labels), 2)
        return dialog

    def assert_applied_candidate(self, index):
        candidates = self.two_candidates()
        chosen = candidates[index][0]
        expected_pixels = chosen.patch.image.copy()
        expected_seed = chosen.request.seed
        directories = [session.request_dir for session, _result in candidates]
        self.choose(index)
        self.assertEqual(self.stack.count(), 1)
        self.assertEqual(self.window_modified_updates, 1)
        record, = self.main.image_patches[self.page_path]
        with Image.open(record["png_path"]) as image:
            np.testing.assert_array_equal(np.asarray(image.convert("RGB")), expected_pixels)
        self.assertEqual(record["metadata"]["seed"], expected_seed)
        self.assertEqual(record["metadata"]["prompt"], chosen.request.prompt)
        self.assertEqual(record["metadata"]["backend"], "comfyui")
        self.assertEqual(record["metadata"]["model_source"], chosen.comfy_spec.profile.diffusion_model)
        self.assertEqual(record["metadata"]["source_crop_sha256"], chosen.prepared.source_hash)
        self.assertFalse(self.controller._candidates)
        self.assertIsNone(self.controller._session)
        self.assertTrue(all(not directory.exists() for directory in directories))
        self.stack.undo()
        self.assertEqual(self.main.image_patches[self.page_path], [])
        self.stack.redo()
        self.assertEqual(len(self.main.image_patches[self.page_path]), 1)

    def test_selecting_a_applies_a_pixels_and_provenance_after_b_finishes(self):
        self.assert_applied_candidate(0)

    def test_selecting_b_applies_b_pixels_and_provenance(self):
        self.assert_applied_candidate(1)

    def test_apply_without_an_undo_stack_retires_both_candidates_and_their_files(self):
        self.main.undo_stacks.pop(self.page_path)
        candidates = self.two_candidates()
        for session, _ in candidates:
            self.assertIsNone(session.undo_stack)
            self.assertTrue(self.controller._source_is_current(session)[0])
        self.choose(0)
        self.controller._warning.assert_called_once()
        self.assertIn("no undo history", self.controller._warning.call_args.args[0].lower())
        self.assertIsNone(self.controller._session)
        self.assertFalse(self.controller._candidates)
        for session, _ in candidates:
            self.assertFalse(session.accept_result)
            self.assertFalse(session.request_dir.exists())
        self.assertEqual(self.main.image_patches, {})
        self.assertEqual(self.stack.count(), 0)
        self.assertEqual(self.window_modified_updates, 0)

    def test_alternative_freezes_input_prompt_and_comfy_settings_with_new_ids(self):
        first, _ = self.complete(self.generate(), "red")
        original = first.comfy_spec
        original_input = Path(first.request.input_path).read_bytes()
        self.panel.set_prompt("Changed prompt must not be captured")
        self.panel.steps_spin.setValue(17)
        changed = ComfySettings(profile=replace(original.profile, diffusion_model="different.safetensors"))
        save_comfy_settings(self.settings, changed)
        self.controller._generate_candidate(first, 202)
        pending = self.controller._session
        self.assertIsNot(pending, first)
        self.assertIsNone(pending.patch)
        self.assertEqual(len(self.controller._candidates), 1)
        self.assertTrue(first.request_dir.is_dir())
        self.assertEqual(Path(pending.request.input_path).read_bytes(), original_input)
        self.assertEqual(pending.prepared.source_hash, first.prepared.source_hash)
        np.testing.assert_array_equal(pending.prepared.model_input, first.prepared.model_input)
        self.assertEqual(pending.request.prompt, original.raster.prompt)
        self.assertEqual(pending.request.steps, original.raster.steps)
        self.assertEqual(pending.request.seed, 202)
        self.assertEqual(pending.comfy_spec.profile, original.profile)
        self.assertEqual(pending.comfy_spec.connection, original.connection)
        self.assertEqual(pending.comfy_spec.lora, original.lora)
        self.assertEqual(pending.comfy_spec.cfg, original.cfg)
        for attribute in ("job_token", "client_id"):
            self.assertNotEqual(getattr(pending.comfy_spec, attribute), getattr(original, attribute))
        for attribute in ("request_id", "input_path", "output_path"):
            self.assertNotEqual(getattr(pending.request, attribute), getattr(first.request, attribute))
        # A late completion from A must not finish or replace the pending B job.
        old_result = ArtisticEditResult(first.request.request_id, first.request.output_path,
                                       first.request.seed, .1, first.request.width, first.request.height)
        self.controller._on_result(old_result)
        self.assertIs(self.controller._session, pending)
        self.assertTrue(self.panel.is_busy())
        self.assertEqual(self.stack.count(), 0)

    def test_local_and_hosted_alternatives_preserve_captured_request_and_input(self):
        self.controller.worker_factory = fixtures._FakeWorker
        self.controller.configured_worker_python = lambda: os.path.abspath(fixtures.__file__)
        self.main.settings_page = SimpleNamespace(get_credentials=lambda _: {"api_key": "fixture-key"})
        for backend in ("local", "bfl"):
            with self.subTest(backend=backend):
                self.panel.backend_combo.setCurrentIndex(self.panel.backend_combo.findData(backend))
                source = self._session()
                source.backend = backend
                source.worker_python = os.path.abspath(__file__)
                source.expected_lettering = "Frozen lettering"
                source.lettering_language = "English"
                self.controller._session = source
                result = ArtisticEditResult(source.request.request_id, source.request.output_path,
                                            source.request.seed, .1, source.request.width, source.request.height)
                self.controller._remember_candidate(source, result)
                self.panel.set_prompt("Changed panel prompt")
                self.panel.steps_spin.setValue(17)
                with (
                    patch("app.controllers.artistic_edit.BflEditJob", _FakeBflJob),
                    patch("app.controllers.artistic_edit.QtWidgets.QMessageBox.question",
                          return_value=QtWidgets.QMessageBox.Yes) as consent,
                ):
                    self.assertTrue(self.controller._generate_candidate(source, 404))
                pending = self.controller._session
                self.assertIsNot(pending, source)
                self.assertEqual(pending.request, replace(source.request,
                    request_id=pending.request.request_id, input_path=pending.request.input_path,
                    output_path=pending.request.output_path, seed=404))
                self.assertNotEqual(pending.request.request_id, source.request.request_id)
                self.assertNotEqual(pending.request_dir, source.request_dir)
                self.assertTrue(source.request_dir.exists())
                self.assertEqual(pending.expected_lettering, "Frozen lettering")
                self.assertEqual(pending.lettering_language, "English")
                with Image.open(pending.request.input_path) as image:
                    np.testing.assert_array_equal(np.asarray(image.convert("RGB")), source.prepared.model_input)
                if backend == "bfl":
                    consent.assert_called_once()
                    self.assertIs(self.controller._bfl_job.request, pending.request)
                else:
                    consent.assert_not_called()
                    self.assertIs(self.controller.worker.request, pending.request)
                    self.assertEqual(self.controller._worker_python, source.worker_python)
                with patch.object(self.controller, "_show_preview"):
                    self.controller._on_cancelled(pending.request.request_id)
                    self.pump()
                self.controller._clear_candidates()
                self.controller._session = None

    def test_declining_paid_regeneration_preserves_originals_without_dispatch(self):
        self.panel.backend_combo.setCurrentIndex(self.panel.backend_combo.findData("bfl"))
        self.main.settings_page = SimpleNamespace(get_credentials=lambda _: {"api_key": "fixture-key"})
        originals = []
        for seed in (101, 202):
            session = self._session()
            session.backend = "bfl"
            session.request = replace(session.request, seed=seed)
            result = ArtisticEditResult(session.request.request_id, session.request.output_path,
                                        seed, .1, session.request.width, session.request.height)
            self.controller._remember_candidate(session, result)
            originals.append(session)
        self.controller._session = originals[1]
        directories = set(originals[0].request_dir.parent.iterdir())
        opened = []

        def factory(*args, **kwargs):
            kwargs.pop("parent", None)
            dialog = ArtisticEditPreviewDialog(*args, **kwargs)
            self.addCleanup(lambda: dialog.deleteLater() if isValid(dialog) else None)
            def interact():
                opened.append((dialog.selected_candidate_index, self.controller._session))
                if len(opened) == 1:
                    dialog.candidate_combo.setCurrentIndex(0)
                    dialog.done(dialog.REGENERATE_NEW_SEED)
                else:
                    dialog.reject()
            QtCore.QTimer.singleShot(0, interact)
            return dialog

        self.controller.preview_factory = factory
        with (
            patch("app.controllers.artistic_edit.BflEditJob") as job,
            patch("app.controllers.artistic_edit.QtWidgets.QMessageBox.question",
                  return_value=QtWidgets.QMessageBox.No) as consent,
        ):
            self.controller._show_preview(self.controller._candidates[1][1])
            self.assertIsNone(self.controller._bfl_job)
            self.assertIsNone(self.controller._candidate_replace_index)
            self.assertEqual(len(self.controller._candidates), 2)
            for index, session in enumerate(originals):
                self.assertIs(self.controller._candidates[index][0], session)
                self.assertTrue(session.accept_result)
                self.assertTrue(session.request_dir.is_dir())
                self.assertIsNotNone(session.patch)
            self.assertEqual(set(originals[0].request_dir.parent.iterdir()), directories)
            self.assertFalse(self.panel.is_busy())
            self.pump()  # Let the controller reopen the preview after confirmation was declined.
        consent.assert_called_once()
        job.assert_not_called()
        self.assertEqual([index for index, _session in opened], [1, 0])
        self.assertIs(opened[0][1], originals[1])
        self.assertIs(opened[1][1], originals[0])
        self.assertEqual(self.stack.count(), 0)

    def test_project_invalidation_during_paid_confirmation_prevents_dispatch_even_after_yes(self):
        self.panel.backend_combo.setCurrentIndex(self.panel.backend_combo.findData("bfl"))
        self.main.settings_page = SimpleNamespace(get_credentials=lambda _: {"api_key": "fixture-key"})
        def invalidate_then_confirm(*_args):
            self.controller.invalidate_project()
            return QtWidgets.QMessageBox.Yes

        for action in ("initial", "alternative"):
            with (
                self.subTest(action=action),
                patch("app.controllers.artistic_edit.BflEditJob") as job,
                patch("app.controllers.artistic_edit.QtWidgets.QMessageBox.question",
                      side_effect=invalidate_then_confirm) as consent,
            ):
                if action == "initial":
                    self.controller.generate_preview(confirm_whole_page=False)
                else:
                    source = self._session()
                    source.backend = "bfl"
                    self.controller._session = source
                    result = ArtisticEditResult(source.request.request_id, source.request.output_path,
                                                source.request.seed, .1, source.request.width, source.request.height)
                    self.controller.preview_factory = lambda *_args, **_kwargs: _CandidateDecision(
                        ArtisticEditPreviewDialog.GENERATE_ALTERNATIVE, 0)
                    self.controller._show_preview(result)
                    self.assertFalse(source.accept_result)
                    self.assertFalse(source.request_dir.exists())
                self.pump()
                consent.assert_called_once()
                job.assert_not_called()
                self.assertIsNone(self.controller._bfl_job)
                self.assertIsNone(self.controller._session)
                self.assertFalse(self.controller._candidates)
                self.assertFalse(self.panel.is_busy())
                self.assertEqual(list((Path(self.temporary.name) / "artistic_edit").glob("*")), [])
                self.assertEqual(self.stack.count(), 0)

    def assert_regeneration_replaces_selected_only(self, decision, expected_seed):
        first, second = self.two_candidates()
        first_dir = first[0].request_dir
        second_dir = second[0].request_dir
        with patch("app.controllers.artistic_edit.secrets.randbelow", return_value=303):
            self.choose(0, decision)
        pending = self.controller._session
        self.assertEqual(pending.request.seed, expected_seed(first[0]))
        self.assertEqual(self.controller._candidate_replace_index, 0)
        self.assertEqual(len(self.controller._candidates), 2)
        self.assertIs(self.controller._candidates[0][0], first[0])
        self.assertIs(self.controller._candidates[1][0], second[0])
        self.assertTrue(first_dir.exists())
        self.assertTrue(second_dir.exists())
        replacement, _ = self.complete(self.controller._comfy_job, "green")
        self.assertEqual(len(self.controller._candidates), 2)
        self.assertIs(self.controller._candidates[0][0], replacement)
        self.assertIs(self.controller._candidates[1][0], second[0])
        self.assertFalse(first_dir.exists())
        self.assertTrue(second_dir.exists())
        self.assertEqual(self.stack.count(), 0)
        self.assertTrue(self.stack.isClean())

    def test_same_seed_regeneration_replaces_selected_a_only_after_success(self):
        self.assert_regeneration_replaces_selected_only(
            ArtisticEditPreviewDialog.REGENERATE_SAME_SEED, lambda session: session.request.seed)

    def test_new_seed_regeneration_replaces_selected_a_only_after_success(self):
        self.assert_regeneration_replaces_selected_only(
            ArtisticEditPreviewDialog.REGENERATE_NEW_SEED, lambda _session: 303)

    def test_failed_replacement_restores_completed_candidate_without_losing_files(self):
        first, second = self.two_candidates()
        self.choose(0, ArtisticEditPreviewDialog.REGENERATE_NEW_SEED)
        pending = self.controller._session
        with patch.object(self.controller, "_show_preview") as show:
            self.controller._comfy_job.finish("failed", {"code": "fixture", "message": "Generation failed"})
            self.pump()
        self.assertEqual(len(self.controller._candidates), 2)
        self.assertIs(self.controller._candidates[0][0], first[0])
        self.assertIs(self.controller._candidates[1][0], second[0])
        self.assertIs(self.controller._session, first[0])
        self.assertTrue(all(session.request_dir.exists() for session, _ in (first, second)))
        self.assertFalse(pending.request_dir.exists())
        self.assertFalse(self.panel.is_busy())
        self.controller._warning.assert_called_once()
        show.assert_called_once()
        self.assertEqual(self.stack.count(), 0)

    def test_cancelled_replacement_rejects_completion_race_and_keeps_both_candidates(self):
        first, second = self.two_candidates()
        self.controller._generate_candidate(second[0], 303, replace_index=1)
        pending = self.controller._session
        job = self.controller._comfy_job
        self.controller.cancel()
        self.assertTrue(job.cancel_requested)
        self.assertFalse(pending.accept_result)
        self.complete(job, "green")
        self.assertEqual(len(self.controller._candidates), 2)
        self.assertIs(self.controller._candidates[0][0], first[0])
        self.assertIs(self.controller._candidates[1][0], second[0])
        self.assertIs(self.controller._session, second[0])
        self.assertTrue(all(session.accept_result for session, _ in (first, second)))
        self.assertTrue(all(session.request_dir.exists() for session, _ in (first, second)))
        self.assertFalse(pending.request_dir.exists())
        self.assertFalse(self.panel.is_busy())
        self.assertEqual(self.stack.count(), 0)

    def test_project_invalidation_purges_completed_candidates_and_rejects_pending_result(self):
        first, second = self.two_candidates()
        self.choose(0, ArtisticEditPreviewDialog.REGENERATE_NEW_SEED)
        pending = self.controller._session
        job = self.controller._comfy_job
        self.controller.invalidate_project()
        self.assertFalse(self.controller._candidates)
        for session, _ in (first, second):
            self.assertFalse(session.accept_result)
            self.assertFalse(session.request_dir.exists())
        self.assertFalse(pending.accept_result)
        raster = job.spec.raster
        Image.new("RGB", (raster.width, raster.height), "green").save(raster.output_path)
        result = ArtisticEditResult(raster.request_id, raster.output_path, raster.seed,
                                    .1, raster.width, raster.height)
        with patch.object(self.controller, "_show_preview") as show:
            job.finish("completed", ComfyCompletion(result, {}))
            self.pump()
        show.assert_not_called()
        self.assertFalse(self.controller._candidates)
        self.assertFalse(pending.request_dir.exists())
        self.assertIsNone(self.controller._session)
        self.assertEqual(self.stack.count(), 0)
        self.assertTrue(self.stack.isClean())

    def wait_until(self, predicate, timeout=2000):
        timer = QtCore.QElapsedTimer()
        timer.start()
        while not predicate() and timer.elapsed() < timeout:
            self.app.processEvents()
            QtTest.QTest.qWait(2)
        self.assertTrue(predicate(), "Queued GUI/worker operation did not finish")

    def test_completed_ocr_cache_follows_candidate_and_replacement_starts_unchecked(self):
        first, second = self.two_candidates()
        first[0].expected_lettering = "FIRST!"
        second[0].expected_lettering = "SECOND!"
        for session, _ in (first, second):
            session.lettering_language = "English"
        self.main.settings_page = SimpleNamespace(
            get_tool_selection=lambda _: "Default", is_gpu_enabled=lambda: False,
            get_credentials=lambda _: {}, ui=SimpleNamespace(tr=lambda text: text))

        def factory(*args, **kwargs):
            kwargs.pop("parent", None)
            dialog = ArtisticEditPreviewDialog(*args, **kwargs)
            self.addCleanup(lambda: dialog.deleteLater() if isValid(dialog) else None)
            display = Mock(wraps=dialog.show_lettering_check_result)
            dialog.show_lettering_check_result = display
            def exercise():
                for index, (session, _) in enumerate((first, second)):
                    self.controller._select_candidate(dialog, index)
                    self.assertIsNone(session.lettering_result)
                    self.assertNotIn("matches the requested wording", dialog.lettering_status_label.text())
                    dialog.lettering_check_button.click()
                    self.wait_until(lambda: self.controller._lettering_job is None)
                    self.assertEqual(session.lettering_result.status, "match")
                    self.assertEqual(session.lettering_result.expected, session.expected_lettering)
                for index in (0, 1, 0):
                    display.reset_mock()
                    self.controller._select_candidate(dialog, index)
                    session = (first, second)[index][0]
                    display.assert_called_once_with(session.lettering_result)
                    self.assertIn("matches the requested wording", dialog.lettering_status_label.text())
                return ArtisticEditPreviewDialog.REGENERATE_SAME_SEED
            dialog.exec = exercise
            return dialog

        self.controller.preview_factory = factory
        with patch("app.controllers.artistic_edit.run_lettering_ocr", side_effect=["FIRST!", "SECOND!"]) as ocr:
            self.controller._show_preview(second[1])
        self.assertEqual(ocr.call_count, 2)
        second_cache = second[0].lettering_result
        pending = self.controller._session
        self.assertEqual(pending.expected_lettering, "FIRST!")
        self.assertIsNone(pending.lettering_result)
        self.assertIsNone(pending.lettering_revision)
        self.assertIsNotNone(first[0].lettering_result)
        replacement, _ = self.complete(self.controller._comfy_job, "green")
        self.assertIs(self.controller._candidates[0][0], replacement)
        self.assertIsNone(replacement.lettering_result)
        self.assertIsNone(replacement.lettering_revision)
        self.assertIs(self.controller._candidates[1][0].lettering_result, second_cache)
        self.assertEqual(self.stack.count(), 0)

    def test_switching_candidates_retires_ocr_even_when_switching_back_before_completion(self):
        first, second = self.two_candidates()
        first[0].expected_lettering = "FIRST!"
        second[0].expected_lettering = "SECOND!"
        for session, _ in (first, second):
            session.lettering_language = "English"
        self.main.settings_page = SimpleNamespace(
            get_tool_selection=lambda _: "Default", is_gpu_enabled=lambda: False,
            get_credentials=lambda _: {}, ui=SimpleNamespace(tr=lambda text: text))
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        def recognize(*_args):
            entered.set()
            if not release.wait(4):
                raise RuntimeError("Test OCR worker was not released")
            return "FIRST!"

        def factory(*args, **kwargs):
            kwargs.pop("parent", None)
            dialog = ArtisticEditPreviewDialog(*args, **kwargs)
            self.addCleanup(lambda: dialog.deleteLater() if isValid(dialog) else None)
            dialog.show_lettering_check_result = Mock(wraps=dialog.show_lettering_check_result)
            def exercise():
                self.controller._select_candidate(dialog, 0)
                dialog.lettering_check_button.click()
                self.wait_until(entered.is_set)
                self.controller._select_candidate(dialog, 1)
                self.assertIs(self.controller._session, second[0])
                self.controller._select_candidate(dialog, 0)
                release.set()
                self.wait_until(lambda: self.controller._lettering_job is None)
                dialog.show_lettering_check_result.assert_not_called()
                self.assertIsNone(first[0].lettering_result)
                self.assertIsNone(second[0].lettering_result)
                self.assertTrue(dialog.lettering_check_button.isEnabled())
                self.controller._select_candidate(dialog, 1)
                dialog.lettering_check_button.click()
                self.wait_until(lambda: self.controller._lettering_job is None)
                dialog.show_lettering_check_result.assert_called_once()
                check = dialog.show_lettering_check_result.call_args.args[0]
                self.assertEqual(check.expected, "SECOND!")
                self.assertEqual(check.status, "mismatch")
                return QtWidgets.QDialog.Rejected
            dialog.exec = exercise
            return dialog

        self.controller.preview_factory = factory
        with patch("app.controllers.artistic_edit.run_lettering_ocr", side_effect=recognize):
            self.controller._show_preview(second[1])
        self.assertFalse(self.controller._candidates)
        self.assertEqual(self.stack.count(), 0)
        self.assertEqual(self.window_modified_updates, 0)


if __name__ == "__main__":
    unittest.main()
