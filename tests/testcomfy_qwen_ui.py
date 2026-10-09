import os
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.ui.artistic_edit_panel import ArtisticEditPanel
from app.ui.dialogs.artistic_edit_preview import ArtisticEditPreviewDialog
from app.ui.dialogs.comfyui_settings import ComfyUISettingsDialog
from modules.artistic_edit.backend_contracts import ComfyCompletion
from modules.artistic_edit.comfy_config import (
    ComfyLoraSelection, ComfySettings, KLEIN_9B_INT8_PROFILE,
    QWEN_IMAGE_21_INT8_PROFILE, QWEN_IMAGE_21_MODEL_KEY, QWEN_IMAGE_21_WORKFLOW_ID,
    save_comfy_settings,
)
from modules.artistic_edit.contracts import ModelKey
from tests import testartistic_edit_controller as harness
from tests import testcomfy_controller as comfy_harness


class QwenSettingsAndPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def dialog(self, config=None):
        dialog = ComfyUISettingsDialog(config or ComfySettings(), Mock())
        self.addCleanup(dialog.deleteLater)
        return dialog

    def panel(self):
        panel = ArtisticEditPanel()
        self.addCleanup(panel.deleteLater)
        return panel

    def test_qwen_preset_selects_existing_matching_files_without_starting_runtime(self):
        dialog = self.dialog()
        dialog.checkpoint_preset.setCurrentIndex(dialog.checkpoint_preset.findData("qwen-image-2.1-int8"))
        config = dialog._read_config()
        self.assertEqual(config.profile, QWEN_IMAGE_21_INT8_PROFILE)
        self.assertEqual(config.profile.workflow_id, QWEN_IMAGE_21_WORKFLOW_ID)
        self.assertTrue(config.profile.experimental)
        self.assertIsNone(config.lora)
        self.assertIsNone(dialog._job)
        self.assertFalse(dialog.qwen_notice.isHidden())
        dialog.runtime.ensure_started.assert_not_called()

    def test_qwen_clears_klein_lora_and_switching_back_preserves_verification(self):
        original = ComfySettings(
            profile=KLEIN_9B_INT8_PROFILE,
            lora=ComfyLoraSelection("klein-ink.safetensors", 0.7, "klein-9b", "a" * 64),
        )
        dialog = self.dialog(original)
        dialog.checkpoint_preset.setCurrentIndex(dialog.checkpoint_preset.findData("qwen-image-2.1-int8"))
        for widget in (dialog.lora, dialog.lora_scale, dialog.lora_path, dialog.lora_file_row):
            self.assertTrue(widget.isEnabled())
        self.assertEqual(dialog.lora.currentText(), "")
        self.assertIsNone(dialog._read_config().lora)
        dialog.checkpoint_preset.setCurrentIndex(dialog.checkpoint_preset.findData("klein-9b-int8"))
        self.assertTrue(dialog.lora.isEnabled())
        self.assertEqual(dialog._read_config(), original)

    def test_saved_qwen_profile_identity_and_custom_files_survive_reopen(self):
        profile = replace(QWEN_IMAGE_21_INT8_PROFILE, profile_id="my-qwen-profile")
        config = ComfySettings(profile=profile)
        dialog = self.dialog(config)
        self.assertEqual(dialog.checkpoint_preset.currentData(), "qwen-image-2.1-int8")
        self.assertEqual(dialog._read_config(), config)
        dialog.diffusion.setCurrentText("custom-qwen.safetensors")
        self.assertEqual(dialog.checkpoint_preset.currentData(), "custom")
        self.assertEqual(dialog._read_config().profile.diffusion_model, "custom-qwen.safetensors")

    def test_catalog_refresh_preserves_qwen_preset_and_empty_lora(self):
        dialog = self.dialog(ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE))
        dialog._job = SimpleNamespace(operation="probe", deleteLater=lambda: None)
        dialog._outcome = (SimpleNamespace(version="test", catalogs={
            "diffusion_models": ("unrelated.safetensors",), "loras": ("klein-ink.safetensors",),
        }), "")
        dialog._on_finished()
        self.assertEqual(dialog.checkpoint_preset.currentData(), "qwen-image-2.1-int8")
        self.assertEqual(dialog._read_config().profile, QWEN_IMAGE_21_INT8_PROFILE)
        self.assertTrue(dialog.lora.isEnabled())
        self.assertEqual(dialog.lora.currentText(), "")

    def test_backend_and_profile_switches_restore_their_sampling_values(self):
        panel = self.panel()
        panel.steps_spin.setValue(8)
        panel.guidance_spin.setValue(3.0)
        panel.backend_combo.setCurrentIndex(panel.backend_combo.findData("comfyui"))
        self.assertEqual((panel.steps_spin.value(), panel.guidance_spin.value()), (4, 1.0))
        self.assertFalse(panel.guidance_spin.isEnabled())
        panel.set_comfyui_profile(QWEN_IMAGE_21_INT8_PROFILE)
        self.assertEqual((panel.steps_spin.value(), panel.guidance_spin.value()), (25, 1.0))
        self.assertEqual(panel.steps_spin.maximum(), 100)
        self.assertTrue(panel.guidance_spin.isEnabled())
        self.assertTrue(panel.model_combo.isHidden())
        self.assertEqual(panel.options().model_key, QWEN_IMAGE_21_MODEL_KEY)
        panel.steps_spin.setValue(37)
        panel.guidance_spin.setValue(2.5)
        panel.set_comfyui_profile(QWEN_IMAGE_21_INT8_PROFILE)
        self.assertEqual((panel.steps_spin.value(), panel.guidance_spin.value()), (37, 2.5))
        panel.set_comfyui_profile(KLEIN_9B_INT8_PROFILE)
        self.assertEqual((panel.steps_spin.value(), panel.guidance_spin.value()), (4, 1.0))
        panel.steps_spin.setValue(6)
        panel.set_comfyui_profile(QWEN_IMAGE_21_INT8_PROFILE)
        self.assertEqual((panel.steps_spin.value(), panel.guidance_spin.value()), (37, 2.5))
        panel.backend_combo.setCurrentIndex(panel.backend_combo.findData("local"))
        self.assertEqual((panel.steps_spin.value(), panel.guidance_spin.value()), (8, 3.0))
        self.assertEqual(panel.steps_spin.maximum(), 20)
        self.assertIsInstance(panel.options().model_key, ModelKey)
        self.assertFalse(panel.model_combo.isHidden())
        panel.backend_combo.setCurrentIndex(panel.backend_combo.findData("bfl"))
        self.assertFalse(panel.guidance_spin.isEnabled())
        self.assertFalse(panel.steps_spin.isEnabled())
        self.assertEqual(panel.model_combo.count(), 2)
        panel.backend_combo.setCurrentIndex(panel.backend_combo.findData("comfyui"))
        self.assertEqual((panel.steps_spin.value(), panel.guidance_spin.value()), (37, 2.5))
        panel.set_comfyui_profile(KLEIN_9B_INT8_PROFILE)
        self.assertEqual((panel.steps_spin.value(), panel.guidance_spin.value()), (6, 1.0))


class QwenControllerTests(unittest.TestCase):
    setUpClass = classmethod(comfy_harness.ComfyControllerTests.setUpClass.__func__)
    _mark_window_update = comfy_harness.ComfyControllerTests._mark_window_update
    pump = comfy_harness.ComfyControllerTests.pump
    generate = comfy_harness.ComfyControllerTests.generate

    def setUp(self):
        comfy_harness.ComfyControllerTests.setUp(self)
        save_comfy_settings(self.settings, ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE))
        self.controller.update_availability()

    def completion(self, job):
        completion = comfy_harness.ComfyControllerTests.completion(self, job)
        return ComfyCompletion(completion.result, {
            "workflow_id": job.spec.profile.workflow_id,
            "model_source": job.spec.profile.diffusion_model,
            "cfg": job.spec.cfg,
        })

    def test_generation_uses_qwen_defaults_and_profile_without_klein_model_cast(self):
        job = self.generate()
        self.assertEqual(job.spec.profile, QWEN_IMAGE_21_INT8_PROFILE)
        self.assertEqual(job.spec.raster.steps, 25)
        self.assertEqual(job.spec.cfg, 1.0)
        self.assertIsNone(job.spec.lora)
        self.assertIn("experimental", self.panel.comfyui_summary.text())
        self.controller.configured_worker_python.assert_not_called()

    def test_regeneration_keeps_frozen_qwen_steps_cfg_profile_and_prompt(self):
        self.controller.preview_factory = lambda *_args, **_kwargs: harness._PreviewDecision(
            ArtisticEditPreviewDialog.REGENERATE_SAME_SEED
        )
        self.panel.steps_spin.setValue(37)
        self.panel.guidance_spin.setValue(2.5)
        job = self.generate()
        original = job.spec
        self.panel.steps_spin.setValue(14)
        self.panel.guidance_spin.setValue(7.0)
        self.panel.set_prompt("Changed UI prompt")
        save_comfy_settings(self.settings, ComfySettings(profile=KLEIN_9B_INT8_PROFILE))
        job.finish("completed", self.completion(job))
        self.pump()
        regenerated = self.controller._comfy_job.spec
        self.assertEqual(regenerated.profile, original.profile)
        self.assertEqual(regenerated.raster.prompt, original.raster.prompt)
        self.assertEqual(regenerated.raster.seed, original.raster.seed)
        self.assertEqual(regenerated.raster.steps, 37)
        self.assertEqual(regenerated.cfg, 2.5)
        self.assertNotEqual(regenerated.job_token, original.job_token)

    def test_regeneration_and_apply_keep_frozen_qwen_adapter_after_settings_change(self):
        adapter = ComfyLoraSelection("QwenImage2.1/ink.safetensors", 0.4,
                                    QWEN_IMAGE_21_MODEL_KEY, "a" * 64)
        config = ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE, lora=adapter)
        save_comfy_settings(self.settings, config)
        self.controller.update_availability()
        summaries = []

        def preview(_before, _after, summary, **_kwargs):
            summaries.append(summary)
            if len(summaries) == 1:
                save_comfy_settings(self.settings, replace(config, lora=None))
                return harness._PreviewDecision(ArtisticEditPreviewDialog.REGENERATE_SAME_SEED)
            return harness._PreviewDecision(QtWidgets.QDialog.Accepted)

        self.controller.preview_factory = preview
        original_job = self.generate()
        self.assertEqual(original_job.spec.lora, adapter)
        original_job.finish("completed", self.completion(original_job))
        self.pump()
        regenerated_job = self.controller._comfy_job
        self.assertIsNotNone(regenerated_job)
        self.assertIsNot(regenerated_job, original_job)
        self.assertEqual(regenerated_job.spec.lora, adapter)
        self.assertEqual(regenerated_job.spec.raster.seed, original_job.spec.raster.seed)
        regenerated_job.finish("completed", self.completion(regenerated_job))
        self.pump()
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(adapter.name in summary for summary in summaries))
        self.assertEqual(self.stack.count(), 1)
        record = self.main.image_patches[self.page_path][0]
        self.assertEqual(record["kind"], "artistic_edit")
        self.assertEqual(record["metadata"]["lora_name"], adapter.name)
        self.assertEqual(record["metadata"]["lora_scale"], adapter.scale)
        self.assertEqual(record["metadata"]["lora_sha256"], adapter.sha256)
        self.stack.undo()
        self.assertEqual(self.main.image_patches[self.page_path], [])
        self.stack.redo()
        self.assertEqual(self.main.image_patches[self.page_path][0]["metadata"]["lora_sha256"], adapter.sha256)

    def test_apply_records_neutral_qwen_kind_cfg_and_supports_undo_redo(self):
        summaries = []
        def preview(_before, _after, summary, **_kwargs):
            summaries.append(summary)
            return harness._PreviewDecision(QtWidgets.QDialog.Accepted)
        self.controller.preview_factory = preview
        self.panel.guidance_spin.setValue(2.5)
        job = self.generate()
        job.finish("completed", self.completion(job))
        self.pump()
        self.assertEqual(self.stack.count(), 1)
        record = self.main.image_patches[self.page_path][0]
        self.assertEqual(record["kind"], "artistic_edit")
        self.assertEqual(record["metadata"]["model_key"], QWEN_IMAGE_21_MODEL_KEY)
        self.assertEqual(record["metadata"]["workflow_id"], QWEN_IMAGE_21_WORKFLOW_ID)
        self.assertEqual(record["metadata"]["steps"], 25)
        self.assertEqual(record["metadata"]["guidance"], 2.5)
        self.assertIn("CFG: 2.5", summaries[0])
        self.stack.undo()
        self.assertEqual(self.main.image_patches[self.page_path], [])
        self.stack.redo()
        self.assertEqual(len(self.main.image_patches[self.page_path]), 1)

    def test_revert_artistic_edits_handles_qwen_and_legacy_klein_together(self):
        self.controller.preview_factory = lambda *_args, **_kwargs: harness._PreviewDecision(QtWidgets.QDialog.Accepted)
        for profile in (QWEN_IMAGE_21_INT8_PROFILE, KLEIN_9B_INT8_PROFILE):
            save_comfy_settings(self.settings, ComfySettings(profile=profile))
            self.controller.update_availability()
            job = self.generate()
            job.finish("completed", self.completion(job))
            self.pump()
        self.assertEqual({p["kind"] for p in self.main.image_patches[self.page_path]},
                         {"artistic_edit", "flux2_edit"})
        self.controller.revert_artistic_edits()
        self.assertEqual(self.main.image_patches[self.page_path], [])
        self.stack.undo()
        self.assertEqual(len(self.main.image_patches[self.page_path]), 2)
        self.stack.redo()
        self.assertEqual(self.main.image_patches[self.page_path], [])


if __name__ == "__main__":
    unittest.main()
