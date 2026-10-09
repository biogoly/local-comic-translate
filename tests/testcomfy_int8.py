import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.ui.dialogs.comfyui_settings import ComfyUISettingsDialog
from modules.artistic_edit.comfy_config import (
    ComfyLoraSelection, ComfyModelProfile, ComfySettings, KLEIN_9B_INT8_PROFILE,
    load_comfy_settings, save_comfy_settings,
)
from modules.artistic_edit.comfy_workflows import build_workflow
from tests.testcomfy_config import MemorySettings
from tests.testcomfy_workflows import make_spec


class ComfyInt8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def dialog(self, config=None):
        dialog = ComfyUISettingsDialog(config or ComfySettings(), Mock())
        self.addCleanup(dialog.deleteLater)
        return dialog

    def select_int8(self, dialog):
        dialog.checkpoint_preset.setCurrentIndex(dialog.checkpoint_preset.findData("klein-9b-int8"))

    def test_opening_previous_settings_does_not_change_them(self):
        config = ComfySettings()
        dialog = self.dialog(config)
        self.assertEqual(dialog.checkpoint_preset.currentData(), "custom")
        self.assertEqual(dialog._read_config(), config)
        dialog.runtime.ensure_started.assert_not_called()

    def test_int8_preset_fills_matching_native_files_without_network_or_runtime(self):
        dialog = self.dialog()
        dialog.model.setCurrentIndex(dialog.model.findData("klein-4b"))
        self.select_int8(dialog)
        self.assertEqual(dialog._read_config().profile, KLEIN_9B_INT8_PROFILE)
        self.assertEqual(dialog.checkpoint_preset.currentData(), "klein-9b-int8")
        self.assertIsNone(dialog._job)
        dialog.runtime.ensure_started.assert_not_called()

    def test_manual_file_or_model_edit_resets_preset_without_overwriting_edit(self):
        for field in ("diffusion", "encoder", "vae", "model"):
            with self.subTest(field=field):
                dialog = self.dialog(ComfySettings(profile=KLEIN_9B_INT8_PROFILE))
                if field == "model":
                    dialog.model.setCurrentIndex(dialog.model.findData("klein-4b"))
                    self.assertEqual(dialog.model.currentData(), "klein-4b")
                else:
                    getattr(dialog, field).setCurrentText("custom.safetensors")
                    self.assertEqual(getattr(dialog, field).currentText(), "custom.safetensors")
                self.assertEqual(dialog.checkpoint_preset.currentData(), "custom")

    def test_custom_preset_leaves_user_files_intact(self):
        dialog = self.dialog(ComfySettings(profile=KLEIN_9B_INT8_PROFILE))
        dialog.checkpoint_preset.setCurrentIndex(0)
        self.assertEqual(dialog._read_config().profile, KLEIN_9B_INT8_PROFILE)

    def test_refresh_preserves_preset_even_when_files_are_not_installed(self):
        dialog = self.dialog(ComfySettings(profile=KLEIN_9B_INT8_PROFILE))
        dialog._job = SimpleNamespace(operation="probe", deleteLater=lambda: None)
        dialog._outcome = (SimpleNamespace(version="0.37.0", catalogs={
            "diffusion_models": ("unrelated.safetensors",), "text_encoders": (), "vae": (),
        }), "")
        dialog._on_finished()
        self.assertEqual(dialog.checkpoint_preset.currentData(), "klein-9b-int8")
        self.assertEqual(dialog._read_config().profile, KLEIN_9B_INT8_PROFILE)
        self.assertGreaterEqual(dialog.diffusion.findText("unrelated.safetensors"), 0)

    def test_existing_int8_profile_identity_and_verified_lora_survive_reopen(self):
        for profile_id in ("klein-9b-native", "my-existing-int8-profile"):
            with self.subTest(profile_id=profile_id):
                config = ComfySettings(
                    profile=replace(KLEIN_9B_INT8_PROFILE, profile_id=profile_id),
                    lora=ComfyLoraSelection("ink.safetensors", 0.7, "klein-9b", "a" * 64),
                )
                dialog = self.dialog(config)
                self.assertEqual(dialog.checkpoint_preset.currentData(), "klein-9b-int8")
                self.assertEqual(dialog._read_config(), config)
                saved = MemorySettings()
                save_comfy_settings(saved, dialog._read_config())
                self.assertEqual(load_comfy_settings(saved), config)

    def test_changing_checkpoint_still_requires_lora_reverification(self):
        config = ComfySettings(lora=ComfyLoraSelection("ink.safetensors", 0.7, "klein-9b", "a" * 64))
        dialog = self.dialog(config)
        self.select_int8(dialog)
        with self.assertRaisesRegex(ValueError, "matching local LoRA"):
            dialog._read_config()

    def test_preset_uses_native_checkpoint_not_an_int8_cast_or_quantization_node(self):
        with TemporaryDirectory() as temporary:
            spec = replace(make_spec(Path(temporary)), profile=KLEIN_9B_INT8_PROFILE)
            graph = build_workflow(spec, "input.png")
        self.assertEqual(graph["1"], {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "flux2Klein9BINT8_9b.safetensors", "weight_dtype": "default"},
        })
        self.assertEqual(graph["2"]["inputs"]["clip_name"], KLEIN_9B_INT8_PROFILE.text_encoder)
        self.assertEqual(graph["3"]["inputs"]["vae_name"], KLEIN_9B_INT8_PROFILE.vae)
        self.assertEqual(len(graph), 17)


if __name__ == "__main__":
    unittest.main()
