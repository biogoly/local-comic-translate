import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.ui.dialogs.comfyui_settings import ComfyUISettingsDialog, _SettingsJob
from modules.artistic_edit.comfy_config import (
    ComfyLoraSelection, ComfySettings, KLEIN_9B_INT8_PROFILE,
    QWEN_IMAGE_21_INT8_PROFILE, QWEN_IMAGE_21_MODEL_KEY,
)


class QwenLoraSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def dialog(self, config=None):
        dialog = ComfyUISettingsDialog(config or ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE), Mock())
        self.addCleanup(dialog.deleteLater)
        return dialog

    @staticmethod
    def select_preset(dialog, preset):
        dialog.checkpoint_preset.setCurrentIndex(dialog.checkpoint_preset.findData(preset))

    @staticmethod
    def run_save(config, path, validation=None, cancelled=False):
        job = _SettingsJob(config, "save", Mock(), str(path))
        outcomes = []
        job.outcome.connect(outcomes.append)
        if cancelled:
            job.cancelled.set()
        with patch("app.ui.dialogs.comfyui_settings.validate_comfy_lora_compatibility",
                   side_effect=validation) as validate:
            job.run()
        return outcomes, validate

    def test_qwen_verified_adapter_reopens_and_keeps_checksum(self):
        config = ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE,
                               lora=ComfyLoraSelection("QwenImage2.1/ink.safetensors", 0.65,
                                                      QWEN_IMAGE_21_MODEL_KEY, "a" * 64))
        dialog = self.dialog(config)
        self.assertEqual(dialog._read_config(), config)
        dialog.lora_scale.setValue(0.8)
        self.assertEqual(dialog._read_config().lora.scale, 0.8)
        self.assertEqual(dialog._read_config().lora.sha256, "a" * 64)
        dialog.diffusion.setCurrentText("other-qwen.safetensors")
        with self.assertRaisesRegex(ValueError, "matching local LoRA"):
            dialog._read_config()

    def test_qwen_new_adapter_requires_file_then_records_verified_checksum(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "ink.safetensors"
            path.write_bytes(b"fixture for independently mocked architecture validation")
            dialog = self.dialog()
            dialog.lora.setCurrentText("QwenImage2.1/ink.safetensors")
            dialog.lora_scale.setValue(0.65)
            with self.assertRaisesRegex(ValueError, "matching local LoRA"):
                dialog._read_config()
            dialog.lora_path.setText(str(path))
            config = dialog._read_config()
            self.assertEqual(config.lora.verified_model_key, QWEN_IMAGE_21_MODEL_KEY)
            self.assertIsNone(config.lora.sha256)
            outcomes, validate = self.run_save(config, path)
            validate.assert_called_once_with(path.resolve(), QWEN_IMAGE_21_MODEL_KEY)
            result, error = outcomes[0]
            self.assertEqual(error, "")
            self.assertEqual(result.lora.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result.lora.scale, 0.65)

    def test_model_switch_restores_each_models_unsaved_adapter_edits(self):
        dialog = self.dialog(ComfySettings(profile=KLEIN_9B_INT8_PROFILE))
        dialog.lora.setCurrentText("klein-ink.safetensors")
        dialog.lora_scale.setValue(0.75)
        dialog.lora_path.setText("klein-file.safetensors")
        self.select_preset(dialog, "qwen-image-2.1-int8")
        self.assertEqual((dialog.lora.currentText(), dialog.lora_scale.value(), dialog.lora_path.text()),
                         ("", 1.0, ""))
        dialog.lora.setCurrentText("qwen-ink.safetensors")
        dialog.lora_scale.setValue(0.4)
        dialog.lora_path.setText("qwen-file.safetensors")
        self.select_preset(dialog, "klein-9b-int8")
        self.assertEqual((dialog.lora.currentText(), dialog.lora_scale.value(), dialog.lora_path.text()),
                         ("klein-ink.safetensors", 0.75, "klein-file.safetensors"))
        dialog.model.setCurrentIndex(dialog.model.findData("klein-4b"))
        self.assertEqual(dialog.lora.currentText(), "")
        self.select_preset(dialog, "qwen-image-2.1-int8")
        self.assertEqual((dialog.lora.currentText(), dialog.lora_scale.value(), dialog.lora_path.text()),
                         ("qwen-ink.safetensors", 0.4, "qwen-file.safetensors"))

    def test_cleared_qwen_adapter_is_not_restored_by_same_preset_or_save(self):
        config = ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE,
                               lora=ComfyLoraSelection("ink.safetensors", 0.5,
                                                      QWEN_IMAGE_21_MODEL_KEY, "a" * 64))
        dialog = self.dialog(config)
        dialog.lora.setCurrentText("")
        self.select_preset(dialog, "klein-9b-int8")
        self.select_preset(dialog, "qwen-image-2.1-int8")
        self.assertIsNone(dialog._read_config().lora)
        outcomes, validate = self.run_save(dialog._read_config(), "ignored.safetensors")
        validate.assert_not_called()
        self.assertIsNone(outcomes[0][0].lora)

    def test_failed_or_cancelled_verification_never_returns_validated_settings(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "ink.safetensors"
            path.write_bytes(b"fixture")
            config = ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE,
                                   lora=ComfyLoraSelection("ink.safetensors", 1.0, QWEN_IMAGE_21_MODEL_KEY))
            outcomes, _ = self.run_save(config, path, ValueError("wrong architecture"))
            self.assertEqual(outcomes, [(None, "wrong architecture")])
            outcomes, validate = self.run_save(config, path, cancelled=True)
            self.assertEqual(outcomes, [])
            validate.assert_not_called()

    def test_changed_file_during_validation_is_rejected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "ink.safetensors"
            path.write_bytes(b"original")
            config = ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE,
                                   lora=ComfyLoraSelection("ink.safetensors", 1.0, QWEN_IMAGE_21_MODEL_KEY))
            def mutate(*_args):
                path.write_bytes(b"different weights written while validation was in progress")
            outcomes, _ = self.run_save(config, path, mutate)
            self.assertIsNone(outcomes[0][0])
            self.assertIn("changed during verification", outcomes[0][1])


if __name__ == "__main__":
    unittest.main()
