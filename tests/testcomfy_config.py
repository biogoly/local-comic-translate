from dataclasses import asdict, replace
import json
import unittest

from modules.artistic_edit.comfy_config import (
    CONFIGURED_KEY, SETTINGS_KEY, ComfyConnectionConfig, ComfyLoraSelection,
    ComfyModelProfile, ComfySettings, load_comfy_settings, normalize_base_url,
    save_comfy_settings, validate_server_name,
)


class MemorySettings:
    def __init__(self):
        self.data = {}

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):
        self.data[key] = value

    def sync(self):
        pass


class ComfyConfigTests(unittest.TestCase):
    def test_loopback_normalization(self):
        self.assertEqual(normalize_base_url("http://localhost:8188/"), "http://127.0.0.1:8188")
        self.assertEqual(normalize_base_url("http://[::1]:8188"), "http://[::1]:8188")
        self.assertEqual(normalize_base_url("https://127.0.0.2"), "https://127.0.0.2")

    def test_nonloopback_credentials_and_url_components_rejected(self):
        for url in ("http://192.168.1.2", "http://example.com", "http://127.0.0.1.evil", "file:///x",
                    "http://user@127.0.0.1", "http://127.0.0.1/x", "http://127.0.0.1?x=1",
                    "http://127.0.0.1#x", "http://127.0.0.1:0", "http://127.0.0.1:99999"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                normalize_base_url(url)

    def test_server_names_are_relative_and_normalized(self):
        self.assertEqual(validate_server_name("styles\\lettering.safetensors"), "styles/lettering.safetensors")
        for name in ("../x", "/x", "C:\\x", "a//x", "a/./x", "x\n.png", "a/../../x", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_server_name(name)

    def test_round_trip_preserves_unknown_selected_filename(self):
        settings = MemorySettings()
        config = ComfySettings(profile=replace(ComfyModelProfile(), diffusion_model="missing/model.safetensors"))
        save_comfy_settings(settings, config)
        self.assertTrue(settings.data[CONFIGURED_KEY])
        self.assertEqual(load_comfy_settings(settings), config)

    def test_unconfigured_defaults_do_not_mark_configured(self):
        settings = MemorySettings()
        self.assertEqual(load_comfy_settings(settings), ComfySettings())
        self.assertNotIn(CONFIGURED_KEY, settings.data)

    def test_invalid_saved_settings_fail_instead_of_silently_replacing(self):
        settings = MemorySettings()
        settings.data[SETTINGS_KEY] = '{"connection": {"base_url": "http://evil.test"}}'
        with self.assertRaisesRegex(ValueError, "Saved ComfyUI settings"):
            load_comfy_settings(settings)

    def test_lora_size_and_gguf_gate(self):
        lora = ComfyLoraSelection("style.safetensors", 0.5, "klein-4b")
        with self.assertRaisesRegex(ValueError, "model family or size"):
            ComfySettings(lora=lora)
        profile = replace(ComfyModelProfile(), loader="gguf", diffusion_model="klein-Q4_K_M.gguf")
        with self.assertRaisesRegex(ValueError, "not been qualified"):
            ComfySettings(profile=profile, lora=replace(lora, verified_model_key="klein-9b"))
        self.assertTrue(profile.experimental)

    def test_managed_fields_validated(self):
        with self.assertRaises(ValueError):
            ComfyConnectionConfig(mode="managed")
        with self.assertRaises(ValueError):
            ComfyConnectionConfig(gpu_index=-1)
        with self.assertRaises(ValueError):
            ComfyConnectionConfig(memory_policy="gpu_only")

    def test_malformed_saved_paths_are_repairable_configuration_errors(self):
        for field, value in (("installation_root", 42), ("installation_root", []), ("python_executable", {})):
            with self.subTest(field=field, value=value):
                data = asdict(ComfySettings())
                data["connection"].update(mode="managed", installation_root="C:/ComfyUI")
                data["connection"][field] = value
                settings = MemorySettings()
                settings.data[SETTINGS_KEY] = json.dumps(data)
                with self.assertRaisesRegex(ValueError, "Saved ComfyUI settings"):
                    load_comfy_settings(settings)


if __name__ == "__main__":
    unittest.main()
