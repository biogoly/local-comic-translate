import os
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import QSettings

from modules.utils import paths
from modules.utils.app_identity import (
    LEGACY_ORGANIZATION_NAME,
    LEGACY_SETTINGS_APPLICATION_NAME,
    LEGACY_USER_DATA_DIR_NAME,
    ORGANIZATION_NAME,
    SETTINGS_APPLICATION_NAME,
    USER_DATA_DIR_NAME,
)
from modules.utils.settings import (
    LOCAL_TRANSLATOR_REPAIR_MARKER,
    MIGRATION_MARKER,
    migrate_settings_values,
    repair_migrated_local_translator,
)
from app.account.auth import token_storage


class ProfileIsolationTests(unittest.TestCase):
    def test_fork_identity_is_distinct_from_upstream(self):
        self.assertNotEqual(ORGANIZATION_NAME, LEGACY_ORGANIZATION_NAME)
        self.assertNotEqual(SETTINGS_APPLICATION_NAME, LEGACY_SETTINGS_APPLICATION_NAME)
        self.assertNotEqual(USER_DATA_DIR_NAME, LEGACY_USER_DATA_DIR_NAME)

    def test_settings_migration_copies_missing_values_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = QSettings(os.path.join(temp_dir, "source.ini"), QSettings.IniFormat)
            destination = QSettings(os.path.join(temp_dir, "destination.ini"), QSettings.IniFormat)
            source.setValue("llm/local_model", "legacy-model")
            source.setValue("theme", "Dark")
            destination.setValue("theme", "Light")
            source.sync()
            destination.sync()

            copied = migrate_settings_values(source, destination)

            self.assertEqual(copied, 1)
            self.assertEqual(destination.value("llm/local_model"), "legacy-model")
            self.assertEqual(destination.value("theme"), "Light")
            self.assertTrue(destination.value(MIGRATION_MARKER, False, type=bool))

            source.setValue("llm/local_model", "changed-upstream-model")
            source.sync()
            self.assertEqual(migrate_settings_values(source, destination), 0)
            self.assertEqual(destination.value("llm/local_model"), "legacy-model")

    def test_user_data_migration_preserves_both_profiles(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ, {"LOCALAPPDATA": temp_dir}
        ), patch.object(paths.platform, "system", return_value="Windows"):
            source = paths.get_user_data_dir(LEGACY_USER_DATA_DIR_NAME)
            destination = paths.get_user_data_dir()
            os.makedirs(os.path.join(source, "models"), exist_ok=True)
            os.makedirs(os.path.join(destination, "models"), exist_ok=True)
            source_model = os.path.join(source, "models", "detector.bin")
            source_shared = os.path.join(source, "models", "existing.bin")
            destination_shared = os.path.join(destination, "models", "existing.bin")
            with open(source_model, "wb") as handle:
                handle.write(b"source-model")
            with open(source_shared, "wb") as handle:
                handle.write(b"upstream-version")
            with open(destination_shared, "wb") as handle:
                handle.write(b"fork-version")

            copied = paths.migrate_legacy_user_data()

            self.assertEqual(copied, 1)
            with open(os.path.join(destination, "models", "detector.bin"), "rb") as handle:
                self.assertEqual(handle.read(), b"source-model")
            with open(destination_shared, "rb") as handle:
                self.assertEqual(handle.read(), b"fork-version")
            with open(source_shared, "rb") as handle:
                self.assertEqual(handle.read(), b"upstream-version")
            self.assertEqual(paths.migrate_legacy_user_data(), 0)

    def test_keyring_token_is_copied_once_and_logout_does_not_restore_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = QSettings(os.path.join(temp_dir, "auth.ini"), QSettings.IniFormat)
            passwords = {
                (token_storage.LEGACY_KEYRING_SERVICE, "access_token"): "legacy-token"
            }

            def get_password(service, name):
                return passwords.get((service, name))

            def set_password(service, name, value):
                passwords[(service, name)] = value

            def delete_password(service, name):
                passwords.pop((service, name), None)

            with patch.object(token_storage, "get_settings", return_value=settings), patch.object(
                token_storage.keyring, "get_password", side_effect=get_password
            ), patch.object(
                token_storage.keyring, "set_password", side_effect=set_password
            ), patch.object(
                token_storage.keyring, "delete_password", side_effect=delete_password
            ):
                self.assertEqual(token_storage.get_token("access_token"), "legacy-token")
                self.assertEqual(
                    passwords[(token_storage.KEYRING_SERVICE, "access_token")],
                    "legacy-token",
                )

                token_storage.delete_token("access_token")

                self.assertIsNone(token_storage.get_token("access_token"))
                self.assertEqual(
                    passwords[(token_storage.LEGACY_KEYRING_SERVICE, "access_token")],
                    "legacy-token",
                )

    def test_migrated_custom_selection_is_repaired_when_local_runtime_is_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = QSettings(os.path.join(temp_dir, "repair.ini"), QSettings.IniFormat)
            settings.setValue(MIGRATION_MARKER, True)
            settings.setValue("tools/translator", "Custom")
            settings.setValue("llm/local_runtime", "managed")
            settings.setValue("llm/llama_model_path", "model.gguf")

            self.assertTrue(repair_migrated_local_translator(settings))
            self.assertEqual(settings.value("tools/translator"), "Local LLM")

            settings.setValue("tools/translator", "Custom")
            self.assertFalse(repair_migrated_local_translator(settings))
            self.assertEqual(settings.value("tools/translator"), "Custom")
            self.assertTrue(
                settings.value(LOCAL_TRANSLATOR_REPAIR_MARKER, False, type=bool)
            )


if __name__ == "__main__":
    unittest.main()
