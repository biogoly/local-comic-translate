"""Fork-isolated QSettings access and one-time legacy migration."""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import QSettings

from modules.utils.app_identity import (
    LEGACY_ORGANIZATION_NAME,
    LEGACY_SETTINGS_APPLICATION_NAME,
    ORGANIZATION_NAME,
    SETTINGS_APPLICATION_NAME,
)
from modules.utils.paths import (
    get_default_project_autosave_dir,
    get_legacy_default_project_autosave_dir,
)


logger = logging.getLogger(__name__)
MIGRATION_MARKER = "migration/legacy_profile_v1"
LOCAL_TRANSLATOR_REPAIR_MARKER = "migration/local_translator_repaired_v1"


def app_settings() -> QSettings:
    """Return the settings store owned exclusively by this fork."""
    return QSettings(ORGANIZATION_NAME, SETTINGS_APPLICATION_NAME)


def legacy_settings() -> QSettings:
    """Return the former shared settings store for migration reads only."""
    return QSettings(LEGACY_ORGANIZATION_NAME, LEGACY_SETTINGS_APPLICATION_NAME)


def migrate_custom_endpoint(settings: QSettings) -> None:
    """Keep an active former Custom connection in Local LLM > External server."""
    if settings.value('tools/translator', '') != 'Custom':
        return
    url = str(settings.value('credentials/Custom_api_url', '') or '')
    model = str(settings.value('credentials/Custom_model', '') or '')
    if url and model:
        settings.setValue('llm/local_runtime', 'external')
        settings.setValue('llm/local_endpoint', url)
        settings.setValue('llm/local_model', model)
        save_keys = settings.value('credentials/save_keys', False, type=bool)
        settings.setValue('llm/local_api_key', settings.value('credentials/Custom_api_key', '') if save_keys else '')
    settings.setValue('tools/translator', 'Local LLM')


def migrate_settings_values(
    source: QSettings,
    destination: QSettings,
    *,
    marker_key: str = MIGRATION_MARKER,
) -> int:
    """Copy missing values once, preserving anything already set in the fork.

    The source remains untouched. Returning a count makes the operation easy to
    report and test without inspecting potentially sensitive values.
    """
    if destination.value(marker_key, False, type=bool):
        return 0

    copied = 0
    for key in source.allKeys():
        if key == marker_key or destination.contains(key):
            continue
        destination.setValue(key, source.value(key))
        copied += 1

    destination.setValue(marker_key, True)
    destination.sync()
    return copied


def migrate_legacy_settings() -> int:
    """Snapshot the old shared profile into the fork's private settings store."""
    try:
        destination = app_settings()
        already_migrated = destination.value(MIGRATION_MARKER, False, type=bool)
        copied = migrate_settings_values(legacy_settings(), destination)
        if not already_migrated:
            autosave_key = "export/project_autosave_folder"
            autosave_folder = str(destination.value(autosave_key, "") or "")
            if autosave_folder and os.path.normcase(os.path.normpath(autosave_folder)) == os.path.normcase(
                os.path.normpath(get_legacy_default_project_autosave_dir())
            ):
                destination.setValue(autosave_key, get_default_project_autosave_dir())
                destination.sync()
        if copied:
            logger.info("Migrated %d legacy settings into the fork profile.", copied)
        return copied
    except Exception as exc:
        # A migration failure should not prevent the application from opening.
        logger.warning("Could not migrate legacy application settings: %s", exc)
        return 0


def repair_migrated_local_translator(settings: QSettings | None = None) -> bool:
    """Restore Local LLM when upstream changed a migrated fork profile to Custom.

    This is deliberately a one-time repair and only applies to a profile that
    completed the legacy migration and still contains a usable local runtime
    configuration. Later user choices are never overridden.
    """
    settings = settings or app_settings()
    if settings.value(LOCAL_TRANSLATOR_REPAIR_MARKER, False, type=bool):
        return False

    migrated = settings.value(MIGRATION_MARKER, False, type=bool)
    translator = str(settings.value("tools/translator", "") or "")
    runtime = str(settings.value("llm/local_runtime", "managed") or "managed")
    managed_ready = bool(str(settings.value("llm/llama_model_path", "") or "").strip())
    external_ready = bool(
        str(settings.value("llm/local_endpoint", "") or "").strip()
        and str(settings.value("llm/local_model", "") or "").strip()
    )
    local_ready = managed_ready if runtime == "managed" else external_ready

    repaired = bool(migrated and translator == "Custom" and local_ready)
    if repaired:
        settings.setValue("tools/translator", "Local LLM")
    settings.setValue(LOCAL_TRANSLATOR_REPAIR_MARKER, True)
    settings.sync()
    return repaired
