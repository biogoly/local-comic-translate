import os
import platform
import shutil

from modules.utils.app_identity import (
    DEFAULT_PROJECT_FOLDER_NAME,
    LEGACY_DEFAULT_PROJECT_FOLDER_NAME,
    LEGACY_USER_DATA_DIR_NAME,
    USER_DATA_DIR_NAME,
)


def get_user_data_dir(app_name: str = USER_DATA_DIR_NAME) -> str:
    """
    Returns the platform-specific user data directory for the application.
    
    Windows: %LOCALAPPDATA%/<app_name>
    macOS: ~/Library/Application Support/<app_name>
    Linux: $XDG_DATA_HOME/<app_name> or ~/.local/share/<app_name>
    """
    system = platform.system()
    
    if system == "Windows":
        base_dir = os.getenv('LOCALAPPDATA')
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    elif system == "Darwin":
        base_dir = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        # Linux / Unix
        base_dir = os.getenv('XDG_DATA_HOME')
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), ".local", "share")
            
    return os.path.join(base_dir, app_name)


def get_default_project_autosave_dir(folder_name: str = DEFAULT_PROJECT_FOLDER_NAME) -> str:
    """
    Returns a user-facing default folder for project auto-save files.

    Windows/macOS/Linux: ~/Documents/<folder_name>
    """
    return os.path.join(os.path.expanduser("~"), "Documents", folder_name)


def get_legacy_default_project_autosave_dir() -> str:
    """Return upstream's default project folder for migration comparisons."""
    return get_default_project_autosave_dir(LEGACY_DEFAULT_PROJECT_FOLDER_NAME)


def _copy_file_if_missing(source: str, destination: str) -> str:
    """Copy one migration file without replacing a fork-owned file."""
    if not os.path.exists(destination):
        shutil.copy2(source, destination)
    return destination


def migrate_legacy_user_data() -> int:
    """Copy the former shared data directory into this fork exactly once.

    Existing destination files always win, which makes an interrupted migration
    safe to retry. The old upstream directory is read-only and remains intact.
    """
    destination = get_user_data_dir()
    marker = os.path.join(destination, ".legacy-profile-migrated-v1")
    if os.path.isfile(marker):
        return 0

    source = get_user_data_dir(LEGACY_USER_DATA_DIR_NAME)
    os.makedirs(destination, exist_ok=True)
    before = {
        os.path.normcase(os.path.relpath(os.path.join(root, name), destination))
        for root, _dirs, files in os.walk(destination)
        for name in files
    }

    if os.path.isdir(source) and os.path.normcase(source) != os.path.normcase(destination):
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            copy_function=_copy_file_if_missing,
        )

    after = {
        os.path.normcase(os.path.relpath(os.path.join(root, name), destination))
        for root, _dirs, files in os.walk(destination)
        for name in files
    }
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write("Legacy ComicTranslate profile migrated without modifying the source.\n")
    return len(after - before)
