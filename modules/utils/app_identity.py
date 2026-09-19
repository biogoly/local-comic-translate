"""Stable product identity for the Local Comic Translate fork.

Keeping these values distinct from upstream prevents two checkouts from sharing
preferences, credentials, model files, autosaves, and Windows taskbar identity.
"""

PRODUCT_NAME = "Local Comic Translate"
ORGANIZATION_NAME = "biogoly"
SETTINGS_APPLICATION_NAME = "LocalComicTranslate"
USER_DATA_DIR_NAME = "LocalComicTranslate"
DEFAULT_PROJECT_FOLDER_NAME = "Local Comic Translate"
WINDOWS_APP_USER_MODEL_ID = "biogoly.LocalComicTranslate"
SINGLE_INSTANCE_PREFIX = "LocalComicTranslate"
KEYRING_SERVICE = "local-comic-translate"

# Read-only migration sources used by releases before the fork had its own
# identity. They must never be used for new writes.
LEGACY_ORGANIZATION_NAME = "ComicLabs"
LEGACY_SETTINGS_APPLICATION_NAME = "ComicTranslate"
LEGACY_USER_DATA_DIR_NAME = "ComicTranslate"
LEGACY_DEFAULT_PROJECT_FOLDER_NAME = "Comic Translate"
LEGACY_KEYRING_SERVICE = "comic-translate"
