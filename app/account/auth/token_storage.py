import keyring
import logging
from typing import Optional

from modules.utils.app_identity import KEYRING_SERVICE, LEGACY_KEYRING_SERVICE
from modules.utils.settings import app_settings

logger = logging.getLogger(__name__)

SETTINGS_GROUP = "auth"
LEGACY_MIGRATION_GROUP = "migration/legacy_keyring"

def get_settings():
    return app_settings()


def _store_keyring_token(service: str, name: str, value: str) -> None:
    chunk_size = 512
    chunks = [value[i:i + chunk_size] for i in range(0, len(value), chunk_size)]
    if len(chunks) == 1:
        keyring.set_password(service, name, value)
        return

    keyring.set_password(service, f"{name}_chunks", str(len(chunks)))
    for index, chunk in enumerate(chunks):
        keyring.set_password(service, f"{name}_chunk_{index}", chunk)


def _read_keyring_token(service: str, name: str) -> Optional[str]:
    try:
        token = keyring.get_password(service, name)
        if token:
            return token
    except Exception:
        pass

    try:
        chunk_count_value = keyring.get_password(service, f"{name}_chunks")
        if not chunk_count_value:
            return None
        chunks = []
        for index in range(int(chunk_count_value)):
            chunk = keyring.get_password(service, f"{name}_chunk_{index}")
            if chunk is None:
                return None
            chunks.append(chunk)
        return "".join(chunks)
    except Exception:
        return None


def _delete_keyring_token(service: str, name: str) -> None:
    try:
        keyring.delete_password(service, name)
    except Exception:
        pass

    try:
        chunk_count_value = keyring.get_password(service, f"{name}_chunks")
        if not chunk_count_value:
            return
        keyring.delete_password(service, f"{name}_chunks")
        for index in range(int(chunk_count_value)):
            try:
                keyring.delete_password(service, f"{name}_chunk_{index}")
            except Exception:
                pass
    except Exception:
        pass

def set_token(name: str, value: str):
    """Securely store a token, chunking it if necessary. Fallback to QSettings if keyring fails."""
    # First, try to clear any existing chunks to avoid stale data
    delete_token(name)
    
    try:
        _store_keyring_token(KEYRING_SERVICE, name, value)
        return
    except Exception as e:
        logger.error(f"Keyring storage failed for {name} (Size: {len(value)}): {e}. Falling back to QSettings.")
        # Fallback: Store in QSettings (Encodings issues handled by QSettings, hopefully)
        # Note: This is less secure, but allows the app to function.
        settings = get_settings()
        settings.setValue(f"{SETTINGS_GROUP}/{name}", value)

def get_token(name: str) -> Optional[str]:
    """Retrieve a token, checking keyring first, then QSettings callback."""
    token = _read_keyring_token(KEYRING_SERVICE, name)
    if token:
        return token

    # Preserve an existing login once, but copy it into the fork's keyring
    # service. The legacy credential is never modified or deleted.
    settings = get_settings()
    migration_key = f"{LEGACY_MIGRATION_GROUP}/{name}"
    if not settings.value(migration_key, False, type=bool):
        token = _read_keyring_token(LEGACY_KEYRING_SERVICE, name)
        settings.setValue(migration_key, True)
        if token:
            try:
                _delete_keyring_token(KEYRING_SERVICE, name)
                _store_keyring_token(KEYRING_SERVICE, name, token)
            except Exception as exc:
                logger.warning("Could not migrate keyring token %s: %s", name, exc)
                settings.setValue(f"{SETTINGS_GROUP}/{name}", token)
            settings.sync()
            return token
        settings.sync()

    # 3. Fallback: QSettings
    if settings.contains(f"{SETTINGS_GROUP}/{name}"):
        return str(settings.value(f"{SETTINGS_GROUP}/{name}"))
        
    return None

def delete_token(name: str):
    """Delete a token and all its potential chunks (keyring & QSettings)."""
    # Only clear the fork's credentials. Upstream keeps its own login intact.
    _delete_keyring_token(KEYRING_SERVICE, name)
    
    # QSettings cleanup
    settings = get_settings()
    if settings.contains(f"{SETTINGS_GROUP}/{name}"):
        settings.remove(f"{SETTINGS_GROUP}/{name}")
