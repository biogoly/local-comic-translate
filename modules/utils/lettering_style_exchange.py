"""Read and write portable lettering styles, and merge them into a project draft."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from uuid import uuid4

from .lettering_styles import (
    MAX_STYLE_NAME, MAX_STYLE_PRESETS, STYLE_PROPERTY_KEYS, normalize_style_properties,
    validate_style_library, validate_style_name,
)


STYLE_FILE_FORMAT = 'local-comic-translate.lettering-styles'
STYLE_FILE_VERSION = 1
MAX_STYLE_FILE_BYTES = 256 * 1024


class StyleExchangeError(ValueError):
    """Stable error code for translation by the editor."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass
class StyleImportResult:
    library: dict
    added: int
    skipped: int
    renamed: int


def _validated_library(library):
    try:
        return validate_style_library(library)
    except (ValueError, OverflowError) as exc:
        raise StyleExchangeError('invalid') from exc


def _portable_entry(name, props):
    family = props.get('font_family', '')
    # Font references are family names. Never interpret or export file locations.
    if '/' in family or '\\' in family or re.match(r'^(?:[a-z]:|file:|data:)', family, re.I):
        raise StyleExchangeError('invalid')
    try:
        # Escaped unpaired surrogates can survive json.loads, but cannot travel in UTF-8.
        name.encode('utf-8')
        family.encode('utf-8')
    except UnicodeError as exc:
        raise StyleExchangeError('invalid') from exc
    return {'name': name, 'properties': props}


def _import_presets(entries):
    if not isinstance(entries, list) or len(entries) > MAX_STYLE_PRESETS:
        raise StyleExchangeError('invalid')
    if not entries:
        raise StyleExchangeError('empty')
    presets = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'name', 'properties'}:
            raise StyleExchangeError('invalid')
        props = entry['properties']
        if not isinstance(props, dict) or not props.keys() <= STYLE_PROPERTY_KEYS:
            raise StyleExchangeError('invalid')
        try:
            name = validate_style_name(entry['name'])
            properties = normalize_style_properties(props, strict=True)
        except (ValueError, OverflowError) as exc:
            raise StyleExchangeError('invalid') from exc
        if not properties:
            raise StyleExchangeError('invalid')
        presets.append(_portable_entry(name, properties))
    return presets


def _unique_keys(pairs):
    mapping = {}
    for key, value in pairs:
        if key in mapping:
            raise StyleExchangeError('format')
        mapping[key] = value
    return mapping


def read_style_file(path) -> list[dict]:
    """Read a bounded UTF-8 file and return validated portable preset entries."""
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_STYLE_FILE_BYTES + 1)
    except OSError as exc:
        raise StyleExchangeError('read') from exc
    if len(raw) > MAX_STYLE_FILE_BYTES:
        raise StyleExchangeError('oversized')
    try:
        data = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError) as exc:
        raise StyleExchangeError('format') from exc
    if (not isinstance(data, dict) or data.get('format') != STYLE_FILE_FORMAT
            or set(data) != {'format', 'version', 'presets'}):
        raise StyleExchangeError('format')
    if type(data['version']) is not int or data['version'] != STYLE_FILE_VERSION:
        raise StyleExchangeError('version')
    return _import_presets(data['presets'])


def write_style_file(path, library) -> None:
    """Write only portable fields, replacing the destination after a complete write."""
    presets = [_portable_entry(p['name'], p['properties']) for p in _validated_library(library)['presets']]
    if not presets:
        raise StyleExchangeError('empty')
    data = {'format': STYLE_FILE_FORMAT, 'version': STYLE_FILE_VERSION, 'presets': presets}
    raw = (json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
           + '\n').encode('utf-8')
    if len(raw) > MAX_STYLE_FILE_BYTES:
        raise StyleExchangeError('oversized')
    destination = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise StyleExchangeError('write') from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Preserve the original write error if the OS also blocks removal.


def _available_name(name, properties, by_name, requested_names):
    """Return the original or numbered name, or None for an existing equivalent."""
    occupants = by_name.get(name.casefold())
    if not occupants:
        return name
    if properties in occupants:
        return None

    available = None
    # Check for a previously imported copy even if an earlier suffix was deleted.
    # Each of the two libraries can occupy at most MAX_STYLE_PRESETS names.
    for index in range(2, 2 * MAX_STYLE_PRESETS + 3):
        suffix = f' ({index})'
        candidate = name[:MAX_STYLE_NAME - len(suffix)].rstrip() + suffix
        folded = candidate.casefold()
        if folded in requested_names:
            continue
        occupants = by_name.get(folded)
        if occupants and properties in occupants:
            return None
        if not occupants and available is None:
            available = candidate
    if available is None:
        raise StyleExchangeError('capacity')
    return available


def merge_imported_styles(library, entries) -> StyleImportResult:
    """Build an independent draft. Any invalid entry or capacity error rejects the merge."""
    existing = _validated_library(library)
    imports = _import_presets(entries)
    requested_names = {p['name'].casefold() for p in imports}
    by_name = {}
    for preset in existing['presets']:
        by_name.setdefault(preset['name'].casefold(), []).append(preset['properties'])
    result = deepcopy(library)
    added = skipped = renamed = 0
    for preset in imports:
        name = _available_name(preset['name'], preset['properties'], by_name, requested_names)
        if name is None:
            skipped += 1
            continue
        if len(result['presets']) >= MAX_STYLE_PRESETS:
            raise StyleExchangeError('capacity')
        if name != preset['name']:
            renamed += 1
        result['presets'].append({'id': str(uuid4()), 'name': name, 'properties': preset['properties']})
        by_name.setdefault(name.casefold(), []).append(preset['properties'])
        added += 1
    return StyleImportResult(result, added, skipped, renamed)
