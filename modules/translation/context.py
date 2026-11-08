"""Versioned project reference material and language-aware term matching.

This module contains plain project data only. It never reads credentials,
starts models, or changes translated text by search-and-replace.
"""
from __future__ import annotations

import json
import re
import unicodedata
import uuid


PROFILE_VERSION = 1
DEFAULT_CONTEXT_TOKEN_LIMIT = 2048


def _text(value):
    return value if isinstance(value, str) else ''


def _identity(value):
    return _text(value).strip() or str(uuid.uuid4())


def _aliases(value):
    if isinstance(value, str):
        value = value.splitlines()
    if not isinstance(value, (tuple, list)):
        return []
    return list(dict.fromkeys(part.strip() for part in value if isinstance(part, str) and part.strip()))


def empty_translation_profile():
    return {'schema_version': PROFILE_VERSION, 'glossary': [], 'characters': [],
            'notes': '', 'summary': '', 'speakers': [], 'context_token_limit': DEFAULT_CONTEXT_TOKEN_LIMIT,
            'include_approved_history': False}


def normalize_translation_profile(value):
    """Return an independent allowlisted profile, with legacy empty defaults."""
    result = empty_translation_profile()
    if not isinstance(value, dict):
        return result
    version = value.get('schema_version', PROFILE_VERSION)
    if version != PROFILE_VERSION:
        raise ValueError(f'Unsupported translation profile version: {version}. This application supports version {PROFILE_VERSION}.')
    result['notes'] = _text(value.get('notes'))
    result['summary'] = _text(value.get('summary'))
    result['include_approved_history'] = value.get('include_approved_history') is True
    try:
        result['context_token_limit'] = max(128, min(32768, int(value.get('context_token_limit', DEFAULT_CONTEXT_TOKEN_LIMIT))))
    except (ValueError, TypeError, OverflowError):
        pass
    for kind in ('glossary', 'characters'):
        rows = value.get(kind, [])
        if not isinstance(rows, (list, tuple)):
            continue
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            identity = _identity(row.get('id'))
            if identity in seen:
                identity = str(uuid.uuid4())
            seen.add(identity)
            item = {'id': identity, 'aliases': _aliases(row.get('aliases')),
                    'target': _text(row.get('target')), 'source_lang': _text(row.get('source_lang')),
                    'target_lang': _text(row.get('target_lang')),
                    'enabled': row.get('enabled', True) is not False,
                    'approved': row.get('approved', True) is True}
            if kind == 'glossary':
                item.update(source=_text(row.get('source')), note=_text(row.get('note')),
                            match=row.get('match') if row.get('match') in ('auto', 'whole_word', 'substring') else 'auto')
            else:
                item.update(name=_text(row.get('name')), notes=_text(row.get('notes')))
            result[kind].append(item)
    characters = {row['id'] for row in result['characters']}
    seen = set()
    assignments = value.get('speakers', [])
    if isinstance(assignments, (list, tuple)):
        for row in assignments:
            if not isinstance(row, dict):
                continue
            page, region, character = (_text(row.get(key)) for key in ('page_id', 'region_id', 'character_id'))
            if page and region and character in characters and (page, region) not in seen:
                result['speakers'].append({'page_id': page, 'region_id': region, 'character_id': character})
                seen.add((page, region))
    return result


def normalized_text(text):
    return unicodedata.normalize('NFKC', _text(text)).casefold()


def canonical_language(value):
    normalized = normalized_text(value).strip().replace('_', '-')
    aliases = {
        'english': 'en', 'japanese': 'ja', 'chinese': 'zh', 'simplified chinese': 'zh', 'zh-cn': 'zh',
        'traditional chinese': 'zh-tw', 'korean': 'ko', 'vietnamese': 'vi', 'french': 'fr',
        'german': 'de', 'spanish': 'es', 'italian': 'it', 'russian': 'ru', 'turkish': 'tr',
        'portuguese': 'pt-pt', 'pt': 'pt-pt', 'european portuguese': 'pt-pt',
        'portuguese (portugal)': 'pt-pt', 'european portuguese (portugal)': 'pt-pt',
        'brazilian portuguese': 'pt-br', 'portuguese (brazil)': 'pt-br',
        'brazilian portuguese (brazil)': 'pt-br',
    }
    return aliases.get(normalized, normalized)


def language_matches(entry, source_lang, target_lang):
    for key, actual in (('source_lang', source_lang), ('target_lang', target_lang)):
        required = canonical_language(entry.get(key))
        if required and required not in ('*', 'any', 'all') and required != canonical_language(actual):
            return False
    return True


def term_matches(term, text, rule='auto', source_lang=''):
    term, text = normalized_text(term).strip(), normalized_text(text)
    if not term:
        return False
    # These scripts normally do not delimit words with spaces. For mixed Auto
    # pages, the term itself is enough to choose the appropriate boundary rule.
    no_spaces = canonical_language(source_lang) in ('ja', 'zh', 'zh-tw', 'th') or any(
        '\u3040' <= char <= '\u30ff' or '\u3400' <= char <= '\u9fff' or '\u0e00' <= char <= '\u0e7f'
        for char in term)
    if rule == 'substring' or (rule == 'auto' and no_spaces):
        return term in text
    return re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text, re.UNICODE) is not None


def compile_context(profile, blocks, source_lang, target_lang, *, page_id=None):
    """Select only relevant approved reference data; return a JSON-safe dict."""
    profile = normalize_translation_profile(profile)
    texts = [_text(getattr(block, 'text', '')) for block in blocks]
    glossary = []
    for entry in profile['glossary']:
        if not entry['enabled'] or not entry['approved'] or not language_matches(entry, source_lang, target_lang):
            continue
        matched = [term for term in [entry['source'], *entry['aliases']]
                   if any(term_matches(term, text, entry['match'], source_lang) for text in texts)]
        if matched and entry['target'].strip():
            glossary.append({'source': entry['source'], 'matched_terms': sorted(set(matched)),
                             'preferred_translation': entry['target'], 'note': entry['note']})
    glossary.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
    characters = {row['id']: row for row in profile['characters']
                  if row['enabled'] and row['approved'] and language_matches(row, source_lang, target_lang)}
    speakers = {}
    for index, block in enumerate(blocks):
        region_id = _text(getattr(block, 'region_id', ''))
        block_page = page_id or getattr(block, '_page_id', None)
        assignments = [row for row in profile['speakers'] if row['region_id'] == region_id
                       and (block_page is None or row['page_id'] == block_page)]
        # Ambiguous region ownership must not guess a character.
        if len(assignments) != 1:
            continue
        character = characters.get(assignments[0]['character_id'])
        if character is not None:
            speakers[f'block_{index}'] = {'name': character['name'], 'preferred_translation': character['target'],
                                         'voice_notes': character['notes']}
    result = {}
    if glossary:
        result['glossary'] = glossary
    if speakers:
        result['speakers'] = speakers
    if profile['notes'].strip():
        result['project_notes'] = profile['notes']
    if profile['summary'].strip():
        result['user_written_summary'] = profile['summary']
    return result
