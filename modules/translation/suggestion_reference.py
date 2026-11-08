"""Bounded, disclosed text references shared by project context suggestions."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from ipaddress import ip_address
import json
from urllib.parse import urlsplit

from .llm.base import BaseLLMTranslation
from .providers import provider_for_translator
from .request import engine_identity, estimate_tokens, sanitized_endpoint


MAX_EXAMPLES = 24
MAX_REFERENCE_TOKENS = 4096
MAX_OUTPUT_TOKENS = 512


class SuggestionError(ValueError):
    """Stable, nonsecret error code for localization by the caller."""
    def __init__(self, code):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SuggestionExample:
    source: str
    target: str


def _destination(engine, provider):
    if provider != 'Local LLM':
        if not engine.api_key:
            raise SuggestionError('credentials')
        return provider, False
    if engine.runtime_mode == 'managed':
        return 'llama.cpp', True
    endpoint = urlsplit(sanitized_endpoint(engine.api_base_url))
    if not endpoint.hostname or endpoint.scheme not in ('http', 'https'):
        raise SuggestionError('credentials')
    host = endpoint.hostname.lower()
    try:
        loopback = ip_address(host).is_loopback
    except ValueError:
        loopback = host.rstrip('.') == 'localhost'
    return f'{endpoint.scheme}://{endpoint.netloc}', loopback


def capture_engine(translator, system_prompt, context_token_limit):
    """Capture a private, text-only adapter and reserve its reply/context overhead."""
    key = translator.translator_key
    provider = 'Local LLM' if key == 'Local LLM' else provider_for_translator(key)
    if not provider or not isinstance(translator.engine, BaseLLMTranslation):
        raise SuggestionError('unsupported')
    engine = copy.copy(translator.engine)
    if isinstance(getattr(engine, 'headers', None), dict):
        engine.headers = engine.headers.copy()
    engine.img_as_llm_input = False
    engine.max_tokens = min(MAX_OUTPUT_TOKENS, int(engine.max_tokens or MAX_OUTPUT_TOKENS))
    destination, local = _destination(engine, provider)
    limit = min(MAX_REFERENCE_TOKENS, int(context_token_limit))
    window = getattr(engine, 'context_size', None)
    if window:
        engine.max_tokens = min(engine.max_tokens, max(1, int(window) // 4))
        limit = min(limit, int(window) - estimate_tokens(system_prompt) - engine.max_tokens - 256)
    return engine, provider, destination, local, limit


def select_reference(approved_examples, limit, serialize):
    """Keep recent whole examples that fit, in their original chronological order."""
    selected, identities = [], []
    found = False
    for entry in reversed(approved_examples):
        found = True
        if len(entry.source) + len(entry.target) > limit * 2:
            continue
        example = SuggestionExample(entry.source, entry.target)
        if estimate_tokens(serialize([example, *selected])) > limit:
            continue
        selected.insert(0, example)
        identities.insert(0, (entry.page_id, entry.region_id))
        if len(selected) == MAX_EXAMPLES:
            break
    if not selected:
        raise SuggestionError('budget' if found else 'no_examples')
    return tuple(selected), tuple(identities), serialize(selected)


def request_fingerprint(engine, provider, system_prompt, user_prompt, identities, *local_scope):
    signature = json.dumps([engine_identity(engine, provider), system_prompt, user_prompt, identities, *local_scope],
                           ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(signature.encode('utf-8')).hexdigest()


def complete_reference(engine, user_prompt, system_prompt):
    """Send one request. Native transport failures may contain credentials or OCR text."""
    try:
        return engine.complete_text_once(user_prompt, system_prompt)
    except Exception:
        raise SuggestionError('request') from None


def decode_object(response):
    if not isinstance(response, str) or len(response) > 16000:
        raise SuggestionError('response')
    text = response.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].lower() in ('```', '```json') and lines[-1] == '```':
        text = '\n'.join(lines[1:-1])

    def unique_keys(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError
        return value

    try:
        obj = json.loads(text, object_pairs_hook=unique_keys)
    except (ValueError, RecursionError):
        raise SuggestionError('response') from None
    if not isinstance(obj, dict):
        raise SuggestionError('response')
    return obj
