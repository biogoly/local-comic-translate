"""Snapshot translation work and commit complete results by region identity."""
from __future__ import annotations

import copy
import hashlib
import json
from modules.translation.history import ApprovedHistory, capture_approved_history, history_is_current

from modules.utils.region_policy import permits_region_operation
from modules.utils.translator_utils import set_upper_case
from modules.translation.spatial import geometry_signature, spatial_context_active


class StaleTranslationError(RuntimeError):
    pass


def _revision(block, spatial=False):
    metadata = getattr(block, 'region_metadata', {})
    value = (block.region_id, getattr(block, 'text', '') or '',
            getattr(block, 'translation', '') or '',
            copy.deepcopy(metadata.get('protections', {})),
            metadata.get('review', {}).get('approved_fingerprints', {}).get('translation'))
    return value + (geometry_signature(block),) if spatial else value


def _epoch(main):
    return (getattr(main, '_translation_project_epoch', 0),
            getattr(main, '_translation_cancel_epoch', 0))


class TranslationJob:
    """Inference touches copies only; the GUI commits against current identities.

    The translator is created on the GUI thread and snapshots its engine/profile.
    A loader may defer original-image decoding until the worker runs.
    """
    def __init__(self, main, translator, blocks, image, extra_context, cache,
                 *, resolve_blocks, path=None, require_current_page=False,
                 upper_case=False, image_loader=None):
        self.main, self.translator, self.cache = main, translator, cache
        self.blocks = [block.deep_copy() for block in blocks]
        for original, copied in zip(blocks, self.blocks):
            if isinstance(getattr(original, '_page_id', None), str):
                copied._page_id = original._page_id
        self.before = {block.region_id: _revision(block) for block in blocks}
        if len(self.before) != len(self.blocks):
            raise ValueError('Translation requires unique region identities.')
        self.resolve_blocks = resolve_blocks
        self.spatial = spatial_context_active(getattr(translator, 'engine', None))
        # Request copies may be in cropped-image coordinates. Capture the live
        # geometry separately, after the caller has restored scene coordinates.
        live = {block.region_id: block for block in resolve_blocks()} if self.spatial else {}
        self.geometry_before = {identity: geometry_signature(live[identity])
                                for identity in self.before if identity in live} if self.spatial else {}
        if self.spatial and set(self.geometry_before) != set(self.before):
            raise StaleTranslationError('Translation regions changed before the request could be prepared.')
        self.image = image.copy() if image is not None else None
        self.image_loader = image_loader
        self.extra_context = str(extra_context)
        self.path, self.require_current_page = path, require_current_page
        self.page_id = getattr(main, 'image_states', {}).get(path, {}).get('page_id')
        self.epoch = _epoch(main)
        self.profile = copy.deepcopy(getattr(main, 'translation_profile', {}))
        self.upper_case = upper_case
        self.request = None
        self.key = None
        self.result = None
        for block in self.blocks:
            if self.page_id and not getattr(block, '_page_id', None):
                block._page_id = self.page_id

    def run(self):
        if self.epoch != _epoch(self.main):
            raise StaleTranslationError('Translation was cancelled or its project was closed.')
        if self.image is None and self.image_loader is not None:
            self.image = self.image_loader()
            if self.image is not None:
                self.image = self.image.copy()
        if self.image is None:
            raise RuntimeError('The original page image is unavailable for translation.')
        self.request = self.translator.prepare_request(self.blocks, self.image, self.extra_context)
        if not history_is_current(self.request, self.main):
            raise StaleTranslationError('Approved translation history changed. Please retry.')
        self.key = self.cache._get_translation_cache_key(
            self.image, self.translator.source_lang, self.translator.target_lang,
            self.translator.translator_key, self.extra_context, request=self.request)
        if self.cache._can_serve_all_blocks_from_translation_cache(self.key, self.blocks):
            self.cache._apply_cached_translations_to_blocks(self.key, self.blocks)
        else:
            self.translator.translate(self.blocks, self.image, self.extra_context, request=self.request)
            self.cache._cache_translation_results(self.key, self.blocks)
        set_upper_case(self.blocks, self.upper_case)
        self.result = {block.region_id: block.translation for block in self.blocks}
        return self

    def commit(self):
        """No partial application if a page, source, target or protection changed."""
        if self.result is None:
            return False
        valid = (self.epoch == _epoch(self.main) and self.profile == getattr(self.main, 'translation_profile', {})
                 and history_is_current(self.request, self.main))
        if self.path is not None:
            valid = valid and self.path in self.main.image_files
            valid = valid and self.main.image_states.get(self.path, {}).get('page_id') == self.page_id
            if self.require_current_page:
                index = self.main.curr_img_idx
                valid = valid and 0 <= index < len(self.main.image_files) and self.main.image_files[index] == self.path
        current = list(self.resolve_blocks()) if valid else []
        by_id = {block.region_id: block for block in current}
        valid = valid and len(by_id) == len(current)
        valid = valid and all(identity in by_id and _revision(by_id[identity]) == before
                              for identity, before in self.before.items())
        if valid and self.request is not None and self.request.spatial_reference_json:
            try:
                valid = all(geometry_signature(by_id[identity]) == before
                            for identity, before in self.geometry_before.items())
            except ValueError:
                valid = False
        if not valid:
            raise StaleTranslationError('Translation inputs changed while the request was running. Please retry.')
        for identity, target in self.result.items():
            by_id[identity].translation = target
        # Keep the saved page mirror in sync without replacing live geometry or
        # formatting. Multi-page UI refreshes consume these saved block objects.
        owners = {block.region_id: getattr(block, '_page_id', self.page_id) for block in self.blocks}
        for path, state in getattr(self.main, 'image_states', {}).items():
            for block in state.get('blk_list', []):
                if block.region_id in self.result and (
                        (owners[block.region_id] and owners[block.region_id] == state.get('page_id'))
                        or (not owners[block.region_id] and path == self.path)):
                    block.translation = self.result[block.region_id]
        return True


def eligible_translation_blocks(blocks, *, single_block=None):
    from modules.utils.manual_regions import has_manual_translation
    if single_block is not None:
        return [single_block]
    return [block for block in blocks if not has_manual_translation(block)
            and permits_region_operation(block, 'translation')]


def speaker_region_ids(main, page_id=None):
    profile = getattr(main, 'translation_profile', {})
    return {row['region_id'] for row in profile.get('speakers', [])
            if page_id is None or row['page_id'] == page_id}


def _page_signature(main, path, spatial=False):
    state = getattr(main, 'image_states', {}).get(path, {})
    blocks = state.get('blk_list', [])
    # In normal mode the current canvas is authoritative, including deletions.
    files, index = getattr(main, 'image_files', []), getattr(main, 'curr_img_idx', -1)
    if (not getattr(main, 'webtoon_mode', False) and 0 <= index < len(files)
            and files[index] == path and hasattr(main, 'blk_list')):
        blocks = main.blk_list
    value = (state.get('page_id'), state.get('source_lang'), state.get('target_lang'),
             sorted((_revision(block, spatial) for block in blocks), key=lambda row: row[0]))
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class BatchTranslationSession:
    """Freeze provider/profile choices before a batch worker starts."""
    def __init__(self, main, paths, translator_factory):
        self.main, self.epoch = main, _epoch(main)
        self.paths = tuple(paths)
        self.profile = copy.deepcopy(getattr(main, 'translation_profile', {}))
        self.extra_context = main.settings_page.get_llm_settings()['extra_context']
        self.translators = {}
        self.history_guards = {}
        self.factory = translator_factory
        for path in self.paths:
            state = main.image_states[path]
            pair = state['source_lang'], state['target_lang']
            if pair not in self.translators:
                self.translators[pair] = translator_factory(main, *pair)
        self.spatial_paths = {path for path in self.paths if spatial_context_active(getattr(
            self.translators[(main.image_states[path]['source_lang'], main.image_states[path]['target_lang'])],
            'engine', None))}
        self.revisions = {path: _page_signature(main, path, path in self.spatial_paths) for path in self.paths}
        self.live_geometry = {}
        if getattr(main, 'webtoon_mode', False):
            from pipeline.translation_geometry import loaded_page_geometry
            for path in self.spatial_paths:
                expected = {block.region_id: geometry_signature(block)
                            for block in main.image_states[path].get('blk_list', [])}
                expected.update(loaded_page_geometry(main, path, expected))
                self.live_geometry[path] = expected

    def validate(self, path=None):
        if self.epoch != _epoch(self.main) or self.profile != getattr(self.main, 'translation_profile', {}):
            raise StaleTranslationError('Translation was cancelled or its project context changed. Please retry.')
        if path in self.revisions:
            try:
                valid = path in self.main.image_files and _page_signature(
                    self.main, path, path in self.spatial_paths) == self.revisions[path]
                if valid and path in self.live_geometry:
                    from pipeline.translation_geometry import loaded_page_geometry
                    expected = self.live_geometry[path]
                    loaded = loaded_page_geometry(self.main, path, expected)
                    valid = all(geometry == expected[identity] for identity, geometry in loaded.items())
            except ValueError:
                valid = False
            if not valid:
                raise StaleTranslationError('This page changed while translation was running. Please retry.')
        if path in self.history_guards:
            current_history = capture_approved_history(self.main)
            if not all(guard.matches(current_history) for guard in self.history_guards[path]):
                raise StaleTranslationError('Approved translation history changed while translation was running. Please retry.')

    def mark_published(self, path):
        """Advance only after our own successful incremental webtoon update."""
        self.revisions[path] = _page_signature(self.main, path, path in self.spatial_paths)
        if path in self.live_geometry:
            # Publish updates saved physical geometry before scene signals are
            # delivered. Do not capture the old live copies as the new baseline.
            self.live_geometry[path] = {block.region_id: geometry_signature(block)
                for block in self.main.image_states[path].get('blk_list', [])}

    def translate(self, blocks, image, source_lang, target_lang, path, cache):
        self.validate(path)
        translator = self.translators.get((source_lang, target_lang))
        if translator is None:
            # Direct worker/test entry points can supply a language pair that
            # was not part of a GUI batch. Normal GUI batches prepare all pairs.
            translator = self.factory(self.main, source_lang, target_lang)
            translator.translation_profile = copy.deepcopy(self.profile)
        spatial = spatial_context_active(getattr(translator, 'engine', None))
        before = {block.region_id: _revision(block, spatial) for block in blocks}
        if len(before) != len(blocks):
            raise StaleTranslationError('Translation requires unique region identities.')
        copies = [block.deep_copy() for block in blocks]
        for original, copied in zip(blocks, copies):
            if isinstance(getattr(original, '_page_id', None), str):
                copied._page_id = original._page_id
        page_id = self.main.image_states.get(path, {}).get('page_id')
        for block in copies:
            if not getattr(block, '_page_id', None) and page_id:
                block._page_id = page_id
        if isinstance(getattr(translator, 'approved_history', None), ApprovedHistory):
            # Earlier pages may already have been published by this batch.
            # Snapshot the still-reviewed wording at this request's boundary.
            request = translator.prepare_request(copies, image, self.extra_context,
                history=capture_approved_history(self.main))
            # A physical webtoon page may have multiple virtual requests with
            # different available context budgets. Validate every contribution.
            self.history_guards.setdefault(path, set()).add(request.history_guard)
        else:
            request = translator.prepare_request(copies, image, self.extra_context)
        key = cache._get_translation_cache_key(image, source_lang, target_lang,
                                               translator.translator_key, self.extra_context, request=request)
        if cache._can_serve_all_blocks_from_translation_cache(key, copies):
            cache._apply_cached_translations_to_blocks(key, copies)
        else:
            translator.translate(copies, image, self.extra_context, request=request)
            cache._cache_translation_results(key, copies)
        self.validate(path)
        # The processor also stages/validates each response; this outer stage
        # prevents a closed/changed project from receiving otherwise valid text.
        by_id = {block.region_id: block for block in blocks}
        try:
            valid = (len(by_id) == len(blocks) and set(by_id) == set(before)
                     and all(_revision(by_id[identity], spatial) == revision
                             for identity, revision in before.items()))
        except ValueError:
            valid = False
        if not valid:
            raise StaleTranslationError('Translation regions changed while the request was running. Please retry.')
        for result in copies:
            by_id[result.region_id].translation = result.translation
        return request
