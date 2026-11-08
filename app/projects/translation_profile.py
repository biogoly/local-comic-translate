"""Project-only translation context lifecycle and portable speaker references."""
from __future__ import annotations

from modules.translation.context import empty_translation_profile, normalize_translation_profile


def reset_translation_project(main) -> None:
    """Invalidate outstanding requests before a project is cleared or loaded.

    This helper is data-only because project codecs also run in a worker.
    The GUI controller separately replaces its draft and undo history.
    """
    main._translation_project_epoch = getattr(main, '_translation_project_epoch', 0) + 1
    main.translation_profile = empty_translation_profile()


def profile_for_project(main, profile=None) -> dict:
    """Return a sanitized independent profile with only owned speaker links."""
    result = normalize_translation_profile(
        getattr(main, 'translation_profile', None) if profile is None else profile
    )
    regions = {
        (state.get('page_id'), getattr(block, 'region_id', None))
        for state in getattr(main, 'image_states', {}).values() if isinstance(state, dict)
        for block in state.get('blk_list', []) or []
    }
    characters = {entry['id'] for entry in result['characters']}
    result['speakers'] = [entry for entry in result['speakers']
                          if (entry['page_id'], entry['region_id']) in regions
                          and entry['character_id'] in characters]
    return result
