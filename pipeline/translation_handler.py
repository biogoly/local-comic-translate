from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from modules.translation.processor import Translator
from modules.utils.region_policy import is_protected
from modules.utils.language_utils import to_canonical_language_name
from modules.translation.spatial import spatial_context_active
from pipeline.webtoon_utils import filter_and_convert_visible_blocks, restore_original_block_coordinates
from .cache_manager import CacheManager

if TYPE_CHECKING:
    from controller import ComicTranslate
    from .main_pipeline import ComicTranslatePipeline

logger = logging.getLogger(__name__)


def refresh_kept_translation_cache(cache_manager, cache_key, blocks, *, exclude=None):
    """Keep user-approved wording authoritative, including an empty target."""
    for block in blocks:
        if block is exclude or not is_protected(block, 'keep_translation'):
            continue
        if getattr(block, 'translation', ''):
            cache_manager.update_translation_cache_for_block(cache_key, block)
        else:
            # Do not resurrect an old target when this protection is removed.
            matched, _ = cache_manager._find_matching_translation_block_id(cache_key, block)
            while matched is not None:
                del cache_manager.translation_cache[cache_key][matched]
                matched, _ = cache_manager._find_matching_translation_block_id(cache_key, block)


class TranslationHandler:
    """Handles translation processing with caching support."""
    
    def __init__(
            self, 
            main_page: ComicTranslate, 
            cache_manager: CacheManager, 
            pipeline: ComicTranslatePipeline,
        ):
        
        self.main_page = main_page
        self.cache_manager = cache_manager
        self.pipeline = pipeline

    def _refresh_kept_translation_cache(self, cache_key, *, exclude=None):
        refresh_kept_translation_cache(self.cache_manager, cache_key, self.main_page.blk_list, exclude=exclude)

    def prepare_job(self, single_block=False, *, visible=False):
        """Capture GUI state before queuing a translation worker."""
        from pipeline.translation_job import TranslationJob, eligible_translation_blocks
        main = self.main_page
        if not main.image_viewer.hasPhoto() or not main.blk_list:
            return None
        source_lang = to_canonical_language_name(main.s_combo.currentText(), main.lang_mapping)
        target_lang = to_canonical_language_name(main.t_combo.currentText(), main.lang_mapping)
        selected = self.pipeline.get_selected_block() if single_block else None
        if single_block and selected is None:
            return None
        translator = Translator(main, source_lang, target_lang) if visible else None
        if visible and spatial_context_active(translator.engine):
            from pipeline.translation_geometry import capture_visible_translation
            image, visible_blocks = capture_visible_translation(main, self.pipeline, single_block)
            if image is None:
                return None
            blocks = visible_blocks if single_block else eligible_translation_blocks(visible_blocks)
        elif visible:
            review = getattr(main, 'review_ctrl', None)
            owners = {block.region_id: review._live_path(block) for block in main.blk_list} if review else {}
            image, mappings = main.image_viewer.get_visible_area_image(include_patches=False)
            if image is None or not mappings:
                return None
            visible_blocks = filter_and_convert_visible_blocks(main, self.pipeline, mappings, single_block)
            try:
                blocks = [b.deep_copy() for b in eligible_translation_blocks(
                    visible_blocks, single_block=selected if single_block else None)]
            finally:
                restore_original_block_coordinates(visible_blocks)
            for block in blocks:
                owner = owners.get(block.region_id)
                if owner is not None:
                    block._page_id = main.image_states.get(owner, {}).get('page_id')
        else:
            image = main.image_viewer.get_image_array(include_patches=False)
            blocks = eligible_translation_blocks(main.blk_list, single_block=selected)
        settings = main.settings_page
        extra = settings.get_llm_settings()['extra_context']
        if not blocks:
            # Legacy entries are discarded for deliberately empty kept targets.
            legacy_key = self.cache_manager._get_translation_cache_key(
                image, source_lang, target_lang, settings.get_tool_selection('translator'), extra)
            self._refresh_kept_translation_cache(legacy_key)
            return None
        paths = getattr(main, 'image_files', [])
        index = getattr(main, 'curr_img_idx', -1)
        path = paths[index] if 0 <= index < len(paths) else None
        translator = translator or Translator(main, source_lang, target_lang)
        return TranslationJob(main, translator, blocks, image, extra, self.cache_manager,
            resolve_blocks=lambda: main.blk_list, path=path, require_current_page=True,
            upper_case=settings.ui.uppercase_checkbox.isChecked())

    def translate_image(self, single_block=False):
        """Synchronous compatibility entry point; GUI callers queue prepared jobs."""
        job = self.prepare_job(single_block)
        if job is not None:
            job.run().commit()

    def translate_webtoon_visible_area(self, single_block=False):
        job = self.prepare_job(single_block, visible=True)
        if job is not None:
            job.run().commit()
