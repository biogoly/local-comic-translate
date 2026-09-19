import logging
from modules.ocr.processor import OCRProcessor
from modules.utils.device import resolve_device
from modules.utils.manual_regions import has_manual_translation, is_manual_region
from modules.utils.language_utils import (
    to_canonical_language_name, get_dominant_page_script, is_supported_script,
)
from pipeline.webtoon_utils import filter_and_convert_visible_blocks, restore_original_block_coordinates

logger = logging.getLogger(__name__)


class OCRHandler:
    """Handles OCR processing with caching support."""
    
    def __init__(self, main_page, cache_manager, pipeline):
        self.main_page = main_page
        self.cache_manager = cache_manager
        self.pipeline = pipeline
        self.ocr = OCRProcessor()

    def OCR_image(self, single_block: bool = False):
        source_lang = to_canonical_language_name(
            self.main_page.s_combo.currentText(),
            self.main_page.lang_mapping,
        )
        if self.main_page.image_viewer.hasPhoto() and self.main_page.image_viewer.rectangles:
            image = self.main_page.image_viewer.get_image_array()
            ocr_model = self.main_page.settings_page.get_tool_selection('ocr')
            device = resolve_device(
                self.main_page.settings_page.is_gpu_enabled()
            )
            cache_key = self.cache_manager._get_ocr_cache_key(image, source_lang, ocr_model, device)
            
            if single_block:
                blk = self.pipeline.get_selected_block()
                if blk is None:
                    return
                
                # An explicit region OCR action is a retry, not a cache lookup.
                # Recognize only that region; don't OCR every other box first.
                self._recognize_selected_block(image, blk, source_lang)
                self.cache_manager.update_ocr_cache_for_block(cache_key, blk)
                logger.info("Re-recognized selected block and refreshed its OCR cache")
            else:
                # A completed manual region may already be cleaned underneath
                # its text layer. Page OCR must not replace its source with an
                # empty reading of that cleaned background.
                blocks = [blk for blk in self.main_page.blk_list if not has_manual_translation(blk)]
                # For full page OCR, check if we can use cached results
                if self.cache_manager._can_serve_all_blocks_from_ocr_cache(cache_key, blocks):
                    # All blocks can be served from cache
                    self.cache_manager._apply_cached_ocr_to_blocks(cache_key, blocks)
                    logger.info("Using cached OCR results for %d unfinished blocks", len(blocks))
                else:
                    # Need to run OCR and cache results
                    self.ocr.initialize(self.main_page, source_lang)
                    if blocks:
                        self.ocr.process(image, blocks)
                        self.cache_manager._cache_ocr_results(cache_key, blocks)
                        logger.info("OCR completed and cached for %d blocks", len(blocks))

    def _recognize_selected_block(self, image, blk, source_lang, context_blocks=None):
        # Work on a copy so a failed request cannot erase existing user text.
        working = blk.deep_copy()
        working.text = ""
        working.texts = []
        working.skipped_small_texts = []
        if source_lang == 'Auto' and not is_supported_script(working.script):
            # Keep the existing page-script fallback when narrowing recognition
            # to one box. This does not override an explicitly selected language.
            context = self.main_page.blk_list if context_blocks is None else context_blocks
            working.script = get_dominant_page_script(context)
        if is_manual_region(working):
            # Hand-drawn boxes may have moved/resized since their previous OCR.
            working.lines = None
            working.direction = ""
        self.ocr.initialize(self.main_page, source_lang)
        self.ocr.process(image, [working])
        for name in ('text', 'texts', 'skipped_small_texts', 'source_lang', 'lines', 'direction'):
            setattr(blk, name, getattr(working, name))

    def OCR_webtoon_visible_area(self, single_block: bool = False):
        """Perform OCR on the visible area in webtoon mode."""
        source_lang = to_canonical_language_name(
            self.main_page.s_combo.currentText(),
            self.main_page.lang_mapping,
        )
        
        if not (self.main_page.image_viewer.hasPhoto() and 
                self.main_page.webtoon_mode):
            logger.warning("OCR_webtoon_visible_area called but not in webtoon mode")
            return
        
        # Get the visible area image and mapping data
        visible_image, mappings = self.main_page.image_viewer.get_visible_area_image()
        if visible_image is None or not mappings:
            logger.warning("No visible area found for OCR")
            return
        
        # Filter blocks to only those in the visible area and convert coordinates
        visible_blocks = filter_and_convert_visible_blocks(
            self.main_page, self.pipeline, mappings, single_block
        )
        if not visible_blocks:
            logger.info("No blocks found in visible area")
            return
        
        # The shared coordinate converter moves outer boxes, not their line
        # metadata. Prepare temporary lines in visible-image coordinates; never
        # leave these attached to a block restored to webtoon scene coordinates.
        blocks = visible_blocks if single_block else [
            blk for blk in visible_blocks if not has_manual_translation(blk)
        ]
        line_states = [(blk, blk.lines, blk.direction) for blk in blocks]
        try:
            for blk in blocks:
                blk.lines = None
                blk.direction = ""
            if single_block:
                self._recognize_selected_block(
                    visible_image, blocks[0], source_lang, visible_blocks,
                )
            else:
                self.ocr.initialize(self.main_page, source_lang)
                if blocks:
                    self.ocr.process(visible_image, blocks)
        finally:
            for blk, lines, direction in line_states:
                blk.lines = lines
                blk.direction = direction
            restore_original_block_coordinates(visible_blocks)
        
        logger.info("OCR completed for %d blocks in visible area", len(blocks))
