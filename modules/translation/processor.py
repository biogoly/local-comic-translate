import numpy as np
import copy

from ..utils.textblock import TextBlock
from ..utils.translator_utils import has_translatable_content
from .base import LLMTranslation
from .factory import TranslationFactory
from .context import normalize_translation_profile
from .history import capture_approved_history, history_is_current
from .request import build_request, StaleTranslationRequest


class Translator:
    """
    Main translator class that orchestrates the translation process.
    
    Supports multiple translation engines including:
    - Traditional translators (e.g Google, Microsoft, DeepL, Yandex)
    - Direct provider APIs and Local LLM
    """
    
    def __init__(self, main_page, source_lang: str = "", target_lang: str = ""):
        """
        Initialize translator with settings and languages.
        
        Args:
            main_page: Main application page with settings
            source_lang: Source language name (localized)
            target_lang: Target language name (localized)
        """
        self.main_page = main_page
        self.settings = main_page.settings_page
        self.translation_profile = normalize_translation_profile(getattr(main_page, 'translation_profile', None))
        
        self.translator_key = self._get_translator_key(self.settings.get_tool_selection('translator'))
        
        self.source_lang = source_lang
        self.source_lang_en = self._get_english_lang(main_page, self.source_lang)
        self.target_lang = target_lang
        self.target_lang_en = self._get_english_lang(main_page, self.target_lang)
        
        # Create appropriate engine using factory
        self.engine = copy.copy(TranslationFactory.create_engine(
            self.settings,
            self.source_lang_en,
            self.target_lang_en,
            self.translator_key
        ))
        # Traditional engines keep language codes rather than both names.
        # A shallow private engine copy freezes configuration without copying
        # provider client/session internals or changing the factory's cache.
        self.engine.source_lang = self.source_lang_en
        self.engine.target_lang = self.target_lang_en
        
        # Track engine type for method dispatching
        self.is_llm_engine = isinstance(self.engine, LLMTranslation)
        self.approved_history = (capture_approved_history(main_page)
                                 if self.is_llm_engine and self.translation_profile['include_approved_history'] else None)
    
    def _get_translator_key(self, localized_translator: str) -> str:
        """
        Map localized translator names to standard keys.
        
        Args:
            localized_translator: Translator name in UI language
            
        Returns:
            Standard translator key
        """
        translator_map = {
            self.settings.ui.tr("Google Gemini"): "Google Gemini",
            self.settings.ui.tr("Anthropic Claude"): "Anthropic Claude",
            "OpenAI": "OpenAI",
            self.settings.ui.tr("Local LLM"): "Local LLM",
            self.settings.ui.tr("Deepseek"): "Deepseek",
            self.settings.ui.tr("GPT-4.1"): "GPT-4.1",
            self.settings.ui.tr("GPT-4.1-mini"): "GPT-4.1-mini",
            self.settings.ui.tr("Claude-4.6-Sonnet"): "Claude-4.6-Sonnet",
            self.settings.ui.tr("Claude-4.5-Haiku"): "Claude-4.5-Haiku",
            self.settings.ui.tr("Gemini-3.1-Flash-Lite"): "Gemini-3.1-Flash-Lite",
            self.settings.ui.tr("Gemini-2.5-Pro"): "Gemini-2.5-Pro",
            self.settings.ui.tr("Microsoft Translator"): "Microsoft Translator",
            self.settings.ui.tr("DeepL"): "DeepL",
            self.settings.ui.tr("Yandex"): "Yandex"
        }
        return translator_map.get(localized_translator, localized_translator)
    
    def _get_english_lang(self, main_page, translated_lang: str) -> str:
        """
        Get English language name from localized language name.
        
        Args:
            main_page: Main application page with language mapping
            translated_lang: Language name in UI language
            
        Returns:
            Language name in English
        """
        return main_page.lang_mapping.get(translated_lang, translated_lang)
    
    def prepare_request(self, blk_list, image=None, extra_context='', *, profile=None, page_id=None, history=None):
        blocks = [block for block in blk_list if has_translatable_content(getattr(block, 'text', ''))]
        return build_request(self.engine, blocks, image, extra_context,
                             profile=self.translation_profile if profile is None else profile,
                             provider=self.translator_key, page_id=page_id,
                             history=self.approved_history if history is None else history,
                             structured_context_supported=self.is_llm_engine)

    def translate(self, blk_list: list[TextBlock], image: np.ndarray = None, extra_context: str = "", *, request=None) -> list[TextBlock]:
        """
        Translate text in text blocks using the configured translation engine.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array (for context in LLM translators)
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        original = list(blk_list)
        translatable_blocks = [block for block in original if has_translatable_content(getattr(block, 'text', ''))]
        request = request or self.prepare_request(translatable_blocks, image, extra_context)
        if not history_is_current(request, self.main_page):
            raise StaleTranslationRequest('Approved translation history changed. Retry the translation.')
        request_blocks = request.assert_current(translatable_blocks)
        before = [(block, getattr(block, 'region_id', ''), block.text, block.translation) for block in original]
        if self.is_llm_engine:
            translations = self.engine.translate_request(request)
        else:
            copies = [request_blocks[identity].deep_copy() for identity in request.region_ids]
            if copies:
                self.engine.translate(copies)
            translations = {f'block_{index}': block.translation for index, block in enumerate(copies)}
        # No field is changed until the complete request succeeds and every
        # original input is still current; reordering is handled by stable IDs.
        if len(blk_list) != len(original) or {id(block) for block in blk_list} != {id(block) for block in original}:
            raise StaleTranslationRequest('Translation regions changed while the request was running. Retry the translation.')
        for block, identity, source, target in before:
            if (getattr(block, 'region_id', ''), block.text, block.translation) != (identity, source, target):
                raise StaleTranslationRequest('Source or target text changed while the request was running. Retry the translation.')
        by_id = request.assert_current(translatable_blocks)
        if not history_is_current(request, self.main_page):
            raise StaleTranslationRequest('Approved translation history changed. Retry the translation.')
        for index, identity in enumerate(request.region_ids):
            by_id[identity].translation = translations[request.expected_keys[index]]
        for block in original:
            if not has_translatable_content(block.text):
                block.translation = ''
        return blk_list
