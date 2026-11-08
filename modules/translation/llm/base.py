from typing import Any
import numpy as np
from abc import abstractmethod
import base64
import imkit as imk

from ..base import LLMTranslation
from ...utils.textblock import TextBlock
from ..request import build_request
from .local_response import request_valid_translation


class BaseLLMTranslation(LLMTranslation):
    """Base class for LLM-based translation engines with shared functionality."""
    MAX_RESPONSE_ATTEMPTS = 1
    
    def __init__(self):
        self.source_lang = None
        self.target_lang = None
        self.api_key = None
        self.api_url = None
        self.model = None
        self.img_as_llm_input = False
        self.spatial_context_enabled = False
        self.temperature = None
        self.top_p = None
        self.max_tokens = None
        self.timeout = 120
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the LLM translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        """
        llm_settings = settings.get_llm_settings()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.img_as_llm_input = llm_settings.get('image_input_enabled', True)
        self.spatial_context_enabled = bool(llm_settings.get('spatial_context_enabled', False))
        self.temperature = 1.0
        self.top_p = 0.95
        self.max_tokens = 5000
        
    def translate(self, blk_list: list[TextBlock], image: np.ndarray, extra_context: str) -> list[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        if not blk_list:
            return blk_list
        request = build_request(self, blk_list, image, extra_context)
        translations = self.translate_request(request)
        by_id = request.assert_current(blk_list)
        for index, identity in enumerate(request.region_ids):
            by_id[identity].translation = translations[request.expected_keys[index]]
        return blk_list

    def translate_request(self, request):
        """All providers validate a whole immutable request before any apply."""
        if not request.expected_keys:
            return {}
        return request_valid_translation(
            lambda prompt: self._perform_translation(prompt, request.system_prompt, request.image),
            request.user_prompt,
            request.expected_keys,
            max_attempts=self.MAX_RESPONSE_ATTEMPTS,
            required_content_keys=request.expected_keys,
        )
    
    def complete_text_once(self, user_prompt: str, system_prompt: str) -> str:
        """One text-only completion through this configured provider."""
        return self._perform_translation(user_prompt, system_prompt, None)

    @abstractmethod
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using specific LLM.
        
        Args:
            user_prompt: User prompt for LLM
            system_prompt: System prompt for LLM
            image: Image as numpy array
            
        Returns:
            Translated JSON text
        """
        pass

    def encode_image(self, image: np.ndarray, ext=".jpg"):
        """
        Encode CV2/numpy image directly to base64 string using cv2.imencode.
        
        Args:
            image: Numpy array representing the image
            ext: Extension/format to encode the image as (".png" by default for higher quality)
                
        Returns:
            Tuple of (Base64 encoded string, mime_type)
        """
        # Direct encoding from numpy/cv2 format to bytes
        buffer = imk.encode_image(image, ext.lstrip('.'))
        
        # Convert to base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Map extension to mime type
        mime_types = {
            ".jpg": "image/jpeg", 
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext.lower(), f"image/{ext[1:].lower()}")
        
        return img_str, mime_type
