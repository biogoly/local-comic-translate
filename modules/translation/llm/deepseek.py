from typing import Any

from .gpt import GPTTranslation
from ...utils.translator_utils import MODEL_MAP


class DeepseekTranslation(GPTTranslation):
    """Translation engine using Deepseek models with OpenAI-compatible API."""
    
    def __init__(self):
        super().__init__()
        self.supports_images = False
        self.api_base_url = "https://api.deepseek.com/v1"
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, model_name: str, **kwargs) -> None:
        """
        Initialize Deepseek translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            model_name: Deepseek model name
        """
        # Call BaseLLMTranslation's initialize
        super(GPTTranslation, self).initialize(settings, source_lang, target_lang, **kwargs)
        
        self.model_name = model_name
        credentials = settings.get_credentials(settings.ui.tr('Deepseek'))
        self.api_key = credentials.get('api_key', '')
        self.model = (credentials.get('model') or MODEL_MAP.get(self.model_name))
        self.supports_images = self.model == "deepseek-flash"

    def _make_api_request(self, payload, headers):
        payload = dict(payload)
        payload["max_tokens"] = payload.pop("max_completion_tokens")
        payload["thinking"] = {"type": "disabled"}
        payload["messages"] = [dict(message) for message in payload["messages"]]
        for message in payload["messages"]:
            content = message["content"]
            if isinstance(content, list) and all(part.get("type") == "text" for part in content):
                message["content"] = "\n".join(part["text"] for part in content)
        return super()._make_api_request(payload, headers)
