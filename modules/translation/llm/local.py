"""Local multimodal translation through an OpenAI-compatible HTTP API."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import requests

from .base import BaseLLMTranslation
from .llama_server import (
    LlamaServerConfig,
    MANAGED_MODEL_ALIAS,
    get_llama_server_runtime,
)
from .local_response import (
    TranslationResponseError,
    build_chat_payload,
    chat_completions_url,
    extract_message_content,
    request_valid_translation,
)
from ...utils.textblock import TextBlock
from ...utils.translator_utils import get_raw_text


class LocalLLMTranslation(BaseLLMTranslation):
    """Translate locally using managed llama.cpp or another compatible server."""

    MAX_RESPONSE_ATTEMPTS = 3

    def __init__(self) -> None:
        super().__init__()
        self.runtime_mode = "managed"
        self.api_base_url = ""
        self.api_key = ""
        self.model = MANAGED_MODEL_ALIAS
        self.llama_server_path = ""
        self.llama_model_path = ""
        self.llama_mmproj_path = ""
        self.context_size = 8192
        self.gpu_layers = 999
        self.startup_timeout = 90
        self.top_k = 40
        self.supports_images = True

    def initialize(
        self,
        settings: Any,
        source_lang: str,
        target_lang: str,
        tr_key: str = "Local LLM",
        **kwargs,
    ) -> None:
        super().initialize(settings, source_lang, target_lang, **kwargs)
        local = settings.get_llm_settings()

        self.runtime_mode = local.get("local_runtime", "managed")
        if self.runtime_mode not in {"managed", "external"}:
            self.runtime_mode = "managed"

        self.api_base_url = str(local.get("local_endpoint", "")).strip().rstrip("/")
        self.api_key = str(local.get("local_api_key", "")).strip()
        self.model = (
            MANAGED_MODEL_ALIAS
            if self.runtime_mode == "managed"
            else str(local.get("local_model", "")).strip()
        )
        self.llama_server_path = str(local.get("llama_server_path", "")).strip()
        self.llama_model_path = str(local.get("llama_model_path", "")).strip()
        self.llama_mmproj_path = str(local.get("llama_mmproj_path", "")).strip()
        self.context_size = _as_int(local.get("local_context_size"), 8192, minimum=1024)
        self.gpu_layers = _as_int(local.get("local_gpu_layers"), 999, minimum=0)
        self.startup_timeout = _as_int(local.get("local_startup_timeout"), 90, minimum=5)
        self.timeout = _as_int(local.get("local_request_timeout"), 300, minimum=10)
        self.max_tokens = _as_int(local.get("local_max_tokens"), 4096, minimum=256)
        self.temperature = _as_float(local.get("local_temperature"), 0.2, minimum=0.0, maximum=2.0)
        self.top_p = _as_float(local.get("local_top_p"), 0.9, minimum=0.0, maximum=1.0)
        self.top_k = _as_int(local.get("local_top_k"), 40, minimum=0)

    def translate(
        self,
        blk_list: list[TextBlock],
        image: np.ndarray | None,
        extra_context: str,
    ) -> list[TextBlock]:
        if not blk_list:
            return blk_list

        expected_keys = [f"block_{index}" for index in range(len(blk_list))]
        raw_text = get_raw_text(blk_list)
        system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
        system_prompt += (
            "\nReturn exactly one valid JSON object. Its keys must be exactly: "
            f"{', '.join(expected_keys)}. Every value must be a non-empty translated string. "
            "Never leave a value blank; preserve a name or symbol verbatim when it should not change."
        )
        base_user_prompt = (
            f"{extra_context.strip()}\n"
            "Use the page image for visual context when one is provided. Correct obvious OCR mistakes "
            "from that context, then make the translation sound natural.\n"
            f"Translate this JSON:\n{raw_text}"
        ).strip()

        try:
            translations = request_valid_translation(
                lambda prompt: self._perform_translation(prompt, system_prompt, image),
                base_user_prompt,
                expected_keys,
                max_attempts=self.MAX_RESPONSE_ATTEMPTS,
                required_content_keys=expected_keys,
            )
        except TranslationResponseError as exc:
            raise RuntimeError(f"The local model returned an {exc}") from exc

        # Commit only after the whole response validates so a failed local
        # request never leaves a page with partially shifted translations.
        for index, block in enumerate(blk_list):
            block.translation = translations[f"block_{index}"]
        return blk_list

    def _perform_translation(
        self,
        user_prompt: str,
        system_prompt: str,
        image: np.ndarray | None,
    ) -> str:
        api_base_url = self._get_api_base_url()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        image_data_url = None
        if self.supports_images and self.img_as_llm_input and image is not None:
            # Preserve small glyph edges for the model's OCR-correction pass.
            encoded_image, mime_type = self.encode_image(image, ext=".png")
            image_data_url = f"data:{mime_type};base64,{encoded_image}"

        payload = build_chat_payload(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_data_url=image_data_url,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )

        url = chat_completions_url(api_base_url)
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as exc:
            details = ""
            if exc.response is not None:
                try:
                    details = f" Server response: {json.dumps(exc.response.json())}"
                except (ValueError, TypeError):
                    details = f" HTTP status: {exc.response.status_code}."
            raise RuntimeError(f"Local LLM request failed: {exc}.{details}") from exc
        except ValueError as exc:
            raise RuntimeError("The local LLM server returned invalid response JSON.") from exc

        try:
            return extract_message_content(response_data)
        except TranslationResponseError as exc:
            raise RuntimeError(f"Invalid local LLM API response: {exc}") from exc

    def _get_api_base_url(self) -> str:
        if self.runtime_mode == "external":
            if not self.api_base_url:
                raise RuntimeError("No external Local LLM endpoint is configured.")
            return self.api_base_url

        config = LlamaServerConfig(
            executable_path=self.llama_server_path,
            model_path=self.llama_model_path,
            mmproj_path=self.llama_mmproj_path,
            context_size=self.context_size,
            gpu_layers=self.gpu_layers,
            startup_timeout=self.startup_timeout,
        )
        return get_llama_server_runtime().ensure_running(config)


def _as_int(value: Any, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        return min(maximum, max(minimum, float(value)))
    except (TypeError, ValueError):
        return default
