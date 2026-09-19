from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtCore import QCoreApplication
from typing import TYPE_CHECKING
from modules.inpainting.lama import LaMa
from modules.inpainting.mi_gan import MIGAN
from modules.inpainting.aot import AOT
from modules.inpainting.schema import Config
from app.ui.messages import Messages
from app.ui.settings.settings_page import SettingsPage
from modules.translation.llm.llama_server import resolve_llama_server
from modules.translation.providers import provider_for_translator

if TYPE_CHECKING:
    from controller import ComicTranslate

inpaint_map = {
    "LaMa": LaMa,
    "MI-GAN": MIGAN,
    "AOT": AOT,
}


def get_inpainter_backend(inpainter_key: str) -> str:
    inpainter_cls = inpaint_map[inpainter_key]
    return getattr(inpainter_cls, "preferred_backend", "onnx")

def get_config(settings_page: SettingsPage):
    strategy_settings = settings_page.get_hd_strategy_settings()
    if strategy_settings['strategy'] == settings_page.ui.tr("Resize"):
        config = Config(hd_strategy="Resize", hd_strategy_resize_limit = strategy_settings['resize_limit'])
    elif strategy_settings['strategy'] == settings_page.ui.tr("Crop"):
        config = Config(hd_strategy="Crop", hd_strategy_crop_margin = strategy_settings['crop_margin'],
                        hd_strategy_crop_trigger_size = strategy_settings['crop_trigger_size'])
    else:
        config = Config(hd_strategy="Original")

    return config

def validate_ocr(main: ComicTranslate):
    """Validate local OCR or direct provider credentials."""
    settings_page = main.settings_page
    settings = settings_page.get_all_settings()
    ocr_tool = settings['tools']['ocr']

    if not ocr_tool:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Text Recognition model"))
        return False
    
    normalized_ocr = settings_page.ui.value_mappings.get(ocr_tool, ocr_tool)
    if normalized_ocr == "Default":
        return True
    service = "Microsoft Azure" if normalized_ocr == "Microsoft OCR" else "Google Gemini"
    fields = ("api_key_ocr", "endpoint") if service == "Microsoft Azure" else ("api_key",)
    return _validate_provider(main, service, fields)


def validate_translator(main: ComicTranslate, target_lang: str):
    """Validate local runtime or the selected direct provider."""
    settings_page = main.settings_page
    tr = settings_page.ui.tr
    settings = settings_page.get_all_settings()
    credentials = settings.get('credentials', {})
    translator_tool = settings['tools']['translator']

    if not translator_tool:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Translator"))
        return False

    normalized_translator = settings_page.ui.value_mappings.get(translator_tool, translator_tool)
    if normalized_translator == "Local LLM":
        local = settings.get('llm', {})
        runtime = local.get('local_runtime', 'managed')
        problems = []

        if runtime == 'external':
            endpoint = str(local.get('local_endpoint', '')).strip()
            parsed = urlparse(endpoint)
            if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
                problems.append("Enter a valid HTTP(S) OpenAI-compatible Base URL.")
            if not str(local.get('local_model', '')).strip():
                problems.append("Enter the model name exposed by the external server.")
        else:
            try:
                resolve_llama_server(str(local.get('llama_server_path', '')))
            except FileNotFoundError as exc:
                problems.append(str(exc))

            model_value = str(local.get('llama_model_path', '')).strip()
            model_path = Path(os.path.expandvars(os.path.expanduser(model_value)))
            if not model_path.is_file():
                problems.append("Select an existing main GGUF model file.")

            mmproj_value = str(local.get('llama_mmproj_path', '')).strip()
            mmproj_path = Path(os.path.expandvars(os.path.expanduser(mmproj_value)))
            if mmproj_value and not mmproj_path.is_file():
                problems.append("The selected vision projector GGUF does not exist.")

        if problems:
            Messages.show_local_llm_not_configured_error(main, "\n".join(problems))
            return False
        return True

    provider = provider_for_translator(normalized_translator)
    if provider is None:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("ToolsPage", "Translator"))
        return False
    return _validate_provider(main, provider, ("api_key", "model"))


def _validate_provider(main, service, fields):
    credentials = main.settings_page.get_credentials(service)
    if all(str(credentials.get(field) or "").strip() for field in fields):
        if "endpoint" not in fields or urlparse(credentials["endpoint"]).scheme == "https":
            return True
    Messages.show_error_with_copy(
        main, QCoreApplication.translate("Messages", "Provider APIs"),
        QCoreApplication.translate("Messages", "Configure the API key and required fields in Settings > Provider APIs: {provider}").format(provider=service),
    )
    return False

def font_selected(main: ComicTranslate):
    if not main.render_settings().font_family:
        Messages.select_font_error(main)
        return False
    return True

def validate_settings(main: ComicTranslate, target_lang: str):
    if not validate_ocr(main):
        return False
    if not validate_translator(main, target_lang):
        return False
    if not font_selected(main):
        return False
    
    return True
