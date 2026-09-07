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
    """Ensure either API credentials are set or the user is authenticated."""
    settings_page = main.settings_page
    settings = settings_page.get_all_settings()
    ocr_tool = settings['tools']['ocr']

    if not ocr_tool:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Text Recognition model"))
        return False
    
    normalized_ocr = settings_page.ui.value_mappings.get(ocr_tool, ocr_tool)
    if normalized_ocr != "Default" and not settings_page.is_logged_in():
        Messages.show_not_logged_in_error(main)
        return False
        
    return True


def validate_translator(main: ComicTranslate, target_lang: str):
    """Ensure either API credentials are set or the user is authenticated, plus check compatibility."""
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

    if not settings_page.is_logged_in():
        Messages.show_not_logged_in_error(main)
        return False

    # Credential checks
    if "Custom" in translator_tool:
        # Custom requires api_key, api_url, and model to be configured LOCALLY
        service = tr('Custom')
        creds = credentials.get(service, {})
        # Check if all required fields are present and non-empty
        if not all([creds.get('api_key'), creds.get('api_url'), creds.get('model')]):
            Messages.show_custom_not_configured_error(main)
            return False
        return True
        
    return True

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
