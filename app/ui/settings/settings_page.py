import os, shutil
import logging
from dataclasses import asdict, is_dataclass

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Signal, QSettings, Qt
from PySide6.QtGui import QFont, QFontDatabase

from app.shortcuts import get_default_shortcuts
from .settings_ui import SettingsPageUI
from modules.utils.device import is_gpu_available
from modules.translation.providers import credential_service, provider_for_translator
from modules.utils.translator_utils import MODEL_MAP
from app.update_checker import UpdateChecker
from modules.utils.paths import get_user_data_dir, get_default_project_autosave_dir
from modules.utils.settings import app_settings, migrate_custom_endpoint


logger = logging.getLogger(__name__)


class SettingsPage(QtWidgets.QWidget):
    theme_changed = Signal(str)
    font_imported = Signal(str)

    def __init__(self, parent=None):
        super(SettingsPage, self).__init__(parent)

        self.ui = SettingsPageUI(self)
        self._setup_connections()
        self._loading_settings = False
        self._is_background_check = False
        self._current_language = None  # Track current language for revert

        # Update Checker
        self.update_checker = UpdateChecker()
        self.update_checker.update_available.connect(self.on_update_available)
        self.update_checker.up_to_date.connect(self.on_up_to_date)
        self.update_checker.error_occurred.connect(self.on_update_error)
        self.update_checker.download_progress.connect(self.on_download_progress)
        self.update_checker.download_finished.connect(self.on_download_finished)
        self.update_dialog = None
        # Use the Settings UI directly; inner content is scrollable on the
        # right side (see settings_ui.py). This keeps the left navbar fixed.
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.ui)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _setup_connections(self):
        # Connect signals to slots
        self.ui.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        self.ui.lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.ui.translator_combo.currentTextChanged.connect(self._sync_extra_context_limit)
        self.ui.font_browser.sig_files_changed.connect(self.import_font)
        self.ui.shortcuts_page.shortcut_changed.connect(self.on_shortcut_changed)
        self.ui.check_update_button.clicked.connect(self.check_for_updates)
        self._sync_extra_context_limit(self.ui.translator_combo.currentText())

    def _sync_extra_context_limit(self, translator: str) -> None:
        self.ui.llms_page.set_extra_context_unlimited(True)
        self.ui.llms_page.set_local_settings_visible(True)

    def on_theme_changed(self, theme: str):
        self.theme_changed.emit(theme)

    def get_language(self):
        return self.ui.lang_combo.currentText()
    
    def get_theme(self):
        return self.ui.theme_combo.currentText()

    def get_tool_selection(self, tool_type):
        tool_combos = {
            'translator': self.ui.translator_combo,
            'ocr': self.ui.ocr_combo,
            'inpainter': self.ui.inpainter_combo,
            'detector': self.ui.detector_combo
        }
        return tool_combos[tool_type].currentText()

    def is_gpu_enabled(self):
        if not is_gpu_available():
            return False
        return self.ui.use_gpu_checkbox.isChecked()

    def get_llm_settings(self):
        local_page = self.ui.llms_page
        return {
            'extra_context': self.ui.extra_context.toPlainText(),
            'image_input_enabled': self.ui.image_checkbox.isChecked(),
            'local_runtime': local_page.local_runtime_combo.currentData() or 'managed',
            'local_endpoint': local_page.local_endpoint.text().strip(),
            'local_model': local_page.local_model.text().strip(),
            'local_api_key': local_page.local_api_key.text(),
            'llama_server_path': local_page.llama_server_path.text().strip(),
            'llama_model_path': local_page.llama_model_path.text().strip(),
            'llama_mmproj_path': local_page.llama_mmproj_path.text().strip(),
            'local_context_size': local_page.local_context_size.value(),
            'local_gpu_layers': local_page.local_gpu_layers.value(),
            'local_startup_timeout': local_page.local_startup_timeout.value(),
            'local_request_timeout': local_page.local_request_timeout.value(),
            'local_max_tokens': local_page.local_max_tokens.value(),
            'local_temperature': local_page.local_temperature.value(),
            'local_top_p': local_page.local_top_p.value(),
            'local_top_k': local_page.local_top_k.value(),
        }

    def get_export_settings(self):
        owner = self.window()
        title_bar = getattr(owner, "title_bar", None)
        settings = app_settings()
        settings.beginGroup('export')
        persisted_autosave_enabled = settings.value('project_autosave_enabled', False, type=bool)
        settings.endGroup()
        if title_bar is not None:
            autosave_enabled = bool(title_bar.autosave_switch.isChecked())
        else:
            autosave_enabled = bool(persisted_autosave_enabled)
        autosave_folder = self.ui.project_autosave_folder_input.text().strip()
        if not autosave_folder:
            autosave_folder = get_default_project_autosave_dir()
        settings = {
            'export_raw_text': self.ui.raw_text_checkbox.isChecked(),
            'export_translated_text': self.ui.translated_text_checkbox.isChecked(),
            'export_inpainted_image': self.ui.inpainted_image_checkbox.isChecked(),
            'project_autosave_enabled': autosave_enabled,
            'project_autosave_interval_min': int(self.ui.project_autosave_interval_spinbox.value()),
            'project_autosave_folder': autosave_folder,
        }
        return settings

    def get_credentials(self, service: str = ""):
        save_keys = self.ui.save_keys_checkbox.isChecked()

        def _text_or_none(widget_key):
            w = self.ui.credential_widgets.get(widget_key)
            return w.text() if w is not None else None

        if service:
            normalized = credential_service(self.ui.value_mappings.get(service, service))
            prefix = normalized + "_"
            return {
                "save_key": save_keys,
                **{key[len(prefix):]: (widget.currentText() if isinstance(widget, QtWidgets.QComboBox) else widget.text()).strip()
                   for key, widget in self.ui.credential_widgets.items() if key.startswith(prefix)},
            }

        # no `service` passed → recurse over all known services
        return {s: self.get_credentials(s) for s in self.ui.credential_services}
        
    def get_hd_strategy_settings(self):
        strategy = self.ui.inpaint_strategy_combo.currentText()
        settings = {
            'strategy': strategy
        }

        if strategy == self.ui.tr("Resize"):
            settings['resize_limit'] = self.ui.resize_spinbox.value()
        elif strategy == self.ui.tr("Crop"):
            settings['crop_margin'] = self.ui.crop_margin_spinbox.value()
            settings['crop_trigger_size'] = self.ui.crop_trigger_spinbox.value()

        return settings
    
    def get_all_settings(self):
        return {
            'language': self.get_language(),
            'theme': self.get_theme(),
            'tools': {
                'translator': self.get_tool_selection('translator'),
                'ocr': self.get_tool_selection('ocr'),
                'detector': self.get_tool_selection('detector'),
                'inpainter': self.get_tool_selection('inpainter'),
                'use_gpu': self.is_gpu_enabled(),
                'hd_strategy': self.get_hd_strategy_settings()
            },
            'llm': self.get_llm_settings(),
            'export': self.get_export_settings(),
            'shortcuts': self.ui.shortcuts_page.get_shortcuts(),
            'credentials': self.get_credentials(),
            'save_keys': self.ui.save_keys_checkbox.isChecked()
        }

    def on_shortcut_changed(self, shortcut_id: str, sequence: str) -> None:
        if self._loading_settings:
            return

        settings = app_settings()
        settings.beginGroup("shortcuts")
        settings.setValue(shortcut_id, sequence)
        settings.endGroup()

        owner = self.window()
        shortcut_ctrl = getattr(owner, "shortcut_ctrl", None)
        if shortcut_ctrl is not None:
            shortcut_ctrl.apply_shortcuts()

    def import_font(self, file_paths: list[str]):

        file_paths = [f for f in file_paths 
                      if f.lower().endswith((".ttf", ".ttc", ".otf", ".woff", ".woff2"))]
        
        # Determine user font directory
        user_font_dir = os.path.join(get_user_data_dir(), "fonts")

        if not os.path.exists(user_font_dir):
            os.makedirs(user_font_dir, exist_ok=True)

        if file_paths:
            loaded_families = []
            for src in file_paths:
                dst = os.path.join(user_font_dir, os.path.basename(src))
                if os.path.normcase(src) != os.path.normcase(dst):
                    shutil.copy(src, dst)
                font_family = self.add_font_family(dst)
                if font_family:
                    loaded_families.append(font_family)

            if loaded_families:
                self.font_imported.emit(loaded_families[0])

    def select_color(self, outline = False):
        default_color = QtGui.QColor('#000000') if not outline else QtGui.QColor('#FFFFFF')
        color_dialog = QtWidgets.QColorDialog()
        color_dialog.setCurrentColor(default_color)
        
        if color_dialog.exec() == QtWidgets.QDialog.Accepted:
            color = color_dialog.selectedColor()
            if color.isValid():
                button = self.ui.color_button if not outline else self.ui.outline_color_button
                button.setStyleSheet(
                    f"background-color: {color.name()}; border: none; border-radius: 5px;"
                )
                button.setProperty('selected_color', color.name())

    # With the mappings, settings are saved with English values and loaded in the selected language
    def save_settings(self):
        settings = app_settings()
        all_settings = self.get_all_settings()

        # Treat an external local-server token like the other optional API keys.
        # The live value is still available to the running translator, but it is
        # persisted only when the user enabled Save Keys.
        if not self.ui.save_keys_checkbox.isChecked():
            all_settings.get('llm', {}).update({'local_api_key': ''})

        def process_group(group_key, group_value, settings_obj: QSettings):
            """Helper function to process a group and its nested values."""
            if is_dataclass(group_value):
                group_value = asdict(group_value)
            if isinstance(group_value, dict):
                settings_obj.beginGroup(group_key)
                for sub_key, sub_value in group_value.items():
                    process_group(sub_key, sub_value, settings_obj)
                settings_obj.endGroup()
            else:
                # Convert value to English using mappings if available
                mapped_value = self.ui.value_mappings.get(group_value, group_value)
                settings_obj.setValue(group_key, mapped_value)

        for key, value in all_settings.items():
            if key != 'credentials':
                process_group(key, value, settings)

        # Remove deprecated export keys from older versions.
        settings.beginGroup('export')
        settings.remove('auto_save')
        settings.remove('archive_save_as')
        settings.endGroup()

        # Remove all old/nested key copies, including the former Custom endpoint.
        settings.remove('credentials')
        settings.beginGroup('credentials')
        save_keys = self.ui.save_keys_checkbox.isChecked()
        settings.setValue('save_keys', save_keys)
        for key, widget in self.ui.credential_widgets.items():
            value = widget.currentText() if isinstance(widget, QtWidgets.QComboBox) else widget.text()
            if (not key.endswith('_api_key') and not key.endswith('_api_key_ocr')) or save_keys:
                settings.setValue(key, value)
        settings.endGroup()

    def load_settings(self):
        self._loading_settings = True
        settings = app_settings()
        migrate_custom_endpoint(settings)

        # Load language
        language = settings.value('language', 'English')
        translated_language = self.ui.reverse_mappings.get(language, language)
        self.ui.lang_combo.setCurrentText(translated_language)

        # Load theme
        theme = settings.value('theme', 'Dark')
        translated_theme = self.ui.reverse_mappings.get(theme, theme)
        self.ui.theme_combo.setCurrentText(translated_theme)
        self.theme_changed.emit(translated_theme)

        # Load tools settings
        settings.beginGroup('tools')
        legacy_translator = settings.value('translator', 'Local LLM')
        translator = provider_for_translator(legacy_translator) or 'Local LLM'
        translated_translator = self.ui.reverse_mappings.get(translator, translator)
        if self.ui.translator_combo.findText(translated_translator) != -1:
            self.ui.translator_combo.setCurrentText(translated_translator)
        else:
            self.ui.translator_combo.setCurrentIndex(-1)

        ocr = settings.value('ocr', 'Default')
        translated_ocr = self.ui.reverse_mappings.get(ocr, ocr)
        if self.ui.ocr_combo.findText(translated_ocr) != -1:
            self.ui.ocr_combo.setCurrentText(translated_ocr)
        else:
            self.ui.ocr_combo.setCurrentIndex(-1)

        inpainter = settings.value('inpainter', 'AOT')
        translated_inpainter = self.ui.reverse_mappings.get(inpainter, inpainter)
        if self.ui.inpainter_combo.findText(translated_inpainter) != -1:
            self.ui.inpainter_combo.setCurrentText(translated_inpainter)
        else:
            self.ui.inpainter_combo.setCurrentIndex(-1)

        detector = settings.value('detector', 'RT-DETR-v2')
        translated_detector = self.ui.reverse_mappings.get(detector, detector)
        if self.ui.detector_combo.findText(translated_detector) != -1:
            self.ui.detector_combo.setCurrentText(translated_detector)
        else:
            self.ui.detector_combo.setCurrentIndex(-1)

        if is_gpu_available():
            self.ui.use_gpu_checkbox.setChecked(settings.value('use_gpu', False, type=bool))
        else:
            self.ui.use_gpu_checkbox.setChecked(False)

        # Load HD strategy settings
        settings.beginGroup('hd_strategy')
        strategy = settings.value('strategy', 'Resize')
        translated_strategy = self.ui.reverse_mappings.get(strategy, strategy)
        if self.ui.inpaint_strategy_combo.findText(translated_strategy) != -1:
            self.ui.inpaint_strategy_combo.setCurrentText(translated_strategy)
        else:
            self.ui.inpaint_strategy_combo.setCurrentIndex(0)

        if strategy == 'Resize':
            self.ui.resize_spinbox.setValue(settings.value('resize_limit', 960, type=int))
        elif strategy == 'Crop':
            self.ui.crop_margin_spinbox.setValue(settings.value('crop_margin', 512, type=int))
            self.ui.crop_trigger_spinbox.setValue(settings.value('crop_trigger_size', 512, type=int))
        settings.endGroup()  # hd_strategy
        settings.endGroup()  # tools

        # Load LLM settings
        settings.beginGroup('llm')
        self.ui.extra_context.setPlainText(settings.value('extra_context', ''))
        self.ui.image_checkbox.setChecked(settings.value('image_input_enabled', False, type=bool))
        local_page = self.ui.llms_page
        local_page.set_local_runtime(settings.value('local_runtime', 'managed', type=str))
        local_page.local_endpoint.setText(settings.value('local_endpoint', 'http://127.0.0.1:11434/v1', type=str))
        local_page.local_model.setText(settings.value('local_model', 'gemma4:e4b', type=str))
        local_page.local_api_key.setText(settings.value('local_api_key', '', type=str))
        local_page.llama_server_path.setText(settings.value('llama_server_path', '', type=str))
        local_page.llama_model_path.setText(settings.value('llama_model_path', '', type=str))
        local_page.llama_mmproj_path.setText(settings.value('llama_mmproj_path', '', type=str))
        local_page.local_context_size.setValue(settings.value('local_context_size', 8192, type=int))
        local_page.local_gpu_layers.setValue(settings.value('local_gpu_layers', 999, type=int))
        local_page.local_startup_timeout.setValue(settings.value('local_startup_timeout', 90, type=int))
        local_page.local_request_timeout.setValue(settings.value('local_request_timeout', 300, type=int))
        local_page.local_max_tokens.setValue(settings.value('local_max_tokens', 4096, type=int))
        local_page.local_temperature.setValue(settings.value('local_temperature', 0.2, type=float))
        local_page.local_top_p.setValue(settings.value('local_top_p', 0.9, type=float))
        local_page.local_top_k.setValue(settings.value('local_top_k', 40, type=int))
        settings.endGroup()

        # Load export settings
        settings.beginGroup('export')
        self.ui.raw_text_checkbox.setChecked(settings.value('export_raw_text', False, type=bool))
        self.ui.translated_text_checkbox.setChecked(settings.value('export_translated_text', False, type=bool))
        self.ui.inpainted_image_checkbox.setChecked(settings.value('export_inpainted_image', False, type=bool))
        autosave_enabled = settings.value('project_autosave_enabled', False, type=bool)
        owner = self.parent()
        title_bar = getattr(owner, "title_bar", None)
        if title_bar is not None:
            title_bar.set_autosave_checked(bool(autosave_enabled))
        self.ui.project_autosave_interval_spinbox.setValue(
            settings.value('project_autosave_interval_min', 3, type=int)
        )
        self.ui.project_autosave_folder_input.setText(
            settings.value('project_autosave_folder', get_default_project_autosave_dir(), type=str)
        )

        settings.endGroup()  # export

        settings.beginGroup('shortcuts')
        default_shortcuts = get_default_shortcuts()
        shortcut_values = {}
        for shortcut_id, default_value in default_shortcuts.items():
            shortcut_values[shortcut_id] = settings.value(shortcut_id, default_value, type=str)
        settings.endGroup()
        self.ui.shortcuts_page.load_shortcuts(shortcut_values)
        owner = self.window()
        shortcut_ctrl = getattr(owner, "shortcut_ctrl", None)
        if shortcut_ctrl is not None:
            shortcut_ctrl.apply_shortcuts()

        # Model choices and non-secret fields are always retained; keys are opt-in.
        settings.beginGroup('credentials')
        save_keys = settings.value('save_keys', False, type=bool)
        self.ui.save_keys_checkbox.setChecked(save_keys)
        for key, widget in self.ui.credential_widgets.items():
            secret = key.endswith('_api_key') or key.endswith('_api_key_ocr')
            default = widget.currentText() if isinstance(widget, QtWidgets.QComboBox) else ''
            value = str(settings.value(key, default) or '') if save_keys or not secret else ''
            if isinstance(widget, QtWidgets.QComboBox):
                # Preserve the precise named model when migrating the old selector.
                service = credential_service(legacy_translator)
                if key == service + '_model' and not settings.contains(key) and legacy_translator in MODEL_MAP:
                    value = MODEL_MAP[legacy_translator]
                widget.setCurrentText(value)
            else:
                widget.setText(value)
        settings.endGroup()

        # Initialize current language tracker after loading
        self._current_language = self.ui.lang_combo.currentText()

        self._loading_settings = False


    def on_language_changed(self, new_language):
        if not self._loading_settings:  
            # Pass the previous language so we can revert if needed
            self.show_restart_dialog(new_language)

    def _show_message_box(self, icon: QtWidgets.QMessageBox.Icon, title: str, text: str):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        ok_btn = msg_box.addButton(self.tr("OK"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        msg_box.setDefaultButton(ok_btn)
        msg_box.exec()

    def _ask_yes_no(self, title: str, text: str, default_yes: bool = False) -> bool:
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        yes_btn = msg_box.addButton(self.tr("Yes"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        no_btn = msg_box.addButton(self.tr("No"), QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg_box.setDefaultButton(yes_btn if default_yes else no_btn)
        msg_box.exec()
        return msg_box.clickedButton() == yes_btn

    def show_restart_dialog(self, new_language):
        from modules.utils.common_utils import restart_application
        
        response = self._ask_yes_no(
            self.tr("Restart Required"),
            self.tr("The application needs to restart for the language changes to take effect.\nRestart now?"),
            default_yes=True
        )
        
        if response:
            # Save settings before restarting
            self.save_settings()
            self._current_language = new_language  # Update tracking
            restart_application()
        else:
            # User declined - revert to previous language
            self._loading_settings = True  # Prevent triggering the handler again
            self.ui.lang_combo.setCurrentText(self._current_language)
            self._loading_settings = False



    def get_min_font_size(self):
        return int(self.ui.min_font_spinbox.value())
    
    def get_max_font_size(self):
        return int(self.ui.max_font_spinbox.value())

    def add_font_family(self, font_input: str) -> QFont:
        # Check if font_input is a file path
        if os.path.splitext(font_input)[1].lower() in [".ttf", ".ttc", ".otf", ".woff", ".woff2"]:
            font_id = QFontDatabase.addApplicationFont(font_input)
            if font_id != -1:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                if font_families:
                    return font_families[0]
        
        # If not a file path or loading failed, treat as font family name
        return font_input
    
    def check_for_updates(self, is_background=False):
        self._is_background_check = is_background
        if not is_background:
            self.ui.check_update_button.setEnabled(False)
            self.ui.check_update_button.setText(self.tr("Checking..."))
        self.update_checker.check_for_updates()

    def on_update_available(self, version, release_url, download_url):
        if not self._is_background_check:
            self.ui.check_update_button.setEnabled(True)
            self.ui.check_update_button.setText(self.tr("Check for Updates"))
        
        # Check ignored version
        settings = app_settings()
        ignored_version = settings.value("updates/ignored_version", "")
        
        if self._is_background_check and version == ignored_version:
            return

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle(self.tr("Update Available"))
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg_box.setText(self.tr("A new version {version} is available.").format(version=version))
        link_text = self.tr("Release Notes")
        msg_box.setInformativeText(f'<a href="{release_url}" style="color: #4da6ff;">{link_text}</a>')
        
        download_btn = msg_box.addButton(self.tr("Yes"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg_box.addButton(self.tr("No"), QtWidgets.QMessageBox.ButtonRole.RejectRole)
        
        dotted_ask_btn = None
        if self._is_background_check:
            dotted_ask_btn = msg_box.addButton(self.tr("Skip This Version"), QtWidgets.QMessageBox.ButtonRole.ApplyRole)
        
        msg_box.setDefaultButton(download_btn)
        msg_box.exec()

        if msg_box.clickedButton() == download_btn:
            self.start_download(download_url)
        elif dotted_ask_btn and msg_box.clickedButton() == dotted_ask_btn:
            settings.setValue("updates/ignored_version", version)
    
    def on_up_to_date(self):
        if self._is_background_check:
            return

        self.ui.check_update_button.setEnabled(True)
        self.ui.check_update_button.setText(self.tr("Check for Updates"))
        self._show_message_box(
            QtWidgets.QMessageBox.Icon.Information,
            self.tr("Up to Date"),
            self.tr("You are using the latest version.")
        )

    def on_update_error(self, message):
        if self._is_background_check:
            logger.error(f"Background update check failed: {message}")
            return

        self.ui.check_update_button.setEnabled(True)
        self.ui.check_update_button.setText(self.tr("Check for Updates"))
        if self.update_dialog:
             self.update_dialog.close()
        
        self._show_message_box(
            QtWidgets.QMessageBox.Icon.Warning,
            self.tr("Update Error"),
            message
        )

    def start_download(self, url):
        # Create a progress dialog
        self.update_dialog = QtWidgets.QProgressDialog(self.tr("Downloading update..."), self.tr("Cancel"), 0, 100, self)
        self.update_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.update_dialog.show()
        
        filename = url.split("/")[-1]
        self.update_checker.download_installer(url, filename)

    def on_download_progress(self, percent):
        if self.update_dialog:
             self.update_dialog.setValue(percent)

    def on_download_finished(self, file_path):
        if self.update_dialog:
             self.update_dialog.close()
        
        # Ask to install
        if self._ask_yes_no(
            self.tr("Download Complete"),
            self.tr("Installer downloaded to {path}. Run it now?").format(path=file_path),
            default_yes=True
        ):
             self.update_checker.run_installer(file_path)

    def shutdown(self):
        """Cleanup resources before app exit."""
        if getattr(self, "_is_shutting_down", False):
            return
        self._is_shutting_down = True
        self.update_checker.shutdown()
    



