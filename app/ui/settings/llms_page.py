from PySide6 import QtWidgets
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.text_edit import MTextEdit
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.spin_box import MDoubleSpinBox, MSpinBox

class LlmsPage(QtWidgets.QWidget):
    DEFAULT_EXTRA_CONTEXT_LIMIT = 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._extra_context_limit: int | None = self.DEFAULT_EXTRA_CONTEXT_LIMIT

        v = QtWidgets.QVBoxLayout(self)
        main_layout = QtWidgets.QHBoxLayout()

        self.image_checkbox = MCheckBox(self.tr("Provide Image as Input to AI"))
        self.image_checkbox.setChecked(False)

        # Left
        left_layout = QtWidgets.QVBoxLayout()
        prompt_label = MLabel(self.tr("Extra Context:"))
        self.extra_context = MTextEdit()
        self.extra_context.setMinimumHeight(200)
        left_layout.addWidget(prompt_label)
        left_layout.addWidget(self.extra_context)
        left_layout.addWidget(self.image_checkbox)
        left_layout.addStretch(1)

        # Right: settings shown only when Local LLM is selected.
        self.local_settings_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(self.local_settings_container)
        self.local_settings_widget = QtWidgets.QGroupBox(self.tr("Local LLM"))
        local_layout = QtWidgets.QVBoxLayout(self.local_settings_widget)

        description = MLabel(self.tr(
            "Run a GGUF model with llama.cpp, or connect to an existing "
            "OpenAI-compatible server such as Ollama or LM Studio."
        ))
        description.setWordWrap(True)
        local_layout.addWidget(description)

        runtime_row = QtWidgets.QFormLayout()
        self.local_runtime_combo = MComboBox().small()
        self.local_runtime_combo.addItem(self.tr("Managed llama.cpp"), "managed")
        self.local_runtime_combo.addItem(self.tr("External server"), "external")
        runtime_row.addRow(self.tr("Runtime:"), self.local_runtime_combo)
        local_layout.addLayout(runtime_row)

        self.managed_settings_widget = QtWidgets.QWidget()
        managed_form = QtWidgets.QFormLayout(self.managed_settings_widget)
        self.llama_server_path = MLineEdit().file()
        self.llama_server_path.setPlaceholderText(self.tr("Optional when llama-server is on PATH"))
        self.llama_model_path = MLineEdit().file([".gguf"])
        self.llama_model_path.setPlaceholderText(self.tr("Main GGUF model (required)"))
        self.llama_mmproj_path = MLineEdit().file([".gguf"])
        self.llama_mmproj_path.setPlaceholderText(self.tr("Vision projector GGUF, when required by the model"))

        self.local_context_size = MSpinBox().small()
        self.local_context_size.setRange(1024, 262144)
        self.local_context_size.setSingleStep(1024)
        self.local_context_size.setValue(8192)
        self.local_gpu_layers = MSpinBox().small()
        self.local_gpu_layers.setRange(0, 999)
        self.local_gpu_layers.setValue(999)
        self.local_startup_timeout = MSpinBox().small()
        self.local_startup_timeout.setRange(5, 600)
        self.local_startup_timeout.setValue(90)
        self.local_startup_timeout.setSuffix(self.tr(" s"))

        managed_form.addRow(self.tr("llama-server:"), self.llama_server_path)
        managed_form.addRow(self.tr("Model:"), self.llama_model_path)
        managed_form.addRow(self.tr("Vision projector:"), self.llama_mmproj_path)
        managed_form.addRow(self.tr("Context size:"), self.local_context_size)
        managed_form.addRow(self.tr("GPU layers:"), self.local_gpu_layers)
        managed_form.addRow(self.tr("Startup timeout:"), self.local_startup_timeout)
        local_layout.addWidget(self.managed_settings_widget)

        self.external_settings_widget = QtWidgets.QWidget()
        external_form = QtWidgets.QFormLayout(self.external_settings_widget)
        self.local_endpoint = MLineEdit("http://127.0.0.1:11434/v1")
        self.local_endpoint.setPlaceholderText("http://127.0.0.1:11434/v1")
        self.local_model = MLineEdit("gemma4:e4b")
        self.local_model.setPlaceholderText(self.tr("Server model name"))
        self.local_api_key = MLineEdit()
        self.local_api_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.local_api_key.setPlaceholderText(self.tr("Optional"))
        self.local_api_key.setToolTip(self.tr("Saved only when Save Keys is enabled in Advanced settings."))
        external_form.addRow(self.tr("Base URL:"), self.local_endpoint)
        external_form.addRow(self.tr("Model name:"), self.local_model)
        external_form.addRow(self.tr("API key:"), self.local_api_key)
        local_layout.addWidget(self.external_settings_widget)

        generation_group = QtWidgets.QGroupBox(self.tr("Generation"))
        generation_form = QtWidgets.QFormLayout(generation_group)
        self.local_max_tokens = MSpinBox().small()
        self.local_max_tokens.setRange(256, 32768)
        self.local_max_tokens.setSingleStep(256)
        self.local_max_tokens.setValue(4096)
        self.local_request_timeout = MSpinBox().small()
        self.local_request_timeout.setRange(10, 3600)
        self.local_request_timeout.setValue(300)
        self.local_request_timeout.setSuffix(self.tr(" s"))
        self.local_temperature = MDoubleSpinBox().small()
        self.local_temperature.setRange(0.0, 2.0)
        self.local_temperature.setSingleStep(0.1)
        self.local_temperature.setValue(0.2)
        self.local_top_p = MDoubleSpinBox().small()
        self.local_top_p.setRange(0.0, 1.0)
        self.local_top_p.setSingleStep(0.05)
        self.local_top_p.setValue(0.9)
        self.local_top_k = MSpinBox().small()
        self.local_top_k.setRange(0, 1000)
        self.local_top_k.setSingleStep(1)
        self.local_top_k.setValue(40)
        self.local_top_k.setToolTip(
            self.tr("Limit sampling to the K most likely tokens; 0 disables Top K")
        )
        generation_form.addRow(self.tr("Max output tokens:"), self.local_max_tokens)
        generation_form.addRow(self.tr("Request timeout:"), self.local_request_timeout)
        generation_form.addRow(self.tr("Temperature:"), self.local_temperature)
        generation_form.addRow(self.tr("Top P:"), self.local_top_p)
        generation_form.addRow(self.tr("Top K:"), self.local_top_k)
        local_layout.addWidget(generation_group)

        right_layout.addWidget(self.local_settings_widget)
        right_layout.addStretch(1)

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.local_settings_container, 1)

        v.addLayout(main_layout)
        v.addStretch(1)

        self.extra_context.textChanged.connect(self._limit_extra_context)
        self.local_runtime_combo.currentIndexChanged.connect(self._sync_runtime_widgets)
        self._sync_runtime_widgets()
        self.set_local_settings_visible(False)

    def set_local_settings_visible(self, visible: bool) -> None:
        self.local_settings_container.setVisible(visible)

    def set_local_runtime(self, runtime: str) -> None:
        index = self.local_runtime_combo.findData(runtime)
        self.local_runtime_combo.setCurrentIndex(index if index >= 0 else 0)

    def _sync_runtime_widgets(self, _index: int | None = None) -> None:
        managed = self.local_runtime_combo.currentData() == "managed"
        self.managed_settings_widget.setVisible(managed)
        self.external_settings_widget.setVisible(not managed)

    def set_extra_context_unlimited(self, enabled: bool) -> None:
        self._extra_context_limit = None if enabled else self.DEFAULT_EXTRA_CONTEXT_LIMIT
        self._limit_extra_context()

    def _limit_extra_context(self):
        max_length = self._extra_context_limit
        if max_length is None:
            return
        text = self.extra_context.toPlainText()
        if len(text) > max_length:
            # Preserve cursor position
            cursor = self.extra_context.textCursor()
            position = cursor.position()
            
            # Truncate
            self.extra_context.setPlainText(text[:max_length])
            
            # Restore cursor (clamped to end)
            new_position = min(position, max_length)
            cursor.setPosition(new_position)
            self.extra_context.setTextCursor(cursor)

