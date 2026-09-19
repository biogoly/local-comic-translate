from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.dayu_widgets import dayu_theme
from modules.artistic_edit.contracts import (
    DevicePolicy,
    ModelKey,
    SelectionMode,
)


@dataclass(frozen=True)
class ArtisticEditOptions:
    prompt: str
    model_key: ModelKey
    selection_mode: SelectionMode
    lora_index: int
    lora_scale: float
    seed: int
    random_seed: bool
    steps: int
    guidance: float
    context_ratio: float
    feather_radius: float
    maximum_side: int
    device_policy: DevicePolicy
    cuda_device_index: int
    edit_margin: int = 0
    backend: str = 'local'


class ArtisticEditPanel(QtWidgets.QWidget):
    """Compact, dependency-free controls for the optional FLUX worker."""

    use_selected_translation_requested = QtCore.Signal()
    generate_requested = QtCore.Signal()
    configure_runtime_requested = QtCore.Signal()
    cancel_requested = QtCore.Signal()
    unload_requested = QtCore.Signal()
    revert_requested = QtCore.Signal()
    refresh_loras_requested = QtCore.Signal()
    open_lora_folder_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._context_available = False
        self._context_reason = ""
        self._busy = False
        self._build_ui()
        self.apply_theme()
        self.prompt_edit.textChanged.connect(self._refresh_generate_button)
        self.area_combo.currentIndexChanged.connect(self._refresh_generate_button)
        self._refresh_generate_button()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItem(self.tr("Local Diffusers"), "local")
        self.backend_combo.addItem("Black Forest Labs API", "bfl")
        layout.addWidget(self.backend_combo)
        self.cloud_notice = QtWidgets.QLabel(self.tr(
            "Cloud editing uploads the selected area plus context and incurs provider charges. "
            "Configure the key in Provider APIs. Local LoRAs and GPU controls are unavailable."
        ))
        self.cloud_notice.setWordWrap(True)
        self.cloud_notice.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        self.cloud_notice.hide()
        layout.addWidget(self.cloud_notice)

        self.prompt_edit = QtWidgets.QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(self.tr("Describe the local image edit…"))
        self.prompt_edit.setMaximumBlockCount(80)
        self.prompt_edit.setMinimumHeight(72)
        self.prompt_edit.setMaximumHeight(120)
        layout.addWidget(self.prompt_edit)

        prompt_row = QtWidgets.QHBoxLayout()
        self.use_translation_button = QtWidgets.QPushButton(self.tr("Use selected translation"))
        self.prompt_template_combo = QtWidgets.QComboBox()
        self.prompt_template_combo.addItem(self.tr("Artistic lettering"), "lettering")
        self.prompt_template_combo.addItem(self.tr("Empty speech bubble"), "empty_bubble")
        prompt_row.addWidget(self.use_translation_button, 1)
        prompt_row.addWidget(self.prompt_template_combo, 1)
        layout.addLayout(prompt_row)

        # Keep the primary action and its disabled reason next to the prompt.
        # The tools sidebar can be quite short, and placing this after every
        # model option made the action look absent even though it was merely
        # below the visible fold (or disabled by a missing worker runtime).
        self.generate_button = QtWidgets.QPushButton(self.tr("Generate Preview"))
        self.generate_button.setMinimumHeight(32)
        layout.addWidget(self.generate_button)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        self.configure_runtime_button = QtWidgets.QPushButton(
            self.tr("Select FLUX Python…")
        )
        self.configure_runtime_button.hide()
        layout.addWidget(self.configure_runtime_button)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        model_form = QtWidgets.QFormLayout()
        model_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        model_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItem(
            self.tr("4B · Apache 2.0 · ~13 GB VRAM"), ModelKey.KLEIN_4B.value
        )
        self.model_combo.addItem(
            self.tr("9B · Non-commercial · gated · ~29 GB VRAM"), ModelKey.KLEIN_9B.value
        )
        model_form.addRow(self.tr("Model"), self.model_combo)

        lora_widget = QtWidgets.QWidget()
        lora_row = QtWidgets.QHBoxLayout(lora_widget)
        lora_row.setContentsMargins(0, 0, 0, 0)
        lora_row.setSpacing(4)
        self.lora_combo = QtWidgets.QComboBox()
        self.lora_combo.addItem(self.tr("None"), None)
        self.lora_strength = QtWidgets.QDoubleSpinBox()
        self.lora_strength.setRange(0.0, 2.0)
        self.lora_strength.setSingleStep(0.05)
        self.lora_strength.setDecimals(2)
        self.lora_strength.setValue(1.0)
        self.lora_strength.setEnabled(False)
        self.lora_strength.setMaximumWidth(72)
        lora_row.addWidget(self.lora_combo, 1)
        lora_row.addWidget(self.lora_strength)
        model_form.addRow(self.tr("LoRA / strength"), lora_widget)

        area_widget = QtWidgets.QWidget()
        area_layout = QtWidgets.QVBoxLayout(area_widget)
        area_layout.setContentsMargins(0, 0, 0, 0)
        area_layout.setSpacing(4)
        self.area_combo = QtWidgets.QComboBox()
        self.area_combo.addItem(self.tr("Selected box"), SelectionMode.SELECTED_BOX.value)
        self.area_combo.addItem(self.tr("Painted area"), SelectionMode.PAINTED_AREA.value)
        self.area_combo.addItem(self.tr("Whole page"), SelectionMode.WHOLE_PAGE.value)
        self.refresh_lora_button = QtWidgets.QToolButton()
        self.refresh_lora_button.setText(self.tr("Refresh"))
        self.refresh_lora_button.setToolTip(self.tr("Refresh compatible LoRAs"))
        self.open_lora_button = QtWidgets.QToolButton()
        self.open_lora_button.setText(self.tr("Folder"))
        self.open_lora_button.setToolTip(self.tr("Open the compatible LoRA folder"))
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.setContentsMargins(0, 0, 0, 0)
        folder_row.addWidget(self.refresh_lora_button, 1)
        folder_row.addWidget(self.open_lora_button, 1)
        area_layout.addWidget(self.area_combo)
        area_layout.addLayout(folder_row)
        model_form.addRow(self.tr("Area"), area_widget)
        self.edit_margin_spin = QtWidgets.QSpinBox()
        self.edit_margin_spin.setRange(0, 512)
        self.edit_margin_spin.setSingleStep(8)
        self.edit_margin_spin.setSuffix(" px")
        self.edit_margin_spin.setToolTip(self.tr(
            "Extra room on each side of the selected box for larger lettering. "
            "Uses original page pixels, regardless of zoom. Start with 32–64 px. "
            "Artwork inside this margin can also change; check the preview. "
            "0 keeps the original box boundary."
        ))
        model_form.addRow(self.tr("Edit margin"), self.edit_margin_spin)
        self.area_combo.currentIndexChanged.connect(self._update_edit_margin_enabled)
        self._update_edit_margin_enabled()
        layout.addLayout(model_form)

        self.advanced_button = QtWidgets.QToolButton()
        self.advanced_button.setText(self.tr("Advanced…"))
        self.advanced_button.setCheckable(True)
        self.advanced_button.setChecked(False)
        layout.addWidget(self.advanced_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.advanced_widget = QtWidgets.QWidget()
        advanced_form = QtWidgets.QFormLayout(self.advanced_widget)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        advanced_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)

        seed_widget = QtWidgets.QWidget()
        seed_row = QtWidgets.QHBoxLayout(seed_widget)
        seed_row.setContentsMargins(0, 0, 0, 0)
        self.seed_edit = QtWidgets.QLineEdit("0")
        self.seed_edit.setValidator(QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-9]{1,19}"), self.seed_edit
        ))
        self.seed_edit.setToolTip(self.tr("Integer from 0 to 9223372036854775807"))
        self.random_seed_checkbox = QtWidgets.QCheckBox(self.tr("Random each run"))
        seed_row.addWidget(self.seed_edit, 1)
        seed_row.addWidget(self.random_seed_checkbox)
        advanced_form.addRow(self.tr("Seed"), seed_widget)

        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(1, 20)
        self.steps_spin.setValue(4)
        advanced_form.addRow(self.tr("Steps"), self.steps_spin)

        self.context_spin = QtWidgets.QSpinBox()
        self.context_spin.setRange(0, 200)
        self.context_spin.setSuffix(" %")
        self.context_spin.setValue(15)
        self.context_spin.setToolTip(self.tr(
            "Surrounding artwork shown to FLUX as context. "
            "To keep lettering beyond the selected box, increase Edit margin."
        ))
        advanced_form.addRow(self.tr("Context"), self.context_spin)

        self.feather_spin = QtWidgets.QDoubleSpinBox()
        self.feather_spin.setRange(0.0, 64.0)
        self.feather_spin.setSingleStep(1.0)
        self.feather_spin.setSuffix(" px")
        self.feather_spin.setValue(8.0)
        advanced_form.addRow(self.tr("Feather"), self.feather_spin)

        self.maximum_side_spin = QtWidgets.QSpinBox()
        self.maximum_side_spin.setRange(256, 8192)
        self.maximum_side_spin.setSingleStep(128)
        self.maximum_side_spin.setValue(2048)
        advanced_form.addRow(self.tr("Max side"), self.maximum_side_spin)

        self.guidance_spin = QtWidgets.QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 20.0)
        self.guidance_spin.setSingleStep(0.1)
        self.guidance_spin.setValue(1.0)
        self.guidance_spin.setToolTip(
            self.tr("Distilled FLUX.2 Klein checkpoints may ignore guidance.")
        )
        advanced_form.addRow(self.tr("Guidance"), self.guidance_spin)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItem(self.tr("Single GPU 0"), DevicePolicy.SINGLE_GPU_0.value)
        self.device_combo.addItem(self.tr("Single GPU 1"), DevicePolicy.SINGLE_GPU_1.value)
        self.device_combo.addItem(self.tr("Model CPU offload"), DevicePolicy.MODEL_CPU_OFFLOAD.value)
        self.device_combo.addItem(self.tr("Sequential CPU offload"), DevicePolicy.SEQUENTIAL_CPU_OFFLOAD.value)
        self.device_combo.addItem(self.tr("Balanced multi-GPU (experimental)"), DevicePolicy.BALANCED.value)
        self.device_combo.addItem(self.tr("CPU"), DevicePolicy.CPU.value)
        self.device_combo.setCurrentIndex(
            self.device_combo.findData(DevicePolicy.MODEL_CPU_OFFLOAD.value)
        )
        advanced_form.addRow(self.tr("Device"), self.device_combo)

        self.offload_gpu_spin = QtWidgets.QSpinBox()
        self.offload_gpu_spin.setRange(0, 15)
        self.offload_gpu_spin.setValue(0)
        self.offload_gpu_spin.setToolTip(
            self.tr(
                "Physical CUDA device used by Model/Sequential CPU offload. "
                "Single-GPU and Balanced policies ignore this value."
            )
        )
        advanced_form.addRow(self.tr("Offload GPU"), self.offload_gpu_spin)
        self.device_combo.currentIndexChanged.connect(self._update_offload_gpu_enabled)
        self._update_offload_gpu_enabled()
        self.advanced_widget.hide()
        layout.addWidget(self.advanced_widget)

        action_row = QtWidgets.QHBoxLayout()
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.unload_button = QtWidgets.QPushButton(self.tr("Unload Model"))
        action_row.addWidget(self.cancel_button, 1)
        action_row.addWidget(self.unload_button, 1)
        layout.addLayout(action_row)

        self.revert_button = QtWidgets.QPushButton(self.tr("Revert Artistic Edits"))
        layout.addWidget(self.revert_button)
        layout.addStretch(1)

        # The editor's tools column is intentionally narrow. Long model/license
        # labels remain available in each combo popup and tooltip, while the
        # controls themselves are allowed to contract instead of forcing the
        # entire right-hand pane several hundred pixels wider.
        narrow_widgets = (
            self.prompt_edit,
            self.use_translation_button,
            self.prompt_template_combo,
            self.model_combo,
            lora_widget,
            self.lora_combo,
            area_widget,
            self.area_combo,
            seed_widget,
            self.device_combo,
            self.offload_gpu_spin,
            self.generate_button,
            self.configure_runtime_button,
            self.cancel_button,
            self.unload_button,
            self.revert_button,
        )
        for widget in narrow_widgets:
            policy = widget.sizePolicy()
            policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
            widget.setSizePolicy(policy)
            widget.setMinimumWidth(0)
        for combo in (
            self.prompt_template_combo,
            self.model_combo,
            self.lora_combo,
            self.area_combo,
            self.device_combo,
        ):
            combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(8)
        self.model_combo.setToolTip(self.model_combo.currentText())
        self.model_combo.currentTextChanged.connect(self.model_combo.setToolTip)

        self.advanced_button.toggled.connect(self.advanced_widget.setVisible)
        self.use_translation_button.clicked.connect(self.use_selected_translation_requested)
        self.generate_button.clicked.connect(self.generate_requested)
        self.configure_runtime_button.clicked.connect(self.configure_runtime_requested)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.unload_button.clicked.connect(self.unload_requested)
        self.revert_button.clicked.connect(self.revert_requested)
        self.refresh_lora_button.clicked.connect(self.refresh_loras_requested)
        self.open_lora_button.clicked.connect(self.open_lora_folder_requested)
        self.lora_combo.currentIndexChanged.connect(self._on_lora_changed)
        self.backend_combo.currentIndexChanged.connect(self._update_backend)
        self._local_model_labels = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]

    def _update_backend(self, *_args) -> None:
        cloud = self.backend_combo.currentData() == "bfl"
        self.cloud_notice.setVisible(cloud)
        self.prompt_edit.setPlaceholderText(
            self.tr("Enter an editing prompt first.") if cloud else self.tr("Describe the local image edit…")
        )
        for widget in (self.lora_combo, self.refresh_lora_button, self.open_lora_button,
                       self.device_combo, self.steps_spin, self.guidance_spin):
            widget.setEnabled(not cloud)
        self.unload_button.setEnabled(not cloud and not self._busy)
        self._update_offload_gpu_enabled()
        self._on_lora_changed()
        for i, label in enumerate(self._local_model_labels):
            self.model_combo.setItemText(i, ("4B" if i == 0 else "9B") if cloud else label)
        self._refresh_generate_button()

    def apply_theme(self) -> None:
        """Give this panel's actions a restrained outline without changing global UI."""
        action_buttons = (
            self.use_translation_button,
            self.generate_button,
            self.configure_runtime_button,
            self.refresh_lora_button,
            self.open_lora_button,
            self.advanced_button,
            self.cancel_button,
            self.unload_button,
            self.revert_button,
        )
        for button in action_buttons:
            button.setProperty("artisticEditAction", True)

        outline_color = dayu_theme.disable_color if dayu_theme.is_dark else dayu_theme.border_color
        self.setStyleSheet(
            f"""
            QPushButton[artisticEditAction="true"],
            QToolButton[artisticEditAction="true"] {{
                background-color: {dayu_theme.background_in_color};
                color: {dayu_theme.primary_text_color};
                border: 1px solid {outline_color};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton[artisticEditAction="true"]:hover,
            QPushButton[artisticEditAction="true"]:focus,
            QToolButton[artisticEditAction="true"]:hover,
            QToolButton[artisticEditAction="true"]:focus {{
                background-color: {dayu_theme.background_out_color};
                border-color: {dayu_theme.primary_5};
            }}
            QPushButton[artisticEditAction="true"]:pressed,
            QToolButton[artisticEditAction="true"]:pressed,
            QToolButton[artisticEditAction="true"]:checked {{
                background-color: {dayu_theme.background_selected_color};
                border-color: {dayu_theme.primary_color};
            }}
            QPushButton[artisticEditAction="true"]:disabled,
            QToolButton[artisticEditAction="true"]:disabled {{
                background-color: {dayu_theme.background_color};
                color: {dayu_theme.disable_color};
                border-color: {dayu_theme.border_color};
            }}
            """
        )

    def options(self) -> ArtisticEditOptions:
        seed_text = self.seed_edit.text().strip()
        seed = int(seed_text) if seed_text else 0
        if seed > (2**63) - 1:
            raise ValueError(self.tr("Seed must be at most 9223372036854775807."))
        if self.backend_combo.currentData() == "bfl" and not self.random_seed_checkbox.isChecked() and seed > 2**32 - 1:
            raise ValueError(self.tr("BFL seeds must be at most 4294967295."))
        return ArtisticEditOptions(
            backend=self.backend_combo.currentData(),
            prompt=self.prompt_edit.toPlainText().strip(),
            model_key=ModelKey(self.model_combo.currentData()),
            selection_mode=SelectionMode(self.area_combo.currentData()),
            lora_index=self.lora_combo.currentIndex() - 1 if self.backend_combo.currentData() == 'local' else -1,
            lora_scale=float(self.lora_strength.value()),
            seed=seed,
            random_seed=self.random_seed_checkbox.isChecked(),
            steps=int(self.steps_spin.value()),
            guidance=float(self.guidance_spin.value()),
            context_ratio=float(self.context_spin.value()) / 100.0,
            feather_radius=float(self.feather_spin.value()),
            maximum_side=int(self.maximum_side_spin.value()),
            device_policy=DevicePolicy(self.device_combo.currentData()),
            cuda_device_index=int(self.offload_gpu_spin.value()),
            edit_margin=(
                self.edit_margin_spin.value()
                if self.area_combo.currentData() == SelectionMode.SELECTED_BOX.value
                else 0
            ),
        )

    def _update_edit_margin_enabled(self, *_args) -> None:
        self.edit_margin_spin.setEnabled(
            self.area_combo.currentData() == SelectionMode.SELECTED_BOX.value
        )

    def _update_offload_gpu_enabled(self, *_args) -> None:
        policy = DevicePolicy(self.device_combo.currentData())
        self.offload_gpu_spin.setEnabled(
            self.backend_combo.currentData() == 'local' and policy
            in {
                DevicePolicy.MODEL_CPU_OFFLOAD,
                DevicePolicy.SEQUENTIAL_CPU_OFFLOAD,
            }
        )

    def set_prompt(self, prompt: str) -> None:
        self.prompt_edit.setPlainText(prompt)
        self.prompt_edit.setFocus()

    def is_busy(self) -> bool:
        return self._busy

    def template_key(self) -> str:
        return str(self.prompt_template_combo.currentData())

    def set_loras(self, entries) -> None:
        self.lora_combo.clear()
        self.lora_combo.addItem(self.tr("None"), None)
        for entry in entries:
            label = entry.display_name
            if entry.sidecar_error:
                label = self.tr("%1 (metadata warning)").replace("%1", label)
            self.lora_combo.addItem(label, entry)
        self._on_lora_changed()

    def _on_lora_changed(self, *_args) -> None:
        entry = self.lora_combo.currentData()
        enabled = entry is not None and self.backend_combo.currentData() == 'local'
        self.lora_strength.setEnabled(enabled)
        if enabled:
            self.lora_strength.setValue(float(entry.default_scale))

    def set_context_available(
        self,
        available: bool,
        reason: str = "",
        *,
        runtime_configuration_needed: bool = False,
    ) -> None:
        self._context_available = bool(available)
        self._context_reason = str(reason or "")
        self.configure_runtime_button.setVisible(bool(runtime_configuration_needed))
        if reason:
            self.set_status(reason)
        elif not self._busy:
            self.set_status("")
        self._refresh_generate_button()

    def set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = bool(busy)
        self.backend_combo.setEnabled(not self._busy)
        self.unload_button.setEnabled(not self._busy and self.backend_combo.currentData() == 'local')
        self.cancel_button.setEnabled(self._busy)
        self.progress_bar.setVisible(self._busy)
        if self._busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        if status:
            self.set_status(status)
        self._refresh_generate_button()

    def set_progress(self, step: int, total: int) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(max(0, min(step, total)))

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))

    def _refresh_generate_button(self) -> None:
        has_prompt = bool(self.prompt_edit.toPlainText().strip())
        enabled = not self._busy and self._context_available and has_prompt
        self.generate_button.setEnabled(enabled)
        if self._busy:
            tooltip = self.tr("An Artistic Edit is already running.")
        elif not self._context_available:
            tooltip = self._context_reason or self.tr("Open a page and select an edit area.")
        elif not has_prompt:
            tooltip = self.tr("Enter an editing prompt first.")
        else:
            tooltip = self.tr("Process the selected area with FLUX.2 Klein.")
        self.generate_button.setToolTip(tooltip)
