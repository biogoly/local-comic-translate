"""Configuration for an optional local ComfyUI server, with background probes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import threading

from PySide6 import QtCore, QtWidgets

from modules.artistic_edit.comfy_config import (
    ComfyConnectionConfig, ComfyLoraSelection, ComfyModelProfile, ComfySettings,
    KLEIN_9B_INT8_PROFILE, QWEN_IMAGE_21_INT8_PROFILE, QWEN_IMAGE_21_MODEL_KEY,
    QWEN_IMAGE_21_WORKFLOW_ID, WORKFLOW_ID,
)
from modules.artistic_edit.comfy_client import probe_comfy
from modules.artistic_edit.comfy_lora import validate_comfy_lora_compatibility


class _SettingsJob(QtCore.QThread):
    outcome = QtCore.Signal(object)

    def __init__(self, config, operation, runtime, lora_path="", parent=None):
        super().__init__(parent)
        self.config = config
        self.operation = operation
        self.runtime = runtime
        self.lora_path = lora_path
        self.cancelled = threading.Event()

    def run(self):
        try:
            if self.cancelled.is_set():
                return
            if self.operation == "probe":
                connection = self.config.connection
                if connection.mode == "managed":
                    url = self.runtime.ensure_started(connection, self.cancelled, lambda _text: None)
                    connection = replace(connection, base_url=url, mode="connect")
                result = probe_comfy(connection, cancelled=self.cancelled)
            else:
                result = self.config
                if result.lora is not None and self.lora_path:
                    path = Path(self.lora_path).expanduser().resolve(strict=True)
                    if path.suffix.lower() != ".safetensors":
                        raise ValueError("Select a safetensors LoRA file.")
                    before = path.stat()
                    validate_comfy_lora_compatibility(path, result.profile.model_key)
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        while chunk := handle.read(1024 * 1024):
                            if self.cancelled.is_set():
                                return
                            digest.update(chunk)
                    after = path.stat()
                    if any(getattr(before, field) != getattr(after, field) for field in
                           ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                        raise ValueError(self.tr("The LoRA file changed during verification. Select it again and retry."))
                    result = replace(result, lora=replace(result.lora, sha256=digest.hexdigest()))
            if not self.cancelled.is_set():
                self.outcome.emit((result, ""))
        except Exception as exc:
            if not self.cancelled.is_set():
                self.outcome.emit((None, str(exc)))


class ComfyUISettingsDialog(QtWidgets.QDialog):
    def __init__(self, config: ComfySettings, runtime, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Configure ComfyUI"))
        self.resize(680, 640)
        self.config = config
        self.runtime = runtime
        self._job = None
        self._outcome = None
        self._reject_pending = False
        layout = QtWidgets.QVBoxLayout(self)
        notice = QtWidgets.QLabel(self.tr(
            "Connect to a local ComfyUI server or start an existing installation. "
            "Models and custom nodes are never installed automatically."
        ))
        notice.setWordWrap(True)
        layout.addWidget(notice)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(form)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItem(self.tr("Connect to running server"), "connect")
        self.mode.addItem(self.tr("Start selected installation"), "managed")
        self.mode.setCurrentIndex(max(0, self.mode.findData(config.connection.mode)))
        form.addRow(self.tr("Connection"), self.mode)
        self.url = QtWidgets.QLineEdit(config.connection.base_url)
        form.addRow(self.tr("Local server URL"), self.url)
        self.installation = QtWidgets.QLineEdit(config.connection.installation_root or "")
        form.addRow(self.tr("Portable installation folder"), self._browse_row(self.installation, folder=True))
        self.python = QtWidgets.QLineEdit(config.connection.python_executable or "")
        form.addRow(self.tr("Python executable (optional)"), self._browse_row(self.python))
        self.gpu = QtWidgets.QSpinBox()
        self.gpu.setRange(0, 15)
        self.gpu.setValue(config.connection.gpu_index)
        form.addRow(self.tr("GPU index"), self.gpu)
        self.memory = QtWidgets.QComboBox()
        self.memory.addItem(self.tr("Default memory manager"), "normal")
        self.memory.addItem(self.tr("Disable execution cache"), "low_cache")
        self.memory.setCurrentIndex(max(0, self.memory.findData(config.connection.memory_policy)))
        form.addRow(self.tr("Memory policy"), self.memory)
        self.checkpoint_preset = QtWidgets.QComboBox()
        self.checkpoint_preset.addItem(self.tr("Custom / native checkpoint"), "custom")
        self.checkpoint_preset.addItem(self.tr("Klein 9B INT8 (prequantized)"), "klein-9b-int8")
        self.checkpoint_preset.addItem(self.tr("Qwen Image 2.1 INT8 (experimental)"), "qwen-image-2.1-int8")
        self.checkpoint_preset.setToolTip(self.tr(
            "Selects existing INT8 model files; no conversion or download. "
            "Each preset uses its matching text encoder and VAE."
        ))
        form.addRow(self.tr("Checkpoint preset"), self.checkpoint_preset)
        self.model = QtWidgets.QComboBox()
        self.model.addItem(self.tr("FLUX.2 Klein 4B distilled (experimental)"), "klein-4b")
        self.model.addItem("FLUX.2 Klein 9B distilled", "klein-9b")
        self.model.addItem(self.tr("Qwen Image 2.1 (experimental)"), QWEN_IMAGE_21_MODEL_KEY)
        self.model.setCurrentIndex(max(0, self.model.findData(config.profile.model_key)))
        form.addRow(self.tr("Model"), self.model)
        model_notice = QtWidgets.QLabel(self.tr(
            "Changing the model does not change the filenames below. Select matching diffusion, "
            "encoder, and VAE files. Untested combinations are experimental."
        ))
        model_notice.setWordWrap(True)
        form.addRow(model_notice)
        self.diffusion = self._catalog_combo(config.profile.diffusion_model)
        self.encoder = self._catalog_combo(config.profile.text_encoder)
        self.vae = self._catalog_combo(config.profile.vae)
        form.addRow(self.tr("Diffusion model"), self.diffusion)
        form.addRow(self.tr("Text encoder"), self.encoder)
        form.addRow(self.tr("VAE"), self.vae)
        self.lora = self._catalog_combo(config.lora.name if config.lora else "")
        form.addRow(self.tr("ComfyUI LoRA (optional)"), self.lora)
        self.lora_scale = QtWidgets.QDoubleSpinBox()
        self.lora_scale.setRange(0, 2)
        self.lora_scale.setSingleStep(0.05)
        self.lora_scale.setValue(config.lora.scale if config.lora else 1.0)
        form.addRow(self.tr("LoRA strength"), self.lora_scale)
        self.lora_path = QtWidgets.QLineEdit()
        self.lora_path.setPlaceholderText(self.tr("Select the local file represented by that ComfyUI LoRA name"))
        self.lora_file_row = self._browse_row(self.lora_path)
        form.addRow(self.tr("LoRA verification file"), self.lora_file_row)
        self._lora_model_key = config.profile.model_key
        self._lora_selections = {}
        self.qwen_notice = QtWidgets.QLabel(self.tr(
            "Qwen Image 2.1 editing and LoRAs are experimental. Use only Qwen Image 2.1 "
            "model-only adapters; text-encoder adapters are not supported."
        ))
        self.qwen_notice.setWordWrap(True)
        layout.addWidget(self.qwen_notice)
        details = QtWidgets.QLabel(self.tr(
            "Native safetensors profiles use the checkpoint's actual format. "
            "A new LoRA needs its matching local file for architecture and checksum verification. "
            "Attached servers retain uploaded and generated images in their own folders."
        ))
        details.setWordWrap(True)
        layout.addWidget(details)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.probe_button = QtWidgets.QPushButton(self.tr("Test Connection / Refresh Models"))
        self.probe_button.clicked.connect(self._probe)
        layout.addWidget(self.probe_button)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.checkpoint_preset.currentIndexChanged.connect(self._apply_checkpoint_preset)
        self.model.currentIndexChanged.connect(self._sync_checkpoint_preset)
        self.model.currentIndexChanged.connect(self._update_model_controls)
        for combo in (self.diffusion, self.encoder, self.vae):
            combo.currentTextChanged.connect(self._sync_checkpoint_preset)
        self._sync_checkpoint_preset()
        self._update_model_controls()
        self._mode_changed()

    @staticmethod
    def _catalog_combo(value):
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.addItem(value)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(12)
        return combo

    def _browse_row(self, edit, *, folder=False):
        widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        button = QtWidgets.QPushButton(self.tr("Browse…"))
        def browse():
            if folder:
                value = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select folder"), edit.text())
            else:
                value, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.tr("Select file"), edit.text())
            if value:
                edit.setText(value)
        button.clicked.connect(browse)
        row.addWidget(edit, 1)
        row.addWidget(button)
        return widget

    def _mode_changed(self):
        managed = self.mode.currentData() == "managed"
        self.url.setEnabled(not managed)
        for widget in (self.installation, self.python, self.gpu, self.memory):
            widget.setEnabled(managed)

    def _apply_checkpoint_preset(self, *_args):
        preset = self._checkpoint_presets().get(self.checkpoint_preset.currentData())
        if preset is None:
            return
        # Block intermediate edits so partial population cannot select Custom.
        with (QtCore.QSignalBlocker(self.model), QtCore.QSignalBlocker(self.diffusion),
              QtCore.QSignalBlocker(self.encoder), QtCore.QSignalBlocker(self.vae)):
            self.model.setCurrentIndex(self.model.findData(preset.model_key))
            self.diffusion.setCurrentText(preset.diffusion_model)
            self.encoder.setCurrentText(preset.text_encoder)
            self.vae.setCurrentText(preset.vae)
        self._sync_checkpoint_preset()
        self._update_model_controls()

    @staticmethod
    def _checkpoint_presets():
        return {"klein-9b-int8": KLEIN_9B_INT8_PROFILE,
                "qwen-image-2.1-int8": QWEN_IMAGE_21_INT8_PROFILE}

    def _sync_checkpoint_preset(self, *_args):
        selected = "custom"
        for key, preset in self._checkpoint_presets().items():
            if self.model.currentData() == preset.model_key and all(
                combo.currentText().strip().replace("\\", "/") == expected
                for combo, expected in ((self.diffusion, preset.diffusion_model),
                                        (self.encoder, preset.text_encoder), (self.vae, preset.vae))
            ):
                selected = key
                break
        with QtCore.QSignalBlocker(self.checkpoint_preset):
            self.checkpoint_preset.setCurrentIndex(self.checkpoint_preset.findData(selected))

    def _update_model_controls(self, *_args):
        model_key = self.model.currentData()
        if model_key != self._lora_model_key:
            # Adapters for different architectures are never selected implicitly.
            # Keep edits in this dialog so returning to a model restores its choice.
            self._lora_selections[self._lora_model_key] = (
                self.lora.currentText(), self.lora_scale.value(), self.lora_path.text(),
            )
            name, scale, path = self._lora_selections.get(model_key, ("", 1.0, ""))
            self.lora.setCurrentText(name)
            self.lora_scale.setValue(scale)
            self.lora_path.setText(path)
            self._lora_model_key = model_key
        self.qwen_notice.setVisible(model_key == QWEN_IMAGE_21_MODEL_KEY)

    def _read_config(self, *, verify_lora=True):
        connection = ComfyConnectionConfig(
            mode=self.mode.currentData(), base_url=self.url.text().strip(),
            installation_root=self.installation.text().strip(),
            python_executable=self.python.text().strip() or None,
            gpu_index=self.gpu.value(), memory_policy=self.memory.currentData(),
        )
        if not verify_lora:
            # Catalog discovery must work before any model filenames are selected.
            return ComfySettings(connection, self.config.profile, None)
        model_key = self.model.currentData()
        preset = self._checkpoint_presets().get(self.checkpoint_preset.currentData())
        profile = ComfyModelProfile(
            profile_id=preset.profile_id if preset is not None else f"{model_key}-native", model_key=model_key,
            diffusion_model=self.diffusion.currentText().strip(),
            text_encoder=self.encoder.currentText().strip(), vae=self.vae.currentText().strip(),
            loader="native", workflow_id=(QWEN_IMAGE_21_WORKFLOW_ID
                                           if model_key == QWEN_IMAGE_21_MODEL_KEY else WORKFLOW_ID),
        )
        if replace(profile, profile_id=self.config.profile.profile_id) == self.config.profile:
            # A UI preset is only a shortcut to files, not a new model identity.
            # Preserve saved provenance and an unchanged LoRA's verification.
            profile = self.config.profile
        name = self.lora.currentText().strip()
        lora = None
        if name and verify_lora:
            existing = self.config.lora
            matches = (existing and existing.name == name and existing.verified_model_key == model_key
                       and existing.sha256 and connection == self.config.connection
                       and profile == self.config.profile)
            if not self.lora_path.text().strip() and not matches:
                raise ValueError(self.tr("Select the matching local LoRA file before saving this adapter."))
            lora = ComfyLoraSelection(name, self.lora_scale.value(), model_key,
                                     existing.sha256 if matches else None)
        return ComfySettings(connection, profile, lora)

    def _start_job(self, operation):
        if self._job is not None:
            return
        try:
            config = self._read_config(verify_lora=operation == "save")
        except (TypeError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self._outcome = None
        self._job = _SettingsJob(config, operation, self.runtime, self.lora_path.text().strip(), self)
        self._job.outcome.connect(self._on_outcome, QtCore.Qt.QueuedConnection)
        self._job.finished.connect(self._on_finished, QtCore.Qt.QueuedConnection)
        self.probe_button.setEnabled(False)
        self.buttons.button(QtWidgets.QDialogButtonBox.Save).setEnabled(False)
        self.status.setText(self.tr("Checking ComfyUI…") if operation == "probe" else self.tr("Validating settings…"))
        self._job.start()

    def _probe(self):
        self._start_job("probe")

    def _save(self):
        self._start_job("save")

    @QtCore.Slot(object)
    def _on_outcome(self, outcome):
        self._outcome = outcome

    @QtCore.Slot()
    def _on_finished(self):
        job = self._job
        self._job = None
        job.deleteLater()
        self.probe_button.setEnabled(True)
        self.buttons.button(QtWidgets.QDialogButtonBox.Save).setEnabled(True)
        if self._reject_pending:
            super().reject()
            return
        if self._outcome is None:
            return
        result, error = self._outcome
        if error:
            self.status.setText(error)
        elif job.operation == "save":
            self.config = result
            self.accept()
        else:
            for combo, key in ((self.diffusion, "diffusion_models"), (self.encoder, "text_encoders"),
                               (self.vae, "vae"), (self.lora, "loras")):
                selected = combo.currentText()
                with QtCore.QSignalBlocker(combo):
                    combo.clear()
                    combo.addItem("")
                    combo.addItems(list(result.catalogs.get(key, ())))
                    combo.setCurrentText(selected)
            self._sync_checkpoint_preset()
            self.status.setText(self.tr("Connected to ComfyUI {version}. Select the matching model files.").format(version=result.version))

    def reject(self):
        if self._job is not None:
            self._reject_pending = True
            self._job.cancelled.set()
            self.status.setText(self.tr("Cancelling connection check…"))
            return
        super().reject()

    def closeEvent(self, event):
        if self._job is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
