from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
import importlib.util
import os
from pathlib import Path
import secrets
import shutil
import sys
from typing import Callable
from uuid import uuid4

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from app.projects.patch_metadata import PATCH_KIND_ARTISTIC_EDIT, PATCH_KIND_FLUX2_EDIT, patch_matches_kind
from app.ui.commands.inpaint import PatchClearCommand, PatchInsertCommand
from app.ui.dialogs.artistic_edit_preview import ArtisticEditPreviewDialog
from app.ui.dayu_widgets.message import MMessage
from modules.artistic_edit.contracts import (
    ArtisticEditRequest,
    ArtisticEditResult,
    DEFAULT_MODEL_SPECS,
    DevicePolicy,
    ModelKey,
    SelectionMode,
)
from modules.artistic_edit.image_ops import (
    ComposedPatch,
    PreparedEdit,
    compose_generated_patch,
    crop_to_bbox,
    mask_bbox,
    normalize_rgb,
    prepare_edit,
    sha256_pixels,
)
from modules.artistic_edit.lora_registry import LoraEntry, LoraRegistry, RegistryError
from modules.artistic_edit.runtime_paths import (
    is_application_executable,
    is_frozen_app,
    lora_root,
)
from modules.artistic_edit.worker_client import FluxWorkerClient
from modules.artistic_edit.bfl_client import BflEditJob
from modules.artistic_edit.backend_contracts import RasterEditSpec, ComfyGenerationSpec
from modules.artistic_edit.comfy_config import (
    ComfySettings, QWEN_IMAGE_21_MODEL_KEY, load_comfy_settings, save_comfy_settings,
)
from modules.artistic_edit.comfy_job import ComfyEditJob
from modules.utils.settings import app_settings


FLUX_PYTHON_ENV = "LOCAL_COMIC_TRANSLATE_FLUX_PYTHON"
FLUX_PYTHONPATH_ENV = "LOCAL_COMIC_TRANSLATE_FLUX_PYTHONPATH"
FLUX_PYTHON_SETTING = "artistic_edit/python_executable"


def build_translation_prompt(source: str, translation: str, template: str) -> str:
    """Build an editable prompt from one explicitly selected OCR block."""
    source = " ".join(str(source or "").split())
    translation = " ".join(str(translation or "").split())
    if not source:
        raise ValueError(QtCore.QCoreApplication.translate(
            "ArtisticEditController", "The selected box has no recognized source text."
        ))
    if not translation:
        raise ValueError(QtCore.QCoreApplication.translate(
            "ArtisticEditController", "The selected box has no translation yet."
        ))
    quoted_source = source.replace('"', '\\"')
    quoted_translation = translation.replace('"', '\\"')
    if template == "empty_bubble":
        return (
            "Remove the lettering \"%s\" and draw an empty speech bubble large enough "
            "for the translation \"%s\". Preserve the original comic style, border "
            "weight, tail direction, surrounding characters, and artwork. Do not put "
            "any text inside the bubble. Leave every other part of the image unchanged."
        ) % (quoted_source, quoted_translation)
    return (
        "Replace only the lettering \"%s\" with exactly \"%s\". Preserve the original "
        "letter style, scale, placement, spacing, texture, outline, shadows, perspective, "
        "and integration with the comic art. Leave every other part of the image unchanged."
    ) % (quoted_source, quoted_translation)


def preview_after_image(prepared: PreparedEdit, patch: ComposedPatch) -> np.ndarray:
    """Place the already masked patch into its context crop for Before/After display."""
    after = prepared.source_crop.copy()
    context_x, context_y, _, _ = prepared.context_bbox
    patch_x, patch_y, width, height = patch.bbox
    local_x = patch_x - context_x
    local_y = patch_y - context_y
    after[local_y : local_y + height, local_x : local_x + width] = patch.image
    return after


@dataclass
class ArtisticEditSession:
    page_path: str
    page_index: int
    selection_mode: SelectionMode
    prepared: PreparedEdit
    request: ArtisticEditRequest | RasterEditSpec
    request_dir: Path
    lora_name: str | None
    feather_radius: float
    cuda_device_index: int = 0
    patch: ComposedPatch | None = None
    backend: str = 'local'
    comfy_spec: ComfyGenerationSpec | None = None
    execution_metadata: dict | None = None
    project_epoch: int | None = None
    undo_stack: object | None = None
    accept_result: bool = True
    temp_root: Path | None = None


class _ComfyStopJob(QtCore.QThread):
    def __init__(self, runtime, parent=None):
        super().__init__(parent)
        self.runtime = runtime
        self.error = ""

    def run(self):
        try:
            self.runtime.shutdown()
        except Exception as exc:
            self.error = str(exc)


class ArtisticEditController(QtCore.QObject):
    """Coordinates page capture, the isolated worker, preview, and patch Apply."""

    def __init__(
        self,
        main,
        *,
        worker_factory: Callable[..., FluxWorkerClient] = FluxWorkerClient,
        preview_factory: Callable[..., ArtisticEditPreviewDialog] = ArtisticEditPreviewDialog,
        comfy_job_factory=ComfyEditJob,
    ) -> None:
        super().__init__(main if isinstance(main, QtCore.QObject) else None)
        self.main = main
        self.panel = main.artistic_edit_panel
        self.worker_factory = worker_factory
        self.preview_factory = preview_factory
        self.comfy_job_factory = comfy_job_factory
        self._comfy_job = None
        self._comfy_outcome = None
        self._comfy_runtime = None
        self._comfy_stop_job = None
        self._project_epoch = 0
        self._close_pending = False
        self.worker: FluxWorkerClient | None = None
        self._bfl_job = None
        self._closing = False
        self._worker_python: str | None = None
        self._worker_policy: DevicePolicy | None = None
        self._worker_cuda_device_index: int | None = None
        self._awaiting_worker_ready = False
        self._session: ArtisticEditSession | None = None
        self._loras: list[LoraEntry] = []
        self.lora_registry = LoraRegistry(lora_root())
        self.refresh_loras()
        self.update_availability()

    def connect_signals(self) -> None:
        panel = self.panel
        panel.use_selected_translation_requested.connect(self.use_selected_translation)
        panel.generate_requested.connect(self.generate_preview)
        panel.configure_runtime_requested.connect(self.select_worker_python)
        panel.configure_comfyui_requested.connect(self.configure_comfyui)
        panel.cancel_requested.connect(self.cancel)
        panel.unload_requested.connect(self.unload)
        panel.revert_requested.connect(self.revert_artistic_edits)
        panel.refresh_loras_requested.connect(self.refresh_loras)
        panel.open_lora_folder_requested.connect(self.open_lora_folder)
        panel.model_combo.currentIndexChanged.connect(self.refresh_loras)
        panel.area_combo.currentIndexChanged.connect(self.update_availability)
        panel.backend_combo.currentIndexChanged.connect(self._on_backend_changed)

    def _on_backend_changed(self, *_args):
        if self._session is not None and self._session.backend != self.panel.backend_combo.currentData():
            self._session.accept_result = False
            self.cancel()
        self.update_availability()

    def _get_comfy_runtime(self):
        if self._comfy_runtime is None:
            from modules.artistic_edit.comfy_runtime import ComfyRuntime
            from modules.utils.paths import get_user_data_dir
            self._comfy_runtime = ComfyRuntime(Path(get_user_data_dir()) / "artistic_edit" / "comfyui")
        return self._comfy_runtime

    def configure_comfyui(self):
        if self.panel.is_busy():
            return
        from app.ui.dialogs.comfyui_settings import ComfyUISettingsDialog
        try:
            settings = app_settings()
            try:
                config = load_comfy_settings(settings)
            except ValueError:
                config = ComfySettings()
            dialog = ComfyUISettingsDialog(config, self._get_comfy_runtime(), self.main)
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                save_comfy_settings(settings, dialog.config)
            self.update_availability()
        except Exception as exc:
            self._warning(str(exc))

    def invalidate_project(self):
        """Retire result acceptance before the page/undo stores are replaced."""
        self._project_epoch += 1
        session = self._session
        if session is not None:
            session.accept_result = False
            if self.panel.is_busy():
                self.cancel()
            else:
                self._discard_session_files(session)
                self._session = None

    def _current_page(self) -> tuple[str, int] | None:
        index = int(getattr(self.main, "curr_img_idx", -1))
        files = getattr(self.main, "image_files", [])
        if index < 0 or index >= len(files):
            return None
        return files[index], index

    def configured_worker_python(self) -> str | None:
        configured = os.environ.get(FLUX_PYTHON_ENV, "").strip()
        if not configured:
            settings = app_settings()
            configured = str(settings.value(FLUX_PYTHON_SETTING, "") or "").strip()
        if configured:
            candidate = Path(configured).expanduser().resolve(strict=False)
            return (
                str(candidate)
                if candidate.is_file() and not is_application_executable(candidate)
                else None
            )
        # A source/dev environment may already contain the optional runtime.
        # sys.executable points to the GUI, not Python, in a packaged build.
        if is_frozen_app():
            return None
        if importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"):
            candidate = Path(sys.executable).resolve(strict=False)
            return str(candidate) if candidate.is_file() else None
        return None

    def _area_is_valid(self, mode: SelectionMode) -> tuple[bool, str]:
        if mode == SelectionMode.WHOLE_PAGE:
            return True, ""
        viewer = self.main.image_viewer
        if mode == SelectionMode.SELECTED_BOX:
            if len(viewer.get_selected_rectangles()) != 1:
                return False, self.tr("Select exactly one box on the page.")
            return True, ""
        mask = viewer.get_mask_for_inpainting()
        if mask is None or mask_bbox(mask) is None:
            return False, self.tr("Paint the area to edit with the cleanup brush.")
        return True, ""

    def update_availability(self, *_args) -> None:
        backend = self.panel.backend_combo.currentData()
        if backend == "comfyui":
            try:
                settings = app_settings()
                configured = settings.value("artistic_edit/comfyui/configured", False, type=bool)
                config = load_comfy_settings(settings)
                self.panel.set_comfyui_profile(config.profile)
                self.panel.set_comfyui_summary(
                    self.tr("{model} | {checkpoint} | LoRA: {lora}").format(
                        model=(self.tr("{model} (experimental)").format(model=config.profile.model_key)
                               if config.profile.experimental else config.profile.model_key),
                        checkpoint=config.profile.diffusion_model,
                        lora=config.lora.name if config.lora else self.tr("None"),
                    ) if configured else self.tr("Configure a local ComfyUI server and model profile first."),
                    managed=config.connection.mode == "managed",
                )
                if not configured:
                    self.panel.set_context_available(False, self.tr("Configure ComfyUI before generating."))
                    return
            except Exception as exc:
                self.panel.set_context_available(False, str(exc))
                return
        if getattr(self.main, "webtoon_mode", False):
            self.panel.set_context_available(
                False, self.tr("Artistic Edit currently requires single-page mode.")
            )
            return
        if self._current_page() is None or not self.main.image_viewer.hasPhoto():
            self.panel.set_context_available(False, self.tr("Open a comic page first."))
            return
        python_path = self.configured_worker_python() if backend == "local" else "external"
        if python_path is None:
            self.panel.set_context_available(
                False,
                self.tr(
                    "Configure a FLUX Python executable in Artistic Edit settings "
                    "or LOCAL_COMIC_TRANSLATE_FLUX_PYTHON."
                ),
                runtime_configuration_needed=True,
            )
            return
        mode = SelectionMode(self.panel.area_combo.currentData())
        valid, reason = self._area_is_valid(mode)
        self.panel.set_context_available(valid, reason)

    def select_worker_python(self) -> None:
        """Let the user choose the isolated FLUX runtime without leaving the editor."""
        current = str(app_settings().value(FLUX_PYTHON_SETTING, "") or "").strip()
        start_dir = str(Path(current).parent) if current else ""
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self.panel,
            self.tr("Select FLUX Python executable"),
            start_dir,
            self.tr("Python executable (python.exe);;All files (*)"),
        )
        if not selected:
            return
        candidate = Path(selected).expanduser().resolve(strict=False)
        if not candidate.is_file():
            self._warning(self.tr("The selected Python executable does not exist."))
            return
        settings = app_settings()
        settings.setValue(FLUX_PYTHON_SETTING, str(candidate))
        settings.sync()
        self.unload()
        self.update_availability()
        self.panel.set_status(
            self.tr("FLUX Python selected. Enter a prompt and choose an edit area.")
        )

    def use_selected_translation(self) -> None:
        viewer = self.main.image_viewer
        if len(viewer.get_selected_rectangles()) != 1:
            self._warning(self.tr("Select exactly one OCR text box first."))
            return
        rect = viewer.get_selected_rectangles()[0]
        mapped_rect = rect.mapRectToScene(rect.rect())
        scene_rect = (
            mapped_rect.boundingRect()
            if hasattr(mapped_rect, "boundingRect")
            else mapped_rect
        )
        block = self.main.rect_item_ctrl.find_corresponding_text_block(
            scene_rect.getCoords(), 0.5
        )
        try:
            prompt = build_translation_prompt(
                getattr(block, "text", ""),
                getattr(block, "translation", ""),
                self.panel.template_key(),
            )
        except ValueError as exc:
            self._warning(str(exc))
            return
        self.panel.set_prompt(prompt)

    def _capture_selection(self, mode: SelectionMode, page: np.ndarray):
        height, width = page.shape[:2]
        if mode == SelectionMode.WHOLE_PAGE:
            return (0, 0, width, height), None
        if mode == SelectionMode.PAINTED_AREA:
            mask = self.main.image_viewer.get_mask_for_inpainting()
            bbox = mask_bbox(mask) if mask is not None else None
            if bbox is None:
                raise ValueError(self.tr("Paint an area before generating."))
            return bbox, mask
        selected = self.main.image_viewer.get_selected_rectangles()
        if len(selected) != 1:
            raise ValueError(self.tr("Select exactly one box before generating."))
        mapped_rect = selected[0].mapRectToScene(selected[0].rect())
        scene_rect = (
            mapped_rect.boundingRect()
            if hasattr(mapped_rect, "boundingRect")
            else mapped_rect
        )
        x, y, width, height = scene_rect.getRect()
        return (x, y, width, height), None

    def _selected_lora(self, options):
        if options.lora_index < 0:
            return None, None
        if options.lora_index >= len(self._loras):
            raise RegistryError(self.tr("The selected LoRA is no longer available; refresh the list."))
        entry = self._loras[options.lora_index]
        return self.lora_registry.to_lora_spec(entry, scale=options.lora_scale), entry.display_name

    def generate_preview(self, *, seed_override: int | None = None, confirm_whole_page: bool = True) -> None:
        if self._session is not None and self.panel.is_busy():
            return
        self.update_availability()
        if not self.panel.generate_button.isEnabled() and seed_override is None:
            return
        current = self._current_page()
        if current is None or getattr(self.main, "webtoon_mode", False):
            return
        try:
            options = self.panel.options()
        except (TypeError, ValueError) as exc:
            self._warning(str(exc))
            return
        if not options.prompt:
            self._warning(self.tr("Enter an edit prompt first."))
            return
        if options.selection_mode == SelectionMode.WHOLE_PAGE and confirm_whole_page:
            answer = QtWidgets.QMessageBox.question(
                self.main,
                self.tr("Whole-page Artistic Edit"),
                self.tr(
                    "A whole-page generation is slower and may alter unrelated artwork. Continue?"
                ),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        if options.backend == "bfl":
            api_key = self.main.settings_page.get_credentials("Black Forest Labs").get("api_key", "").strip()
            if not api_key:
                self._warning(self.tr("Configure a Black Forest Labs API key in Settings > Provider APIs."))
                return
            answer = QtWidgets.QMessageBox.question(
                self.main, self.tr("Cloud Artistic Edit"),
                self.tr("Send the selected area and surrounding context (or the whole page if selected) to Black Forest Labs? "
                        "This is a paid request under your provider account. Content restrictions apply. "
                        "Cancel stops waiting but may not stop billing. Continue?"),
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        page_path, page_index = current
        page = self.main.image_ctrl.get_composited_page_image(page_path)
        if page is None:
            self._warning(self.tr("The current page could not be read."))
            return
        request_dir: Path | None = None
        try:
            comfy_config = load_comfy_settings(app_settings()) if options.backend == "comfyui" else None
            selection_bbox, edit_mask = self._capture_selection(options.selection_mode, page)
            prepared = prepare_edit(
                page,
                selection_bbox,
                edit_mask=edit_mask,
                edit_margin=options.edit_margin,
                context_ratio=options.context_ratio,
                maximum_model_size=((min(options.maximum_side, 1984),) * 2 if options.backend == 'bfl' else (options.maximum_side, options.maximum_side)),
            )
            lora, lora_name = self._selected_lora(options) if options.backend == "local" else (None, None)
            if comfy_config is not None and comfy_config.lora is not None:
                lora_name = comfy_config.lora.name
            python_path = self.configured_worker_python() if options.backend == "local" else "external"
            if python_path is None:
                raise RuntimeError(self.tr("No valid FLUX Python executable is configured."))

            request_id = str(uuid4())
            request_dir = Path(self.main.temp_dir) / "artistic_edit" / request_id
            request_dir.mkdir(parents=True, exist_ok=False)
            input_path = request_dir / "input.png"
            output_path = request_dir / "output.png"
            Image.fromarray(prepared.model_input, mode="RGB").save(input_path)
            seed = (
                int(seed_override)
                if seed_override is not None
                else secrets.randbelow(2**32 if options.backend == "bfl" else 2**63)
                if options.random_seed
                else options.seed
            )
            raster_fields = dict(
                request_id=request_id, input_path=str(input_path.resolve()),
                output_path=str(output_path.resolve()), prompt=options.prompt,
                width=prepared.model_input.shape[1], height=prepared.model_input.shape[0],
                steps=options.steps, seed=seed,
            )
            if comfy_config is not None:
                request = RasterEditSpec(**raster_fields)
                comfy_spec = ComfyGenerationSpec(
                    raster=request, connection=comfy_config.connection,
                    profile=comfy_config.profile, lora=comfy_config.lora,
                    job_token=str(uuid4()), client_id=str(uuid4()),
                    cfg=(options.guidance if comfy_config.profile.model_key == QWEN_IMAGE_21_MODEL_KEY else 1.0),
                )
            else:
                request = ArtisticEditRequest(
                    **raster_fields, guidance=options.guidance,
                    model=DEFAULT_MODEL_SPECS[options.model_key], lora=lora,
                    device_policy=options.device_policy,
                )
                comfy_spec = None
        except Exception as exc:
            if request_dir is not None:
                shutil.rmtree(request_dir, ignore_errors=True)
            self._warning(str(exc))
            return

        self._discard_session_files()
        self._session = ArtisticEditSession(
            page_path=page_path,
            page_index=page_index,
            backend=options.backend,
            selection_mode=options.selection_mode,
            prepared=prepared,
            request=request,
            request_dir=request_dir,
            lora_name=lora_name,
            feather_radius=options.feather_radius,
            cuda_device_index=(
                1
                if options.device_policy == DevicePolicy.SINGLE_GPU_1
                else 0
                if options.device_policy == DevicePolicy.SINGLE_GPU_0
                else options.cuda_device_index
            ),
            comfy_spec=comfy_spec,
            project_epoch=self._project_epoch,
            undo_stack=self.main.undo_stacks.get(page_path),
            temp_root=Path(self.main.temp_dir).resolve(),
        )
        if options.backend == "comfyui":
            self._dispatch_comfy_request()
            return
        if options.backend == "bfl":
            self.panel.set_busy(True, self.tr("Waiting for Black Forest Labs…"))
            try:
                self._bfl_job = BflEditJob(request, api_key, self)
                self._bfl_job.result.connect(self._on_result)
                self._bfl_job.error.connect(self._on_bfl_error)
                self._bfl_job.cancelled.connect(self._on_cancelled)
                self._bfl_job.start()
            except Exception as exc:
                self._release_bfl_job()
                self._generation_failed(str(exc), code="bfl_api")
            return
        self.panel.set_busy(True, self.tr("Starting FLUX worker…"))
        try:
            self._dispatch_request(
                python_path,
                options.device_policy,
                options.cuda_device_index,
            )
        except Exception as exc:
            self._generation_failed(str(exc))

    def _dispatch_comfy_request(self):
        session = self._session
        if session is None or session.comfy_spec is None:
            return
        self.panel.set_busy(True, self.tr("Connecting to ComfyUI…"))
        self._comfy_outcome = None
        self._comfy_session = session
        try:
            job = self.comfy_job_factory(session.comfy_spec, self, runtime=self._get_comfy_runtime())
            self._comfy_job = job
            job.event.connect(self._on_comfy_event, QtCore.Qt.QueuedConnection)
            job.finished.connect(self._on_comfy_finished, QtCore.Qt.QueuedConnection)
            job.start()
        except Exception as exc:
            self._comfy_job = None
            self._comfy_session = None
            self._generation_failed(str(exc), code="comfyui")

    @QtCore.Slot(object)
    def _on_comfy_event(self, event):
        session = getattr(self, "_comfy_session", None)
        if session is None or session.comfy_spec is None:
            return
        spec = session.comfy_spec
        if event.request_id != spec.raster.request_id or event.job_token != spec.job_token:
            return
        if self._comfy_outcome is not None:
            return
        if event.kind in {"completed", "cancelled", "failed"}:
            self._comfy_outcome = event
            return
        if self._closing or self._session is not session or not session.accept_result:
            return
        if event.kind == "status":
            statuses = {
                "Starting ComfyUI…": self.tr("Starting ComfyUI…"),
                "Connecting to ComfyUI…": self.tr("Connecting to ComfyUI…"),
                "Uploading edit area…": self.tr("Uploading edit area…"),
                "Submitting ComfyUI edit…": self.tr("Submitting ComfyUI edit…"),
                "Waiting for ComfyUI generation…": self.tr("Waiting for ComfyUI generation…"),
                "Connection lost; checking the same ComfyUI edit…": self.tr("Connection lost; checking the same ComfyUI edit…"),
                "Fetching ComfyUI preview…": self.tr("Fetching ComfyUI preview…"),
            }
            self.panel.set_status(statuses.get(str(event.payload), str(event.payload)))

    @QtCore.Slot()
    def _on_comfy_finished(self):
        job = self._comfy_job
        if job is None or (self.sender() is not None and self.sender() is not job):
            return
        session = self._comfy_session
        outcome = self._comfy_outcome
        self._comfy_job = None
        self._comfy_session = None
        self._comfy_outcome = None
        job.deleteLater()
        if self._close_pending:
            self._discard_session_files(session)
            if self._session is session:
                self._session = None
            self.panel.set_busy(False)
            QtCore.QTimer.singleShot(0, self.main.close)
            return
        if self._closing or self._session is not session:
            self._discard_session_files(session)
            return
        if not session.accept_result or (outcome is not None and outcome.kind == "cancelled"):
            payload = outcome.payload if outcome is not None and outcome.kind == "cancelled" else {}
            continues = bool(payload.get("server_may_continue", False)) if isinstance(payload, Mapping) else False
            self._discard_session_files(session)
            self._session = None
            self.panel.set_busy(False)
            self.update_availability()
            self.panel.set_status(self.tr("Stopped waiting. ComfyUI may still be computing this request.") if continues else self.tr("Artistic Edit cancelled."))
            return
        if outcome is None or outcome.kind == "failed":
            payload = outcome.payload if outcome is not None else {}
            self._generation_failed(
                str(payload.get("message", self.tr("ComfyUI stopped without a result."))),
                code=str(payload.get("code", "comfyui")),
            )
            return
        completion = outcome.payload
        session.execution_metadata = dict(completion.metadata)
        self._on_result(completion.result)

    def _regenerate_comfy(self, session, seed):
        current, reason = self._source_is_current(session)
        if not current:
            self._warning(reason)
            self._discard_session_files(session)
            self._session = None
            return
        request_dir = Path(self.main.temp_dir) / "artistic_edit" / str(uuid4())
        try:
            request_dir.mkdir(parents=True, exist_ok=False)
            input_path = request_dir / "input.png"
            Image.fromarray(session.prepared.model_input, mode="RGB").save(input_path)
            raster = replace(session.request, request_id=request_dir.name,
                             input_path=str(input_path.resolve()),
                             output_path=str((request_dir / "output.png").resolve()), seed=seed)
            spec = replace(session.comfy_spec, raster=raster, job_token=str(uuid4()), client_id=str(uuid4()))
            new_session = replace(session, request=raster, request_dir=request_dir,
                                  comfy_spec=spec, patch=None, execution_metadata=None)
        except Exception as exc:
            shutil.rmtree(request_dir, ignore_errors=True)
            self._generation_failed(str(exc), code="comfyui")
            return
        self._discard_session_files(session)
        self._session = new_session
        self._dispatch_comfy_request()

    def _worker_callback_is_current(self):
        sender = self.sender()
        if sender is not None and sender is not self.worker:
            return False
        return not self._closing and (
            self._session is not None and self._session.backend == "local" and self._session.accept_result
        )

    @QtCore.Slot(str)
    def _on_worker_loading(self, text):
        if self._worker_callback_is_current():
            self._on_loading(text)

    @QtCore.Slot(int, int)
    def _on_worker_progress(self, step, total):
        if self._worker_callback_is_current():
            self.panel.set_progress(step, total)

    def _new_worker(
        self,
        python_path: str,
        policy: DevicePolicy,
        cuda_device_index: int,
    ) -> FluxWorkerClient:
        if policy == DevicePolicy.SINGLE_GPU_1:
            cuda_device_index = 1
        elif policy == DevicePolicy.SINGLE_GPU_0:
            cuda_device_index = 0
        worker = self.worker_factory(
            Path(self.main.temp_dir),
            python_executable=python_path,
            python_path_entries=self.configured_worker_python_paths(),
            cuda_device_index=cuda_device_index,
            allow_downloads=False,
            parent=self,
        )
        worker.ready.connect(self._on_worker_ready)
        worker.loading.connect(self._on_worker_loading)
        worker.progress.connect(self._on_worker_progress)
        worker.result.connect(self._on_result)
        worker.cancelled.connect(self._on_cancelled)
        worker.error.connect(self._on_worker_error)
        worker.worker_crashed.connect(self._on_worker_crashed)
        worker.diagnostics.connect(lambda text: print(text, end=""))
        self.worker = worker
        self._worker_python = python_path
        self._worker_policy = policy
        self._worker_cuda_device_index = cuda_device_index
        return worker

    def _on_loading(self, text: str) -> None:
        # The separate worker has no Qt translator; localize its known progress
        # messages here while retaining any unexpected diagnostic verbatim.
        if text == "Importing PyTorch and Diffusers…":
            text = self.tr("Importing PyTorch and Diffusers…")
        elif text.startswith("Preparing ") and text.endswith(" (BF16) runtime…"):
            model = text[len("Preparing "):-len(" (BF16) runtime…")]
            text = self.tr("Preparing {model} (BF16) runtime…").format(model=model)
        elif text.startswith("Loading ") and text.endswith(" BF16 weights…"):
            model = text[len("Loading "):-len(" BF16 weights…")]
            text = self.tr("Loading {model} BF16 weights…").format(model=model)
        elif text.startswith("Configuring ") and text.endswith("…"):
            policy = text[len("Configuring "):-1].replace(" ", "_")
            index = self.panel.device_combo.findData(policy)
            label = self.panel.device_combo.itemText(index) if index >= 0 else policy
            text = self.tr("Configuring {device}…").format(device=label)
        self.panel.set_status(text)

    @staticmethod
    def configured_worker_python_paths() -> tuple[str, ...]:
        raw = os.environ.get(FLUX_PYTHONPATH_ENV, "")
        return tuple(part for part in raw.split(os.pathsep) if part.strip())

    def _dispatch_request(
        self,
        python_path: str,
        policy: DevicePolicy,
        cuda_device_index: int,
    ) -> None:
        effective_device_index = (
            1
            if policy == DevicePolicy.SINGLE_GPU_1
            else 0
            if policy == DevicePolicy.SINGLE_GPU_0
            else max(0, int(cuda_device_index))
        )
        if self.worker is not None and (
            self._worker_python != python_path
            or self._worker_policy != policy
            or self._worker_cuda_device_index != effective_device_index
        ):
            previous_worker = self.worker
            self.worker = None
            previous_worker.close()
        worker = self.worker or self._new_worker(
            python_path,
            policy,
            effective_device_index,
        )
        if worker.is_running():
            self._awaiting_worker_ready = False
            worker.generate(self._session.request)
            return
        self._awaiting_worker_ready = True
        worker.start(policy)

    def _on_worker_ready(self, _capabilities: dict) -> None:
        if self.sender() is not None and self.sender() is not self.worker:
            return
        if not self._awaiting_worker_ready:
            if self._session is None and self.panel.backend_combo.currentData() == "local" and _capabilities.get("loaded_model", object()) is None:
                self.panel.set_status(self.tr("FLUX model unloaded."))
            return
        if self._session is None or self.worker is None or self._session.backend != "local" or not self._session.accept_result:
            return
        self._awaiting_worker_ready = False
        try:
            self.worker.generate(self._session.request)
        except Exception as exc:
            self._generation_failed(str(exc))

    @QtCore.Slot(str)
    def _on_bfl_error(self, message):
        if self.sender() is not None and self.sender() is not self._bfl_job:
            return
        if self._session is None or self._session.backend != "bfl":
            return
        self._release_bfl_job()
        if not self._closing:
            self._generation_failed(message, code="bfl_api")

    def _release_bfl_job(self):
        if self._bfl_job is not None:
            self._bfl_job.wait()
            self._bfl_job.deleteLater()
            self._bfl_job = None

    def _on_result(self, result: ArtisticEditResult) -> None:
        if self._closing:
            return
        session = self._session
        if session is None or result.request_id != session.request.request_id:
            return
        if not session.accept_result:
            self._on_cancelled(result.request_id)
            return
        if session.backend == "bfl":
            self._release_bfl_job()
        try:
            result.validate_path(session.request_dir)
            if Path(result.output_path).resolve() != Path(session.request.output_path).resolve():
                raise ValueError(self.tr("The result does not belong to this request."))
            with Image.open(result.output_path) as image:
                generated = np.asarray(image.convert("RGB"))
            session.patch = compose_generated_patch(
                session.prepared,
                generated,
                feather_radius=session.feather_radius,
            )
        except Exception as exc:
            self._generation_failed(str(exc))
            return
        self.panel.set_busy(False, self.tr("Preview ready."))
        self._show_preview(result)

    def _source_is_current(self, session: ArtisticEditSession) -> tuple[bool, str]:
        if not session.accept_result or (session.project_epoch is not None and session.project_epoch != self._project_epoch):
            return False, self.tr("This preview belongs to a closed or replaced project. Regenerate it.")
        if session.undo_stack is not None and self.main.undo_stacks.get(session.page_path) is not session.undo_stack:
            return False, self.tr("The page undo history changed. Regenerate the preview.")
        current = self._current_page()
        if current is None or current[0] != session.page_path or current[1] != session.page_index:
            return False, self.tr("The preview belongs to another page. Return and regenerate it.")
        page = self.main.image_ctrl.get_composited_page_image(session.page_path)
        if page is None:
            return False, self.tr("The page can no longer be read.")
        try:
            source_crop = crop_to_bbox(normalize_rgb(page), session.prepared.context_bbox)
        except Exception:
            return False, self.tr("The page geometry changed. Regenerate the preview.")
        if sha256_pixels(source_crop) != session.prepared.source_hash:
            return False, self.tr("The source pixels changed. Regenerate before applying.")
        return True, ""

    def _show_preview(self, result: ArtisticEditResult) -> None:
        session = self._session
        if session is None or session.patch is None:
            return
        current, reason = self._source_is_current(session)
        model = session.request.model if session.backend != "comfyui" else None
        lora = session.lora_name or self.tr("None")
        percent = int(round(session.prepared.working_scale * 100))
        summary = self.tr(
            "Model: {model} | Precision: BF16 | LoRA: {lora} | "
            "Seed: {seed} | Steps: {steps} | "
            "Elapsed: {elapsed:.1f}s | Processed at {percent}% resolution\nPrompt: {prompt}"
        ).format(
            model=model.key.value if model is not None else session.comfy_spec.profile.model_key,
            lora=lora,
            seed=result.seed,
            steps=session.request.steps,
            elapsed=result.elapsed_seconds,
            percent=percent,
            prompt=session.request.prompt,
        )
        if session.backend == "bfl":
            summary = self.tr("Provider: Black Forest Labs | Model: {model} | Seed: {seed} | Elapsed: {elapsed:.1f}s | Prompt: {prompt}").format(
                model=model.key.value, seed=result.seed, elapsed=result.elapsed_seconds, prompt=session.request.prompt)
        elif session.backend == "comfyui":
            metadata = session.execution_metadata or {}
            summary = self.tr(
                "Backend: ComfyUI | Model: {model} | Checkpoint: {checkpoint} | LoRA: {lora} | "
                "Seed: {seed} | Steps: {steps} | CFG: {cfg} | Elapsed: {elapsed:.1f}s | Processed at {percent}% resolution\nPrompt: {prompt}"
            ).format(model=session.comfy_spec.profile.model_key,
                     checkpoint=metadata.get("model_source", session.comfy_spec.profile.diffusion_model),
                     lora=lora, seed=result.seed, steps=session.request.steps, cfg=session.comfy_spec.cfg,
                     elapsed=result.elapsed_seconds, percent=percent, prompt=session.request.prompt)
        dialog = self.preview_factory(
            session.prepared.source_crop,
            preview_after_image(session.prepared, session.patch),
            summary,
            apply_available=current,
            stale_reason=reason,
            parent=self.main,
        )
        decision = dialog.exec()
        if decision == QtWidgets.QDialog.DialogCode.Accepted:
            current, reason = self._source_is_current(session)
            if not current:
                self._warning(reason)
                self._discard_session_files()
                self._session = None
                return
            self._apply_session(session)
            return
        if decision == ArtisticEditPreviewDialog.REGENERATE_SAME_SEED:
            seed = session.request.seed
            if session.backend == "comfyui":
                self._regenerate_comfy(session, seed)
                return
            self._discard_session_files()
            self._session = None
            self.generate_preview(seed_override=seed, confirm_whole_page=False)
            return
        if decision == ArtisticEditPreviewDialog.REGENERATE_NEW_SEED:
            if session.backend == "comfyui":
                self._regenerate_comfy(session, secrets.randbelow(2**63))
                return
            self._discard_session_files()
            self._session = None
            self.generate_preview(seed_override=secrets.randbelow(2**32 if session.backend == 'bfl' else 2**63), confirm_whole_page=False)
            return
        self._discard_session_files()
        self._session = None

    def _apply_session(self, session: ArtisticEditSession) -> None:
        current, reason = self._source_is_current(session)
        if not current:
            self._warning(reason)
            return
        patch = session.patch
        stack = self.main.undo_stacks.get(session.page_path)
        if patch is None or stack is None:
            self._warning(self.tr("The current page has no undo history."))
            return
        request = session.request
        metadata = {
            "backend": session.backend,
            "prompt": request.prompt,
            "model_key": session.comfy_spec.profile.model_key if session.backend == "comfyui" else request.model.key.value,
            "model_source": (session.comfy_spec.profile.diffusion_model if session.backend == "comfyui"
                             else request.model.source if session.backend == "local" else "https://api.bfl.ai"),
            "cuda_device_index": session.cuda_device_index if session.backend == 'local' else None,
            "seed": request.seed,
            "steps": request.steps if session.backend in {"local", "comfyui"} else None,
            "guidance": (session.comfy_spec.cfg if session.backend == "comfyui"
                         else request.guidance if session.backend == "local" else None),
            "lora_name": session.lora_name,
            "lora_sha256": (session.comfy_spec.lora.sha256 if session.comfy_spec.lora else None) if session.backend == "comfyui" else request.lora.sha256 if request.lora else None,
            "lora_scale": (session.comfy_spec.lora.scale if session.comfy_spec.lora else None) if session.backend == "comfyui" else request.lora.scale if request.lora else None,
            "selection_mode": session.selection_mode.value,
            "source_crop_sha256": session.prepared.source_hash,
            "working_scale": session.prepared.working_scale,
        }
        if session.backend == "comfyui":
            # Execution details are client-produced primitive provenance, not URLs/credentials.
            metadata.update(session.execution_metadata or {})
            metadata["backend"] = "comfyui"
            metadata["selection_mode"] = session.selection_mode.value
            metadata["source_crop_sha256"] = session.prepared.source_hash
        command = PatchInsertCommand(
            self.main,
            [{"bbox": list(patch.bbox), "image": patch.image}],
            session.page_path,
            display=True,
            kind=(PATCH_KIND_ARTISTIC_EDIT if session.backend == "comfyui"
                  and session.comfy_spec.profile.model_key == QWEN_IMAGE_21_MODEL_KEY
                  else PATCH_KIND_FLUX2_EDIT),
            text=self.tr("Artistic edit"),
            metadata=metadata,
        )
        if session.selection_mode == SelectionMode.PAINTED_AREA:
            stack.beginMacro(self.tr("Artistic edit"))
            try:
                stack.push(command)
                self.main.image_viewer.clear_brush_strokes()
            finally:
                stack.endMacro()
        else:
            stack.push(command)
        try:
            self.main._update_window_modified()
        except Exception:
            pass
        self.panel.set_status(self.tr("Artistic edit applied. Use Undo to reverse it."))
        self._discard_session_files()
        self._session = None

    def cancel(self) -> None:
        if self._session is not None:
            self._session.accept_result = False
        if self._comfy_job is not None:
            self._comfy_job.cancel()
            self.panel.set_status(self.tr("Cancelling Artistic Edit…"))
            return
        if self._bfl_job is not None and self._bfl_job.isRunning():
            self._bfl_job.cancel()
            self.panel.set_status(self.tr("Cancelling Artistic Edit…"))
            return
        if self._awaiting_worker_ready and self._session is not None and self._session.backend == "local":
            # No generate command exists yet, so the worker cannot acknowledge a
            # generation cancellation. Retire the pending dispatch locally.
            self._awaiting_worker_ready = False
            self._on_cancelled(self._session.request.request_id)
            return
        try:
            if self.worker is not None and self.worker.cancel():
                self.panel.set_status(self.tr("Cancelling Artistic Edit…"))
        except Exception as exc:
            self._warning(str(exc))

    def _on_cancelled(self, _request_id: str) -> None:
        if self._closing:
            return
        if self._session is None or _request_id != self._session.request.request_id:
            return
        if self._session.backend == "bfl":
            self._release_bfl_job()
        self.panel.set_busy(False, self.tr("Artistic Edit cancelled."))
        self._discard_session_files()
        self._session = None
        self.update_availability()

    def _on_worker_error(self, code: str, message: str, details: str) -> None:
        if self._session is None or self._session.backend != "local":
            return
        if self.sender() is not None and self.sender() is not self.worker:
            return
        text = f"{message}\n\n{details}" if details else message
        self._generation_failed(text, code=code)

    def _on_worker_crashed(self, message: str) -> None:
        if self._session is None or self._session.backend != "local":
            return
        if self.sender() is not None and self.sender() is not self.worker:
            return
        self._generation_failed(message, code="worker_crashed")

    def _generation_failed(self, message: str, *, code: str = "artistic_edit_error") -> None:
        self.panel.set_busy(False, self.tr("Artistic Edit failed."))
        self._discard_session_files()
        self._session = None
        self.update_availability()
        self._warning(f"{code}: {message}")

    def unload(self) -> None:
        if self.panel.backend_combo.currentData() == "comfyui":
            try:
                managed = load_comfy_settings(app_settings()).connection.mode == "managed"
            except ValueError as exc:
                self._warning(str(exc))
                return
            if managed and self._comfy_runtime is not None and self._comfy_runtime.is_running():
                self._stop_comfy_runtime()
            else:
                if self._comfy_runtime is not None and not managed:
                    self._comfy_runtime.disconnect()
                self.panel.set_status(self.tr("Disconnected. The external ComfyUI server remains running."))
            return
        if self.worker is None or not self.worker.is_running():
            self.panel.set_status(self.tr("The FLUX model is not loaded."))
            return
        try:
            self.worker.unload()
            self.panel.set_status(self.tr("Unloading FLUX model…"))
        except Exception as exc:
            self._warning(str(exc))

    def revert_artistic_edits(self) -> None:
        current = self._current_page()
        if current is None:
            return
        file_path, _ = current
        patches = self.main.image_patches.get(file_path, [])
        if not any(patch_matches_kind(patch, PATCH_KIND_ARTISTIC_EDIT) for patch in patches):
            self._warning(self.tr("There are no artistic edits to revert on this page."))
            return
        stack = self.main.undo_stacks.get(file_path)
        if stack is not None:
            stack.push(
                PatchClearCommand(
                    self.main,
                    file_path,
                    kind=PATCH_KIND_ARTISTIC_EDIT,
                    text=self.tr("Revert artistic edits"),
                )
            )
            try:
                self.main._update_window_modified()
            except Exception:
                pass

    def refresh_loras(self, *_args) -> None:
        try:
            model_key = ModelKey(self.panel.model_combo.currentData())
            self._loras = self.lora_registry.scan(model_key)
            self.panel.set_loras(self._loras)
        except Exception as exc:
            self._loras = []
            self.panel.set_loras([])
            self.panel.set_status(self.tr("Could not scan LoRAs: %1").replace("%1", str(exc)))

    def open_lora_folder(self) -> None:
        model_key = ModelKey(self.panel.model_combo.currentData())
        folder = self.lora_registry.folder_for(model_key)
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _discard_session_files(self, session=None) -> None:
        session = session if session is not None else self._session
        if session is None:
            return
        try:
            request_dir = session.request_dir.resolve(strict=False)
            temp_root = ((session.temp_root or Path(self.main.temp_dir)) / "artistic_edit").resolve(strict=False)
            request_dir.relative_to(temp_root)
        except (ValueError, OSError):
            return
        shutil.rmtree(request_dir, ignore_errors=True)

    def _stop_comfy_runtime(self):
        if self._comfy_stop_job is not None:
            return
        self.panel.set_busy(True, self.tr("Stopping ComfyUI…"))
        self._comfy_stop_job = _ComfyStopJob(self._comfy_runtime, self)
        self._comfy_stop_job.finished.connect(self._on_comfy_stopped, QtCore.Qt.QueuedConnection)
        self._comfy_stop_job.start()

    @QtCore.Slot()
    def _on_comfy_stopped(self):
        job = self._comfy_stop_job
        if job is None:
            return
        self._comfy_stop_job = None
        error = job.error
        job.deleteLater()
        self.panel.set_busy(False)
        self.update_availability()
        if error:
            self.cancel_close()
            self._warning(error)
        else:
            self.panel.set_status(self.tr("ComfyUI stopped."))
            if self._close_pending:
                QtCore.QTimer.singleShot(0, self.main.close)

    def prepare_close(self) -> bool:
        """Finish Comfy cancellation/owned-process shutdown before temp files disappear."""
        if self._comfy_job is not None:
            self._close_pending = True
            self.cancel()
            return False
        if self._comfy_stop_job is not None:
            self._close_pending = True
            return False
        if self._comfy_runtime is not None and self._comfy_runtime.is_running():
            self._close_pending = True
            self._stop_comfy_runtime()
            return False
        self._close_pending = False
        return True

    def cancel_close(self):
        self._close_pending = False
        self.main._artistic_close_approval = None

    def shutdown(self) -> None:
        self._closing = True
        if self._comfy_job is not None:
            self._comfy_job.cancel()
            # Normal window close uses prepare_close asynchronously. This is an
            # emergency/aboutToQuit fallback; HTTP stages have bounded timeouts.
            self._comfy_job.wait()
            self._comfy_job.deleteLater()
            self._comfy_job = None
        if self._comfy_stop_job is not None:
            self._comfy_stop_job.wait()
            self._comfy_stop_job.deleteLater()
            self._comfy_stop_job = None
        if self._comfy_runtime is not None:
            try:
                self._comfy_runtime.shutdown()
            except Exception as exc:
                # Never terminate someone else's work to make shutdown succeed.
                print(f"ComfyUI runtime was left running: {exc}")
        if self._bfl_job is not None:
            self._bfl_job.cancel()
            self._bfl_job.wait()  # Network calls have bounded connect/read timeouts.
            self._bfl_job = None
        self._discard_session_files()
        self._session = None
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def _warning(self, text: str) -> None:
        MMessage.warning(text, parent=self.main, duration=6)
