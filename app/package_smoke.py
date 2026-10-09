"""Offline, disposable-profile acceptance checks for a frozen desktop package.

Run in a fresh process, before ``comic.main`` and its profile migration. This
does not download models, start llama-server/FLUX, or contact provider APIs.
The report intentionally contains no exception messages, settings, or paths.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import logging
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
from unittest.mock import patch


def _varint(value: int) -> bytes:
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number: int, value: int | bytes | str) -> bytes:
    """Encode the two protobuf wire types needed by our tiny ONNX fixture."""
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _matmul_model() -> bytes:
    """ONNX IR 8/opset 13: two runtime float32 [2, 2] inputs -> MatMul.

    No onnx/protobuf build dependency, external data, weights or constant folding.
    Field numbers follow ONNX ModelProto/GraphProto/ValueInfoProto definitions.
    """
    shape = _field(1, _field(1, 2)) * 2
    tensor_type = _field(1, 1) + _field(2, shape)

    def value_info(name: str) -> bytes:
        return _field(1, name) + _field(2, _field(1, tensor_type))

    node = (
        _field(1, "x") + _field(1, "w") + _field(2, "y")
        + _field(3, "package_matmul") + _field(4, "MatMul")
    )
    graph = (
        _field(1, node) + _field(2, "package_smoke")
        + _field(11, value_info("x")) + _field(11, value_info("w"))
        + _field(12, value_info("y"))
    )
    return _field(1, 8) + _field(2, "local-comic-translate") + _field(7, graph) + _field(8, _field(2, 13))


@contextmanager
def _isolated_environment(root: Path):
    """Redirect all app-controlled storage before importing application modules."""
    directories = {
        "LOCALAPPDATA": root / "local", "APPDATA": root / "roaming",
        "USERPROFILE": root / "home", "HOME": root / "home",
        "XDG_DATA_HOME": root / "data", "XDG_CONFIG_HOME": root / "config",
        "XDG_CACHE_HOME": root / "cache", "HF_HOME": root / "hf",
        "TEMP": root / "tmp", "TMP": root / "tmp",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    (root / "home" / "Documents").mkdir(parents=True, exist_ok=True)
    environment = {name: str(path) for name, path in directories.items()}
    environment.update({
        "QT_QPA_PLATFORM": "offscreen", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    })
    with patch.dict(os.environ, environment), patch.object(tempfile, "tempdir", str(root / "tmp")):
        yield


class _Checks:
    def __init__(self):
        self.results: list[dict] = []

    def run(self, name, operation):
        try:
            detail = operation()
            result = {"name": name, "status": "passed"}
            if detail is not None:
                result["detail"] = detail
            self.results.append(result)
            return True
        except Exception as exc:
            # Third-party exceptions often include home paths or request data.
            self.results.append({"name": name, "status": "failed", "error_type": type(exc).__name__})
            logging.getLogger(__name__).exception("Package smoke check failed: %s", name)
            return False


def _onnx_inference(root: Path, *, cuda: bool) -> dict:
    import numpy as np
    import onnxruntime as ort
    from modules.utils.device import _preload_onnx_runtime_dependencies

    _preload_onnx_runtime_dependencies()
    provider = "CUDAExecutionProvider" if cuda else "CPUExecutionProvider"
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if cuda:
        # Constructor fallback must not turn a GPU acceptance test into a CPU pass.
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(root / "cuda-profile")
    session = ort.InferenceSession(_matmul_model(), sess_options=options, providers=[provider])
    session.disable_fallback()
    if provider not in session.get_providers():
        raise AssertionError("Requested provider did not load")
    output = session.run(None, {
        "x": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "w": np.array([[5, 6], [7, 8]], dtype=np.float32),
    })[0]
    np.testing.assert_array_equal(output, np.array([[19, 22], [43, 50]], dtype=np.float32))
    if cuda:
        events = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        providers = {
            event.get("args", {}).get("provider") for event in events
            if event.get("cat") == "Node" and event.get("args", {}).get("provider")
        }
        if providers != {"CUDAExecutionProvider"}:
            raise AssertionError("The test node did not execute exclusively on CUDA")
    return {"provider": provider, "version": ort.__version__, "verified_result": True}


def _check_language_ocr() -> dict:
    """Check frozen imports and explicit-language routing without model loading."""
    from types import SimpleNamespace
    from modules.ocr.factory import OCRFactory
    from modules.ocr.ppocr import PPOCRv5Engine
    from modules.ocr.vietnamese import VietnameseOCREngine
    from modules.utils.language_utils import language_codes

    settings = SimpleNamespace(is_gpu_enabled=lambda: False)
    for language, code in (("Vietnamese", "vi"), ("Portuguese", "pt"),
                           ("Brazilian Portuguese", "pt-br")):
        if language_codes.get(language) != code:
            raise AssertionError("Language identifier changed")
        engine_class = VietnameseOCREngine if language == "Vietnamese" else PPOCRv5Engine
        # Only suppress initialization; construction and factory dispatch are real.
        # This must not open sessions, download weights, or use the user's cache.
        with patch.object(engine_class, "initialize") as initialize:
            engine = OCRFactory._create_new_engine(settings, language, "Default", "onnx")
            if type(engine) is not engine_class:
                raise AssertionError("Explicit source language selected the wrong OCR engine")
            expected = {"device": "cpu", "use_text_lines": True}
            if language != "Vietnamese":
                expected["lang"] = "latin"
            initialize.assert_called_once_with(**expected)
            if engine.rec_sess is not None or engine.det_sess is not None:
                raise AssertionError("OCR construction unexpectedly loaded a model")
    return {"languages": 3, "models_loaded": False, "inference_tested": False}


def _check_qwen_lora(root: Path) -> dict:
    """Exercise packaged Qwen graph and header validation without ComfyUI/GPU."""
    from dataclasses import replace
    from uuid import uuid4
    from modules.artistic_edit.backend_contracts import RasterEditSpec, ComfyGenerationSpec
    from modules.artistic_edit.comfy_config import (
        ComfyLoraSelection, ComfySettings, QWEN_IMAGE_21_INT8_PROFILE,
        QWEN_IMAGE_21_MODEL_KEY,
    )
    from modules.artistic_edit.comfy_lora import validate_comfy_lora_compatibility
    from modules.artistic_edit.comfy_workflows import build_workflow, FINAL_OUTPUT_NODE
    from modules.artistic_edit.lora_registry import LoraCompatibilityError

    # Synthetic header only, not model weights or a usable adapter file.
    prefix = "diffusion_model.transformer_blocks.0.img_mlp.gate_up"
    header = {
        prefix + ".lora_A.weight": {"shape": [1, 4096], "dtype": "BF16", "data_offsets": [0, 0]},
        prefix + ".lora_B.weight": {"shape": [24576, 1], "dtype": "BF16", "data_offsets": [0, 0]},
    }
    fixture = root / "qwen-header-fixture.safetensors"

    def write_header():
        payload = json.dumps(header).encode("utf-8")
        fixture.write_bytes(len(payload).to_bytes(8, "little") + payload)

    write_header()
    validate_comfy_lora_compatibility(fixture, QWEN_IMAGE_21_MODEL_KEY)
    header[prefix + ".lora_B.weight"]["shape"] = [12288, 1]
    write_header()
    try:
        validate_comfy_lora_compatibility(fixture, QWEN_IMAGE_21_MODEL_KEY)
    except LoraCompatibilityError:
        pass
    else:
        raise AssertionError("Qwen LoRA validator accepted a mismatched shape")

    lora = ComfyLoraSelection("QwenImage2.1/smoke.safetensors", 0.4, QWEN_IMAGE_21_MODEL_KEY)
    config = ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE, lora=lora)
    spec = ComfyGenerationSpec(
        RasterEditSpec(str(uuid4()), str(root / "qwen-input.png"), str(root / "qwen-output.png"),
                       "Replace HOLA with HELLO", 272, 400, 42, steps=25),
        config.connection, config.profile, lora, str(uuid4()), str(uuid4()),
    )
    graph = build_workflow(spec, "local-comic-translate/fixture.png")
    if graph["18"] != {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["1", 0], "lora_name": lora.name, "strength_model": lora.scale,
    }} or graph["7"]["inputs"]["model"] != ["18", 0]:
        raise AssertionError("Qwen LoRA must patch the model before its cache")
    if graph[FINAL_OUTPUT_NODE]["class_type"] != "SaveImage":
        raise AssertionError("Missing designated Qwen output")
    disabled = build_workflow(replace(spec, lora=None), "local-comic-translate/fixture.png")
    if "18" in disabled or disabled["7"]["inputs"]["model"] != ["1", 0]:
        raise AssertionError("Disabling Qwen LoRA left an adapter in the workflow")
    return {"header_validation": True, "workflow_nodes": len(graph),
            "runtime_started": False, "inference_tested": False}


def _exercise_package(root: Path, checks: _Checks, *, require_cuda: bool) -> None:
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QTranslator
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is not None:
        raise RuntimeError("Package smoke checks require a fresh process")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope, folder in ((QSettings.Scope.UserScope, "settings-user"), (QSettings.Scope.SystemScope, "settings-system")):
        directory = root / folder
        directory.mkdir()
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(directory))
    app = QApplication(["LocalComicTranslate-package-smoke"])
    app.setOrganizationName("biogoly")
    app.setApplicationName("LocalComicTranslate")
    # CPython may ask cmd.exe for the Windows version on first platform access.
    # Cache that harmless system query before forbidding application subprocesses.
    platform.uname()

    blocked_attempts: list[str] = []
    qt_exceptions: list[str] = []

    def blocked(category):
        def reject(*_args, **_kwargs):
            blocked_attempts.append(category)
            raise RuntimeError("External operations are forbidden during package smoke checks")
        return reject

    with ExitStack() as guards:
        def qt_exception_hook(exc_type, exc, traceback):
            qt_exceptions.append(exc_type.__name__)
            logging.getLogger(__name__).error("Unhandled Qt smoke callback", exc_info=(exc_type, exc, traceback))

        guards.enter_context(patch.object(sys, "excepthook", qt_exception_hook))
        for target, category in (
            ("requests.sessions.Session.request", "network"),
            ("urllib.request.urlopen", "network"),
            ("socket.create_connection", "network"),
            ("socket.socket.connect", "network"),
            ("socket.socket.connect_ex", "network"),
            ("subprocess.Popen", "subprocess"),
            ("keyring.get_password", "keyring"),
            ("keyring.get_credential", "keyring"),
            ("keyring.set_password", "keyring"),
            ("keyring.delete_password", "keyring"),
            ("modules.utils.download_file.download_url_to_file", "model_download"),
            ("modules.utils.download.ModelDownloader.get", "model_download"),
        ):
            guards.enter_context(patch(target, side_effect=blocked(category)))

        def isolated_settings():
            # QSettings(organization, application) explicitly selects NativeFormat
            # on Windows even after setDefaultFormat(). Use its format overload.
            settings = QSettings(
                QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                "biogoly", "LocalComicTranslate",
            )
            settings.setFallbacksEnabled(False)
            return settings

        guards.enter_context(patch("modules.utils.settings.app_settings", isolated_settings))
        guards.enter_context(patch("modules.utils.settings.legacy_settings", side_effect=blocked("legacy_settings")))

        from modules.utils.settings import app_settings
        from modules.utils.paths import get_default_project_autosave_dir, get_user_data_dir

        def check_isolation():
            settings = app_settings()
            for path in (settings.fileName(), get_user_data_dir(), get_default_project_autosave_dir()):
                if not Path(path).resolve().is_relative_to(root.resolve()):
                    raise AssertionError("Profile escaped temporary storage")
            if settings.allKeys():
                raise AssertionError("Smoke profile unexpectedly contains existing settings")
            settings.setValue("export/project_autosave_enabled", False)
            settings.sync()

        # Do not continue if QSettings isolation failed: the real window saves settings.
        if not checks.run("isolated_profile", check_isolation):
            return

        import comic
        resources = Path(comic.__file__).resolve().parent / "resources"

        def check_resources():
            if QIcon(str(resources / "icons" / "icon.ico")).isNull():
                raise AssertionError("Missing application icon")
            if QPixmap(str(resources / "icons" / "splash.png")).isNull():
                raise AssertionError("Missing splash")
            from PySide6.QtSvg import QSvgRenderer
            for filename in ("clone-stamp.svg", "restore-brush.svg", "eyedropper.svg"):
                if not QSvgRenderer(str(resources / "static" / filename)).isValid():
                    raise AssertionError("Missing SVG resource")
            for filename in ("main.qss", "base.json", "dark.json", "light.json"):
                if not (resources / "static" / filename).is_file():
                    raise AssertionError("Missing theme resource")

        checks.run("resources", check_resources)

        def check_translations():
            languages = ("de", "es", "fr", "it", "ja", "ko", "ru", "tr", "zh-CN")
            for language in languages:
                translator = QTranslator()
                if not translator.load(str(resources / "translations" / "compiled" / f"ct_{language}.qm")):
                    raise AssertionError("Translation catalog failed to load")
            return {"catalogs": len(languages)}

        checks.run("translations", check_translations)

        def check_native_modules():
            import mahotas
            import msgpack
            import numpy as np
            import pyclipper
            import rarfile
            from shapely.geometry import Polygon
            from app.controllers.psd_support import require_photoshopapi
            from azure.ai.vision.imageanalysis import ImageAnalysisClient
            from azure.ai.vision.imageanalysis.models import VisualFeatures

            if not callable(ImageAnalysisClient) or not VisualFeatures.READ:
                raise AssertionError("Azure OCR SDK failed to load")

            require_photoshopapi()
            labels, count = mahotas.label(np.ones((3, 3), dtype=np.uint8))
            if count != 1 or labels.shape != (3, 3):
                raise AssertionError("Connected components failed")
            if Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]).area != 4:
                raise AssertionError("Polygon calculation failed")
            if pyclipper.Area([(0, 0), (2, 0), (2, 2), (0, 2)]) != 4:
                raise AssertionError("Clipper calculation failed")
            if msgpack.unpackb(msgpack.packb({"smoke": 1})) != {"smoke": 1}:
                raise AssertionError("MessagePack roundtrip failed")
            return {"rar_backend_extraction_tested": False}

        checks.run("native_modules", check_native_modules)

        import numpy as np
        from PIL import Image
        import imkit as imk

        fixture = np.zeros((48, 64, 3), dtype=np.uint8)
        fixture[:, :, 0] = np.arange(64, dtype=np.uint8) * 4
        fixture[:, :, 1] = np.arange(48, dtype=np.uint8)[:, None] * 5
        fixture[:, :, 2] = 123
        image_dir = root / "images"
        image_dir.mkdir()
        png_path = image_dir / "page-001.png"

        def check_image():
            imk.write_image(str(png_path), fixture)
            np.testing.assert_array_equal(imk.read_image(str(png_path)), fixture)
            if QPixmap(str(png_path)).isNull():
                raise AssertionError("Qt cannot decode fixture")

        checks.run("png_roundtrip", check_image)

        def check_codecs():
            import io
            for format_name in ("JPEG", "BMP", "WEBP", "AVIF"):
                stream = io.BytesIO()
                Image.fromarray(fixture).save(stream, format=format_name)
                stream.seek(0)
                with Image.open(stream) as decoded:
                    decoded.load()
                    if decoded.size != (64, 48):
                        raise AssertionError("Image codec roundtrip failed")

        checks.run("image_codecs", check_codecs)

        from modules.utils import archives

        def check_archive(kind):
            output = root / f"fixture.{kind}"
            getattr(archives, f"make_{kind}")(str(image_dir), output_path=str(output))
            entries = archives.list_archive_image_entries(str(output))
            if len(entries) != 1:
                raise AssertionError("Archive page count mismatch")
            extracted = root / f"extracted-{kind}.png"
            if not archives.materialize_archive_entry(str(output), entries[0], str(extracted)):
                raise AssertionError("Archive materialization failed")
            np.testing.assert_array_equal(imk.read_image(str(extracted)), fixture)

        for kind in ("cbz", "cb7", "pdf"):
            checks.run(f"{kind}_roundtrip", lambda kind=kind: check_archive(kind))

        def check_psd():
            from app.controllers.psd_exporter import PsdPageData, export_psd_pages
            from app.controllers.psd_importer import import_psd_files
            page = PsdPageData(str(png_path), fixture, {}, [])
            output = export_psd_pages(str(root), [page], "fixture", str(root / "fixture.psd"))
            imported = import_psd_files([output])
            if len(imported) != 1:
                raise AssertionError("PSD page count mismatch")
            np.testing.assert_array_equal(imported[0].rgb_image, fixture)

        checks.run("psd_roundtrip", check_psd)
        checks.run("onnx_cpu", lambda: _onnx_inference(root, cuda=False))
        if require_cuda:
            checks.run("onnx_cuda", lambda: _onnx_inference(root, cuda=True))
        checks.run("language_ocr_routing", _check_language_ocr)
        checks.run("optional_qwen_lora", lambda: _check_qwen_lora(root))

        def check_comfy_optional():
            # Import and construct the integration without a Comfy installation,
            # GPU runtime, server, model download, or background process.
            from uuid import uuid4
            from modules.artistic_edit.backend_contracts import RasterEditSpec, ComfyGenerationSpec
            from modules.artistic_edit.comfy_config import ComfySettings
            from modules.artistic_edit.comfy_runtime import ComfyRuntime
            from modules.artistic_edit.comfy_workflows import build_workflow, FINAL_OUTPUT_NODE
            from app.ui.dialogs.comfyui_settings import ComfyUISettingsDialog
            config = ComfySettings()
            runtime = ComfyRuntime(root / "comfy-smoke")
            spec = ComfyGenerationSpec(
                RasterEditSpec(str(uuid4()), str(root / "input.png"), str(root / "output.png"),
                               "Replace HOLA with HELLO", 272, 400, 42),
                config.connection, config.profile, None, str(uuid4()), str(uuid4()),
            )
            graph = build_workflow(spec, "local-comic-translate/fixture.png")
            if graph[FINAL_OUTPUT_NODE]["class_type"] != "SaveImage":
                raise AssertionError("Missing designated Comfy output")
            dialog = ComfyUISettingsDialog(config, runtime)
            dialog.reject()
            dialog.deleteLater()
            runtime.shutdown()
            if runtime.is_running() or (root / "comfy-smoke").exists():
                raise AssertionError("Optional backend started or wrote runtime state during construction")
            return {"runtime_started": False, "workflow_nodes": len(graph)}

        checks.run("optional_comfyui", check_comfy_optional)

        if getattr(sys, "frozen", False):
            def check_worker_source():
                from modules.artistic_edit.runtime_paths import worker_source_root
                worker_source_root()
                return {"runtime_loaded": False}
            checks.run("optional_worker_source", check_worker_source)

        def check_window():
            from controller import ComicTranslate
            from app.ui.main_window.window import THEME_PRESETS
            window = ComicTranslate()
            try:
                if window.isVisible():
                    raise AssertionError("Smoke window must remain hidden")
                for theme in THEME_PRESETS:
                    window.apply_theme(theme)
                window.image_viewer.setPhoto(QPixmap(str(png_path)))
                if not window.image_viewer.hasPhoto():
                    raise AssertionError("Canvas failed to load image")
                app.processEvents()
                return {"themes": len(THEME_PRESETS), "shown": False}
            finally:
                window._skip_close_prompt = True
                window.close()
                # Drain callbacks while their widgets still exist. Processing
                # timers after DeferredDelete would create artificial failures.
                app.processEvents()
                window.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

        checks.run("main_window", check_window)
        archives.close_pdf_cache()

        def check_offline():
            if blocked_attempts:
                raise AssertionError("Forbidden external operation was attempted")
            return {"network_requests": 0, "model_downloads": 0, "keyring_accesses": 0}

        checks.run("offline_guards", check_offline)

        def check_qt_callbacks():
            if qt_exceptions:
                raise AssertionError("A Qt callback raised during the smoke check")

        checks.run("qt_callbacks", check_qt_callbacks)
    app.quit()


def run_smoke_test(report_path: str, require_cuda: bool = False) -> int:
    """Write a path-free JSON acceptance report; return 0 only when all pass."""
    destination = Path(report_path).expanduser().resolve()
    checks = _Checks()
    with tempfile.TemporaryDirectory(prefix="local-comic-package-smoke-") as temporary:
        root = Path(temporary)
        with _isolated_environment(root):
            checks.run("package_checks", lambda: _exercise_package(root, checks, require_cuda=require_cuda))
    report = {
        "schema_version": 1,
        "frozen": bool(getattr(sys, "frozen", False)),
        "architecture_bits": struct.calcsize("P") * 8,
        "require_cuda": bool(require_cuda),
        "checks": checks.results,
        "ok": bool(checks.results) and all(item["status"] == "passed" for item in checks.results),
    }
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return 2
    return 0 if report["ok"] else 1
