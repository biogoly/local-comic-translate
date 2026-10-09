"""Bounded local ComfyUI transport. Generation is submitted once; reads recover it."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping
from uuid import UUID

from PIL import Image
import requests

from .backend_contracts import ComfyCompletion, ComfyGenerationSpec
from .comfy_config import ComfyConnectionConfig, QWEN_IMAGE_21_MODEL_KEY, validate_server_name
from .comfy_workflows import FINAL_OUTPUT_NODE, WORKFLOW_VERSION, build_workflow, combo_values, validate_node_schemas, workflow_fingerprint
from .contracts import ArtisticEditResult

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
# These exact runtimes provide atomic job-ID cancellation and caller-supplied prompt IDs.
QUALIFIED_TARGETED_CANCEL_VERSIONS = frozenset({"0.37.0", "0.38.0"})


class ComfyClientError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ComfyCancelled(Exception):
    def __init__(self, server_may_continue=False):
        self.server_may_continue = server_may_continue
        super().__init__("Stopped waiting; the server may continue this edit." if server_may_continue else "Artistic Edit cancelled.")


@dataclass(frozen=True)
class ComfyServerInfo:
    version: str
    nodes: Mapping[str, Any]
    catalogs: Mapping[str, tuple[str, ...]]
    targeted_cancel: bool


class _Transport:
    def __init__(self, config, http, cancelled, clock=time.monotonic):
        self.base_url = config.base_url
        self.http = http
        self.http.trust_env = False
        self.cancelled = cancelled
        self.clock = clock

    def check_cancel(self):
        if self.cancelled.is_set():
            raise ComfyCancelled()

    def read(self, method, path, *, limit=MAX_JSON_BYTES, check_cancel=True, **kwargs):
        for attempt in range(2 if method == "GET" else 1):
            try:
                return self._read_once(method, path, limit=limit, check_cancel=check_cancel, **kwargs)
            except ComfyClientError as exc:
                if method != "GET" or attempt or exc.code != "connection_error":
                    raise
                if check_cancel and self.cancelled.wait(0.15):
                    raise ComfyCancelled() from None

    def _read_once(self, method, path, *, limit, check_cancel, **kwargs):
        if check_cancel:
            self.check_cancel()
        try:
            deadline = self.clock() + 90
            with self.http.request(method, self.base_url + path, timeout=(3.05, 10), allow_redirects=False, stream=True, **kwargs) as response:
                if response.status_code != 200:
                    raise ComfyClientError("http_error", f"ComfyUI returned HTTP {response.status_code}.", status_code=response.status_code)
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > limit:
                    raise ComfyClientError("response_too_large", "ComfyUI response exceeds the size limit.")
                data = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    if self.clock() > deadline:
                        raise ComfyClientError("response_timeout", "ComfyUI response exceeded the read deadline.")
                    if check_cancel:
                        self.check_cancel()
                    data.extend(chunk)
                    if len(data) > limit:
                        raise ComfyClientError("response_too_large", "ComfyUI response exceeds the size limit.")
                return bytes(data)
        except requests.RequestException:
            raise ComfyClientError("connection_error", "Could not communicate with the local ComfyUI server.") from None

    def json(self, method, path, **kwargs):
        raw = self.read(method, path, **kwargs)
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ComfyClientError("invalid_response", "ComfyUI returned invalid JSON.") from exc
        if not isinstance(value, dict):
            raise ComfyClientError("invalid_response", "ComfyUI returned an unexpected response.")
        return value


def _catalog(nodes, class_type, field):
    try:
        schema = nodes[class_type]["input"]["required"][field]
    except (KeyError, TypeError, IndexError):
        return ()
    return combo_values(schema)


def _probe(transport):
    stats = transport.json("GET", "/system_stats")
    nodes = transport.json("GET", "/object_info")
    version = str(stats.get("system", {}).get("comfyui_version", "unknown"))[:128]
    catalogs = {
        "diffusion_models": _catalog(nodes, "UNETLoader", "unet_name"),
        "gguf_models": _catalog(nodes, "UnetLoaderGGUF", "unet_name"),
        "text_encoders": _catalog(nodes, "CLIPLoader", "clip_name"),
        "vae": _catalog(nodes, "VAELoader", "vae_name"),
        "loras": _catalog(nodes, "LoraLoaderModelOnly", "lora_name"),
    }
    return ComfyServerInfo(version, nodes, catalogs, version in QUALIFIED_TARGETED_CANCEL_VERSIONS)


def probe_comfy(config: ComfyConnectionConfig, *, http=None, cancelled=None, clock=time.monotonic) -> ComfyServerInfo:
    owned = http is None
    http = http or requests.Session()
    try:
        return _probe(_Transport(config, http, cancelled or threading.Event(), clock))
    finally:
        if owned:
            http.close()


def _check_catalog(info, profile, lora):
    validate_node_schemas(info.nodes, profile, lora)
    names = {
        "gguf_models" if profile.loader == "gguf" else "diffusion_models": profile.diffusion_model,
        "text_encoders": profile.text_encoder,
        "vae": profile.vae,
    }
    if lora is not None:
        names["loras"] = lora.name
    registered = {}
    for category, selected in names.items():
        matches = [value for value in info.catalogs[category] if value.replace("\\", "/") == selected]
        if not matches:
            raise ComfyClientError("model_missing", f"Selected ComfyUI resource is unavailable: {selected}.")
        # Comfy validates combos against the exact registered string. Windows
        # catalogs may use backslashes even though persisted selections use '/'.
        registered[category] = selected if selected in matches else matches[0]
        validate_server_name(registered[category])
    return registered


def _history_state(record):
    if not isinstance(record, dict):
        return None
    status = record.get("status", {})
    messages = status.get("messages", [])
    kinds = {item[0] for item in messages if isinstance(item, (tuple, list)) and item}
    if "execution_interrupted" in kinds:
        return "cancelled"
    if status.get("status_str") == "error" or "execution_error" in kinds:
        return "failed"
    if status.get("completed") is True and status.get("status_str") == "success":
        return "completed"
    return None


def _queue_contains(queue, prompt_id):
    entries = []
    for key in ("queue_pending", "queue_running"):
        value = queue.get(key)
        if not isinstance(value, list) or any(
            not isinstance(item, (list, tuple)) or len(item) < 2 or not isinstance(item[1], str)
            for item in value
        ):
            raise ComfyClientError("invalid_response", "ComfyUI returned an invalid queue response.")
        entries.extend(value)
    return any(item[1] == prompt_id for item in entries)


def _cancel_server(transport, prompt_id, info, on_terminal, clock):
    targeted = info is not None and info.targeted_cancel
    try:
        if targeted:
            try:
                transport.json("POST", f"/api/jobs/{prompt_id}/cancel", json={}, check_cancel=False)
            except ComfyClientError as exc:
                if exc.status_code not in {404, 405}:
                    raise
                targeted = False
        if not targeted:
            # This deletes only our pending job, never globally interrupts running work.
            transport.read("POST", "/queue", json={"delete": [prompt_id]}, check_cancel=False)
        deadline = clock() + 10
        while True:
            history = transport.json("GET", f"/history/{prompt_id}", check_cancel=False)
            state = _history_state(history.get(prompt_id))
            queue = transport.json("GET", "/queue", check_cancel=False)
            if state is not None or not _queue_contains(queue, prompt_id):
                if on_terminal:
                    on_terminal(prompt_id)
                return False
            if not targeted or clock() >= deadline:
                return True
            time.sleep(0.2)
    except (ComfyClientError, ValueError, TypeError):
        return True


def _output_descriptor(record):
    try:
        images = record["outputs"][FINAL_OUTPUT_NODE]["images"]
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError("Expected one final image")
        image = images[0]
        name = validate_server_name(image["filename"], allow_subfolders=False)
        subfolder = image.get("subfolder", "")
        if subfolder:
            subfolder = validate_server_name(subfolder)
        if image.get("type") not in {"output", "temp"} or not name.lower().endswith(".png"):
            raise ValueError("Unexpected output type")
        return {"filename": name, "subfolder": subfolder, "type": image["type"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise ComfyClientError("invalid_output", "ComfyUI did not return one valid final PNG.") from exc


def _save_output(data, raster, cancelled):
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (raster.width, raster.height):
            raise ComfyClientError("invalid_output", "ComfyUI returned an unexpected image format or dimensions.")
        image.load()
        if cancelled.is_set():
            raise ComfyCancelled()
        rgb = image.convert("RGB")
    output = Path(raster.output_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".comfy-", suffix=".png", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            rgb.save(handle, format="PNG")
        if cancelled.is_set():
            raise ComfyCancelled()
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generate_comfy(spec: ComfyGenerationSpec, cancelled, *, http=None, clock=time.monotonic,
                   on_event=None, on_submitted=None, on_terminal=None, timeout=600, poll_interval=0.5) -> ComfyCompletion:
    """Execute one correlated transaction; on_event receives (kind, payload)."""
    owned = http is None
    http = http or requests.Session()
    transport = _Transport(spec.connection, http, cancelled, clock)
    prompt_id = None
    info = None
    server_terminal = False
    registered_prompt = None
    started = clock()

    def event(kind, payload):
        if on_event:
            on_event(kind, payload)

    def terminal():
        nonlocal server_terminal
        server_terminal = True

    def submitted():
        nonlocal registered_prompt
        if registered_prompt != prompt_id and on_submitted:
            on_submitted(prompt_id)
        registered_prompt = prompt_id

    def check_deadline():
        transport.check_cancel()
        if clock() - started > timeout:
            raise ComfyClientError("timeout", "ComfyUI edit timed out. It was not resubmitted; server work may continue.")

    try:
        event("status", "Connecting to ComfyUI…")
        info = _probe(transport)
        registered = _check_catalog(info, spec.profile, spec.lora)
        raster = spec.raster
        with Image.open(raster.input_path) as source:
            if source.format != "PNG" or source.mode != "RGB" or source.size != (raster.width, raster.height):
                raise ComfyClientError("invalid_input", "Prepared ComfyUI input must be an RGB PNG with the requested dimensions.")
            source.verify()
        if Path(raster.input_path).stat().st_size > MAX_IMAGE_BYTES:
            raise ComfyClientError("invalid_input", "Prepared ComfyUI image exceeds the upload limit.")
        check_deadline()
        event("status", "Uploading edit area…")
        with open(raster.input_path, "rb") as image_file:
            uploaded = transport.json("POST", "/upload/image", files={"image": ("input.png", image_file, "image/png")},
                                      data={"type": "input", "subfolder": f"local-comic-translate/{raster.request_id}", "overwrite": "false"})
        if uploaded.get("type") != "input":
            raise ComfyClientError("invalid_upload", "ComfyUI returned an unexpected upload location.")
        upload_name = validate_server_name(uploaded.get("name"), allow_subfolders=False)
        subfolder = uploaded.get("subfolder", "")
        if subfolder:
            upload_name = validate_server_name(subfolder) + "/" + upload_name
        graph = build_workflow(spec, upload_name)
        graph["1"]["inputs"]["unet_name"] = registered["gguf_models" if spec.profile.loader == "gguf" else "diffusion_models"]
        graph["2"]["inputs"]["clip_name"] = registered["text_encoders"]
        graph["3"]["inputs"]["vae_name"] = registered["vae"]
        if spec.lora is not None:
            graph["18"]["inputs"]["lora_name"] = registered["loras"]
        check_deadline()
        event("status", "Submitting ComfyUI edit…")
        try:
            response = transport.json("POST", "/prompt", json={"prompt": graph, "client_id": spec.client_id,
                                       "prompt_id": raster.request_id, "extra_data": {"local_comic_translate_request": raster.request_id}}, check_cancel=False)
            prompt_id = str(UUID(response["prompt_id"]))
        except (ComfyClientError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ComfyClientError) and exc.status_code is not None and 400 <= exc.status_code < 500:
                raise ComfyClientError("prompt_rejected", "ComfyUI rejected the workflow. Check the selected models and required nodes.") from exc
            # The tested server honors client-supplied IDs. Reads can find acceptance;
            # neither a UUID nor a timeout licenses a second POST.
            prompt_id = raster.request_id
            submitted()
            try:
                history = transport.json("GET", f"/history/{prompt_id}", check_cancel=False)
                queue = transport.json("GET", "/queue", check_cancel=False)
                found = info.targeted_cancel and (prompt_id in history or _queue_contains(queue, prompt_id))
            except ComfyClientError:
                found = False
            if not found:
                raise ComfyClientError("submission_uncertain", "ComfyUI submission could not be confirmed. It was not resubmitted; inspect the server before retrying.") from exc
        submitted()
        event("submitted", prompt_id)
        event("status", "Waiting for ComfyUI generation…")
        reconnecting = False
        while True:
            check_deadline()
            try:
                history = transport.json("GET", f"/history/{prompt_id}")
            except ComfyClientError as exc:
                if exc.code != "connection_error" and exc.status_code not in {502, 503, 504}:
                    raise
                if not reconnecting:
                    event("status", "Connection lost; checking the same ComfyUI edit…")
                    reconnecting = True
                if cancelled.wait(poll_interval):
                    raise ComfyCancelled()
                continue
            if reconnecting:
                event("status", "Waiting for ComfyUI generation…")
                reconnecting = False
            record = history.get(prompt_id)
            state = _history_state(record)
            if state is not None:
                terminal()
                if state == "cancelled":
                    raise ComfyCancelled()
                if state == "failed":
                    raise ComfyClientError("execution_failed", "ComfyUI could not generate the edit. Check its model or memory error log.")
                break
            if cancelled.wait(poll_interval):
                raise ComfyCancelled()
        descriptor = _output_descriptor(record)
        event("status", "Fetching ComfyUI preview…")
        data = transport.read("GET", "/view", params=descriptor, limit=MAX_IMAGE_BYTES)
        _save_output(data, raster, cancelled)
        metadata = {
            "backend": "comfyui", "model_key": spec.profile.model_key,
            "model_source": spec.profile.diffusion_model, "text_encoder": spec.profile.text_encoder,
            "vae": spec.profile.vae, "loader": spec.profile.loader,
            "precision": "GGUF (experimental)" if spec.profile.loader == "gguf" else "Native checkpoint",
            "workflow_id": spec.profile.workflow_id, "workflow_version": WORKFLOW_VERSION,
            "workflow_sha256": workflow_fingerprint(graph), "runtime_version": info.version,
            "seed": raster.seed, "steps": raster.steps, "guidance": spec.cfg,
            "sampler": "euler", "scheduler": "simple" if spec.profile.model_key == QWEN_IMAGE_21_MODEL_KEY else "flux2",
            "width": raster.width, "height": raster.height,
            "lora_name": spec.lora.name if spec.lora else None,
            "lora_scale": spec.lora.scale if spec.lora else None,
            "lora_sha256": spec.lora.sha256 if spec.lora else None,
        }
        if spec.profile.model_key == QWEN_IMAGE_21_MODEL_KEY:
            metadata.update(sampling_width=raster.width + (-raster.width) % 32,
                            sampling_height=raster.height + (-raster.height) % 32,
                            reference_resolution=0, output_alpha="composited_over_input")
        return ComfyCompletion(ArtisticEditResult(raster.request_id, raster.output_path, raster.seed,
                               clock() - started, raster.width, raster.height), metadata)
    except ComfyCancelled:
        may_continue = False
        if prompt_id and not server_terminal:
            may_continue = _cancel_server(transport, prompt_id, info, on_terminal, clock)
        raise ComfyCancelled(may_continue) from None
    finally:
        if owned:
            http.close()
        if server_terminal and on_terminal:
            on_terminal(prompt_id)
