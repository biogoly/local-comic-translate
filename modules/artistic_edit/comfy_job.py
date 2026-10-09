"""Qt job boundary for the optional ComfyUI client and managed runtime."""
from __future__ import annotations

from dataclasses import replace
import threading

from PySide6 import QtCore

from .backend_contracts import ComfyJobEvent
from .comfy_client import ComfyCancelled, ComfyClientError, generate_comfy


class ComfyEditJob(QtCore.QThread):
    event = QtCore.Signal(object)

    def __init__(self, spec, parent=None, *, runtime=None, generator=generate_comfy):
        super().__init__(parent)
        self.spec = spec
        self.runtime = runtime
        self._generate = generator
        self._cancel = threading.Event()
        self._pending_prompt = None
        self._registered_prompts = set()

    def cancel(self):
        self._cancel.set()

    def _emit(self, kind, payload):
        self.event.emit(ComfyJobEvent(self.spec.raster.request_id, self.spec.job_token, kind, payload))

    def _submitted(self, prompt_id):
        self._pending_prompt = prompt_id
        if self.runtime is not None and self.spec.connection.mode == "managed":
            self.runtime.register_prompt(prompt_id)
            self._registered_prompts.add(prompt_id)

    def _terminal(self, prompt_id):
        self._pending_prompt = None

    def run(self):
        try:
            spec = self.spec
            if self._cancel.is_set():
                raise ComfyCancelled()
            if self.runtime is not None:
                base_url = self.runtime.ensure_started(spec.connection, self._cancel, lambda text: self._emit("status", text))
                spec = replace(spec, connection=replace(spec.connection, base_url=base_url))
            elif spec.connection.mode == "managed":
                raise ComfyClientError("runtime_missing", "The managed ComfyUI runtime is unavailable.")
            completion = self._generate(spec, self._cancel, on_event=self._emit,
                                        on_submitted=self._submitted, on_terminal=self._terminal)
            if self._cancel.is_set():
                raise ComfyCancelled()
            self._emit("completed", completion)
        except ComfyCancelled as exc:
            self._emit("cancelled", {"message": str(exc), "server_may_continue": exc.server_may_continue})
        except Exception as exc:
            if self._cancel.is_set():
                may_continue = self._pending_prompt is not None
                self._emit("cancelled", {"message": str(ComfyCancelled(may_continue)), "server_may_continue": may_continue})
            else:
                self._emit("failed", {"code": getattr(exc, "code", "comfyui_error"), "message": str(exc)})
        finally:
            if self.runtime is not None:
                for prompt_id in self._registered_prompts:
                    self.runtime.finish_prompt(prompt_id)
            self._registered_prompts.clear()
