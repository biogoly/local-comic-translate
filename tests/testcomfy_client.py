from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.parse import urlsplit
from uuid import uuid4

from PIL import Image
from PySide6 import QtCore
import requests

from modules.artistic_edit.comfy_client import ComfyCancelled, ComfyClientError, generate_comfy
from modules.artistic_edit.comfy_job import ComfyEditJob
from tests.testcomfy_workflows import fake_nodes, make_spec


class Response:
    def __init__(self, data=None, *, content=None, status=200):
        self.status_code = status
        self.content = content if content is not None else json.dumps(data).encode()
        self.headers = {"Content-Length": str(len(self.content))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, size):
        yield self.content


class FakeServer:
    def __init__(self, spec):
        self.spec = spec
        self.calls = []
        self.version = "0.37.0"
        self.nodes = fake_nodes(spec.profile)
        self.id = str(uuid4())
        self.foreign_id = str(uuid4())
        self.cancelled = False
        self.post_error = None
        self.recover_submission = False
        self.history_failures = 0
        self.running = False
        self.on_prompt = None
        self.on_view = None
        self.upload = {"name": "input (1).png", "subfolder": "local-comic-translate/test", "type": "input"}
        self.descriptor = {"filename": "final.png", "subfolder": "local-comic-translate", "type": "output"}
        buffer = BytesIO()
        Image.new("RGB", (spec.raster.width, spec.raster.height), "red").save(buffer, format="PNG")
        self.png = buffer.getvalue()

    def request(self, method, url, **kwargs):
        path = urlsplit(url).path
        self.calls.append((method, path, kwargs))
        if path == "/system_stats":
            return Response({"system": {"comfyui_version": self.version}})
        if path == "/object_info":
            return Response(self.nodes)
        if path == "/upload/image":
            return Response(self.upload)
        if path == "/prompt":
            if self.on_prompt:
                self.on_prompt()
            if self.post_error:
                if self.recover_submission:
                    self.id = kwargs["json"]["prompt_id"]
                raise self.post_error
            return Response({"prompt_id": self.id, "number": 0, "node_errors": {}})
        if path.startswith("/api/jobs/"):
            self.cancelled = True
            return Response({"cancelled": True})
        if path == "/queue":
            if method == "POST":
                return Response(content=b"")
            pending = []
            if self.running and not self.cancelled:
                pending.append([0, self.id])
            return Response({"queue_running": pending, "queue_pending": [[1, self.foreign_id]]})
        if path.startswith("/history/"):
            if self.history_failures:
                self.history_failures -= 1
                raise requests.ConnectionError("private transport text")
            if self.running and not self.cancelled:
                return Response({})
            if self.post_error and not self.recover_submission:
                return Response({})
            status = {"completed": True, "status_str": "success", "messages": []}
            if self.cancelled:
                status = {"completed": False, "status_str": "error", "messages": [["execution_interrupted", {}]]}
            return Response({self.id: {"status": status, "outputs": {"17": {"images": [self.descriptor]}, "preview": {"images": [{"filename": "wrong.png"}]}}}})
        if path == "/view":
            if self.on_view:
                self.on_view()
            return Response(content=self.png)
        raise AssertionError((method, path))


class ComfyClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.spec = make_spec(Path(self.temp.name))
        Image.new("RGB", (272, 400), "white").save(self.spec.raster.input_path)
        self.http = FakeServer(self.spec)
        self.cancel = threading.Event()

    def generate(self, **kwargs):
        return generate_comfy(self.spec, self.cancel, http=self.http, poll_interval=0, **kwargs)

    def test_success_uses_returned_upload_and_designated_output(self):
        result = self.generate()
        posts = [call for call in self.http.calls if call[:2] == ("POST", "/prompt")]
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0][2]["json"]["prompt"]["4"]["inputs"]["image"], "local-comic-translate/test/input (1).png")
        self.assertEqual(self.http.calls[-1][2]["params"]["filename"], "final.png")
        with Image.open(result.result.output_path) as image:
            self.assertEqual(image.size, (272, 400))
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result.metadata["precision"], "Native checkpoint")
        self.assertFalse(self.http.trust_env)
        self.assertTrue(all(call[2]["allow_redirects"] is False for call in self.http.calls))

    def test_wrong_dimensions_never_overwrite_output_and_release_after_fetch(self):
        Path(self.spec.raster.output_path).write_bytes(b"existing")
        buffer = BytesIO()
        Image.new("RGB", (256, 256)).save(buffer, format="PNG")
        self.http.png = buffer.getvalue()
        terminal = []
        self.http.on_view = lambda: self.assertEqual(terminal, [])
        with self.assertRaisesRegex(ComfyClientError, "dimensions"):
            self.generate(on_terminal=terminal.append)
        self.assertEqual(Path(self.spec.raster.output_path).read_bytes(), b"existing")
        self.assertEqual(terminal, [self.http.id])

    def test_descriptor_traversal_never_downloaded(self):
        self.http.descriptor["subfolder"] = "../outside"
        with self.assertRaises(ComfyClientError):
            self.generate()
        self.assertFalse(any(path == "/view" for _, path, _ in self.http.calls))

    def test_upload_traversal_fails_before_submit(self):
        self.http.upload["name"] = "../input.png"
        with self.assertRaises(ValueError):
            self.generate()
        self.assertFalse(any(path == "/prompt" for _, path, _ in self.http.calls))

    def test_missing_model_fails_before_upload(self):
        self.http.nodes["UNETLoader"]["input"]["required"]["unet_name"] = [[]]
        with self.assertRaisesRegex(ComfyClientError, "unavailable"):
            self.generate()
        self.assertFalse(any(method == "POST" for method, _, _ in self.http.calls))

    def test_windows_catalog_names_keep_exact_server_spelling(self):
        profile = replace(self.spec.profile, diffusion_model="models/klein.safetensors",
                          text_encoder="encoders/qwen.safetensors", vae="vae/flux.safetensors")
        self.spec = replace(self.spec, profile=profile)
        for class_type, field, value in (("UNETLoader", "unet_name", profile.diffusion_model),
                                         ("CLIPLoader", "clip_name", profile.text_encoder),
                                         ("VAELoader", "vae_name", profile.vae)):
            self.http.nodes[class_type]["input"]["required"][field] = [[value.replace("/", "\\")]]
        self.generate()
        graph = next(kwargs["json"]["prompt"] for method, path, kwargs in self.http.calls if path == "/prompt")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "models\\klein.safetensors")
        self.assertEqual(graph["2"]["inputs"]["clip_name"], "encoders\\qwen.safetensors")
        self.assertEqual(graph["3"]["inputs"]["vae_name"], "vae\\flux.safetensors")

    def test_uncertain_submission_is_never_reposted_or_silently_retried(self):
        self.http.post_error = requests.Timeout("private transport text")
        with self.assertRaisesRegex(ComfyClientError, "not resubmitted") as caught:
            self.generate()
        self.assertEqual(sum(path == "/prompt" for _, path, _ in self.http.calls), 1)
        self.assertNotIn("private", str(caught.exception))

    def test_accepted_ambiguous_submission_recovers_by_id_without_repost(self):
        self.http.post_error = requests.Timeout()
        self.http.recover_submission = True
        submitted = []
        self.generate(on_submitted=submitted.append)
        self.assertEqual(submitted, [self.spec.raster.request_id])
        self.assertEqual(sum(path == "/prompt" for _, path, _ in self.http.calls), 1)

    def test_transient_history_loss_recovers_same_job(self):
        self.http.history_failures = 3
        self.generate()
        self.assertEqual(sum(path == "/prompt" for _, path, _ in self.http.calls), 1)

    def test_targeted_cancel_leaves_foreign_queue_and_no_output(self):
        self.http.running = True
        self.http.on_prompt = self.cancel.set
        with self.assertRaises(ComfyCancelled) as caught:
            self.generate()
        self.assertFalse(caught.exception.server_may_continue)
        self.assertIn(("POST", f"/api/jobs/{self.http.id}/cancel"), [(m, p) for m, p, _ in self.http.calls])
        self.assertFalse(any(path in {"/free", "/interrupt"} for _, path, _ in self.http.calls))
        self.assertFalse(Path(self.spec.raster.output_path).exists())

    def test_legacy_cancel_deletes_only_ours_and_reports_running_work(self):
        self.http.version = "0.36.0"
        self.http.running = True
        self.http.on_prompt = self.cancel.set
        with self.assertRaises(ComfyCancelled) as caught:
            self.generate()
        self.assertTrue(caught.exception.server_may_continue)
        posts = [(path, kwargs["json"]) for method, path, kwargs in self.http.calls if method == "POST" and path == "/queue"]
        self.assertEqual(posts, [("/queue", {"delete": [self.http.id]})])
        self.assertFalse(any(path == "/interrupt" for _, path, _ in self.http.calls))

    def test_malformed_queue_cannot_confirm_cancellation(self):
        original = self.http.request

        def malformed(method, url, **kwargs):
            response = original(method, url, **kwargs)
            if method == "GET" and (urlsplit(url).path.startswith("/history/") or urlsplit(url).path == "/queue"):
                return Response({})
            return response

        self.http.request = malformed
        self.http.on_prompt = self.cancel.set
        terminal = []
        with self.assertRaises(ComfyCancelled) as caught:
            self.generate(on_terminal=terminal.append)
        self.assertTrue(caught.exception.server_may_continue)
        self.assertEqual(terminal, [])

    def test_cancel_before_transfer(self):
        self.cancel.set()
        with self.assertRaises(ComfyCancelled):
            self.generate()
        self.assertEqual(self.http.calls, [])

    def test_qt_job_releases_observer_on_failure_and_emits_one_terminal(self):
        class Runtime:
            active = set()

            def ensure_started(runtime, config, cancelled, on_status):
                return config.base_url

            def register_prompt(runtime, prompt_id):
                runtime.active.add(prompt_id)

            def finish_prompt(runtime, prompt_id):
                runtime.active.discard(prompt_id)

        runtime = Runtime()
        spec = replace(self.spec, connection=replace(self.spec.connection, mode="managed", installation_root=self.temp.name))

        def failing_generator(spec, cancelled, *, on_submitted, **kwargs):
            on_submitted("accepted-prompt")
            self.assertEqual(runtime.active, {"accepted-prompt"})
            raise ComfyClientError("connection_error", "Connection lost")

        job = ComfyEditJob(spec, runtime=runtime, generator=failing_generator)
        events = []
        job.event.connect(events.append, QtCore.Qt.ConnectionType.DirectConnection)
        job.run()
        self.assertEqual(runtime.active, set())
        self.assertEqual([event.kind for event in events], ["failed"])
        self.assertEqual(events[0].job_token, spec.job_token)


if __name__ == "__main__":
    unittest.main()
