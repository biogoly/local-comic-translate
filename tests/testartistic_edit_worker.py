import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import threading
import unittest
from unittest import mock
from unittest.mock import patch
from uuid import uuid4

from PIL import Image
from PySide6.QtCore import QCoreApplication, QObject, QProcess, Signal

from modules.artistic_edit.contracts import (
    ArtisticEditRequest,
    DevicePolicy,
    FluxModelSpec,
    LoraSpec,
    ModelKey,
    encode_json_line,
)
from modules.artistic_edit.flux_worker import (
    FluxModelManager,
    FluxWorkerServer,
    GenerationCancelled,
    RuntimeDependencies,
    classify_error,
)
from modules.artistic_edit.worker_client import FluxWorkerClient


class FakeCuda:
    def __init__(self, events, device_count=2):
        self.events = events
        self._device_count = device_count

    def is_available(self):
        return True

    def device_count(self):
        return self._device_count

    def synchronize(self):
        self.events.append(("cuda_synchronize",))

    def empty_cache(self):
        self.events.append(("cuda_empty_cache",))


class FakeGenerator:
    def __init__(self, events, device):
        self.events = events
        self.device = device

    def manual_seed(self, seed):
        self.events.append(("seed", self.device, seed))
        return self


class FakeTorch:
    bfloat16 = "fake-bfloat16"

    def __init__(self, events, device_count=2):
        self.events = events
        self.cuda = FakeCuda(events, device_count=device_count)

    def Generator(self, device):
        return FakeGenerator(self.events, device)


class FakeVae:
    def __init__(self, events):
        self.events = events

    def enable_tiling(self):
        self.events.append(("vae_tiling",))


class FakePipeline:
    def __init__(self, events):
        self.events = events
        self.vae = FakeVae(events)
        self.call_kwargs = None

    def to(self, device):
        self.events.append(("to", device))
        return self

    def enable_model_cpu_offload(self):
        self.events.append(("model_cpu_offload",))

    def enable_sequential_cpu_offload(self):
        self.events.append(("sequential_cpu_offload",))

    def load_lora_weights(self, path, adapter_name):
        self.events.append(("load_lora", Path(path).name, adapter_name))

    def set_adapters(self, names, adapter_weights):
        self.events.append(("set_adapters", tuple(names), tuple(adapter_weights)))

    def unload_lora_weights(self):
        self.events.append(("unload_lora",))

    def remove_all_hooks(self):
        self.events.append(("remove_hooks",))

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        self.events.append(
            (
                "generate",
                kwargs["width"],
                kwargs["height"],
                kwargs["num_inference_steps"],
                kwargs["guidance_scale"],
            )
        )
        callback = kwargs["callback_on_step_end"]
        for step in range(kwargs["num_inference_steps"]):
            callback(self, step, step, {"latents": object()})
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "red")])


def fake_dependencies(events, *, device_count=2):
    class FakePipelineClass:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            events.append(("from_pretrained", source, kwargs))
            return FakePipeline(events)

    return RuntimeDependencies(
        torch=FakeTorch(events, device_count=device_count),
        pipeline_class=FakePipelineClass,
    )


class FluxWorkerModelManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.events = []
        self.dependencies = fake_dependencies(self.events)
        self.manager = FluxModelManager(dependencies=self.dependencies)
        self.model4 = FluxModelSpec(ModelKey.KLEIN_4B, "model-4b", "Apache 2.0")
        self.model9 = FluxModelSpec(ModelKey.KLEIN_9B, "model-9b", "Non-commercial")
        self.cancel_event = threading.Event()
        self.status_events = []

    def status(self, event, **fields):
        self.status_events.append((event, fields))

    def ensure(
        self,
        model=None,
        lora=None,
        policy=DevicePolicy.SINGLE_GPU_0,
    ):
        return self.manager.ensure_loaded(
            model or self.model4,
            lora,
            policy,
            self.cancel_event,
            self.status,
        )

    def make_lora(self, name, content, scale=1.0, model=ModelKey.KLEIN_4B):
        path = self.root / name
        width = 3072 if model == ModelKey.KLEIN_4B else 4096
        header = json.dumps(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": {
                    "dtype": "F32",
                    "shape": [8, width],
                    "data_offsets": [0, 0],
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        content = len(header).to_bytes(8, "little") + header + content
        path.write_bytes(content)
        return LoraSpec(
            str(path),
            model,
            scale,
            hashlib.sha256(content).hexdigest(),
        )

    def make_request(self, **overrides):
        input_path = self.root / "input.png"
        Image.new("RGB", (32, 16), "blue").save(input_path)
        values = {
            "request_id": str(uuid4()),
            "input_path": str(input_path),
            "output_path": str(self.root / "output.png"),
            "prompt": "Replace the lettering",
            "width": 32,
            "height": 16,
            "steps": 4,
            "guidance": 1.0,
            "seed": 123,
            "model": self.model4,
            "lora": None,
            "device_policy": DevicePolicy.SINGLE_GPU_0,
        }
        values.update(overrides)
        return ArtisticEditRequest(**values)

    def test_heavy_dependencies_are_imported_only_on_first_load(self):
        manager = FluxModelManager()
        manager.unload()
        with patch(
            "modules.artistic_edit.flux_worker.import_runtime_dependencies",
            return_value=self.dependencies,
        ) as importer:
            manager.ensure_loaded(
                self.model4,
                None,
                DevicePolicy.CPU,
                self.cancel_event,
                self.status,
            )
        importer.assert_called_once_with()

    def test_same_pipeline_key_is_reused_and_vae_tiling_is_enabled(self):
        first = self.ensure()
        second = self.ensure()
        self.assertIs(first, second)
        self.assertEqual(sum(event[0] == "from_pretrained" for event in self.events), 1)
        self.assertEqual(sum(event[0] == "vae_tiling" for event in self.events), 1)
        self.assertIn(("to", "cuda:0"), self.events)

    def test_model_switch_unloads_old_pipeline_before_new_load(self):
        self.ensure()
        self.ensure(model=self.model9)
        removal = self.events.index(("remove_hooks",))
        second_load = [
            index for index, event in enumerate(self.events) if event[0] == "from_pretrained"
        ][1]
        self.assertLess(removal, second_load)

    def test_balanced_and_offload_policies_use_explicit_paths(self):
        self.ensure(policy=DevicePolicy.BALANCED)
        load_event = next(event for event in self.events if event[0] == "from_pretrained")
        self.assertEqual(load_event[2]["device_map"], "balanced")
        self.assertEqual(load_event[2]["max_memory"], {0: "21GiB", 1: "21GiB"})
        self.manager.unload()
        self.ensure(policy=DevicePolicy.SEQUENTIAL_CPU_OFFLOAD)
        self.assertIn(("sequential_cpu_offload",), self.events)

    def test_lora_load_scale_change_replacement_and_removal_order(self):
        first = self.make_lora("first.safetensors", b"first", scale=0.5)
        first_rescaled = LoraSpec(first.path, first.base_model, 1.25, first.sha256)
        second = self.make_lora("second.safetensors", b"second", scale=0.8)
        self.ensure(lora=first)
        self.ensure(lora=first_rescaled)
        self.ensure(lora=second)
        self.ensure(lora=None)
        lora_events = [event for event in self.events if "lora" in event[0] or event[0] == "set_adapters"]
        self.assertEqual(
            lora_events,
            [
                ("load_lora", "first.safetensors", "artistic_edit"),
                ("set_adapters", ("artistic_edit",), (0.5,)),
                ("set_adapters", ("artistic_edit",), (1.25,)),
                ("unload_lora",),
                ("load_lora", "second.safetensors", "artistic_edit"),
                ("set_adapters", ("artistic_edit",), (0.8,)),
                ("unload_lora",),
            ],
        )

    def test_changed_lora_file_is_rejected_before_diffusers_load(self):
        lora = self.make_lora("changed.safetensors", b"before")
        Path(lora.path).write_bytes(b"after")
        with self.assertRaisesRegex(ValueError, "changed after selection"):
            self.ensure(lora=lora)
        self.assertFalse(any(event[0] == "load_lora" for event in self.events))

    def test_generation_forwards_parameters_reports_progress_and_writes_verified_png(self):
        request = self.make_request()
        result = self.manager.generate(
            request,
            self.cancel_event,
            self.status,
        )
        self.assertEqual(result.seed, 123)
        self.assertEqual((result.width, result.height), (32, 16))
        with Image.open(result.output_path) as output:
            self.assertEqual(output.size, (32, 16))
            self.assertEqual(output.convert("RGB").getpixel((0, 0)), (255, 0, 0))
        self.assertIn(("seed", "cpu", 123), self.events)
        self.assertEqual(
            [fields["step"] for event, fields in self.status_events if event == "progress"],
            [1, 2, 3, 4],
        )

    def test_callback_cancellation_does_not_write_output_and_pipeline_remains_reusable(self):
        request = self.make_request()

        def cancelling_status(event, **fields):
            if event == "progress" and fields["step"] == 1:
                self.cancel_event.set()

        with self.assertRaises(GenerationCancelled):
            self.manager.generate(request, self.cancel_event, cancelling_status)
        self.assertFalse(Path(request.output_path).exists())
        original_pipeline = self.manager.pipeline
        self.cancel_event.clear()
        second = self.make_request(output_path=str(self.root / "second.png"))
        self.manager.generate(second, self.cancel_event, self.status)
        self.assertIs(self.manager.pipeline, original_pipeline)

    def test_unload_releases_hooks_and_cuda_cache(self):
        self.ensure()
        self.manager.unload()
        self.assertIsNone(self.manager.pipeline)
        self.assertIn(("remove_hooks",), self.events)
        self.assertIn(("cuda_synchronize",), self.events)
        self.assertIn(("cuda_empty_cache",), self.events)


class FluxWorkerServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = io.StringIO()

    def messages(self):
        return [json.loads(line) for line in self.output.getvalue().splitlines() if line]

    def test_ping_returns_capabilities_as_one_json_object(self):
        server = FluxWorkerServer(
            self.root,
            DevicePolicy.CPU,
            manager=FluxModelManager(dependencies=fake_dependencies([])),
            stdout=self.output,
        )
        request_id = str(uuid4())
        server.process_command({"id": request_id, "op": "ping"})
        self.assertEqual(self.messages()[0]["event"], "ready")
        self.assertEqual(self.messages()[0]["id"], request_id)
        self.assertEqual(self.messages()[0]["capabilities"]["protocol_version"], 1)
    def test_load_command_parses_model_lora_and_policy(self):
        server = FluxWorkerServer(
            self.root,
            DevicePolicy.CPU,
            manager=FluxModelManager(dependencies=fake_dependencies([])),
            stdout=self.output,
        )
        model = FluxModelSpec(ModelKey.KLEIN_4B, "model", "license")
        base = {
            "id": str(uuid4()),
            "op": "load",
            "model": model.to_dict(),
            "lora": None,
            "device_policy": DevicePolicy.CPU.value,
        }
        parsed_model, parsed_lora, parsed_policy = server._parse_load(base)
        self.assertEqual(parsed_model, model)
        self.assertIsNone(parsed_lora)
        self.assertEqual(parsed_policy, DevicePolicy.CPU)

    def test_invalid_request_returns_structured_error(self):
        server = FluxWorkerServer(
            self.root,
            DevicePolicy.CPU,
            manager=FluxModelManager(dependencies=fake_dependencies([])),
            stdout=self.output,
        )
        request_id = str(uuid4())
        server.process_command({"id": request_id, "op": "generate"})
        message = self.messages()[0]
        self.assertEqual(message["event"], "error")
        self.assertEqual(message["code"], "invalid_request")

    def test_prequeued_cancel_aborts_generation(self):
        request_id = str(uuid4())
        model = FluxModelSpec(ModelKey.KLEIN_4B, "model", "license")
        input_path = self.root / "input.png"
        Image.new("RGB", (16, 16)).save(input_path)
        request = ArtisticEditRequest(
            request_id,
            str(input_path),
            str(self.root / "output.png"),
            "edit",
            16,
            16,
            1,
            1.0,
            0,
            model,
            None,
            DevicePolicy.CPU,
        )
        stdin = io.StringIO(
            encode_json_line(request.to_command())
            + encode_json_line({"id": request_id, "op": "cancel"})
        )
        server = FluxWorkerServer(
            self.root,
            DevicePolicy.CPU,
            manager=FluxModelManager(dependencies=fake_dependencies([])),
            stdin=stdin,
            stdout=self.output,
        )
        server.serve()
        events = [message["event"] for message in self.messages() if message["id"] == request_id]
        self.assertEqual(events[-1], "cancelled")
        self.assertNotIn("result", events)

    def test_reader_waits_off_pipe_during_runtime_setup_then_resumes(self):
        request_id = str(uuid4())
        model = FluxModelSpec(ModelKey.KLEIN_4B, "model", "license")
        input_path = self.root / "input.png"
        Image.new("RGB", (16, 16)).save(input_path)
        request = ArtisticEditRequest(
            request_id,
            str(input_path),
            str(self.root / "output.png"),
            "edit",
            16,
            16,
            1,
            1.0,
            0,
            model,
            None,
            DevicePolicy.CPU,
        )
        observed_reader_states = []

        class PausingManager:
            def generate(
                manager_self,
                _request,
                cancel_event,
                _status,
                runtime_ready,
            ):
                observed_reader_states.append(server._reader_resume.is_set())
                runtime_ready()
                if not cancel_event.wait(1):
                    raise RuntimeError("Reader did not resume to receive cancellation.")
                raise GenerationCancelled()

            def unload(manager_self):
                pass

        stdin = io.StringIO(
            encode_json_line(request.to_command())
            + encode_json_line({"id": request_id, "op": "cancel"})
        )
        server = FluxWorkerServer(
            self.root,
            DevicePolicy.CPU,
            manager=PausingManager(),
            stdin=stdin,
            stdout=self.output,
        )
        server.serve()

        self.assertEqual(observed_reader_states, [False])
        events = [message["event"] for message in self.messages() if message["id"] == request_id]
        self.assertEqual(events, ["cancelled"])

    def test_error_classifier_recognizes_cuda_oom(self):
        code, message = classify_error(RuntimeError("CUDA out of memory"))
        self.assertEqual(code, "cuda_oom")
        self.assertIn("GPU memory", message)

    def test_error_classifier_recognizes_incompatible_lora(self):
        from modules.artistic_edit.lora_registry import LoraCompatibilityError

        code, _message = classify_error(LoraCompatibilityError("wrong width"))
        self.assertEqual(code, "lora_incompatible")

    def test_subprocess_ping_uses_json_stdout_without_optional_torch_import(self):
        import subprocess

        ping_id = str(uuid4())
        shutdown_id = str(uuid4())
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "modules.artistic_edit.flux_worker",
                "--temp-root",
                str(self.root),
                "--device-policy",
                "cpu",
            ],
            input=(
                encode_json_line({"id": ping_id, "op": "ping"})
                + encode_json_line({"id": shutdown_id, "op": "shutdown"})
            ),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        messages = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([message["id"] for message in messages], [ping_id, shutdown_id])
        self.assertEqual([message["event"] for message in messages], ["ready", "ready"])


class FakeProcess(QObject):
    started = Signal()
    readyReadStandardOutput = Signal()
    readyReadStandardError = Signal()
    finished = Signal(int, object)
    errorOccurred = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = QProcess.ProcessState.NotRunning
        self.program = None
        self.arguments = None
        self.environment = None
        self.working_directory = None
        self.writes = []

    def state(self):
        return self._state

    def setProcessEnvironment(self, environment):
        self.environment = environment

    def setWorkingDirectory(self, directory):
        self.working_directory = directory

    def setProgram(self, program):
        self.program = program

    def setArguments(self, arguments):
        self.arguments = arguments

    def setProcessChannelMode(self, _mode):
        pass

    def start(self):
        self._state = QProcess.ProcessState.Running
        self.started.emit()

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def readAllStandardOutput(self):
        return b""

    def readAllStandardError(self):
        return b""

    def waitForFinished(self, _timeout):
        return True

    def kill(self):
        self._state = QProcess.ProcessState.NotRunning


class FluxWorkerClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.client = FluxWorkerClient(
            self.root,
            python_executable=sys.executable,
            process_factory=FakeProcess,
        )

    def test_start_uses_unbuffered_module_worker_and_selected_gpu(self):
        self.client.start(DevicePolicy.SINGLE_GPU_1)
        process = self.client.process
        self.assertEqual(process.program, str(Path(sys.executable).resolve()))
        self.assertEqual(process.arguments[:3], ["-u", "-m", "modules.artistic_edit.flux_worker"])
        self.assertEqual(process.environment.value("CUDA_VISIBLE_DEVICES"), "1")
        ping = json.loads(process.writes[0])
        self.assertEqual(ping["op"], "ping")

    def test_worker_can_receive_isolated_python_path_entries(self):
        extra = self.root / "flux-site-packages"
        with mock.patch.dict(
            os.environ,
            {"PYTHONPATH": "poisoned-main-app-path", "PYTHONHOME": "poisoned-home"},
        ):
            client = FluxWorkerClient(
                self.root,
                python_executable=sys.executable,
                python_path_entries=[extra],
                process_factory=FakeProcess,
            )
            client.start(DevicePolicy.CPU)
        value = client.process.environment.value("PYTHONPATH")
        self.assertEqual(value, str(extra.resolve()))
        self.assertFalse(client.process.environment.contains("PYTHONHOME"))

    def test_worker_does_not_inherit_main_application_python_paths(self):
        with mock.patch.dict(
            os.environ,
            {"PYTHONPATH": "poisoned-main-app-path", "PYTHONHOME": "poisoned-home"},
        ):
            self.client.start(DevicePolicy.CPU)
        self.assertFalse(self.client.process.environment.contains("PYTHONPATH"))
        self.assertFalse(self.client.process.environment.contains("PYTHONHOME"))

    def test_mismatched_or_late_results_are_ignored(self):
        active_id = str(uuid4())
        late_id = str(uuid4())
        self.client._active_generation_id = active_id
        self.client._pending[late_id] = "generate"
        received = []
        self.client.result.connect(received.append)
        self.client._consume_stdout_line(
            encode_json_line(
                {
                    "id": late_id,
                    "event": "result",
                    "output_path": str(self.root / "late.png"),
                    "seed": 0,
                    "elapsed_seconds": 1.0,
                    "width": 16,
                    "height": 16,
                }
            )
        )
        self.assertEqual(received, [])

    def test_active_result_is_validated_and_emitted(self):
        request_id = str(uuid4())
        self.client._active_generation_id = request_id
        self.client._pending[request_id] = "generate"
        received = []
        self.client.result.connect(received.append)
        self.client._consume_stdout_line(
            encode_json_line(
                {
                    "id": request_id,
                    "event": "result",
                    "output_path": str(self.root / "result.png"),
                    "seed": 5,
                    "elapsed_seconds": 2.0,
                    "width": 16,
                    "height": 16,
                }
            )
        )
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].seed, 5)
        self.assertIsNone(self.client._active_generation_id)


if __name__ == "__main__":
    unittest.main()
