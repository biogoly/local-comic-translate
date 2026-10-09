import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.package_smoke import (
    _Checks, _check_language_ocr, _check_qwen_lora, _field,
    _isolated_environment, _matmul_model, _varint, run_smoke_test,
)


class PackageSmokeTests(unittest.TestCase):
    def test_language_routing_smoke_does_not_load_models(self):
        with patch("modules.utils.download.ModelDownloader.ensure", side_effect=AssertionError("download")), \
                patch("modules.ocr.vietnamese.make_session", side_effect=AssertionError("session")):
            result = _check_language_ocr()
        self.assertEqual(result, {"languages": 3, "models_loaded": False, "inference_tested": False})

    def test_language_routing_smoke_detects_incorrect_factory_dispatch(self):
        with patch("modules.ocr.factory.OCRFactory._create_new_engine", return_value=None):
            with self.assertRaisesRegex(AssertionError, "wrong OCR engine"):
                _check_language_ocr()

    def test_qwen_lora_smoke_exercises_header_and_graph_without_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("subprocess.Popen", side_effect=AssertionError("process")), \
                    patch("requests.sessions.Session.request", side_effect=AssertionError("network")):
                result = _check_qwen_lora(Path(temporary))
        self.assertTrue(result["header_validation"])
        self.assertFalse(result["runtime_started"])
        self.assertFalse(result["inference_tested"])

    def test_qwen_lora_smoke_detects_missing_shape_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("modules.artistic_edit.comfy_lora.validate_comfy_lora_compatibility"):
                with self.assertRaisesRegex(AssertionError, "mismatched shape"):
                    _check_qwen_lora(Path(temporary))

    def test_minimal_protobuf_encoding(self):
        self.assertEqual(_varint(300), b"\xac\x02")
        self.assertEqual(_field(1, 8), b"\x08\x08")
        self.assertEqual(_field(2, "hi"), b"\x12\x02hi")
        self.assertIn(b"MatMul", _matmul_model())
        self.assertLess(len(_matmul_model()), 256)

    def test_onnx_fixture_executes_on_cpu(self):
        import numpy as np
        import onnxruntime as ort
        session = ort.InferenceSession(_matmul_model(), providers=["CPUExecutionProvider"])
        output = session.run(None, {
            "x": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "w": np.array([[5, 6], [7, 8]], dtype=np.float32),
        })[0]
        np.testing.assert_array_equal(output, [[19, 22], [43, 50]])

    def test_environment_isolation_and_restoration(self):
        original = dict(os.environ)
        with tempfile.TemporaryDirectory() as temporary:
            original_tempdir = tempfile.tempdir
            root = Path(temporary)
            with _isolated_environment(root):
                self.assertEqual(Path(os.environ["USERPROFILE"]), root / "home")
                self.assertEqual(Path(os.environ["LOCALAPPDATA"]), root / "local")
                self.assertEqual(Path(tempfile.gettempdir()), root / "tmp")
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(dict(os.environ), original)
            self.assertEqual(tempfile.tempdir, original_tempdir)

    def test_error_report_does_not_expose_exception_message(self):
        checks = _Checks()
        def fail():
            raise ValueError("private-path-and-secret-key")
        with self.assertLogs("app.package_smoke", level="ERROR"):
            self.assertFalse(checks.run("fixture", fail))
        self.assertNotIn("private-path-and-secret-key", json.dumps(checks.results))
        self.assertEqual(checks.results[0]["error_type"], "ValueError")

    def test_report_contract_and_cuda_argument(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            def exercise(root, checks, *, require_cuda):
                self.assertTrue(require_cuda)
                self.assertTrue(root.is_dir())
                checks.run("fixture", lambda: None)
            with patch("app.package_smoke._exercise_package", side_effect=exercise):
                self.assertEqual(run_smoke_test(str(output), require_cuda=True), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["ok"])
            self.assertTrue(report["require_cuda"])
            self.assertNotIn(temporary, json.dumps(report))

    def test_failed_checks_return_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            with patch("app.package_smoke._exercise_package", side_effect=ImportError("private")):
                with self.assertLogs("app.package_smoke", level="ERROR"):
                    self.assertEqual(run_smoke_test(str(output)), 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["ok"])


if __name__ == "__main__":
    unittest.main()
