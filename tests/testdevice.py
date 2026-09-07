import tempfile
import unittest
from unittest.mock import patch

from modules.utils import device


class DeviceProviderTests(unittest.TestCase):
    def test_cuda_selection_does_not_probe_tensorrt(self):
        available = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch.object(device.ort, "get_available_providers", return_value=available):
            self.assertEqual(
                device.get_providers("cuda"),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

    def test_default_selection_prefers_cuda_without_tensorrt(self):
        available = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch.object(device.ort, "get_available_providers", return_value=available):
            self.assertEqual(
                device.get_providers(),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

    def test_cuda_selection_falls_back_to_cpu_when_unavailable(self):
        with patch.object(
            device.ort,
            "get_available_providers",
            return_value=["CPUExecutionProvider"],
        ):
            self.assertEqual(device.get_providers("cuda"), ["CPUExecutionProvider"])

    def test_tensorrt_is_available_only_when_explicitly_requested(self):
        available = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(device.ort, "get_available_providers", return_value=available),
                patch.object(device, "get_user_data_dir", return_value=temp_dir),
            ):
                providers = device.get_providers("tensorrt")

        self.assertEqual(providers[0][0], "TensorrtExecutionProvider")
        self.assertEqual(
            providers[1:],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
