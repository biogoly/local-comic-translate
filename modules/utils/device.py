from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

import onnxruntime as ort

from .paths import get_user_data_dir


logger = logging.getLogger(__name__)


def _preload_onnx_runtime_dependencies() -> None:
    """Load CUDA/cuDNN DLLs supplied by ONNX Runtime's Python extras.

    Passing an empty directory tells recent ONNX Runtime releases to search the
    NVIDIA packages installed in site-packages. Older and CPU-only releases do
    not necessarily expose this helper, so startup remains compatible with
    those installations.
    """
    preload_dlls = getattr(ort, "preload_dlls", None)
    if not callable(preload_dlls):
        return

    try:
        preload_dlls(directory="")
    except TypeError:
        # Compatibility with ONNX Runtime versions that lack ``directory``.
        try:
            preload_dlls()
        except Exception as exc:
            logger.debug("Could not preload ONNX Runtime dependencies: %s", exc)
    except Exception as exc:
        logger.debug("Could not preload ONNX Runtime dependencies: %s", exc)


_preload_onnx_runtime_dependencies()


def torch_available() -> bool:
    """Check if torch is available without raising import errors."""
    try:
        import torch
        return True
    except Exception:
        return False


def _get_available_torch_accelerators() -> list[str]:
    """Return supported non-CPU torch accelerator names that are currently usable."""
    try:
        import torch
    except ImportError:
        return []

    accelerators: list[str] = []

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerators.append("mps")
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            accelerators.append("cuda")
    except Exception:
        pass

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            accelerators.append("xpu")
    except Exception:
        pass

    return accelerators


def resolve_device(use_gpu: bool, backend: str = "onnx") -> str:
    """Return the best available device string for the specified backend.

    Args:
        use_gpu: Whether to use GPU acceleration
        backend: Backend to use ('onnx' or 'torch')

    Returns:
        Device string compatible with the specified backend
    """
    if not use_gpu:
        return "cpu"

    if backend.lower() == "torch":
        return _resolve_torch_device(fallback_to_onnx=True)
    else:
        return _resolve_onnx_device()


def _resolve_torch_device(fallback_to_onnx: bool = False) -> str:
    """Resolve the best available PyTorch device."""
    if not torch_available():
        # Torch not available, fallback to ONNX resolution if requested
        if fallback_to_onnx:
            return _resolve_onnx_device()
        return "cpu"

    accelerators = _get_available_torch_accelerators()
    if accelerators:
        return accelerators[0]

    # Fallback to CPU
    return "cpu"


def _resolve_onnx_device() -> str:
    """Resolve the best available ONNX device."""
    providers = ort.get_available_providers() 

    if not providers:
        return "cpu"

    if "CUDAExecutionProvider" in providers:
        return "cuda"
    
    if "TensorrtExecutionProvider" in providers:
        return "tensorrt"

    if "CoreMLExecutionProvider" in providers:
        return "coreml"
    
    if "ROCMExecutionProvider" in providers:
        return "rocm"

    if "OpenVINOExecutionProvider" in providers:
        return "openvino"

    # Fallback to CPU
    return "cpu"

def tensors_to_device(data: Any, device: str) -> Any:
    """Move tensors in nested containers to device; returns the same structure.
    Supports dict, list/tuple, and tensors. Other objects are returned as-is.
    """
    try:
        import torch
    except Exception:
        # Torch is not available; return data unchanged
        return data

    # Map unknown device strings (onnx-driven) to torch-compatible device
    torch_device = device
    if isinstance(device, str):
        low = device.lower()
        if low in ("cpu", "cuda", "mps", "xpu"):
            torch_device = low
        else:
            # Unknown or ONNX-specific device -> fallback to cpu for torch tensors
            torch_device = "cpu"

    if isinstance(data, torch.Tensor):
        return data.to(torch_device)
    if isinstance(data, Mapping):
        return {k: tensors_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        seq = [tensors_to_device(v, device) for v in data]
        return type(data)(seq) if isinstance(data, tuple) else seq
    return data

def get_providers(device: Optional[str] = None) -> list[Any]:
    """Return a providers list for ONNXRuntime (optionally with provider options).

    Rules:
    - If device is the string 'cpu' (case-insensitive) -> return ['CPUExecutionProvider']
    - Select only the requested accelerator and its required fallbacks
    - TensorRT is used only when explicitly requested; CUDA does not require it
    - If no providers are available, fall back to ['CPUExecutionProvider']
    """
    try:
        available = ort.get_available_providers()
    except Exception:
        available = []

    requested = device.lower().split(":", 1)[0] if isinstance(device, str) else None
    if requested == 'cpu':
        return ['CPUExecutionProvider']

    if not available:
        return ['CPUExecutionProvider']

    if requested is None:
        requested = _resolve_onnx_device()

    provider_names = {
        'cuda': 'CUDAExecutionProvider',
        'tensorrt': 'TensorrtExecutionProvider',
        'coreml': 'CoreMLExecutionProvider',
        'rocm': 'ROCMExecutionProvider',
        'openvino': 'OpenVINOExecutionProvider',
    }
    primary = provider_names.get(requested)
    if primary not in available:
        return ['CPUExecutionProvider']

    configured: list[Any]
    if requested == 'openvino':
        cache_dir = os.path.join(
            get_user_data_dir(), 'models', 'onnx-gpu-cache', 'openvino'
        )
        os.makedirs(cache_dir, exist_ok=True)
        configured = [(primary, {
            'device_type': 'GPU',
            'precision': 'FP32',
            'cache_dir': cache_dir,
        })]
    elif requested == 'tensorrt':
        cache_dir = os.path.join(
            get_user_data_dir(), 'models', 'onnx-gpu-cache', 'tensorrt'
        )
        os.makedirs(cache_dir, exist_ok=True)
        configured = [(primary, {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,
        })]
        # TensorRT delegates unsupported nodes to CUDA before falling back to CPU.
        if 'CUDAExecutionProvider' in available:
            configured.append('CUDAExecutionProvider')
    elif requested == 'coreml':
        cache_dir = os.path.join(
            get_user_data_dir(), 'models', 'onnx-gpu-cache', 'coreml'
        )
        os.makedirs(cache_dir, exist_ok=True)
        configured = [(primary, {'ModelCacheDirectory': cache_dir})]
    else:
        configured = [primary]

    if 'CPUExecutionProvider' not in configured:
        configured.append('CPUExecutionProvider')

    return configured


def is_gpu_available() -> bool:
    """Check if either ONNX or torch can use a supported non-CPU accelerator."""
    try:
        providers = ort.get_available_providers()
    except Exception:
        providers = []

    ignored_providers = {'AzureExecutionProvider', 'CPUExecutionProvider'}
    available = set(providers)

    if not available.issubset(ignored_providers):
        return True

    return bool(_get_available_torch_accelerators())
