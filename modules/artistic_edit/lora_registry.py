"""Model-specific, lazy-hashing LoRA discovery for FLUX.2 Klein."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, BinaryIO

from .contracts import LoraSpec, ModelKey, validate_lora_scale


MAX_SIDECAR_BYTES = 64 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_MODEL_DIRECTORIES = {
    ModelKey.KLEIN_4B: "4b",
    ModelKey.KLEIN_9B: "9b",
}


class RegistryError(ValueError):
    """Raised when a LoRA file or sidecar cannot be handled safely."""


class LoraCompatibilityError(RegistryError):
    """Raised when tensor metadata does not match the selected Klein model."""


_SINGLE_BLOCK_RE = re.compile(
    r"(?:single_transformer_blocks\.|single_blocks[._])(\d+)"
)
_DOUBLE_BLOCK_RE = re.compile(
    r"(?<!single_)transformer_blocks\.(\d+)|(?<!single_)double_blocks[._](\d+)"
)


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read only a safetensors JSON header, never tensor bodies."""
    candidate = Path(path)
    try:
        file_size = candidate.stat().st_size
        with candidate.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise LoraCompatibilityError("LoRA has a truncated safetensors header.")
            header_length = int.from_bytes(length_bytes, "little", signed=False)
            if (
                header_length <= 0
                or header_length > MAX_SAFETENSORS_HEADER_BYTES
                or header_length > file_size - 8
            ):
                raise LoraCompatibilityError("LoRA has an invalid safetensors header length.")
            raw_header = handle.read(header_length)
    except LoraCompatibilityError:
        raise
    except OSError as exc:
        raise LoraCompatibilityError(f"Could not read LoRA metadata: {exc}") from exc

    def reject_constant(value: str) -> None:
        raise LoraCompatibilityError(
            f"Non-finite safetensors metadata is not allowed: {value}."
        )

    try:
        header = json.loads(raw_header.decode("utf-8"), parse_constant=reject_constant)
    except LoraCompatibilityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LoraCompatibilityError("LoRA safetensors header is not valid JSON.") from exc
    if not isinstance(header, dict):
        raise LoraCompatibilityError("LoRA safetensors header must be a JSON object.")
    return header


def validate_lora_compatibility(
    path: str | Path,
    model_key: ModelKey | str,
) -> None:
    """Reject obvious FLUX.1 and Klein 4B/9B architecture mismatches.

    Klein 4B uses a 3072-wide transformer with 5 double and 20 single
    blocks; Klein 9B uses width 4096 with 8 double and 24 single blocks.
    Ambiguous headers are rejected so Diffusers never partially applies an
    adapter merely because some tensor names happen to match.
    """
    try:
        selected_model = ModelKey(model_key)
    except (TypeError, ValueError) as exc:
        raise LoraCompatibilityError(f"Unknown FLUX model key: {model_key!r}.") from exc
    header = read_safetensors_header(path)
    dimensions: set[int] = set()
    double_blocks: set[int] = set()
    single_blocks: set[int] = set()
    tensor_count = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            continue
        shape = metadata.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in shape
        ):
            continue
        tensor_count += 1
        dimensions.update(shape)
        single_match = _SINGLE_BLOCK_RE.search(name)
        if single_match:
            single_blocks.add(int(single_match.group(1)))
        else:
            double_match = _DOUBLE_BLOCK_RE.search(name)
            if double_match:
                block_text = double_match.group(1) or double_match.group(2)
                double_blocks.add(int(block_text))

    if not tensor_count:
        raise LoraCompatibilityError("LoRA contains no recognizable tensor metadata.")
    has_4b_width = 3072 in dimensions
    has_9b_width = 4096 in dimensions
    if has_4b_width and has_9b_width:
        raise LoraCompatibilityError("LoRA mixes 4B and 9B transformer dimensions.")

    max_double = max(double_blocks, default=-1)
    max_single = max(single_blocks, default=-1)
    if selected_model == ModelKey.KLEIN_4B:
        if has_9b_width:
            raise LoraCompatibilityError("This is a Klein 9B LoRA; select the 9B model.")
        if not has_4b_width:
            raise LoraCompatibilityError("Could not verify this LoRA as Klein 4B compatible.")
        if max_double > 4 or max_single > 19:
            raise LoraCompatibilityError(
                "This 3072-wide LoRA targets more blocks than Klein 4B and is likely FLUX.1."
            )
    else:
        if has_4b_width:
            raise LoraCompatibilityError("This is a 4B/FLUX.1-width LoRA; select a Klein 9B LoRA.")
        if not has_9b_width:
            raise LoraCompatibilityError("Could not verify this LoRA as Klein 9B compatible.")
        if max_double > 7 or max_single > 23:
            raise LoraCompatibilityError("This LoRA targets more blocks than Klein 9B.")


@dataclass(frozen=True)
class LoraEntry:
    path: str
    filename: str
    display_name: str
    base_model: ModelKey
    trigger_words: tuple[str, ...]
    default_scale: float
    description: str
    file_size: int
    mtime_ns: int
    sidecar_path: str | None = None
    sidecar_error: str | None = None
    sha256: str | None = None


class LoraRegistry:
    """Discover LoRAs under ``<root>/4b`` or ``<root>/9b`` only.

    Directory placement is authoritative. A malformed sidecar never moves an
    adapter into another model list and never prevents the adapter itself from
    appearing with safe fallback metadata.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self._checksum_cache: dict[str, tuple[int, int, str]] = {}

    @staticmethod
    def _model_key(value: ModelKey | str) -> ModelKey:
        try:
            return ModelKey(value)
        except (TypeError, ValueError) as exc:
            raise RegistryError(f"Unknown FLUX model key: {value!r}.") from exc

    def folder_for(self, model_key: ModelKey | str) -> Path:
        key = self._model_key(model_key)
        return self.root / _MODEL_DIRECTORIES[key]

    def _path_is_inside(self, path: Path, folder: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(folder.resolve(strict=False))
        except ValueError:
            return False
        return True

    @staticmethod
    def _fallback_metadata(path: Path, model_key: ModelKey) -> dict[str, Any]:
        return {
            "display_name": path.stem,
            "base_model": model_key,
            "trigger_words": (),
            "default_scale": 1.0,
            "description": "",
        }

    @staticmethod
    def _read_limited(handle: BinaryIO, maximum: int) -> bytes:
        data = handle.read(maximum + 1)
        if len(data) > maximum:
            raise RegistryError(f"LoRA sidecar exceeds {maximum} bytes.")
        return data

    def _sidecar_metadata(
        self,
        sidecar: Path,
        model_key: ModelKey,
    ) -> dict[str, Any]:
        with sidecar.open("rb") as handle:
            raw = self._read_limited(handle, MAX_SIDECAR_BYTES)

        def reject_constant(value: str) -> None:
            raise RegistryError(f"Non-finite sidecar value is not allowed: {value}.")

        try:
            metadata = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)
        except RegistryError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryError("LoRA sidecar is not valid UTF-8 JSON.") from exc
        if not isinstance(metadata, dict):
            raise RegistryError("LoRA sidecar must contain a JSON object.")

        allowed = {
            "display_name",
            "base_model",
            "trigger_words",
            "default_scale",
            "description",
        }
        unexpected = set(metadata) - allowed
        if unexpected:
            raise RegistryError(
                f"Unexpected LoRA sidecar fields: {', '.join(sorted(unexpected))}."
            )

        base_model = metadata.get("base_model", model_key.value)
        try:
            sidecar_model = ModelKey(base_model)
        except (TypeError, ValueError) as exc:
            raise RegistryError("LoRA sidecar base_model is invalid.") from exc
        if sidecar_model != model_key:
            raise RegistryError(
                f"LoRA sidecar targets {sidecar_model.value}, not {model_key.value}."
            )

        display_name = metadata.get("display_name")
        if display_name is not None and (
            not isinstance(display_name, str) or not display_name.strip()
        ):
            raise RegistryError("LoRA sidecar display_name must be a non-empty string.")

        trigger_words = metadata.get("trigger_words", [])
        if not isinstance(trigger_words, list) or any(
            not isinstance(word, str) or not word.strip() for word in trigger_words
        ):
            raise RegistryError("LoRA sidecar trigger_words must be an array of strings.")
        cleaned_triggers = tuple(word.strip() for word in trigger_words)

        description = metadata.get("description", "")
        if not isinstance(description, str):
            raise RegistryError("LoRA sidecar description must be a string.")

        default_scale = metadata.get("default_scale", 1.0)
        try:
            scale = validate_lora_scale(default_scale)
        except ValueError as exc:
            raise RegistryError(str(exc)) from exc
        if not math.isfinite(scale):
            raise RegistryError("LoRA sidecar default_scale must be finite.")

        return {
            "display_name": display_name.strip() if display_name is not None else None,
            "base_model": model_key,
            "trigger_words": cleaned_triggers,
            "default_scale": scale,
            "description": description.strip(),
        }

    def scan(self, model_key: ModelKey | str) -> list[LoraEntry]:
        """List active-model ``.safetensors`` files without reading their bodies."""
        key = self._model_key(model_key)
        folder = self.folder_for(key)
        if not folder.is_dir():
            return []

        entries: list[LoraEntry] = []
        for path in sorted(folder.iterdir(), key=lambda candidate: candidate.name.casefold()):
            if path.suffix.lower() != ".safetensors" or not path.is_file():
                continue
            resolved = path.resolve(strict=False)
            if not self._path_is_inside(resolved, folder):
                continue
            try:
                stat = resolved.stat()
            except OSError:
                continue

            fallback = self._fallback_metadata(resolved, key)
            sidecar = path.with_suffix(".json")
            sidecar_path: str | None = None
            sidecar_error: str | None = None
            metadata = fallback
            if sidecar.is_file() and self._path_is_inside(sidecar, folder):
                sidecar_path = str(sidecar.resolve(strict=False))
                try:
                    parsed = self._sidecar_metadata(sidecar, key)
                    metadata = {
                        **fallback,
                        **{name: value for name, value in parsed.items() if value is not None},
                    }
                except (OSError, RegistryError) as exc:
                    sidecar_error = str(exc)

            entries.append(
                LoraEntry(
                    path=str(resolved),
                    filename=resolved.name,
                    display_name=metadata["display_name"],
                    base_model=metadata["base_model"],
                    trigger_words=metadata["trigger_words"],
                    default_scale=metadata["default_scale"],
                    description=metadata["description"],
                    file_size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sidecar_path=sidecar_path,
                    sidecar_error=sidecar_error,
                )
            )
        return entries

    def _validate_entry_path(self, entry: LoraEntry) -> Path:
        model_key = self._model_key(entry.base_model)
        folder = self.folder_for(model_key)
        path = Path(entry.path).expanduser().resolve(strict=False)
        if path.suffix.lower() != ".safetensors" or not self._path_is_inside(path, folder):
            raise RegistryError("LoRA path is outside its model-specific registry folder.")
        if not path.is_file():
            raise RegistryError("LoRA file no longer exists.")
        return path

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def checksum(self, entry: LoraEntry) -> str:
        """Compute SHA-256 on selection and cache it by file size and mtime."""
        path = self._validate_entry_path(entry)
        for _attempt in range(2):
            before = path.stat()
            key = str(path)
            cached = self._checksum_cache.get(key)
            if cached is not None and cached[:2] == (before.st_size, before.st_mtime_ns):
                return cached[2]
            digest = self._hash_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                self._checksum_cache[key] = (after.st_size, after.st_mtime_ns, digest)
                return digest
        raise RegistryError("LoRA file changed while its checksum was being calculated.")

    def with_checksum(self, entry: LoraEntry) -> LoraEntry:
        return replace(entry, sha256=self.checksum(entry))

    def to_lora_spec(self, entry: LoraEntry, *, scale: float | None = None) -> LoraSpec:
        selected_scale = entry.default_scale if scale is None else scale
        return LoraSpec(
            path=entry.path,
            base_model=entry.base_model,
            scale=selected_scale,
            sha256=self.checksum(entry),
        )
