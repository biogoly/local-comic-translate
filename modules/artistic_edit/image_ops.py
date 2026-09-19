"""CPU-only image preparation and composition for Artistic Edit.

The FLUX worker is intentionally given a rectangular RGB crop.  The app keeps
the authoritative edit mask and uses it when the generated crop returns, so
the selected/painted region, optional edit margin, and feathering determine
which pixels are replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageOps


BBox = tuple[int, int, int, int]
CropBox = tuple[int, int, int, int]


class ImageOperationError(ValueError):
    """Raised when an image, mask, or crop cannot be prepared safely."""


def _as_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            finite = finite * 255.0
        return np.ascontiguousarray(np.clip(np.rint(finite), 0, 255).astype(np.uint8))

    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def normalize_rgb(
    image: Image.Image | np.ndarray,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Return a contiguous HxWx3 uint8 RGB array.

    Transparent pixels are composited over ``background`` instead of silently
    losing their alpha channel. EXIF orientation is honored for PIL inputs.
    """
    if isinstance(image, Image.Image):
        pil_image = ImageOps.exif_transpose(image)
        if pil_image.mode in {"RGBA", "LA", "P"} or "transparency" in pil_image.info:
            rgba = pil_image.convert("RGBA")
            backdrop = Image.new("RGBA", rgba.size, (*background, 255))
            return np.asarray(Image.alpha_composite(backdrop, rgba).convert("RGB"))
        return np.asarray(pil_image.convert("RGB"))

    array = _as_uint8(np.asarray(image))
    if array.ndim == 2:
        return np.ascontiguousarray(np.repeat(array[:, :, None], 3, axis=2))
    if array.ndim != 3 or array.shape[2] not in {1, 2, 3, 4}:
        raise ImageOperationError("Image must be grayscale, RGB, RGBA, LA, or single-channel.")
    if array.shape[2] == 1:
        return np.ascontiguousarray(np.repeat(array, 3, axis=2))
    if array.shape[2] == 3:
        return np.ascontiguousarray(array)

    if array.shape[2] == 2:
        rgb = np.repeat(array[:, :, :1], 3, axis=2).astype(np.float32)
        alpha = array[:, :, 1:2].astype(np.float32) / 255.0
    else:
        rgb = array[:, :, :3].astype(np.float32)
        alpha = array[:, :, 3:4].astype(np.float32) / 255.0
    backdrop = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    composed = (rgb * alpha) + (backdrop * (1.0 - alpha))
    return np.ascontiguousarray(np.clip(np.rint(composed), 0, 255).astype(np.uint8))


def normalize_mask(mask: Image.Image | np.ndarray, *, binary: bool = True) -> np.ndarray:
    """Return a contiguous HxW uint8 mask."""
    if isinstance(mask, Image.Image):
        pil_mask = ImageOps.exif_transpose(mask)
        if pil_mask.mode in {"RGBA", "LA"}:
            array = np.asarray(pil_mask.getchannel("A"))
        else:
            array = np.asarray(pil_mask.convert("L"))
    else:
        array = _as_uint8(np.asarray(mask))
        if array.ndim == 3:
            if array.shape[2] == 1:
                array = array[:, :, 0]
            elif array.shape[2] in {2, 4}:
                array = array[:, :, -1]
            elif array.shape[2] == 3:
                array = array.max(axis=2)
            else:
                raise ImageOperationError("Mask has an unsupported channel count.")
        elif array.ndim != 2:
            raise ImageOperationError("Mask must be a 2D or channel image.")

    array = _as_uint8(array)
    if binary:
        array = np.where(array > 0, 255, 0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _finite_coordinate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ImageOperationError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ImageOperationError(f"{name} must be finite.")
    return number


def normalize_bbox(
    bbox: Sequence[float],
    image_size: tuple[int, int],
) -> BBox:
    """Floor/ceil and clamp an ``(x, y, width, height)`` box to an image."""
    if len(bbox) != 4:
        raise ImageOperationError("Bounding box must contain x, y, width, and height.")
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ImageOperationError("Image dimensions must be positive.")

    x, y, width, height = (
        _finite_coordinate(value, name)
        for value, name in zip(bbox, ("x", "y", "width", "height"), strict=True)
    )
    left = math.floor(min(x, x + width))
    top = math.floor(min(y, y + height))
    right = math.ceil(max(x, x + width))
    bottom = math.ceil(max(y, y + height))
    left = max(0, min(image_width, left))
    top = max(0, min(image_height, top))
    right = max(0, min(image_width, right))
    bottom = max(0, min(image_height, bottom))
    if right <= left or bottom <= top:
        raise ImageOperationError("Bounding box does not overlap the image.")
    return left, top, right - left, bottom - top


def expand_bbox(
    bbox: Sequence[float],
    image_size: tuple[int, int],
    *,
    context_ratio: float = 0.15,
    minimum_context: int = 64,
    maximum_context: int | None = None,
) -> BBox:
    """Expand a normalized selection equally on all sides, clamped to the page."""
    x, y, width, height = normalize_bbox(bbox, image_size)
    ratio = _finite_coordinate(context_ratio, "context_ratio")
    if ratio < 0:
        raise ImageOperationError("context_ratio cannot be negative.")
    if minimum_context < 0 or (maximum_context is not None and maximum_context < 0):
        raise ImageOperationError("Context limits cannot be negative.")
    context = max(minimum_context, math.ceil(max(width, height) * ratio))
    if maximum_context is not None:
        context = min(context, maximum_context)
    image_width, image_height = image_size
    left = max(0, x - context)
    top = max(0, y - context)
    right = min(image_width, x + width + context)
    bottom = min(image_height, y + height + context)
    return left, top, right - left, bottom - top


def make_rect_mask(image_size: tuple[int, int], bbox: Sequence[float]) -> np.ndarray:
    image_width, image_height = image_size
    x, y, width, height = normalize_bbox(bbox, image_size)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask[y : y + height, x : x + width] = 255
    return mask


def mask_bbox(mask: Image.Image | np.ndarray) -> BBox | None:
    normalized = normalize_mask(mask)
    ys, xs = np.nonzero(normalized)
    if not xs.size:
        return None
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return left, top, right - left, bottom - top


def crop_to_bbox(array: np.ndarray, bbox: BBox) -> np.ndarray:
    x, y, width, height = bbox
    if array.ndim not in {2, 3}:
        raise ImageOperationError("Only 2D and 3D arrays can be cropped.")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ImageOperationError("Crop box must be positive and inside the array.")
    if x + width > array.shape[1] or y + height > array.shape[0]:
        raise ImageOperationError("Crop box exceeds the array bounds.")
    return np.ascontiguousarray(array[y : y + height, x : x + width])


def pad_to_multiple(
    array: np.ndarray,
    *,
    multiple: int = 16,
    mode: str = "edge",
    constant_value: int = 0,
) -> tuple[np.ndarray, CropBox]:
    """Symmetrically pad spatial dimensions and return the restoration crop box."""
    source = np.asarray(array)
    if source.ndim not in {2, 3} or source.shape[0] == 0 or source.shape[1] == 0:
        raise ImageOperationError("Array must have non-empty height and width dimensions.")
    if not isinstance(multiple, int) or isinstance(multiple, bool) or multiple <= 0:
        raise ImageOperationError("multiple must be a positive integer.")
    if mode not in {"edge", "constant"}:
        raise ImageOperationError("Padding mode must be 'edge' or 'constant'.")

    height, width = source.shape[:2]
    add_width = (-width) % multiple
    add_height = (-height) % multiple
    left = add_width // 2
    right = add_width - left
    top = add_height // 2
    bottom = add_height - top
    pad_width = ((top, bottom), (left, right))
    if source.ndim == 3:
        pad_width += ((0, 0),)
    kwargs = {"constant_values": constant_value} if mode == "constant" else {}
    padded = np.pad(source, pad_width, mode=mode, **kwargs)
    return np.ascontiguousarray(padded), (left, top, left + width, top + height)


def crop_padding(array: np.ndarray, crop_box: CropBox) -> np.ndarray:
    left, top, right, bottom = crop_box
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ImageOperationError("Padding crop box is invalid.")
    if right > array.shape[1] or bottom > array.shape[0]:
        raise ImageOperationError("Padding crop box exceeds the array bounds.")
    return np.ascontiguousarray(array[top:bottom, left:right])


def resize_array(
    array: np.ndarray,
    size: tuple[int, int],
    *,
    is_mask: bool = False,
) -> np.ndarray:
    """Resize to an exact ``(width, height)`` using image-appropriate sampling."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ImageOperationError("Resize dimensions must be positive.")
    source = normalize_mask(array, binary=False) if is_mask else normalize_rgb(array)
    mode = "L" if is_mask else "RGB"
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    resized = Image.fromarray(source, mode=mode).resize((width, height), resample)
    result = np.asarray(resized)
    return normalize_mask(result, binary=False) if is_mask else normalize_rgb(result)


def resize_to_fit(
    array: np.ndarray,
    maximum_size: tuple[int, int],
    *,
    is_mask: bool = False,
) -> tuple[np.ndarray, float]:
    """Downscale within a ceiling without changing aspect ratio or upscaling."""
    max_width, max_height = maximum_size
    if max_width <= 0 or max_height <= 0:
        raise ImageOperationError("Maximum dimensions must be positive.")
    source = normalize_mask(array, binary=False) if is_mask else normalize_rgb(array)
    height, width = source.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    target_width = max(1, min(max_width, int(round(width * scale))))
    target_height = max(1, min(max_height, int(round(height * scale))))
    if target_width == width and target_height == height:
        return source.copy(), 1.0
    return resize_array(source, (target_width, target_height), is_mask=is_mask), scale


def dilate_mask(mask: Image.Image | np.ndarray, radius: int) -> np.ndarray:
    normalized = normalize_mask(mask)
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ImageOperationError("Mask dilation radius must be a non-negative integer.")
    if radius == 0:
        return normalized
    return np.asarray(
        Image.fromarray(normalized, mode="L").filter(ImageFilter.MaxFilter((radius * 2) + 1))
    )


def mask_to_alpha(
    mask: Image.Image | np.ndarray,
    *,
    dilation: int = 0,
    feather_radius: float = 0.0,
) -> np.ndarray:
    """Build a uint8 alpha mask, optionally dilated and feathered."""
    feather = _finite_coordinate(feather_radius, "feather_radius")
    if feather < 0:
        raise ImageOperationError("feather_radius cannot be negative.")
    alpha = dilate_mask(mask, dilation)
    if feather:
        alpha = np.asarray(
            Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(feather))
        )
    return np.ascontiguousarray(alpha)


def blend_with_mask(
    source: Image.Image | np.ndarray,
    generated: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    *,
    dilation: int = 0,
    feather_radius: float = 0.0,
) -> np.ndarray:
    original = normalize_rgb(source)
    edited = normalize_rgb(generated)
    if edited.shape != original.shape:
        raise ImageOperationError("Generated and source images must have identical dimensions.")
    alpha = mask_to_alpha(mask, dilation=dilation, feather_radius=feather_radius)
    if alpha.shape != original.shape[:2]:
        raise ImageOperationError("Mask dimensions must match the source image.")
    alpha_float = alpha.astype(np.float32)[:, :, None] / 255.0
    blended = (edited.astype(np.float32) * alpha_float) + (
        original.astype(np.float32) * (1.0 - alpha_float)
    )
    return np.ascontiguousarray(np.clip(np.rint(blended), 0, 255).astype(np.uint8))


def sha256_pixels(image: Image.Image | np.ndarray) -> str:
    """Hash normalized pixel dimensions and bytes for stale-result detection."""
    normalized = normalize_rgb(image)
    digest = hashlib.sha256()
    digest.update(normalized.shape[1].to_bytes(4, "big"))
    digest.update(normalized.shape[0].to_bytes(4, "big"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedEdit:
    """CPU-side generation input plus data needed for safe recomposition."""

    selection_bbox: BBox
    context_bbox: BBox
    source_crop: np.ndarray
    mask_crop: np.ndarray
    model_input: np.ndarray
    model_mask: np.ndarray
    padding_crop_box: CropBox
    working_scale: float
    source_hash: str


@dataclass(frozen=True)
class ComposedPatch:
    """The smallest RGB patch affected by app-side mask composition."""

    bbox: BBox
    image: np.ndarray
    alpha: np.ndarray


def prepare_edit(
    page: Image.Image | np.ndarray,
    selection_bbox: Sequence[float],
    *,
    edit_mask: Image.Image | np.ndarray | None = None,
    edit_margin: int = 0,
    context_ratio: float = 0.15,
    minimum_context: int = 64,
    maximum_context: int | None = None,
    maximum_model_size: tuple[int, int] = (2048, 2048),
    multiple: int = 16,
) -> PreparedEdit:
    """Prepare a context crop without importing Qt, Torch, or Diffusers."""
    source = normalize_rgb(page)
    page_size = (source.shape[1], source.shape[0])
    selection = normalize_bbox(selection_bbox, page_size)
    if isinstance(edit_margin, bool) or not isinstance(edit_margin, int) or edit_margin < 0:
        raise ImageOperationError("Edit margin must be a non-negative integer.")
    if edit_mask is not None and edit_margin:
        raise ImageOperationError("Edit margin is only supported for rectangular selections.")
    editable = expand_bbox(
        selection, page_size, context_ratio=0, minimum_context=edit_margin,
    )
    context = expand_bbox(
        editable,
        page_size,
        context_ratio=context_ratio,
        minimum_context=minimum_context,
        maximum_context=maximum_context,
    )

    if edit_mask is None:
        full_mask = make_rect_mask(page_size, editable)
    else:
        full_mask = normalize_mask(edit_mask)
        if full_mask.shape != source.shape[:2]:
            raise ImageOperationError("Edit mask dimensions must match the page.")
    source_crop = crop_to_bbox(source, context)
    mask_crop = crop_to_bbox(full_mask, context)
    if mask_bbox(mask_crop) is None:
        raise ImageOperationError("Edit mask does not affect the selected context crop.")

    resized_input, working_scale = resize_to_fit(source_crop, maximum_model_size)
    resized_mask = resize_array(
        mask_crop,
        (resized_input.shape[1], resized_input.shape[0]),
        is_mask=True,
    )
    resized_mask = normalize_mask(resized_mask)
    model_input, crop_box = pad_to_multiple(resized_input, multiple=multiple, mode="edge")
    model_mask, mask_crop_box = pad_to_multiple(
        resized_mask,
        multiple=multiple,
        mode="constant",
        constant_value=0,
    )
    if mask_crop_box != crop_box:
        raise ImageOperationError("Image and mask padding geometry diverged.")
    return PreparedEdit(
        selection_bbox=selection,
        context_bbox=context,
        source_crop=source_crop,
        mask_crop=mask_crop,
        model_input=model_input,
        model_mask=model_mask,
        padding_crop_box=crop_box,
        working_scale=working_scale,
        source_hash=sha256_pixels(source_crop),
    )


def compose_generated_patch(
    prepared: PreparedEdit,
    generated: Image.Image | np.ndarray,
    *,
    dilation: int = 0,
    feather_radius: float = 0.0,
) -> ComposedPatch:
    """Restore a worker result and confine it to the retained app-side mask."""
    worker_image = normalize_rgb(generated)
    if worker_image.shape != prepared.model_input.shape:
        raise ImageOperationError("Worker output dimensions do not match its generation input.")
    unpadded = crop_padding(worker_image, prepared.padding_crop_box)
    restored = resize_array(
        unpadded,
        (prepared.source_crop.shape[1], prepared.source_crop.shape[0]),
    )
    alpha = mask_to_alpha(
        prepared.mask_crop,
        dilation=dilation,
        feather_radius=feather_radius,
    )
    blended = blend_with_mask(
        prepared.source_crop,
        restored,
        prepared.mask_crop,
        dilation=dilation,
        feather_radius=feather_radius,
    )
    affected = mask_bbox(alpha)
    if affected is None:
        raise ImageOperationError("Composed edit has an empty alpha mask.")
    local_x, local_y, width, height = affected
    context_x, context_y, _, _ = prepared.context_bbox
    return ComposedPatch(
        bbox=(context_x + local_x, context_y + local_y, width, height),
        image=crop_to_bbox(blended, affected),
        alpha=crop_to_bbox(alpha, affected),
    )
