# FLUX.2 Klein Artistic Edit Integration Plan

Status: implementation handoff plan

Target branch baseline: `main` at `c0b9088` (`Restore text box resizing after inline formatting`)

Reference prototype: `C:\AI\flux2-batch-edit\batch_edit_flux2.py`

## 1. Objective

Add an optional, fully local **Artistic Edit** stage to Local Comic Translate using the Diffusers `Flux2KleinPipeline`.

This tool is for edits where deterministic cleanup and ordinary text layers are not enough:

- replacing a stylized title, sign, logo-like caption, or sound effect while preserving its visual treatment;
- repairing art around lettering;
- drawing or enlarging a speech bubble so a translation fits;
- making an explicitly prompted local correction to comic art.

It must not replace the existing detect → recognize → translate → segment → clean → render workflow. Normal speech bubbles should continue to use deterministic cleanup plus editable text layers. FLUX is a user-invoked pixel editor for exceptional regions.

The safest first release is **manual, region-scoped, preview-before-apply, and undoable**. Do not add it to Automatic Translation or run it across a book in the MVP.

## 2. Product decisions

### 2.1 Placement

Add a divider and panel named `Artistic Edit (FLUX.2 Klein)` immediately below the current Inpainting controls in the right-hand scroll area built by `app/ui/main_window/builders/workspace.py`.

Keep the panel collapsed by default if a compact disclosure widget is easy to implement. If not, use a permanently visible compact layout. Do not add a menu-indicator chevron to a text-only button; the project previously had to hide that visual artifact on the Sort control.

### 2.2 Intended workflow

1. The user selects one text rectangle, paints an irregular edit mask, or explicitly chooses Whole Page.
2. The app captures the current page pixels with committed patch edits included, but excludes selection rectangles, brush guides, and editable Qt text items.
3. The app crops the requested region with surrounding context, pads it to a multiple of 16, and writes a temporary RGB PNG.
4. The user enters an edit instruction or inserts a prompt based on the selected OCR block's source and translated text.
5. The user selects 4B or 9B and, optionally, one compatible LoRA and its strength.
6. FLUX runs in an isolated local worker and returns an image to a Before/After preview.
7. `Apply` creates a normal page patch on the current undo stack. `Discard` changes nothing. `Regenerate` may reuse or randomize the seed.
8. Undo/redo, project save/load, normal export, and webtoon patch composition treat the accepted edit like other page patches.

### 2.3 Recommended use of FLUX text rendering

FLUX can reproduce artistic lettering remarkably well, but exact spelling is not guaranteed. The official model cards explicitly warn that generated text can be inaccurate or distorted.

Use two distinct prompt templates:

- **Artistic lettering replacement:** ask FLUX to replace the source lettering with the exact translated phrase while preserving size, angle, texture, outline, shadows, color, and integration with the surrounding art. Always inspect the preview character by character.
- **Speech-bubble creation/repair:** ask FLUX to draw an empty bubble of an appropriate size and style. Then place exact translated text with the existing editable text tool. This is more reliable than asking a generative model to produce a paragraph perfectly.

### 2.4 MVP scope

The first usable milestone includes:

- regular single-page mode;
- selected rectangle, painted mask, and opt-in whole-page input;
- FLUX.2 Klein 4B and 9B choices;
- one local `.safetensors` LoRA at a time;
- LoRA strength;
- prompt, seed, four-step default, working-size limit, and context padding;
- local preview, apply, discard, regenerate, cancel, and unload;
- undo/redo and `.ctpr` persistence;
- no model download or model import during normal application startup.

Defer these until the MVP is stable:

- automatic artistic-text detection;
- book-wide/batch FLUX processing;
- webtoon multi-page/visible-area editing;
- multiple reference images;
- stacked LoRAs in the UI;
- quantized loading;
- tiling a single edit over several generative calls;
- background model downloads;
- training LoRAs.

In webtoon mode, disable Generate with a concise message such as `Artistic Edit currently requires single-page mode.` This is preferable to getting page/scene coordinates partly right.

## 3. Verified upstream constraints

These facts were checked against the model cards and current Diffusers documentation on 2026-09-07:

- The correct class is `diffusers.Flux2KleinPipeline`.
- The pipeline accepts a PIL image or list of images through `image=`, an instruction through `prompt=`, output width/height, steps, a generator, and a step-end callback.
- The current pipeline signature does **not** expose an inpainting mask. Region restriction therefore belongs to Local Comic Translate: crop before inference and composite the generated crop through the user's app-side mask afterward.
- Both selected checkpoints are distilled around four inference steps. Start at `num_inference_steps=4` and `guidance_scale=1.0`. Diffusers notes that guidance is ignored for step-distilled models, so guidance belongs in Advanced settings and must not be presented as an important artistic control.
- FLUX.2 Klein 4B is Apache 2.0, is advertised at about 13 GB VRAM, and is the default.
- FLUX.2 Klein 9B uses the FLUX non-commercial license, requires accepting gated access on Hugging Face, and is advertised at about 29 GB VRAM. Label those facts directly in the UI; do not bundle or select 9B by default.
- Diffusers exposes the `Flux2LoraLoaderMixin`, including `load_lora_weights()`, `set_adapters()`, and `unload_lora_weights()`.
- A LoRA must match the base architecture and size. A FLUX.1, SDXL, 4B-on-9B, or 9B-on-4B LoRA must be rejected or clearly blocked before generation.

Official references:

- <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>
- <https://huggingface.co/black-forest-labs/FLUX.2-klein-9B>
- <https://huggingface.co/docs/diffusers/api/pipelines/flux2>
- <https://huggingface.co/docs/diffusers/optimization/memory>
- <https://huggingface.co/docs/diffusers/main/using-diffusers/loading_adapters>

The local reference prototype previously used this known-working stack:

- `torch 2.11.0+cu128`
- `diffusers 0.38.0.dev0`, Git commit `b8aebf4c1261c4b887778e0882e56ed19e686518`
- `transformers 5.5.4`
- `accelerate 1.13.0`
- `peft 0.20.0`
- `safetensors 0.8.0rc0`

Do not assume GitHub `main` will remain compatible. During the spike, recreate a working environment and pin the tested Diffusers commit or a released version that contains `Flux2KleinPipeline` and `Flux2LoraLoaderMixin`.

The development machine currently exposes two RTX 3090 Ti GPUs with 24 GB each. The 4B model should use one card. The 9B model cannot be assumed to fit on one 24 GB card and needs either tested `device_map="balanced"` placement across both cards or single-GPU model CPU offload.

## 4. Architecture

Keep PyTorch and Diffusers outside the Qt GUI process. A persistent worker process is slightly more work than a direct import, but it avoids four serious problems:

- PyTorch/CUDA packages do not have to become mandatory startup dependencies for every user.
- A CUDA out-of-memory or native-library failure is less likely to take down the editor and lose unsaved work.
- The worker can keep a large pipeline warm across several edits and still be terminated to reclaim all GPU memory reliably.
- The worker environment can use a PyTorch CUDA build independently of the app's ONNX Runtime CUDA packages.

Recommended data flow:

```text
ArtisticEditPanel
        |
        v
ArtisticEditController ----> page/selection capture + crop/mask helpers
        |
        v
FluxWorkerClient (QProcess, JSON Lines)
        |
        v
flux_worker.py ----> Flux2KleinPipeline ----> temporary output PNG
        |
        v
Before/After Preview ----Apply----> typed page patch ----> QUndoStack/.ctpr
```

The worker is local. No page image is uploaded. A Hugging Face network connection may be needed to acquire model files initially, but inference must work with local files/cache and support `local_files_only=True`.

### 4.1 Process protocol

Launch the worker with `QProcess`, Python unbuffered mode, and no visible console window on Windows. Never use `shell=True`. Reserve stdout for one JSON object per line; route Python, Diffusers, and model logs to stderr so logging cannot corrupt the protocol.

Minimum commands:

```json
{"id":"uuid","op":"ping"}
{"id":"uuid","op":"load","model":{},"device":{},"lora":null}
{"id":"uuid","op":"generate","input_path":"...","output_path":"...","prompt":"...","width":1024,"height":1024,"steps":4,"guidance":1.0,"seed":0}
{"id":"uuid","op":"cancel"}
{"id":"uuid","op":"unload"}
{"id":"uuid","op":"shutdown"}
```

Minimum events:

```json
{"id":"uuid","event":"ready","capabilities":{}}
{"id":"uuid","event":"loading","message":"Loading FLUX.2 Klein 4B"}
{"id":"uuid","event":"progress","step":2,"total":4}
{"id":"uuid","event":"result","output_path":"...","seed":0,"elapsed_seconds":12.3}
{"id":"uuid","event":"cancelled"}
{"id":"uuid","event":"error","code":"cuda_oom","message":"...","details":"..."}
```

Give every operation an ID and ignore late events for an operation that the controller no longer owns. Validate every input/output path and refuse an output path outside the application temporary directory.

The worker should have a stdin-reader thread so it can receive `cancel` while the main worker thread is inside inference. The `callback_on_step_end` callback checks a thread-safe cancellation event and raises a private cancellation exception. If graceful cancellation does not complete promptly, the GUI may terminate and recreate the worker.

### 4.2 Backend interface

Keep the UI and controller independent of Diffusers details. Define a request/result contract similar to:

```python
@dataclass(frozen=True)
class FluxModelSpec:
    key: Literal["klein-4b", "klein-9b"]
    source: str
    license_label: str

@dataclass(frozen=True)
class LoraSpec:
    path: str
    base_model: Literal["klein-4b", "klein-9b"]
    scale: float
    sha256: str

@dataclass(frozen=True)
class ArtisticEditRequest:
    request_id: str
    input_path: str
    output_path: str
    prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    model: FluxModelSpec
    lora: LoraSpec | None
    device_policy: str
```

Do not pass a live Qt object, NumPy array, or `TextBlock` into the worker. Use immutable values and temporary PNGs.

### 4.3 Model lifecycle

The worker caches one base pipeline keyed by model source, dtype, and device policy.

- Load lazily on the first Generate.
- Reuse the pipeline for later requests with the same key.
- On 4B/9B or device-policy change, unload the old pipeline before loading the new one.
- Use BF16 by default on supported NVIDIA hardware.
- Call `vae.enable_tiling()` when available.
- Use `torch.Generator(device="cpu").manual_seed(seed)` as in the working prototype.
- On LoRA change, call `unload_lora_weights()` and then load the selected adapter. Do not use hotswap in the MVP because LoRAs can have different ranks and targets.
- On Unload or Shutdown, remove pipeline references, collect Python garbage, synchronize if appropriate, and clear the CUDA allocator cache. The process boundary remains the final guarantee that memory can be reclaimed.

### 4.4 Device policies

Implement explicit policies rather than an opaque `Auto` that changes silently:

- `Single GPU 0` and `Single GPU 1`: best for 4B. Set the worker's `CUDA_VISIBLE_DEVICES` before importing Torch, then use its local `cuda:0`.
- `Balanced multi-GPU`: intended for 9B on the current dual-3090-Ti machine. Load with `device_map="balanced"` and a conservative `max_memory`, initially 21 GiB per 24 GB card, preserving activation headroom.
- `CPU offload`: call `enable_model_cpu_offload()` targeting one GPU. It should work when a model does not fit, but the UI must warn that transfers can make it much slower.

For the first end-to-end implementation, make 4B + Single GPU the supported path. Add 9B + CPU offload next. Treat balanced multi-GPU as a separately measured optimization; Diffusers' balanced device mapping splits pipeline components and does not guarantee an ideal split for every architecture.

Do not add quantization in the MVP. It multiplies dependency and compatibility combinations before the basic edit path is proven.

Before starting FLUX, the controller should:

1. save the current page/editor state;
2. call `pipeline.release_model_caches()` to release ONNX-backed detector/OCR/inpainting sessions;
3. call `shutdown_local_runtime()` for the app-managed llama.cpp server;
4. never terminate a user-managed external llama.cpp server;
5. log available/free GPU memory and the chosen policy.

Initially provide an explicit `Unload Model` button and unload on application exit. If keeping FLUX warm causes conflicts with later OCR/translation operations, add a small GPU resource coordinator rather than scattering more cache-clearing calls through unrelated controllers.

## 5. Selection, crop, mask, and composition

### 5.1 Input source

Build the input from `ImageStateController.get_composited_page_image(file_path)`. That produces the base image plus accepted patches while excluding editable text and UI guides. Do not use a screenshot of the viewport; zoom must not affect inference resolution.

Store a SHA-256 digest of the exact source crop supplied to FLUX. Before Apply, recapture the same crop and compare its digest. If it changed because of undo, inpainting, text flattening, page navigation, or another edit, disable Apply and require regeneration. A result generated from stale pixels must never be composited silently.

### 5.2 Selection modes

Implement these modes in order:

1. **Selected box** — default. Require exactly one selected `MoveableRectItem`. Convert its scene rectangle to page coordinates, normalize, round to integers, and clamp to image bounds.
2. **Painted area** — use the existing inpainting brush mask. Compute the nonzero bounding rectangle, add context, and keep the cropped binary mask for final composition. Clear only the strokes consumed by a successful Apply; Discard preserves them.
3. **Whole page** — explicit opt-in with a confirmation because it is slow and can alter unrelated art.

If a selected rectangle and strokes both exist, follow the selected mode shown in the panel rather than guessing.

### 5.3 Context and padding

FLUX needs surrounding art to continue style correctly. Expand the requested edit bounds with context, initially:

- default `15%` of the larger selected dimension;
- minimum `64 px`;
- maximum configurable in Advanced settings;
- clamp to the page.

Pad the context crop to a multiple of 16 using edge padding, preserving the crop-back coordinates. Reuse and unit-test the logic from the batch editor's `pad_image_to_multiple()` rather than importing that script at runtime.

### 5.4 Working size

Preserve the crop's native resolution when it fits the configured working-size ceiling. If it must be reduced:

- preserve aspect ratio;
- record the scale factor;
- show `Processed at NN% resolution` in the preview;
- restore the result to the original crop size with a high-quality resize before patch creation;
- never let rounding change the final page dimensions.

Large full-page edits should fail early with an actionable memory estimate instead of repeatedly attempting unsafe allocations.

### 5.5 Applying the result

Because the current FLUX pipeline does not consume a mask, the app performs final composition:

- Selected box: composite the generated pixels through the selected rectangle, with an optional small feather (default 8 px) on the edge.
- Painted area: composite through the cropped brush mask, with configurable feathering/dilation to avoid seams.
- Whole page: use the full generated page.

Create an RGB patch whose `bbox` is the smallest affected page rectangle. Existing patch display code does not support alpha, so blend against the captured source before creating the patch.

FLUX patches appear at the current patch z-level below editable text. This supports the useful pattern `generate empty bubble → render exact editable text above it`.

## 6. Patch history and project persistence

The current `image_patches` list is ordered and already participates in rendering, undo/redo, lazy loading, webtoon display, export, and `.ctpr` blobs. Extend it rather than creating an unrelated second history system.

### 6.1 Typed generic patches

Add these optional fields to the patch record:

```python
{
    "patch_id": "uuid",
    "kind": "inpaint" | "flux2_edit",
    "bbox": [x, y, width, height],
    "png_path": "...",
    "hash": "...",
    "metadata": {
        "prompt": "...",
        "model_key": "klein-4b",
        "model_source": "black-forest-labs/FLUX.2-klein-4B",
        "seed": 0,
        "steps": 4,
        "guidance": 1.0,
        "lora_name": null,
        "lora_sha256": null,
        "lora_scale": null,
        "selection_mode": "selected_box",
        "source_crop_sha256": "...",
        "working_scale": 1.0
    }
}
```

Do not persist absolute LoRA paths in a shareable project; use the display name and checksum. Keep metadata JSON/msgpack-safe and bounded. Limit prompt length before persistence.

Treat every legacy patch without `kind` as `inpaint`.

### 6.2 Commands

Refactor `PatchInsertCommand` to accept:

- `kind="inpaint"` by default;
- a translated undo label, defaulting to `Inpaint`;
- optional metadata copied into every generated patch record.

The FLUX controller pushes it with `kind="flux2_edit"` and label `Artistic edit`.

Refactor `PatchClearCommand` to clear only one requested kind. `Revert All Inpainting` must never delete accepted artistic edits. Add `Revert Artistic Edits` inside the FLUX panel, using the same filtered command and remaining fully undoable.

Keep one canonical ordered patch list so compositing order remains obvious. Do not put FLUX edits in `image_history`; that older full-image history is a separate mechanism.

### 6.3 Serialization

Update both v2 and legacy project paths:

- `app/projects/project_state_v2.py` must write/read `kind` and a whitelisted/sanitized `metadata` object in each patch reference.
- `app/projects/project_state.py` must do the same for legacy ZIP/msgpack projects.
- Unknown future metadata keys should survive a save/load cycle if they are primitive msgpack values.
- Project loading must continue to materialize PNG blobs lazily.
- Existing projects with only bbox/path/hash/ID must load unchanged.

Add a regression test that saves a mixed ordered list of legacy inpaint, typed inpaint, and FLUX patches and verifies order, kind, metadata, and PNG hashes after reload.

## 7. UI specification

Create a dedicated `ArtisticEditPanel` widget instead of adding another large block of construction code directly to `workspace.py`. The builder should instantiate the panel and add it below the inpainting slider.

Suggested compact layout:

```text
Artistic Edit (FLUX.2 Klein)
[ multi-line edit prompt                                      ]
[Use selected translation] [Prompt template v]

Model [4B (Apache 2.0) v]   LoRA [None v]   Strength [1.00]
Area  [Selected box v]      [Refresh LoRAs] [Open LoRA folder]

[Advanced...]
  Seed [0] [Random each run]  Steps [4]
  Context [15%]  Feather [8 px]  Max side [2048]
  Device [Single GPU 0 v]

[Generate Preview] [Cancel] [Unload Model]
[Revert Artistic Edits]
```

Preview dialog:

- synchronized Before and After panes with fit/zoom;
- prompt, model, LoRA, seed, elapsed time, and working-resolution summary;
- `Apply`, `Regenerate Same Seed`, `Regenerate New Seed`, and `Discard`;
- Apply disabled if the source hash is stale;
- no accepted state mutation until Apply.

Controls and validation:

- Generate disabled without a page, valid area, nonblank prompt, configured worker interpreter, and resolvable model.
- 4B display text includes `Apache 2.0 · ~13 GB VRAM`.
- 9B display text includes `Non-commercial · gated · ~29 GB VRAM` and shows a one-time acknowledgement.
- LoRA strength is a double spin box from `0.0` to `2.0`, step `0.05`, default `1.0`; disable it for None.
- Accept only finite numeric values.
- Steps range initially `1–20`; default `4`.
- Seed accepts a nonnegative 64-bit integer. Random mode generates the seed before dispatch and records the actual value.
- Guidance remains in Advanced, defaults to `1.0`, and has a tooltip saying distilled Klein checkpoints may ignore it.
- Show worker/model-loading/progress state in the existing top progress UI or a small panel progress row. Do not freeze the Qt event loop.

`Use selected translation` should inspect exactly one selected OCR rectangle/block and create a prompt such as:

```text
Replace only the lettering "UNA REVOLUCIÓN LLAMADA" with exactly
"A REVOLUTION CALLED". Preserve the original letter style, scale, placement,
spacing, texture, outline, shadows, perspective, and integration with the comic
art. Leave every other part of the image unchanged.
```

Do not send OCR or translation automatically when the button is not used. The prompt is always editable before generation.

## 8. Model and LoRA configuration

### 8.1 Optional environment

Add `requirements-flux.txt`; do not add Torch/Diffusers to base `requirements.txt`.

Document creation of a dedicated environment and let the app store a `Flux Python executable` in `QSettings`. Default to the current interpreter only when an import probe succeeds. The settings page needs a `Test Environment` action that runs a small probe and reports:

- Python path/version;
- Torch version and CUDA build;
- visible GPUs and VRAM;
- Diffusers version;
- whether `Flux2KleinPipeline` and FLUX.2 LoRA loading are available.

The existing `C:\AI\flux2-batch-edit\.venv` currently records a Python interpreter location that no longer exists. Do not hardcode or depend on that environment. Recreate or repair it during the spike and make the environment test catch this condition cleanly.

Pin the final tested dependencies. Install the appropriate Torch CUDA wheel separately because it comes from the PyTorch wheel index and depends on the host. A candidate requirements file after validation is:

```text
diffusers @ git+https://github.com/huggingface/diffusers.git@b8aebf4c1261c4b887778e0882e56ed19e686518
transformers==5.5.4
accelerate==1.13.0
peft==0.20.0
safetensors>=0.8.0rc0
sentencepiece
pillow
numpy
huggingface_hub
```

These are candidate pins from the proven prototype, not immutable requirements. The spike must install them in a fresh environment and run 4B image editing and LoRA smoke tests before committing the pins.

### 8.2 Model sources

Store settings for each model separately:

- 4B model source, default `black-forest-labs/FLUX.2-klein-4B`;
- 9B model source, default `black-forest-labs/FLUX.2-klein-9B`;
- `Local files only` toggle;
- Hugging Face cache directory override, optional.

A source may be an existing local Diffusers directory or a Hugging Face model ID. Never download weights at application startup. A Generate request may offer to fetch missing files only after explicit confirmation, with expected size and license shown.

Do not set `trust_remote_code=True`.

### 8.3 LoRA folders

Create this repository structure:

```text
models/loras/flux2-klein/
├── README.md
├── 4b/
│   └── .gitkeep
└── 9b/
    └── .gitkeep
```

Ignore weight files in Git:

```gitignore
models/loras/**/*.safetensors
models/loras/**/*.bin
models/loras/**/*.pt
```

Only offer `.safetensors` in the application. Do not deserialize pickle-based `.bin` or `.pt` LoRAs.

The dropdown scans only the active model's folder. Optionally support a sidecar JSON file with the same stem:

```json
{
  "display_name": "Manga Lettering",
  "base_model": "klein-4b",
  "trigger_words": ["manga-lettering"],
  "default_scale": 0.7,
  "description": "Optional user note"
}
```

If no sidecar exists, list the filename, infer compatibility from the 4B/9B folder, and calculate SHA-256 lazily on selection. Refresh must not block the GUI on many large files; use file size/mtime caching for hashes.

The working batch prototype already demonstrates the correct basic calls:

```python
pipe.load_lora_weights(path, adapter_name="artistic_edit")
pipe.set_adapters(["artistic_edit"], adapter_weights=[scale])
```

## 9. File-level implementation map

### New files

- `app/ui/artistic_edit_panel.py` — compact controls, validation signals, and settings binding.
- `app/ui/dialogs/artistic_edit_preview.py` — Before/After preview and apply/regenerate/discard actions.
- `app/controllers/artistic_edit.py` — selection capture, preflight, request dispatch, stale check, preview, patch application, and cleanup.
- `modules/artistic_edit/contracts.py` — dataclasses/enums and JSON serialization.
- `modules/artistic_edit/image_ops.py` — clamped bounds, context expansion, mask crop, multiple-of-16 padding, resizing, feathering, and final RGB composition.
- `modules/artistic_edit/lora_registry.py` — model-specific discovery, sidecar validation, and checksums.
- `modules/artistic_edit/worker_client.py` — QProcess lifecycle and JSON-lines protocol.
- `modules/artistic_edit/flux_worker.py` — lazy Diffusers imports, pipeline/model/LoRA lifecycle, generation, progress, cancellation, and structured errors.
- `requirements-flux.txt` — optional tested dependencies.
- `models/loras/flux2-klein/README.md` and model-size subfolders.
- `tests/testartistic_edit_image_ops.py`
- `tests/testartistic_edit_contracts.py`
- `tests/testartistic_edit_loras.py`
- `tests/testartistic_edit_worker.py`
- `tests/testartistic_edit_undo.py`
- `tests/testartistic_edit_project_state.py`

### Modified files

- `app/ui/main_window/builders/workspace.py` — insert the new panel below Inpainting.
- `controller.py` — instantiate/connect/close `ArtisticEditController`; route Cancel when an artistic edit is active.
- `app/ui/commands/inpaint.py` — generalize insertion/clear commands with `kind`, label, and metadata while retaining inpaint defaults.
- `app/ui/commands/base.py` — extend the patch property type and, if useful, store kind in graphics-item data for diagnostics.
- `app/controllers/image.py` — generic patch callback or an artistic-specific wrapper; filter Revert Inpainting by kind; preserve ordered composition.
- `app/projects/project_state.py` — legacy patch metadata round trip.
- `app/projects/project_state_v2.py` — v2 patch metadata round trip.
- `app/ui/canvas/webtoons/scene_items/patch_manager.py` — carry optional kind/metadata even though generation is initially disabled in webtoon mode.
- `app/ui/settings/settings_ui.py` and `app/ui/settings/settings_page.py` — optional Flux environment/model/cache/device settings and probes.
- `.gitignore` — ignore model/LoRA weights, worker outputs, and probe artifacts.
- `README.md` — after the feature works, document optional installation, local-only behavior, model licenses, and basic use.

Avoid adding FLUX methods to `pipeline/main_pipeline.py`: Artistic Edit is not a translation pipeline stage and should remain independently optional.

## 10. Error handling and recovery

Map worker failures to short user actions while preserving technical details in the terminal log:

- `worker_python_missing` — select or create a Flux environment.
- `dependency_missing` — run the optional Flux dependency install instructions.
- `model_not_found` — choose a local model folder or allow the model download.
- `model_access_denied` — log into Hugging Face and accept the 9B license, or use 4B.
- `lora_incompatible` — move/select a LoRA matching the active 4B/9B model.
- `cuda_oom` — unload other models, reduce max side, switch to 4B, choose CPU offload, or select another GPU.
- `worker_crashed` — retain the page and prompt, discard the incomplete output, restart the worker, and allow retry.
- `cancelled` — make no project/history change.
- `stale_source` — keep the preview visible for reference but require regeneration before Apply.

Write outputs to a unique request directory below `self.temp_dir`. Generate into a temporary filename and atomically rename it only after PIL can reopen it and verify the expected dimensions. Delete incomplete request directories on cancellation/crash and clean old previews at app shutdown.

Never overwrite the source comic image. The only durable effect is an accepted patch saved through the existing project/export systems.

## 11. Test strategy

All automated tests must run without a GPU, model download, Hugging Face login, or installed Diffusers environment.

### 11.1 Unit tests

Image operations:

- bounds normalize, round, and clamp correctly;
- a selection touching each page edge gets valid context;
- padding to 16 and crop-back reproduce the original dimensions and pixels;
- downscale/upscale preserves exact requested dimensions;
- binary-mask and feather composition affect only intended pixels;
- empty masks and zero-size rectangles are rejected;
- RGB/RGBA/grayscale inputs normalize correctly.

Protocol/contracts:

- JSON serialization round trips every request/result field;
- malformed messages, unknown operations, NaN/Infinity, and path traversal are rejected;
- late/mismatched request IDs are ignored;
- stdout remains valid JSON while diagnostic logs go to stderr.

LoRA registry:

- 4B and 9B folders never cross-populate;
- only `.safetensors` appears;
- sidecar defaults and invalid sidecars are handled safely;
- strength is finite and within the UI range;
- checksum caching invalidates on size/mtime change.

Worker/model manager, using fake Torch/Diffusers modules:

- lazy load and same-key reuse;
- model switch unloads first;
- LoRA load, scale, change, and removal calls occur in the correct order;
- `vae.enable_tiling()` is called when available;
- seed/steps/size are forwarded;
- cancellation and structured errors work;
- shutdown releases the cached pipeline.

Patch/history:

- Apply creates exactly one undo entry named Artistic edit;
- Undo removes only its patch and Redo restores it;
- Revert Inpainting leaves FLUX patches intact;
- Revert Artistic Edits leaves inpaint patches intact;
- visually identical patches remain distinct by `patch_id`;
- patch order remains stable.

Project state:

- mixed legacy/inpaint/FLUX patches survive v2 save/load;
- metadata survives without absolute LoRA paths;
- old project fixtures still load as inpaint patches;
- lazy PNG materialization remains functional.

Controller:

- Discard never marks the project dirty;
- Apply marks it dirty;
- stale source disables Apply;
- page navigation during generation cannot apply to the wrong page;
- editor overlays never enter the FLUX input image;
- webtoon mode disables generation.

### 11.2 Manual GPU acceptance tests

Keep the user's example pages outside Git and reference them through a local test checklist:

1. A cover with large stylized Spanish lettering: replace only the requested title, inspect exact spelling, and compare continuity to the existing FLUX output.
2. A conventional translated comic page: generate an empty enlarged bubble in one selected region and render exact text above it with the existing text tool.
3. An irregular sound effect/sign: use a painted mask and verify no rectangular seam.
4. Undo/redo each applied edit five times, save a project, close the app, reopen, and repeat undo/redo.
5. Generate, alter the page before completion, and verify Apply is blocked as stale.
6. Cancel during model load and during denoising; verify the page and project remain unchanged.
7. Switch 4B → 9B → 4B and confirm GPU memory is reclaimed between loads.
8. Select a 4B LoRA with 4B, adjust scale, then confirm it cannot be selected with 9B.
9. Run 4B on GPU 0 and GPU 1. Measure cold load, warm generation, peak VRAM, and output dimensions.
10. Run 9B with CPU offload, then separately test balanced multi-GPU. Record whether either is reliable enough to expose without an Experimental label.
11. After an artistic edit, run OCR, local Gemma translation, and ONNX inpainting to confirm no GPU-memory collision.
12. Launch the application with no Flux environment configured and confirm every existing feature still works.

## 12. Acceptance criteria

The feature is complete only when all of these are true:

- The application starts and existing workflows operate without Torch/Diffusers installed.
- The FLUX panel is below Inpainting and does not crowd the editor at ordinary window sizes.
- A selected region can be edited with 4B and previewed without modifying project state.
- Apply, Undo, Redo, Revert Artistic Edits, export, save, close, and reload all preserve the expected pixels.
- Revert All Inpainting never removes an artistic patch.
- A result cannot apply to a different page or to a changed source revision.
- The source file is never overwritten.
- LoRA discovery is model-size-specific, `.safetensors`-only, and scale is recorded.
- 9B's gated non-commercial license and memory requirement are visible before use.
- Cancellation and worker crashes leave the project usable.
- Tests perform no network or real-model operations.
- The pre-existing test suite remains green.

## 13. Implementation milestones and commit sequence

Each milestone should be a reviewable commit. Run the full existing test suite after every milestone. At baseline this repository had 49 passing tests; record the current count before beginning because it may change.

### Milestone 0 — reproducible spike

- Create a fresh optional Flux environment; do not repair the existing app venv in place.
- Run the current batch reference against one untracked test crop with 4B.
- Verify current Diffusers image-edit call, callback signature, LoRA calls, BF16, VAE tiling, and exact dependency revisions.
- Test one cancellation callback in a standalone harness.
- Test 9B CPU offload and balanced multi-GPU separately; record peak VRAM and elapsed time.
- Commit only the resulting dependency decision/documentation or a small untracked diagnostic script if appropriate. Do not add model weights or example comics.

Suggested commit: `Document validated FLUX.2 Klein runtime`

### Milestone 1 — generic typed patch foundation

- Add patch kind/metadata with legacy defaults.
- Scope clear/revert by patch kind.
- Extend v2 and legacy persistence.
- Add undo and project-state tests.
- Make no visible FLUX UI yet.

Suggested commit: `Generalize image patches for artistic edits`

### Milestone 2 — pure image and LoRA helpers

- Add request/result contracts.
- Add crop/context/pad/resize/mask/blend helpers.
- Add model-specific LoRA registry and sidecars.
- Add exhaustive CPU-only unit tests.

Suggested commit: `Add artistic edit data and image helpers`

### Milestone 3 — isolated worker

- Add optional requirements.
- Implement JSON-lines worker and QProcess client.
- Implement model/LoRA lifecycle, progress, cancellation, and structured errors.
- Use fake modules for automated tests.
- Manually prove 4B end to end before continuing.

Suggested commit: `Add isolated FLUX.2 Klein worker`

### Milestone 4 — UI, preview, and apply

- Add the panel below Inpainting.
- Add selection validation and selected-translation prompt helper.
- Add preview dialog.
- Apply accepted output through the typed patch command.
- Add stale-source and wrong-page protections.
- Disable in webtoon mode.

Suggested commit: `Add previewable FLUX artistic edit tool`

### Milestone 5 — settings, licenses, and resource behavior

- Add Flux Python/model/cache/device settings and environment probe.
- Add 9B license/memory warnings.
- Release managed llama.cpp and ONNX caches before FLUX.
- Add Unload and shutdown cleanup.
- Prove that returning to OCR/translation/inpainting is reliable.

Suggested commit: `Add FLUX runtime settings and GPU safeguards`

### Milestone 6 — documentation and stabilization

- Run the full manual matrix.
- Fix project reload, cancellation, OOM, or seam issues found.
- Update README and LoRA folder instructions.
- Document limitations and recommended workflows.

Suggested commit: `Document and stabilize artistic editing`

Do not combine all milestones into one commit. Stop and report after Milestones 0, 1, 3, and 4 so a human can test the risky boundaries.

## 14. Instructions for the implementing coding agent

Use this plan as the source of truth, but inspect the current code before each phase because later commits may have changed line numbers.

Important repository constraints:

- Work in your `local-comic-translate` checkout.
- Begin from `main` at or after `c0b9088`.
- Preserve unrelated user changes.
- The untracked `src/local_comic_translate/__init__.py` was created by a separate `uv init` experiment; do not add, edit, delete, or commit `src/` unless the user explicitly changes this instruction.
- Do not commit or push model weights, LoRAs, source comics, generated examples, tokens, Hugging Face cache files, or virtual environments.
- Do not alter the existing OCR/detection/translation/inpainting behavior as part of this feature.
- Do not add FLUX to Automatic Translation.
- Do not make Torch/Diffusers mandatory for startup.
- Prefer small methods and pure functions around coordinate/mask processing.
- Keep Qt updates on the GUI thread.
- Add tests before or with every behavior change.
- Do not push until the user has tested and explicitly asks.

Suggested first prompt for Qwen:

```text
Work in your local-comic-translate checkout. Read
docs/FLUX2_KLEIN_ARTISTIC_EDIT_PLAN.md completely before changing code.

Implement Milestone 0 only. Inspect the current repository and the reference
prototype at C:\AI\flux2-batch-edit\batch_edit_flux2.py. Record git status and
the baseline test count first. Recreate a clean optional Flux environment if
needed; do not modify the application's normal .venv. Validate the exact pinned
Diffusers/Torch/Transformers/Accelerate/PEFT versions with FLUX.2 Klein 4B image
editing and a local compatible LoRA. Test 9B CPU-offload and balanced multi-GPU
only if the local model is already available; do not download gated weights
without asking. Record cold-load time, warm inference time, peak VRAM, output
dimensions, and any callback/cancellation findings. Do not implement the GUI
or patch-system changes yet. Do not touch or commit src/. Stop with a concise
report, proposed dependency pins, files changed, tests run, and any deviations
you recommend before starting Milestone 1.
```

After Milestone 0 is reviewed, hand Qwen one milestone at a time. This integration crosses UI state, project persistence, CUDA runtimes, a large generative model, and undo history; incremental commits and human image review are part of the design, not optional cleanup.

## 15. Deferred adjacent enhancement: coordinate-aware local translation

This is deliberately outside the FLUX Artistic Edit milestones and should be
implemented separately after that integration is stable.

### Current limitation

The local multimodal translator sends the projector a page image and sends the
language model a separate JSON object mapping `block_N` identifiers to OCR
text. It does not currently send `TextBlock.xyxy`, `bubble_xyxy`, or
`text_class`. The vision model therefore sees the page layout implicitly but
must infer which visible region corresponds to each `block_N` by matching the
OCR text. Repeated dialogue, ambiguous OCR, unusual reading order, or closely
spaced bubbles can make that association uncertain.

### Preferred implementation

- Extend only the local-LLM input schema; keep the required response schema as
  the existing exact `block_N: translated string` mapping.
- For each block, send its OCR source text, normalized `[x1, y1, x2, y2]`
  coordinates, and `text_class`. Include the page width and height or document
  clearly that coordinates are normalized to `0.0–1.0`.
- Prefer normalized coordinates so prompts remain stable across page sizes and
  webtoon crops. Preserve the same block order and stable block identifiers
  used to commit translations.
- Tell the model explicitly that coordinates bind each OCR string to its page
  region and that the image is context for speaker attribution, reading order,
  and correction of obvious OCR errors.
- Consider an optional second, numbered/boxed annotation image only if prompt
  coordinates prove insufficient. Do not draw annotations onto the sole
  high-quality source image because they may obscure lettering or artwork.
- Keep image-disabled local translation working with the same structured block
  metadata.

Example input shape:

```json
{
  "page_size": [1200, 1800],
  "blocks": {
    "block_0": {
      "text": "¿Dónde estás?",
      "bbox_normalized": [0.1183, 0.1767, 0.3375, 0.2372],
      "text_class": "text_bubble"
    }
  }
}
```

Add regression tests for repeated identical source strings in different
regions, coordinate normalization and clamping, stable output-to-block
assignment, webtoon-local coordinates, image-disabled requests, and backward
compatibility with cached translations. Do not include coordinates in the
translated output or use model-inferred geometry to move existing text boxes.
