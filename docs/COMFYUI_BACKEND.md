# Local ComfyUI artistic editing

ComfyUI is an optional third Artistic Edit backend. **Local Diffusers** and
**Black Forest Labs API** continue to work as before. ComfyUI uses its own Python,
PyTorch, model files, and memory manager; it does not load those libraries into
the editor's Python environment. There is no separate Comfy browser window to
operate during normal editing.

## Setup

1. Select **Local ComfyUI** in the Artistic Edit backend dropdown, then
   **Configure ComfyUI…**.
2. Choose one connection mode:
   - **Connect to running server**: enter its loopback URL, normally
     `http://127.0.0.1:8188`. Start ComfyUI yourself, with the GPU and memory
     options you prefer. The editor never stops this server or unloads its models.
   - **Start selected installation**: select an existing Windows portable
     installation (the folder containing `ComfyUI` and `python_embeded`), or the
     ComfyUI source folder and its separate Python executable. Select the physical
     NVIDIA GPU index. The editor starts a hidden, dedicated server on a private
     loopback port. Initial managed-runtime qualification is Windows/NVIDIA only.
3. Click **Test Connection / Refresh Models**. In managed mode this starts the
   server, but does not load a generation model yet.
4. Select the **distilled** Klein model size and its matching diffusion model,
   Qwen text encoder, and FLUX.2 VAE from the server's catalogs. Changing the model
   size does not automatically choose different files. Alternatively, select
   **Checkpoint preset > Klein 9B INT8 (prequantized)** to fill in the tested
   `flux2Klein9BINT8_9b.safetensors`, `qwen_3_8b_fp8mixed.safetensors`, and
   `flux2-vae.safetensors` combination. These files must already be installed in
   ComfyUI; the preset neither downloads nor converts weights. The diffusion
   model is INT8; the encoder uses mixed FP8 precision. Editing any of these
   selections returns the preset selector to **Custom / native checkpoint**.
   Save the configuration.
5. Use the existing selection/painted-area/whole-page controls and prompt, then
   **Generate Preview**. Inspect the result before **Apply**. Apply creates the
   same reversible image patch as the other backends; Undo and Revert Artistic
   Edits remain available. Editable text overlays are not baked into the input.

No ComfyUI installation, packages, custom nodes, models, or LoRAs are installed or
downloaded by this feature. Use a trusted, already-working installation. Models
and third-party software retain their own licenses. Nothing is uploaded to a
cloud provider by this backend; LAN/cloud ComfyUI URLs are not supported.

## Model formats and LoRAs

This backend uses reviewed native-node Klein and experimental Qwen image-edit graphs, not an
imported arbitrary workflow. It preserves the editor's exact prepared image
dimensions and applies the editor's crop/mask/feathering when composing a preview.
The model can redraw surrounding pixels inside that effective mask; check the
preview carefully, especially for lettering.

Native `.safetensors` checkpoints retain their native format. The editor does
**not** quantize a BF16 checkpoint on load with bitsandbytes. Choose a compatible
prequantized checkpoint that your ComfyUI installation already supports. A
filename is not proof of compatibility or memory requirements.

The measured initial configuration is **Klein 9B native INT8**, the mixed-FP8
Qwen 3 8B encoder, and `flux2-vae.safetensors`. See
[validation details](COMFYUI_RUNTIME_VALIDATION.md). Klein 4B mappings are present
but experimental until inference-tested with matching weights. GGUF/custom-node
profiles and other quantization formats are not qualified or exposed in this
release. The API option remains useful when local hardware
is insufficient.

LoRAs are selected from **ComfyUI's** catalog, independently of the Diffusers
LoRA folder. For a new adapter, also select the matching local `.safetensors` file
to validate its model architecture and record its checksum. You must ensure
that this really is the file registered under that server name; the standard
server catalog does not attest the remote file's checksum. LoRAs for Klein 4B
and 9B, and Qwen Image 2.1 adapters, are not interchangeable. Verification is required again when changing
the connection or model profile. A same-seed regeneration retains the submitted
configuration instead of silently using newly selected settings.

## Experimental Qwen Image 2.1

To try Qwen instead of Klein, open **Configure ComfyUI…** and choose
**Checkpoint preset > Qwen Image 2.1 INT8 (experimental)**. Refresh the model
catalog and save. The preset selects these already-installed ComfyUI files:

- Diffusion: `qwen_image_2.1_int8_convrot.safetensors`
- Text encoder: `qwen3vl_8b_int8_convrot.safetensors`
- VAE: `qwen_image_2.1_vae_bf16.safetensors`

Both the diffusion model and encoder are prequantized INT8; the VAE is BF16.
This does not convert weights or install anything. ComfyUI must expose the
native `TextEncodeQwenImage21` and `QwenImage21Cache` nodes. Older Qwen Image/Edit
weights, encoders, and VAEs are not substitutes for the 2.1 files. Refer to
[Comfy-Org's model files and workflow](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
for setup; model licenses apply independently of the editor's license.

The initial settings follow the installed official edit template: **25 steps,
CFG 1.0, Euler / Simple**. Steps and CFG are adjustable under Advanced, with
separate in-session values for each backend/model. Same-seed regeneration keeps
the submitted profile, settings, and input. The optional prompt-enhancement
model is not required and is not invoked.

Selection, painted masks, preview, Apply, Undo/Redo, and project persistence
work as for Klein. The graph pads to multiples of 32 without resizing the
prepared crop, then removes that padding before composition. Pixels outside
the editor's effective edit mask remain protected by the editor, not by the
model. Generated transparency is composited over the original input rather
than exposing hidden RGB colors. Output remains opaque RGB; this is not a
background-removal/alpha tool.

**Experimental Qwen Image 2.1 model-only LoRAs are supported.** In the same
configuration dialog, select a file from **ComfyUI LoRA (optional)**, set its
strength, and use **LoRA verification file** to select the corresponding local
`.safetensors` file. Save checks its tensor names and dimensions without loading
the weights into the editor, then records a SHA-256 checksum. You must select
the actual file represented by the server's catalog entry; its name alone does
not prove that the server and local file are identical.

The native layouts qualified against the Qwen Image 2.1 INT8 preset are fused
`diffusion_model.*` adapters and split `transformer.*`/unprefixed Qwen adapters.
The fused adapter layout needs a checkpoint exposing fused gate/up weights;
arbitrary native checkpoint variants need separate qualification. Older Qwen Image/Edit adapters,
Klein adapters, text-encoder adapters, DoRA/unknown layouts, incomplete pairs,
and attention-only adapters whose architecture cannot be established are
rejected. A rejection means the adapter is not supported by this validator,
not necessarily that it could never work with a different workflow.

Clear the LoRA selection to disable it; strength zero also leaves the model
unmodified. While the dialog is open, each model keeps its own adapter fields,
so switching model families never silently selects an incompatible adapter.
Regeneration keeps the submitted adapter/strength/checksum even if settings
are changed before applying the result.

Acceleration adapters may need different sampling settings: change **Steps**
and **Guidance** under Advanced according to the adapter's instructions. No
step count is inferred from its filename. Live tests covered a regular rank-32
adapter at strengths 0.4/0.8 and the installed Turbo rank-256 adapter at six
steps, CFG 1, strength 1. Those checks demonstrate loading and reversible
switching, not improved comic-lettering quality. See the [validation record](COMFYUI_RUNTIME_VALIDATION.md#qwen-image-21-loras-2026-10-06).

Select the Klein preset again to return to its existing LoRA workflow.
The Diffusers and BFL API backends remain Klein-only and unchanged.

Qwen needs separate quality and hardware evaluation: a successful small crop
does not establish full-page performance or support for 8/12 GB GPUs. Inspect
every preview, particularly exact lettering and artwork continuity. During
lengthy encoding/loading, or on an unqualified server version, cancellation
may return "Stopped waiting" before the server becomes idle; wait before
stopping a managed server. No other client's work is interrupted.

## Memory and process ownership

- The optional **Disable execution cache** policy trades reuse/speed for less
  retained graph state. It is not a guarantee that a model fits a particular GPU.
- Start with a selected region and modest processing resolution. Image size,
  encoder precision, checkpoint format, and LoRA patches all affect memory use.
- Close or unload other GPU applications yourself if needed. This feature does
  not stop llama-server, other ComfyUI instances, or other applications.
- **Stop ComfyUI** stops only the idle server started by this editor. If its queue
  contains another client's work, unloading/closing is refused until it is safe.
- **Disconnect** from an attached server leaves it running. Cancellation is
  targeted to this request when the server supports it. Older servers can stop
  the editor waiting while continuing computation; the status makes this clear.
  No global queue clearing, interrupt, or model-unload command is sent.

Managed launches use separate input, output, temporary, user, database, and cache
directories beneath the fork's user-data folder. Installed ComfyUI profiles and
model files are not rewritten. Custom nodes and API nodes are disabled for these
initial managed profiles. Existing external-model-path configuration can expose
additional model locations; install/profile isolation is not an OS security
sandbox. Only run installations you trust.

Uploaded/generated images are retained in the server's own folders (including
the isolated folders of managed launches). This version does not automatically
sweep those folders or delete shared-server history. Normal project patches and
undo history are separate from server artifacts.

## Troubleshooting and scope

- **Missing model/node**: refresh the catalogs and select a matching installed
  checkpoint, encoder, and VAE. Managed mode needs the verified safe startup
  flags available in ComfyUI 0.37.0 or later; a version number alone does not
  establish compatibility. Use connect mode for a different supported layout.
- **Wrong model size**: 4B uses its matching Qwen 3 4B encoder; 9B uses its matching
  Qwen 3 8B encoder. Base/undistilled variants are not supported by this graph.
- **Out of memory**: lower processing resolution, remove the LoRA, try an
  appropriate smaller/prequantized model, or use the hosted backend. No automatic
  destructive retry, model conversion, or server restart is performed.
- **Preview rejected after page/project changes**: regenerate it against the
  current page. Old results must never be applied to a replacement project.
- Progress currently shows connection, upload, queued/executing, and retrieval
  stages rather than a fabricated per-step percentage. Server history is the
  authority for completion, not a disconnected progress stream.

Real 8/12 GB VRAM and low-system-RAM qualification, macOS managed launch, GGUF,
custom workflows, batch generation, and webtoon editing remain outside
this first tested configuration. See the
[implementation plan](COMFYUI_BACKEND_PLAN.md) for the broader roadmap.
