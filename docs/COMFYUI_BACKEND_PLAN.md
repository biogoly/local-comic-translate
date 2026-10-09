# Optional ComfyUI backend: implementation and validation plan

> **2026-10-03 extension:** Experimental native Qwen Image 2.1 editing has now
> been added as a separate versioned workflow, using its own encoder/VAE and
> INT8 preset. The original Klein milestones and historical observations below
> are retained. Current setup/limits are in [the backend guide](COMFYUI_BACKEND.md#experimental-qwen-image-21);
> measured acceptance results are in [the validation record](COMFYUI_RUNTIME_VALIDATION.md).
> Qwen patches use the neutral `artistic_edit` kind; Revert Artistic Edits also
> includes legacy `flux2_edit` patches without migrating existing projects.

Plan date: 2026-09-26. Implementation update: 2026-09-27.

**Initial native ComfyUI backend implemented and inference-tested.** C0's
existing-checkpoint proof, C1 client, C2 GUI/controller integration, and C3 managed
runtime are implemented. See [setup](COMFYUI_BACKEND.md) and the
[validation record](COMFYUI_RUNTIME_VALIDATION.md). The detailed sections below
retain the original design and its acceptance targets; they are not a claim
that every proposed profile has been qualified.

Delivery adjustments and remaining qualification:

- The available checkpoint was **native INT8 Klein 9B**, not a BF16/FP8 diffusion
  checkpoint. It passed with the mixed-FP8 Qwen 3 8B encoder, FLUX.2 VAE, and a
  matching LoRA. No new weights or dependencies were downloaded.
- C1 uses bounded HTTP history polling rather than WebSocket progress. Completion
  is authoritative, cancellation is targeted where supported, and progress is
  honestly stage-based. WebSocket progress can be layered on later.
- C4 is **partial**: the installed native INT8 combination is tested. The 4B
  mapping is explicitly experimental. GGUF/custom-node profiles are not exposed;
  their compatibility, LoRAs, and lower-VRAM/lower-RAM hardware qualification
  remain pending. Do not advertise an 8/12 GB hardware guarantee from these tests.
- C5 covers regression/localization/docs and optional-backend package smoke
  checks. A public beta release and broader hardware testing remain separate
  release work; this change does not publish a package or resolve unrelated
  dependency-redistribution reviews.

## 1. Objective and boundaries

Add **Local ComfyUI** as an optional Artistic Edit backend alongside the working
**Local Diffusers** and **Black Forest Labs API** backends. The purpose is to make
local editing practical on more modest GPU **and system-memory** configurations,
reuse existing ComfyUI models/LoRAs, and reduce the work needed to support future
image-editing models. This is not a replacement for the current working setup.

The first supported model family remains FLUX.2 Klein **4B and 9B distilled**.
Support the existing selected-box, painted-area, and whole-page editing modes,
including edit margin, context, feathering, preview, regeneration, Apply, Undo,
and Revert Artistic Edits. Keep automatic translation, OCR, segmentation,
inpainting, clone/restore tools, and translation-provider settings unchanged.

Initial delivery includes connecting to an existing local ComfyUI server and
optionally starting an existing installation in the background. It does **not**
install/update ComfyUI, Python, CUDA, custom nodes, or models automatically.

Deferred: Qwen Image 2.1 execution, Klein base/undistilled variants, arbitrary
workflow imports, cloud/LAN Comfy endpoints, batch generation, webtoon support,
multi-GPU scheduling, model conversion, automatic model downloads, and packaging
ComfyUI itself. No changes to the current Diffusers/BF16 configuration or return
to on-load bitsandbytes quantization are required.

## 2. Findings from the current checkout

This plan reflects the **working files**, including uncommitted Windows packaging
changes, not just Git HEAD (`4a61c24`). Existing README, splash, packaging, worker,
controller, and other local changes must be preserved. Before implementation,
record the actual starting diff and agree on a reviewable integration baseline;
do not reset, stash, pull over, or commit unrelated work as part of this feature.

| Existing component | Reuse / change needed |
| --- | --- |
| `app/ui/artistic_edit_panel.py` | Append backend ID `comfyui`; keep existing IDs `local` and `bfl`. Replace two-backend enable/disable assumptions with explicit capabilities. |
| `app/controllers/artistic_edit.py` | Reuse capture, `prepare_edit`, result composition, preview, Apply, and undo. Add explicit dispatch and job identity handling. |
| `modules/artistic_edit/image_ops.py` | Keep authoritative crop, padding, edit mask, scaling, edit margin, and feathering in the app. |
| `modules/artistic_edit/contracts.py` | Preserve the strict existing Diffusers worker wire format. Do not send Comfy-specific fields into it. |
| `modules/artistic_edit/bfl_client.py` | Useful asynchronous job/result pattern, not a Comfy transport implementation. Preserve cloud upload consent and credentials behavior. |
| `modules/artistic_edit/lora_registry.py` | Reuse bounded safetensors-header and Klein architecture checks; Comfy selection uses its own registered filenames. |
| `modules/utils/external_process.py` | Reuse environment/DLL isolation for launching an external Python from the frozen Windows app. |
| `modules/artistic_edit/runtime_paths.py` | Preserve frozen Diffusers worker discovery; reuse GUI-executable rejection where appropriate. |
| `modules/utils/settings.py` | Use fork-owned `app_settings()` and a new `artistic_edit/comfyui/*` namespace. |
| `app/projects/patch_metadata.py` | Metadata accepts flat primitive values, at most 32 keys and 2,048 characters per string. Do not persist a workflow graph here. |
| `packaging/windows/` and `app/package_smoke.py` | Package lightweight client code/templates only. Preserve existing external FLUX worker payload and exclusions. |

Specific controller assumptions to remove carefully:

- `update_availability()` and `generate_preview()` currently treat non-`local`
  backends as cloud in runtime checks.
- The preview defaults to the label **BF16**; Apply records every nonlocal model
  source as BFL and omits its steps/guidance.
- Results and cancellation have request-ID checks, but loading/progress/errors
  from a retained Diffusers worker could affect a different backend's session.
  Gate **every** asynchronous callback by backend, runtime generation, and job ID.
- `Regenerate Same Seed` currently rereads panel options. A Comfy regeneration
  must retain its submitted configuration snapshot, not a refreshed model list
  or silently changed workflow.
- `get_composited_page_image()` supplies page pixels plus accepted image patches,
  not a screenshot of editable text overlays or selection handles. Preserve this;
  neither bake overlays into the input nor delete overlapping text items on Apply.

Read-only inspection found an existing Windows portable installation at
`C:\path\to\ComfyUI_windows_portable`,
with `python_embeded/python.exe`, `ComfyUI/main.py`, declared ComfyUI **0.37.0**,
and Python DLL version **3.13.14**. This verifies layout/source, **not successful
imports, model loading, or inference**. Treat it as the initial test candidate,
not an automatically qualified minimum version.

The earlier runtime report describes **two separate 24 GB RTX 3090 Ti GPUs** and
approximately 128 GB system memory. Do not treat these as one pooled 48 GB GPU,
or use this host alone to establish low-RAM compatibility.

## 3. User experience and settings

Keep Generate Preview near the prompt and preserve the compact, themed panel.
Normal editing should not require opening ComfyUI's browser/node editor.

- **Backend:** Local Diffusers / Local ComfyUI / Black Forest Labs API.
- **Configure ComfyUI...:** a separate dialog for connection, model profiles, and
  diagnostics; do not repurpose the existing Select FLUX Python control.
- **Connection mode:** Connect to running server, or Start selected installation.
  Connect mode defaults to loopback port 8188; managed mode uses its own available
  loopback port and starts only on an explicit connection/generation action.
- **Model:** Klein 4B / Klein 9B; distilled only in this release.
- **Model profile:** a validated combination of diffusion checkpoint, loader,
  text encoder, VAE, and workflow. Display actual format, e.g. FP8 or GGUF Q4_K_M,
  not a generic checkbox that purports to quantize arbitrary weights.
- **LoRA / strength:** list compatible Comfy-registered `.safetensors` adapters;
  clearly explain missing, incompatible, or unverified adapters. Start with one
  LoRA, matching the existing UI, with the existing 0.0-2.0 strength range.
- Retain prompt templates, selection, seed, steps where supported, context,
  edit margin, feather, and maximum side. Hide/disable unsupported guidance with
  an explanation; do not equate Diffusers guidance with Comfy CFG automatically.
- **Memory options:** backend-specific. Existing Diffusers device/offload options
  remain untouched. A managed Comfy process may expose a physical GPU choice and
  tested memory/cache presets. An attached process inherits its owner's device
  policy; the app cannot change it per request.
- **Status:** Not configured / Connecting / Starting / Uploading / Queued /
  Loading model / Generating / Fetching preview / Cancelling / Error. Indeterminate
  progress is preferable to assigning another client's progress to our job.
- **Unload:** managed mode releases the app-owned runtime; attached mode offers
  Disconnect instead, explaining that the server and its models remain running.

Persist connection mode, install/interpreter paths, endpoint, selected profile,
and backend-specific options under `artistic_edit/comfyui/*`. Keep the default
backend `local` for existing installations and preserve their FLUX Python setting.
Store model catalog names as stable IDs, never translated labels or combo indexes.
Missing selections should remain visibly unresolved rather than silently switching
to a different checkpoint. Do not copy settings into the upstream app's profile.

All new visible text/tooltips/errors must be translated and compiled for the
existing nine non-English catalogs: de, es, fr, it, ja, ko, ru, tr, zh-CN. Test
all six themes, narrow sidebars, scaling, and long localized labels.

## 4. Architecture and contracts

```text
Existing page + accepted patches + selection
                  |
        prepare_edit / frozen session
                  |
          backend-specific execution
          /            |           \
   Diffusers       ComfyUI         BFL API
   unchanged    workflow + client   unchanged
          \            |           /
       validated local PNG + execution details
                  |
       app-side mask/feather composition
                  |
           Preview -> Apply -> Undo
```

Keep the backend boundary small; do not introduce a general plugin framework.
Proposed additions under `modules/artistic_edit/`:

- `backend_contracts.py`: immutable common raster-input spec, execution details,
  backend capabilities, and correlated job events. The existing
  `ArtisticEditResult` can remain the shared validated PNG result.
- `comfy_config.py`: validated connection/settings types and model-profile data.
- `comfy_workflows.py`: versioned profile registry, required-node/schema checks,
  and pure API-graph construction from validated inputs.
- `comfy_client.py`: bounded HTTP operations, WebSocket observation, history
  reconciliation, result validation, and targeted cancellation.
- `comfy_job.py`: Qt worker adapter; emits correlated events and performs all
  network/image-transfer work off the GUI thread.
- `comfy_runtime.py`: installation resolution and ownership-aware QProcess
  startup/readiness/shutdown. It does not implement generation itself.
- `workflows/comfyui/`: reviewed API-format JSON graphs and a manifest, bundled
  as package data. Keep upstream source URLs/revisions and applicable notices.
- `app/ui/dialogs/comfyui_settings.py`: configuration/test-connection dialog.

Do not import Torch, Diffusers, Comfy modules, or custom-node code into the GUI.
Use existing `requests` for HTTP. If WebSocket observation uses `websocket-client`,
add only that lightweight client dependency with a tested pin, packaging entry,
and notice; do not add PySide6 Addons merely for QtWebSockets. HTTP polling must
remain a functional fallback, not trigger resubmission.

Freeze at submission: app request UUID, client UUID, runtime-instance token,
page/project identity, prepared image and mask, source hash, resolved seed,
model/LoRA IDs and strengths, workflow version/hash, dimensions, and all settings.
Map this to the returned Comfy `prompt_id`; all subsequent events must match.
Cancel, backend switch, project close, or runtime replacement immediately revokes
permission to preview/apply results. Keep **job-lifecycle ownership** until its
correlated terminal acknowledgement and cleanup complete; do not discard the
cancellation acknowledgement merely because results are now invalid. A detached
job finishes cleanup without changing another session's UI. One app editing job
at a time; never mix unrelated server jobs.

Keep the existing `ArtisticEditRequest` serializer/parser unchanged for the
Diffusers worker and adapt common fields into it. Give Comfy its own execution
spec instead of placing server filenames into the absolute-path `LoraSpec`.

Persist compact execution details: backend, model family/size, actual checkpoint
name/known fingerprint, encoder/format, VAE, workflow ID/version/hash, runtime
version, loader revision if known, LoRA name/hash/strength when verified, seed,
effective steps/guidance, dimensions, selection mode, and source hash. Fit within
the current flat metadata limits. Do not claim unknown hashes/precision are known;
never save authentication data, complete graphs, or private endpoint URLs.

For this Klein-only addition, keep persisted patch kind `flux2_edit`; both existing
and Comfy-generated Klein patches participate in Revert Artistic Edits. Do not
rename legacy patch records. A future non-FLUX adapter should introduce a neutral
artistic-patch grouping that also recognizes legacy records, with its own migration
tests. Accepted patches must reopen/export without Comfy or the model installed.

## 5. Workflow/model profiles

Use reviewed, versioned **API-format** graphs derived from the official Klein
distilled image-edit templates. Do not fetch mutable graphs at runtime, accept
arbitrary imported node graphs, or assume a UI workflow JSON is directly executable.

Each manifest entry declares: model family/variant, workflow version/hash, tested
Comfy revision, exact required node classes and input schemas, diffusion/encoder/
VAE loader choices, allowed checkpoint formats, LoRA support, supported parameters,
dimension constraints, and the single authoritative output node. Build graphs by
setting typed JSON fields, not raw string substitution with prompts/filenames.

Qualification order:

1. **Native BF16/FP8, 4B and 9B distilled.** Use compatible Comfy-format files;
   existing Hugging Face Diffusers directories are not automatically interchangeable.
   Start at four steps, batch size one, with official distilled conditioning and
   scheduler settings. Verify behavior rather than copying numerical guidance
   values from the Diffusers API.
2. **GGUF Q8_0 and Q4_K_M.** Separate loader-specific profiles with a tested
   ComfyUI-GGUF revision and matched encoder. Support one validated combination
   first, then expand; do not advertise every file with `.gguf` as compatible.
3. **Native INT8/ConvRot.** Qualify exact checkpoints/runtime combinations if
   available. Existence of an INT8 loader/checkpoint is not proof that Klein,
   editing, and the user's LoRAs work together. Leave unqualified profiles disabled.

FP8, INT8/ConvRot, and GGUF Q8/Q4 are distinct formats. Never relabel them as one
universal precision option or force native FP8 compute on an unsupported GPU.
Prequantized files avoid a requirement to quantize a full BF16 model at app startup,
but inference can still dequantize tensors and allocate substantial working memory.

Model discovery comes from the selected server's `/object_info` loader inputs
and supported model-catalog endpoints. Resolve names for the correct loader,
including subdirectories and duplicate basenames. Never upload a model or LoRA
through the image-upload API. Do not duplicate or convert the user's weights.

LoRA validation must include architecture/size, format, strength changes, removal,
and cache invalidation. Reuse local header checks when a verified local model-root
mapping is available; reject cross-size and FLUX.1 adapters. If an attached server
cannot provide enough information to verify an adapter, require an explicit local
file mapping/profile validation rather than presenting it as compatible. Matching
dimensions alone do not prove base/distilled training compatibility or quality.
ComfyUI-GGUF labels its LoRA loading experimental: a profile may be released as
**LoRA unavailable** until actual LoRA tests pass. Never accept silently skipped
adapter tensors as a successful LoRA application.

## 6. Geometry and image integrity

The app, not ComfyUI, remains responsible for defining which page pixels may change.

- Upload the exact prepared RGB PNG crop, not the full page unless selected, and
  not scene overlays. Keep `PreparedEdit` and its source hash for composition.
- Distinguish **context** (model input) from **edit margin** (editable pixels).
  Preserve the current expanded-lettering behavior and app-side feather support.
- Official example workflows include `ImageScaleToTotalPixels`; do not let an
  unnoticed internal resize redefine output geometry. The adapter must explicitly
  control reference/latent dimensions and remove or configure implicit resizing.
- Confirm model-specific alignment/minimum sizes in milestone C0. Current
  preparation pads to multiples of 16. If a validated Comfy graph requires further
  padding, record an exact pad/crop-back transform; never stretch a mismatched result
  merely to make it fit. Return exactly the prepared dimensions to common composition.
- Retrieve only the manifest's designated final output, not an intermediate
  preview or the first arbitrary image in history. Validate format, byte limit,
  pixel count, channel handling, and dimensions before writing an atomic local PNG.
- Recheck page/project identity and source pixels before showing/applying the result.
  Reject stale results and explain why; no page or undo-stack mutation on failure.
- Test that pixels outside the **effective feathered alpha support** are unchanged.
  A feather can intentionally affect pixels beyond a hard selection boundary.
- No new flattening/deletion of editable text overlays. Existing text remains
  editable even if it overlaps an artistic patch.

## 7. API transaction and failure handling

Initial scope is local loopback only: `127.0.0.1`, `localhost` resolving to loopback,
or `[::1]`, with validated scheme/port and no URL credentials/query/fragments.
Do not inherit HTTP proxy settings for this local connection or follow redirects
to other hosts. Remote authenticated servers require a separate privacy/security
design later. Use only allowlisted local nodes; no cloud/API nodes in shipped graphs.

| Stage | Contract |
| --- | --- |
| Preflight | Check `/system_stats`, required `/object_info` schemas and selected catalog names. Missing capability blocks before upload/generation. No silent install/update. |
| Upload | Multipart `POST /upload/image` with a unique app/request subfolder/name and overwrite disabled. Use the returned name/subfolder/type; collision handling may rename it. |
| Submit | `POST /prompt` with the versioned graph, unique `client_id`, and correlation metadata. Retain `prompt_id` and report validation/node errors. Submit once. |
| Observe | Match `/ws` events to this prompt; use job-specific history/status polling as recovery. Server-wide queue length is not our job progress; a node `executed` event is not whole-job success. |
| Complete | Require a successful terminal state/history and the expected final output descriptor. Handle cached-node execution, which may skip normal step events. |
| Download | Fetch through `/view` using validated descriptor fields and original-image output, not arbitrary URLs or filesystem reads. Enforce time/size limits and validate PNG geometry. |
| Finish | Emit exactly one correlated terminal acknowledgement to the owning job, including after cancellation. Only a still-valid result may reach preview/Apply. Keep local files until preview/apply or final disposal. |

Use separate bounded connection/read timeouts and stage deadlines. Model startup
can be slow: surface elapsed time and cancellation instead of a frozen window.
Probe, clock, HTTP transport, socket, and process factory must be injectable for tests.

On an ambiguous submission timeout, **do not automatically POST again**. A prompt
UUID is correlation, not a verified server-side idempotency guarantee. Reconcile
queue/history or the job endpoint; if acceptance cannot be determined, report the
uncertainty and require an explicit retry decision. After a socket failure, resume
observation/download of the same job. Bounded retries of safe reads are acceptable.

OOM, invalid LoRA, missing nodes, process exit, timeout, malformed output, and
cancellation must leave the page unchanged and restore usable controls. Do not
silently switch model, lower resolution, send data to an API, or loop on retries.
Offer actionable choices such as a smaller crop/profile or restarting an owned
runtime; keep technical log tails bounded and redact credentials/private content.

## 8. Cancellation, ownership, and managed startup

Keep **connection ownership** separate from endpoint address. An existing server
on localhost is not automatically ours. Never kill processes by name or port.

The inspected 0.37.0 source provides `POST /api/jobs/{job_id}/cancel`, using
`interrupt_if_running(prompt_id)` for atomic targeted cancellation. Qualify this
capability against a tested server revision/schema. Legacy `/interrupt` is not a
safe shared-server fallback, including implementations that first inspect a queue
snapshot and then issue a global interruption. `/free` is also server-wide.

| Action | App-started dedicated runtime | Attached existing server |
| --- | --- | --- |
| Cancel queued/running job | Prefer verified targeted job cancellation. | Same, only for this app's prompt ID. |
| Older server without targeted running cancellation | Remove only our queued job; if needed offer stopping our child to abort active work. | Remove only our queued job. For active work, stop waiting/discard the result and explicitly say server computation may continue. |
| Unload/disconnect | Initial implementation stops the idle owned runtime, releasing all its model memory; next run starts it again. No global `/free` needed. | Disconnect only. No `/free`, global `/interrupt`, queue clearing, or termination. |
| Close app | Bounded cancellation/transport shutdown, then stop only our tracked child if no foreign-work conflict; cleanup after it stops. | Cancel this app's job where safe, disconnect, leave server/other jobs untouched. |

Stop-after-unload is deliberate for the first implementation: it offers reliable
memory release without promising a per-model unload operation Comfy does not expose.
Before stopping an owned runtime, detect unexpected foreign queued/running work.
If found, refuse normal unload/stop and defer app closure: offer Wait or Cancel
Close, not a silent kill. A separate force-stop action must explicitly warn that
it would stop the other work and require confirmation. Do not offer detachment
unless independently implemented/tested: destroying a live QProcess can kill its
child. Retain files while the runtime is still using them.

A queue snapshot is not an atomic shutdown lock against a new external submission.
Managed runtimes must be documented as dedicated, not shared services; do not
advertise their private endpoint for other clients. Strict noninterference with
other clients is guaranteed by **attach mode's refusal to stop/unload the server**,
not by claiming ownership makes every managed shutdown race impossible. Never
broaden a failure recovery into terminating an external server.

Managed mode requirements:

1. Resolve a user-selected portable root to its exact interpreter and `main.py`;
   allow explicitly configured venv Python for other installations later. Reject
   the frozen GUI executable, missing files, and ambiguous layouts.
2. Launch Python directly, using QProcess argument arrays and no shell/batch file.
   Set cwd to the ComfyUI directory, use `--listen 127.0.0.1`, a dedicated available
   port, and `--disable-auto-launch`; portable builds retain the appropriate
   `--windows-standalone-build` flag. Do not open a browser or visible console.
3. Reuse `sanitized_external_environment()` and `external_process_environment()`;
   do not leak bundled Qt/CUDA DLL paths or inherited PYTHONPATH/PYTHONHOME into
   Comfy. Verify restoration after successful and failed process creation.
4. Put input/output/temp/user data in app-owned writable profile/session directories
   using tested Comfy directory flags. Temp/input/output roots must be unique to
   the owned runtime so its startup cleanup cannot erase another instance's files.
   Set an explicit **app-owned `--database-url`** as well: in inspected 0.37.0,
   changing `--user-directory` alone can trigger legacy database migration that
   renames/copies the existing installation's database. Validate database/Windows
   path syntax in C0 and test the original database and profile remain unchanged.
   If another version cannot safely isolate its state, refuse managed launch for
   it and offer attach mode instead. Reuse existing model paths read-only; no
   installation into the packaged app folder or edits to Comfy's global config.
5. Core profiles start without custom nodes/Manager where supported. GGUF profiles
   explicitly enable only their tested installed loader dependency when the runtime
   supports an allowlist. In inspected 0.37.0 the whitelist matches directory
   **basenames across every configured custom-node root**, not a unique full path.
   Resolve all configured roots, reject duplicate allowed basenames, verify the
   selected loader path/revision, and review its prestartup behavior before launch.
   Otherwise require a separately configured trusted runtime; do not weaken the
   baseline by blindly importing every custom node.
6. Poll readiness asynchronously while checking child liveness. Verify the expected
   server instance/port and capabilities. For inspected 0.37.0, correlate
   `/system_stats.system.argv` with this launch's unique directory/port arguments
   and child liveness; a healthy port and matching version alone are insufficient.
   Port collisions retry boundedly or report failure; **never silently attach** to
   the process that occupied the port.
7. Retain the process handle/runtime token. Do not restart a child while its job may
   still be executing or report cancellation before late results are invalidated.
   Windows termination may be forced; do not label QProcess termination graceful.
8. No automatic stopping of llama-server, another Comfy process, or unrelated GPU
   jobs. Show memory pressure and ask the user to free resources when appropriate.

Cleanup: app-generated server files live under an owned namespace. For managed
mode, remove only verified files under that root after no process/job uses them,
with a bounded retention policy for failures. For attached mode, clean local copies
only; explain that uploads/results may remain in the user's Comfy input/temp/output
folders. Do not assume a standard remote file-deletion API or edit/delete other
users' Comfy files. Never sweep an entire installation's input/output/history.

## 9. Low-memory accessibility and benchmark plan

ComfyUI is an execution option, not a guarantee of low memory. Qualification must
cover diffusion weights **and** encoder, VAE, activations, caching, LoRA patching,
startup transients, and Windows shared-GPU-memory/pagefile use.

The installed version explicitly says `--lowvram` has no effect when dynamic VRAM
is enabled. Start with its supported default memory manager; benchmark quantized
weights, smaller bounded crops, tiled VAE decoding if qualified, and cache policy.
`--cache-none`, VRAM headroom/reserve controls, and disk/offload behavior are
version-specific candidates, not universal presets. Never force `--gpu-only` or
high-RAM caching for the low-memory audience. Model-switch peaks matter too.

Start with these **test targets, not promised minimum requirements**:

| Target machine | Candidate tests |
| --- | --- |
| 8 GB GPU / 16 GB RAM | Klein 4B Q4 candidate; 512 and 768 max-side crops; one LoRA only if qualified. Record failure honestly if insufficient. |
| 12 GB GPU / 32 GB RAM | Klein 4B FP8/Q8/Q4 at 768 and 1024; test a 9B Q4 profile only if feasible. |
| 16 GB GPU / 32 GB RAM | Klein 4B profiles and 9B Q4/Q8 candidates, 1024 crops and controlled larger inputs. |
| 24 GB GPU / 32-64 GB RAM | Native and quantized Klein comparisons at 1024/1536, matching settings/crops. |
| Maintainer host | Regression and performance reference on one explicitly selected 24 GB GPU; not proof of any low-RAM tier. |

GPU architecture and operating system affect results; VRAM capacity alone is not
enough. Tests with artificial reservations on the maintainer machine can reveal
problems, but are **not** substitutes for real 16/32 GB-RAM machines. Mark untested
tiers experimental and recruit testers before advertising supported minimums.

Record per run: runtime/template/loader revisions, GPU and driver, RAM/pagefile,
exact model formats, encoder/VAE, LoRA, crop/resolution, seed/steps, first-load time,
first-generation and warm-generation times, peak dedicated/shared GPU memory,
peak process-tree RAM/commit, and output quality. Separate cold filesystem-cache
and warm-cache measurements; do not compare a cold Comfy run with a warm Diffusers
run or compare different model variants/steps.

Use privacy-safe synthetic fixtures plus user-selected example crops: an artistic
title, Japanese sound-effect lettering, colored/nonwhite background, dense small
text, and an expanded speech bubble. Do not commit the user's comic pages or LoRAs.
Judge exact translated spelling, style continuity, edit-boundary preservation,
LoRA effect, and legibility at native page scale. No cross-backend bit-exact output
requirement; fixed seeds support controlled comparisons, not a determinism promise.

## 10. Milestones and stop/go gates

Each milestone should be a small reviewable change with automated tests and an
explicit test handoff. Do not implement every phase in one untested change.

### C0 - Reproducible workflow proof and baseline

- Record existing app regression results and dirty baseline without modifying it.
- Record exact installed Comfy commit/version, relevant loader versions, and GPU
  availability; do not update/install anything silently.
- Build a minimal distilled Klein core-node API workflow from a pinned official
  template using existing compatible weights. Validate prompt, seed, dimensions,
  selected reference image, output retrieval, and targeted cancel.
- Before any test-managed launch, isolate input/output/temp/user/database paths
  explicitly and verify the original installation's database/profile are unchanged.
- Test one 4B profile and one 9B profile where weights are available, then a real
  compatible LoRA. Missing models/LoRAs become explicit prerequisites, not automatic
  downloads. Start with small private fixtures and the user's chosen GPU.
- Produce `docs/COMFYUI_RUNTIME_VALIDATION.md` with exact evidence and unresolved
  items. Resolve resizing, cancellation, and loader/LoRA assumptions before GUI work.

**Gate:** a repeatable, bounded edit returns the expected-size PNG; runtime/model
requirements and measured baseline are documented. No claim of low-memory support yet.

### C1 - Client, workflow profiles, and offline tests

- Add configuration/contracts, pure workflow builder, HTTP/WS client, and Qt job.
- Add fake transport/clock/socket/server fixtures for all transaction/failure paths.
- Preflight required schemas; submit once; retrieve only the designated final output;
  implement correlated terminal events, read recovery, and safe targeted cancellation.
- No automatic launch and no changes to the main page until this standalone path works.

**Gate:** offline tests run without Comfy/Torch/GPU/network. An opt-in real-server
smoke test produces the C0 result through the new client.

### C2 - GUI integration and page safety

- Add Local ComfyUI, connection/model-profile dialog, backend-specific controls,
  persistent settings, visible disabled reasons, and execution-aware preview metadata.
- Use the common preparation/composition/preview/Apply/Undo route. Guard every
  existing worker callback against the wrong backend/session.
- Keep existing Diffusers/BFL behavior, source language settings, and text overlays.
- Translate strings introduced in this milestone; do not leave a partially English UI.

**Gate:** selected box, painted area, whole page, edit margin, regeneration,
cancel/discard, Undo/Redo/Revert, save/reopen/export, and stale-page tests pass.
User can test a running Comfy server before automatic startup is introduced.

### C3 - Optional managed runtime

- Add installation resolver, async startup/readiness, dedicated loopback port,
  environment isolation, app-owned directories, bounded logs, and ownership controls.
- Implement managed stop/unload and safe app shutdown. Connect mode never acquires
  ownership of an external process. No automatic installer/updater.
- Exercise source and frozen Windows paths, including spaces/non-ASCII characters.

**Gate:** open editor -> generate -> preview -> unload/restart -> close works without
opening Comfy's browser. An already-running external server survives all app actions.

### C4 - Quantized profiles and low-memory qualification

- Qualify native FP8 and GGUF Q4/Q8 combinations separately for 4B/9B, diffusion
  model and encoder, with/without real compatible LoRAs and strength/removal changes.
- Qualify INT8/ConvRot only if a concrete candidate passes; otherwise document it
  as deferred. Do not block stable native profiles on an experimental quantizer.
- Add measured memory presets with honest tradeoffs, actionable OOM recovery, and
  test results on lower-RAM machines. No silent model/resolution fallback.

**Gate:** publish exact verified configurations; at least one genuinely lower-memory
configuration must be demonstrated before claiming the accessibility objective met.
Untested hardware tiers and unverified LoRA combinations remain clearly labeled.

### C5 - Packaging, regression, documentation, and beta handoff

- Bundle only lightweight client code, templates/manifests, translations, and
  required notices. No Comfy runtime/model/custom-node downloads during build or launch.
- Update the Windows offline smoke checks, source tests, translated READMEs/setup
  instructions, and troubleshooting. Keep existing optional Diffusers worker working.
- Document trusted custom-node requirements, retained server image files, software
  and model licenses, known hardware profiles, and how to return to Local Diffusers.
- Run real frozen Windows generation against a separately installed Comfy runtime,
  then have the user test a representative manual comic-editing session.

**Gate:** no regression of current backends; original projects open without Comfy;
new patches survive export; cancellation does not affect another client's work;
no application/runtime/model updates or public release occur without approval.

## 11. Required automated coverage

Proposed tests: `testcomfy_config.py`, `testcomfy_workflows.py`,
`testcomfy_client.py`, `testcomfy_runtime.py`, plus extensions to existing
`testartistic_edit_controller.py`, `testartistic_edit_image_ops.py`,
`testartistic_edit_undo.py`, `testartistic_edit_project_state.py`,
`testexternal_process.py`, `testui_translations.py`, and `testpackage_smoke.py`.

- Configuration migration/defaults; unavailable checkpoints retain selection;
  loopback validation including IPv6; proxy/redirect rejection; missing runtime.
- Workflow shape/schema and node allowlist; filenames/subdirectories; Unicode and
  quotes/newlines in prompts; seed boundaries; no prompt injection into graph JSON;
  unknown precision/loader/base-model combinations fail preflight.
- Upload collisions, malformed response, node validation errors, duplicate events,
  cached executions, other-client messages, ambiguous POST acceptance, network loss,
  reconnect to completed history, missing history, and bounded result downloads.
- Final-output-node selection; traversal/absolute-path descriptors; malformed image,
  decompression/pixel limit, wrong dimensions, truncated output, atomic output save.
- Cancel during upload, queueing, loading, sampling, and download; race with job
  completion/next foreign job; legacy cancel fallback; exactly one terminal event;
  revoked result acceptance still allows the owning job's final cleanup acknowledgement.
- Owned/attached server distinctions, occupied-port/startup races, failed imports,
  process crash, environment restoration, shutdown timeout, isolated database/user
  directories without legacy migration (including original DB/WAL files), duplicate
  custom-node basenames, foreign-work close refusal, and per-instance cleanup timing.
- Backend switch followed by late Diffusers/Comfy error/progress/result; no job can
  clear another backend's session or write to a newly opened page/project.
- Tiny/odd-sized, tall/wide, edge-of-page and resized crops; edit margin versus
  context; feather support; all pixels outside allowed support unchanged.
- Apply is one undoable operation; painted mask behavior remains correct; existing
  text overlays remain editable; mixed inpaint/Diffusers/Comfy patches round-trip;
  legacy projects and unknown metadata still work without a running backend.
- LoRA strength zero, enabled, changed, removed, switching model/profile, and invalid
  adapters cannot reuse a stale patched model. Real efficacy is an integration test.
- All supported UI catalogs/themes; missing-Comfy editor startup; frozen builds with
  no optional image runtime installed. Existing BFL consent/key handling unchanged.

Automated tests must default to offline fixtures. Real GPU tests are opt-in and
must not stop unrelated services, download gated weights, overwrite projects, or
spend API credit. Plan creation itself does not run these tests.

## 12. Future-model extension, release scope, and references

The backend should not hardcode model family into network transport, preview, or
process management. Adding Qwen later would require a separate profile/adapter,
model-specific conditioning/sizing, license review, and quality/memory tests, but
should reuse connection, job lifecycle, selection, preview, Apply, and Undo.

Qwen Image 2.1 already has Comfy-Org model files/workflows, including quantized
options. Its 7B visual component is accompanied by an 8B Qwen3-VL encoder; this
does not prove it needs less total memory than either Klein variant. Future mask
conditioning or RGBA-layer editing would need explicit app contracts/composition
work; they are not automatically enabled by connecting ComfyUI.

Comfy integration does not resolve the separate NVIDIA redistribution review for
the app's ONNX detection/OCR/inpainting bundle. Keep that release gate independent.
Do not bundle model weights or imply their licenses are inherited from the app.

Primary references (reviewed 2026-09-26; pin revisions during C0):

1. [ComfyUI local server routes](https://docs.comfy.org/development/comfyui-server/comms_routes)
2. [ComfyUI WebSocket messages](https://docs.comfy.org/development/comfyui-server/comms_messages)
3. [Server implementation: jobs/cancel, queue, upload, free](https://github.com/Comfy-Org/ComfyUI/blob/master/server.py)
4. [Official WebSocket API example](https://github.com/Comfy-Org/ComfyUI/blob/master/script_examples/websockets_api_example.py)
5. [Windows portable installation](https://docs.comfy.org/installation/comfyui_portable_windows)
6. [ComfyUI command-line flags](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/cli_args.py)
7. [Official Klein workflows and model requirements](https://docs.comfy.org/tutorials/flux/flux-2-klein)
8. [4B distilled edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_flux2_klein_image_edit_4b_distilled.json)
9. [9B distilled edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_flux2_klein_image_edit_9b_distilled.json)
10. [Native LoRA node implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py)
11. [ComfyUI-GGUF, including experimental LoRA caveat](https://github.com/city96/ComfyUI-GGUF)
12. [Example prequantized Klein GGUF files](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/tree/main)
13. [Native INT8 support change](https://github.com/Comfy-Org/ComfyUI/pull/14636)
14. [Example Klein INT8/ConvRot candidate, not yet qualified here](https://huggingface.co/wraps/FLUX.2-klein-4B-INT8-ConvRot-ComfyUI)
15. [Comfy-Org Qwen Image 2.1 files/workflows](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
16. [Qwen Image 2.1 announcement](https://qwen.ai/blog?id=qwen-image-2.1)

Recommended next action after plan approval: **C0 only**, followed by its evidence
review. Then implement C1/C2 for an initial user test before managed startup and
low-memory profile expansion. Do not treat all milestones as already authorized
to change the user's external runtime or publish a release.
