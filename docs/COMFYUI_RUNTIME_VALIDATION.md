# ComfyUI backend validation

Initial validation date: 2026-09-27; Qwen extension tested 2026-10-03 below.
This records observed tests, not universal hardware
requirements or a claim that every checkpoint with a similar filename works.

## Initial real workflow qualification (C0)

Existing Windows portable installation, unchanged dependencies and weights:

| Component | Tested value |
| --- | --- |
| ComfyUI | 0.37.0, commit `93810483a4739a1588236919a3128d3070244146` |
| Embedded Python | 3.13.14 |
| PyTorch | 2.13.0+cu130 |
| comfy-aimdo / comfy-kitchen | 0.5.5 / 0.2.35 |
| Official workflow-template package | `comfyui_workflow_templates_json` 0.1.95 |
| Diffusion checkpoint | `flux2Klein9BINT8_9b.safetensors`, 10,093,675,518 bytes |
| Text encoder | `qwen_3_8b_fp8mixed.safetensors` |
| VAE | `flux2-vae.safetensors` |
| LoRA | `Flux2_colorManga.safetensors`, Klein 9B metadata, rank 32, strength 0.7 |
| Hardware | One RTX 3090 Ti 24 GB (physical GPU 1), approximately 128 GB host RAM |

The checkpoint header contains integer quantized tensors and Comfy quantization
metadata. It was not loaded as BF16 and then quantized by this application. The
filename/header does not establish a specific quantization algorithm or training
provenance. The host's second GPU was not pooled with the test GPU; GPU 0's
existing workload was left alone.

The test used generated, non-copyrighted cream-panel fixtures with the word
`HOLA`, and the instruction to replace it with `HELLO`. The output was visually
checked for that replacement. No user comic pages were changed. Four steps,
Euler, CFG 1.0, reference-latent conditioning, one designated SaveImage output.

| Case | Observed result |
| --- | --- |
| 512 x 384 edit | Valid PNG, exact requested dimensions, readable HELLO |
| 272 x 400 edit | Exact dimensions; no hidden one-megapixel resize |
| LoRA enabled | Successful output, 112 model patches reported by ComfyUI |
| LoRA removed next job | Successful output, zero model patches reported |
| Cancel queued job while another runs | Only queued job removed; running job preserved |
| Cancel running job | Targeted endpoint acknowledged; history recorded interruption |
| Managed startup isolation | Readable original profile/database/config files unchanged |

One pre-existing locked profile file could not be read for hashing, so its
contents were **not** verified. Both test-owned server processes were stopped
after their tests. There were no automatic downloads/updates. A missing
`model.lm_head` warning was logged by the encoder; inference nevertheless
completed. No skipped LoRA-key warnings were observed.

Reference elapsed times including HTTP polling/transfer were 9.03 seconds for the
first 512 x 384 edit in one process and 1.02 seconds for its warm 272 x 400 edit.
A second process took 5.03 seconds for its first 512 x 384 edit, 1.02 seconds for
the smaller warm edit, 3.02 seconds with LoRA, and 2.02 seconds after removal.
OS caches were not flushed; these are **not cold-storage benchmarks**, nor an
apples-to-apples speed comparison with Diffusers. Peak RAM/VRAM was not measured
by these initial probes. They do not prove 8/12 GB or low-RAM compatibility.

Graph source-template SHA-256 references are preserved in
`modules/artistic_edit/comfy_workflows.py`. The production graph intentionally
removes implicit reference-image rescaling and uses the app-prepared dimensions.

## Production integration checks

The implemented `ComfyRuntime`, `generate_comfy` client, Qt job, and editor
controller were then exercised against the same installed models, using separate
test settings and generated images. Managed startup, live catalog discovery,
upload, single submission, history polling, designated-output retrieval, and
exact-size validation passed for 512 x 384 and 272 x 400 inputs. LoRA-on and
LoRA-off generations passed through the production client. Targeted cancellation
also passed through the real Qt job; the managed observer was released, and the
same persistent server successfully processed the next controller edit.

The actual controller captured the page, ran its production background job,
delivered the preview on the GUI thread, and applied one patch. Undo removed
that patch and Redo restored it. This was an offscreen automated controller
test, not a substitute for manually checking the on-screen editing experience.
The owned server stopped and the readable original profile/config hashes
remained unchanged. The new optional-backend package smoke check also passed
from source with network access and child-process startup forbidden.

The regression suite now passes 371 tests (302 before this integration), including
new client, workflow, configuration, runtime, and controller tests. All nine Qt
catalogs are compiled and checked for active-message coverage and placeholders.
The Windows x64 frozen build also passed its offline smoke checks, including
optional ComfyUI construction, the external Diffusers worker source payload, the
main window, and all six themes. No external server is started by that offline
package check. Real inference above was exercised from source; a frozen-app
live-runtime test remains separate. No package was published.

Integration testing caught and fixed a real schema difference: ComfyUI 0.37.0's
Euler sampler exposes a V3 `COMBO` schema, while some other nodes use the legacy
list schema. Catalog and enum validation now handle both forms explicitly.

## Limits of qualification

Only the above native 9B INT8 combination and LoRA were inference-tested in the
initial qualification. Matching 4B weights, native BF16/FP8 diffusion weights,
GGUF/custom-node revisions, other adapters, lower-memory hardware, and macOS
require separate tests. A saved model profile is a user's selection, not a
compatibility certification. Future model families require a reviewed workflow
and their own acceptance tests.

## Experimental Qwen Image 2.1: 2026-10-03

Tested against the user's existing installation and weights, without downloads,
dependency changes, or modifying the ComfyUI installation/profile:

| Component | Tested value |
| --- | --- |
| ComfyUI | 0.38.0, commit `b65d1ffaca35fcda5889e120dcd0c8520553c7b2` |
| PyTorch | 2.13.0+cu130 |
| GPU | Physical GPU 1, RTX 3090 Ti, 24 GB; GPU 0 left alone |
| Diffusion | `qwen_image_2.1_int8_convrot.safetensors` |
| Encoder | `qwen3vl_8b_int8_convrot.safetensors` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |
| Sampling | 25 steps, CFG 1.0, Euler / Simple, seed 12345 |
| Official template | `image_qwen_image_2_1_image_edit.json`, template package 0.1.99 |
| Template SHA-256 | `c1e6551759de131f083a040da072ac6f032c3c8401ceca202412cd912897d0fc` |

The production runtime/client edited a synthetic cream-colored panel from
"HOLA" to "HELLO". Both 512 x 384 and 272 x 400 RGB outputs matched their
requested dimensions and were visually inspected. The latter exercised
padding to 288 x 416 and cropping back without rescaling. A diagnostic-only
extra SaveImage node retained the decoded RGBA for inspection; the subsequent
controller test used the unchanged production graph.

The narrow crop exposed transparent output with purple hidden RGB values.
Discarding alpha would reveal those colors. The final graph instead composites
decoded RGB over the original input using generated alpha, before applying the
editor's normal mask. Raw diagnostic alpha ranged from 0 to 255 on that crop;
the corrected final RGB image retained the original beige background.

The final live run verified targeted cancellation of a running 100-step Qwen
job (`server_may_continue=False`, server logged interruption), followed by a
successful generation on the same persistent server. ComfyUI 0.38.0's atomic
job-specific cancellation route and caller-supplied UUID behavior were inspected
and added to the explicit qualified-version list alongside 0.37.0. Unknown
versions retain the conservative pending-job-only fallback.

The real controller captured a page, generated through its Qt job, delivered
the preview on the GUI thread, and applied a neutral `artistic_edit` patch.
Undo removed it; Redo restored it. Automated persistence tests also verified
project reload and mixed Qwen/legacy-Klein reverts without removing inpainting.
This is offscreen controller validation, not a full manual comic-page trial.

The managed server stopped afterward. Readable original ComfyUI profile/config
hashes were unchanged. Local evidence is retained in the task workspace under
`work/comfy-c0/qwen-c6d8838fdfd14032b7b29f4e5edc368a/report.json` and its input,
output, and diagnostic folders; no private user paths or images are bundled here.

For these small fixtures only, managed startup took 7.33 s and client-observed
generation took 7.23 s for the first crop and 2.62 s for the second, warm crop.
One-second NVIDIA samples observed physical GPU 1 total usage between 2,791 and
19,734 MiB, including existing non-test use. These are not isolated model-memory
measurements, minimum hardware requirements, or full-page speed claims.

All 404 regression tests pass. Coverage includes model-specific settings/defaults, separate sampling
values across backends, frozen regeneration, pre-upload node validation, RGB/RGBA
composition, cancellation/recovery, undo, and persistence. All nine UI catalogs
cover 963 active/runtime messages with matching compiled resources. The source
offline application smoke test passes, including the main window and all six
themes; no new frozen Windows package was built or published for this addition.

At the time of the October 3 test, unqualified configurations included Qwen
LoRAs (tested separately below), other Qwen weight formats/variants,
8/12 GB GPUs, low-system-RAM machines, large-page quality/performance, macOS,
and arbitrary workflows. Exact lettering and surrounding artwork still require
human inspection before Apply.

## Qwen Image 2.1 LoRAs: 2026-10-06

Used the same existing INT8 Qwen 2.1 diffusion/encoder and BF16 VAE, on physical
GPU 1 (RTX 3090 Ti, 24 GB). The installed ComfyUI was 0.38.0, commit
`5c460d8172fe30761ff67c0df3d5643bb74e0d70`; no installation, dependency, model,
or original-profile changes were made. All five installed Qwen 2.1 adapters
passed bounded header-only compatibility checks. Two were inference-tested:

| Adapter | Layout | Live test settings |
| --- | --- | --- |
| `Qwen_VNCCS_QI2_PoseStudioV1.1.safetensors` | Generic fused model-only, rank 32 | 25 steps / CFG 1; strengths 0, 0.4, 0.8 |
| `Qwen2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors` | Split Qwen model-only, rank 256 | 6 steps / CFG 1; strength 1 |

The production graph routes UNETLoader through LoraLoaderModelOnly **before**
QwenImage21Cache. No text-encoder weights are patched. ComfyUI reported 192
model weight targets patched for the regular adapter and 195 for Turbo, versus
zero with the adapter removed; no unloaded-key or shape-mismatch warnings were
found in the retained output. The application still uses the exact prepared
crop and original-input alpha composition established above.

A synthetic 512 x 384 cream panel was edited from "HOLA" to "HELLO", with
seed 12345 held constant. The regular adapter at strength 0.4 produced different
pixels from baseline, and 0.8 differed from 0.4. Strength zero matched baseline
exactly. After removing each adapter on the same persistent server, the
25-step output matched baseline pixel-for-pixel, showing no residual adapter
effect. Both regular and Turbo outputs were visually inspected; the lettering
was correct. This is a load/effect/switching test, not evidence that a pose or
acceleration adapter improves artistic lettering on real comics.

The real settings-validation background job verified architecture and SHA-256.
The actual controller then generated with the regular adapter, showed its
preview on the GUI thread, applied an `artistic_edit` patch with adapter name,
strength and checksum, and passed Undo/Redo. Fixture tests additionally cover
changing settings during a same-seed preview: regeneration and Apply retain
the originally submitted adapter. Family switching restores separate in-dialog
choices rather than passing Klein adapters to Qwen. Existing Klein validation
is delegated unchanged.

Client-observed fixture times were 9.41 s for initial baseline, 4.14/4.20 s for
the regular adapter at 0.4/0.8, and 3.12 s for the six-step Turbo run. Different
step counts and cache states make these unsuitable as a quality/speed benchmark.
The isolated managed server stopped afterward; readable original ComfyUI
profile/config hashes were unchanged. Evidence is retained in the local task
workspace at `work/comfy-c0/qwen-lora-07bc3b475c02464eb77493b5c9a3bab1/report.json`.

Compatibility checks deliberately reject unknown layouts, attention-only
ambiguous adapters, text-encoder/DoRA payloads, malformed/incomplete pairs,
cross-family dimensions, and overlapping patches. Standard server catalogs
do not attest file hashes: users must choose the corresponding local file for
verification. No LoRA or model weights are bundled, downloaded, or published.
Large-page quality, other adapters, and lower-memory hardware need further
testing. Qwen remains an optional experimental alternative, not a replacement
for Klein.

All 430 regression tests pass. All nine UI catalogs cover 964 active/runtime
messages with matching compiled resources. The source offline application smoke
test passes, including the main window and all six themes. No new frozen Windows
package was built or published for this addition.
