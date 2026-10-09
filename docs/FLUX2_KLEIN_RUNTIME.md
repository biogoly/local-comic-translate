# FLUX.2 Klein Runtime Validation (Milestone 0)

Date: 2026-09-08 (local). Hardware: 2x NVIDIA GeForce RTX 3090 Ti (24 GB), 125 GB system RAM, Windows.
Sole durable intent: validate that the pinned FLUX.2 Klein runtime from `docs/FLUX2_KLEIN_ARTISTIC_EDIT_PLAN.md` runs image-editing inference on this machine, record measured characteristics, and replace proposed pins with validated pins (or record why a pin cannot be validated). No application `src/`, `app/`, or `.venv` changes were made.

## Environment

- Isolated spike venv: `C:\AI\flux2-batch-edit\.venv-flux-spike` (created with `uv`, Python 3.12.10). The application `.venv` was **not** touched.
- Weights: `black-forest-labs/FLUX.2-klein-4B` and `FLUX.2-klein-9B`, already present in the local Hugging Face cache; all loads used `local_files_only=True`. No gated weights were downloaded.
- All GPU runs pinned to physical **GPU 1** (`CUDA_VISIBLE_DEVICES=1`) because llama-server permanently occupies GPU 0 and must not be stopped.
- Diagnostic harness (untracked, outside the repo): `C:\AI\flux2-batch-edit\spike_m0\flux_spike.py` (+ `inspect_lora_headers.py`, `make_synth_lora_4b.py`), JSON results in `spike_m0/*.json`.
- Protocol: `guidance_scale=1.0`, 4 steps, bf16, synthetic comic-like crops, fixed seeds — matching the `C:\AI\flux2-batch-edit` prototype.

## Validated versions

| Package | Pin (plan §2) | Status |
| --- | --- | --- |
| torch | `2.11.0+cu128` (CUDA build 12.8) | **validated** — imports and runs inference |
| diffusers | git `b8aebf4c1261c4b887778e0882e56ed19e686518` (0.38.0.dev0) | **validated** — `Flux2KleinPipeline` present; commit verified in `direct_url.json` |
| transformers | `5.5.4` | **validated** |
| accelerate | `1.13.0` | **validated** |
| peft | `0.20.0` | **validated** |
| safetensors | `>=0.8.0rc0` | **validated** (`0.8.0-rc.0` installed) |
| huggingface_hub | `1.11.0` (via diffusers pin) | present, works with cached snapshots |

Baseline app test suite before any work: **55 tests, OK** (`unittest discover -s tests -p "test*.py"`, `QT_QPA_PLATFORM=offscreen`) at `c0b9088`. Git status preserved: only the user's pre-existing unstaged `app/controllers/*.py` edits and untracked files; nothing committed by this milestone.

## API deviations found vs plan

1. `Flux2LoraLoaderMixin` is **not** importable from the top-level `diffusers` namespace (`ImportError`). Correct path: `from diffusers.loaders import Flux2LoraLoaderMixin`. It is already in `Flux2KleinPipeline.__mro__`, so `pipe.load_lora_weights(...)` / `pipe.set_adapters(...)` / `pipe.unload_lora_weights()` work directly on the pipeline.
2. `pipeline.release_model_caches()` does **not** exist on `Flux2KleinPipeline`. The plan's cleanup step must use `del pipeline` + `gc.collect()` + `torch.cuda.empty_cache()` (optionally `pipe.remove_all_hooks()` when offload hooks are installed).
3. `image=` in `Flux2KleinPipeline.__call__` requires a **PIL `Image.Image`**; a `str`/`Path` raises `ValueError: Image must be a PIL.Image.Image` from `image_processor.check_image_input`. Milestone 1 loaders must open + `convert("RGB")` themselves (the prototype already does).
4. There is **no inpainting-mask argument**; artistic edits are whole-image with a text instruction. Padding-to-multiple-of-16 + crop-back remains required.

## Measured runtime characteristics

Protocol per generation: 4 steps, guidance 1.0, bf16, `local_files_only=True`. "Peak" = `torch.cuda.max_memory_allocated` during the generation (GPU 1). "Cold" = first generation after pipeline construction; "warm" = subsequent.

| Config (GPU 1) | Pipeline load | Cold gen | Warm gen | Peak VRAM (torch) | Output |
| --- | --- | --- | --- | --- | --- |
| 4B **sequential** CPU offload, 512x512 | 0.9–1.1 s | 15.4 s | **9.2–9.6 s** | **992 MiB** | 512x512 OK |
| 4B **model** CPU offload, 512x512 | 0.4–0.8 s | 10.9 s | **5.3 s** | **8.3 GB** | 512x512 OK |
| 4B seq offload + LoRA, 384x384 | ~1 s (+0.3 s LoRA) | 21.9 s | — | 991 MiB | 384x384 OK |
| 9B model offload + real LoRA, 384x384 | 0.6 s (+0.14 s LoRA) | 81 s (first run 651 s, page-cache cold) | — | **18.1 GB** | 384x384 OK |
| 4B full CPU, 256x256 | 0.7 s | 14.6 s | — | n/a | 256x256 OK |

Interpretation:
- **4B + sequential offload is the sweet spot on this host**: < 1 GB VRAM, ~9–16 s/edit warm — coexists easily with llama-server holding ~17–23 GB on the same GPU.
- 4B model offload is faster (5–11 s) but needs an ~8.3 GB contiguous VRAM window; on GPU 1 this works only because Windows WDDM overcommits (see below).
- 9B model offload **works** (validated end-to-end with a real trained LoRA) but is slow (81 s warm at 384x384, far worse cold) and its 18.1 GB peak only materializes via WDDM shared-memory spillover; it functioned but is not recommended for interactive use alongside llama-server.
- **Load-time caveat**: the ~1 s pipeline load times reflect a warm 125 GB page cache (weights mmapped). Cold load after reboot/RAM eviction will be disk-bound (tens of GB); keep a long-lived pipeline instance between jobs.
- Windows WDDM: `torch.cuda.mem_get_info()` over-reports free VRAM (~23.3 GB/GPU even when `nvidia-smi` shows ~1 GB free on GPU 0). `nvidia-smi` is authoritative here; never size offload policies from `mem_get_info`.

## Callback & cancellation

- `callback_on_step_end(pipeline, step_index, timestep, callback_kwargs)` fires **every step** (observed steps `0,1,2,3` for 4 steps). `callback_kwargs` contained exactly one key: `latents`. Returning `callback_kwargs` continues normally.
- **Cancellation: raising an exception from `callback_on_step_end` cleanly aborts the current generation** (verified: aborted at step 2 of 4). The pipeline instance remains fully usable — an immediate regeneration with the same pipeline succeeded in 9.94 s. This is the recommended GUI cancel mechanism for Milestone 1.

## LoRA findings

- All ~20 FLUX.2-Klein-tagged LoRAs in the local ComfyUI `models\loras` are **9B-shaped** (inner dim 4096; ff 12288/24576). Two files named `FLUX2_*` (`FLUX2_RaveGothGFLadyMk1`, `FLUX2_Scene_girls`) are actually FLUX.1-dev LoRAs (3072 width, 19 double / 38 single blocks).
- **User hypothesis tested ("most FLUX.2 9B LoRAs work on 4B regardless of what BFL says"): rejected on this runtime.** Loading `Flux2_uncensorMangaKlein_V2.safetensors` (rank-32, 9B shapes) into the 4B transformer via `pipe.load_lora_weights` fails with peft strict `size mismatch` on every adapted projection (checkpoint `4096/12288/24576/36864/16384` vs 4B model `3072/9216/18432/27648/12288`). ComfyUI tolerates this by silently **skipping** mismatched tensors (partial, degraded application — likely the source of the "it works" folk belief); diffusers/peft refuses outright. BFL's statement holds for the diffusers path.
- After a failed LoRA load the base pipeline remains fully functional (4B base generation succeeded, 14.55 s).
- No trained local 4B-shape LoRA exists on this machine (searched all drives). Mechanism validation on 4B therefore used a **synthetic zero-delta rank-8 LoRA** with correct 4B shapes (`make_synth_lora_4b.py`): `load_lora_weights` + `set_adapters` + generation all succeeded (peak 991 MiB). Positive 4B test with a *trained* community Klein-4B LoRA was **explicitly waived by the user** (2026-09-08) — no ungated download performed; it remains the single open M0 item and Milestone 1 should re-test it whenever such an adapter becomes locally available.
- Real trained LoRA (`Flux2_uncensorMangaKlein_V2`) on the **9B** model: `load_lora_weights` + `set_adapters` + generation **validated** (see 9B row above).

## Multi-GPU

`device_map="balanced"` across both GPUs (plan §5, `max_memory={0: "21GiB", 1: "21GiB"}`) **cannot be validated on this host right now**: llama-server permanently holds GPU 0 (~23.2 GB used, ~1 GB free per `nvidia-smi`) and cannot be stopped. All successful runs above used GPU 1 only. Recorded here as the plan's sanctioned "pin/config not validatable on this hardware" outcome; defer to a maintenance window. The measured 4B configs make single-GPU + CPU offload a viable Milestone 1 design anyway.

## Milestone 1 recommendations

1. Pin the exact validated versions above (diffusers via git commit) in a **separate optional extra** (e.g. `.[flux2]`) or dedicated venv — transformers 5.x / torch 2.11 are too invasive for the app venv's current stack.
2. Always pass PIL RGB images to `image=`; keep pad-to-16 + crop-back; write `.tmp.png` + verify + atomic replace (prototype pattern; harness size assertions confirmed correct output dimensions).
3. Call LoRA methods directly on the pipeline (`load_lora_weights`/`set_adapters`/`unload_lora_weights`); import the mixin from `diffusers.loaders` only if type references are needed.
4. Implement LoRA pre-flight: read safetensors headers and reject adapters whose inner dim doesn't match the model (3072 vs 4096) with a clear user-facing error — never rely on ComfyUI-style partial loading.
5. Cancellation: raise from `callback_on_step_end`; the pipeline demonstrably recovers for the next job.
6. Default runtime target: **FLUX.2 Klein 4B, model CPU offload**. This matches the standalone batch editor and also lets a 24 GB GPU load the BF16 9B model as an explicit slow-path option. Sequential CPU offload remains available for lower-VRAM systems.
7. Keep a persistent pipeline worker process; reload cost is page-cache-dependent.

## Milestone 3 worker validation

Date: 2026-09-09. The isolated JSON-lines worker was exercised end to end with
the cached 4B model, physical GPU 1, BF16, sequential CPU offload, four steps,
guidance 1.0, and a 256x256 RGB PNG. The worker emitted one `loading` event,
four ordered `progress` events, and one verified `result` event. Generation and
PNG verification took 20.70 seconds after pipeline construction; the output
was RGB and exactly 256x256. No model files were downloaded and the running
llama.cpp service was not stopped or modified.

The old `.venv-flux-spike` launcher still points to the removed system Python
recorded above. For this smoke test, the bundled Codex Python 3.12 interpreter
loaded the already-installed spike packages through `PYTHONPATH`. This proves
the worker implementation and validated package set, but it is not the final
user configuration. Before the Milestone 4 UI is exercised, create or repair a
dedicated FLUX virtual environment and select its working Python executable.

Header-only compatibility checks also accepted the synthetic Klein 4B adapter
and the trained local Klein 9B adapter, while automated tests reject cross-size,
ambiguous, malformed, and FLUX.1-shaped adapters before Diffusers loading.
