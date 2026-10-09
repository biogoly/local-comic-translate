# Local Comic Translate 0.1.0b2

Windows x64 beta refresh, October 9, 2026. This update keeps the existing manual
workflow, local Diffusers worker, and direct provider APIs available.

## Changes

- Optional Local ComfyUI artistic editing: connect to an existing local server
  or launch an existing Windows portable installation in the background. Model
  files and ComfyUI are not installed automatically.
- Native INT8 Klein 9B support and experimental Qwen Image 2.1 editing, including
  validated model-only Qwen LoRAs, strength controls, and separate model-family
  settings. Preview, Apply, Undo/Redo, and saved project patches remain supported.
- Explicit Portuguese (Portugal), Portuguese (Brazil), and Vietnamese source
  choices. Portuguese retains the existing Latin OCR; Vietnamese uses a separate
  local recognition model downloaded on demand (about 60 MB).
- Portuguese dialect instructions for local/cloud LLMs and corrected language
  mappings for supported translation services. Existing saved language IDs and
  source/target selection persistence are retained.
- Updated GUI translations in all nine existing non-English catalogs, refreshed
  documentation, splash artwork, and frozen-runtime integration safeguards.
- Repeatable Windows x64 packaging with a dedicated dependency lock, isolated
  offline acceptance checks, source revision metadata, and SHA-256 checksums.

See [ComfyUI setup](COMFYUI_BACKEND.md), [runtime validation](COMFYUI_RUNTIME_VALIDATION.md),
and [language support](LANGUAGE_SUPPORT.md) for details.

## Testing and limitations

- Select **Vietnamese** explicitly with **Default** OCR. Auto still uses the
  existing script-based routing and does not select the Vietnamese recognizer.
  Vietnamese comic-page accuracy remains experimental; current OCR evidence is
  from synthetic printed-text images, not a comic benchmark.
- Klein 4B mappings, GGUF/custom-node combinations, and lower-memory hardware
  still need qualification. Successful INT8 tests do not establish a minimum
  VRAM requirement for every resolution, model, or LoRA.
- Generated artistic edits should be reviewed before applying; spelling and
  artwork preservation are not guaranteed. External API usage is billed by the
  selected provider.
- This is an unsigned portable build. Extract it to a **new folder**, close any
  other running copy, and back up important projects before testing. Source and
  packaged copies of this fork share the fork's existing settings/model cache;
  they are not isolated profiles. The upstream app uses a separate profile.
- Python, Qt, and ONNX runtime libraries are bundled; model weights, private
  settings, API keys, llama.cpp, ComfyUI, and the optional Diffusers environment
  are not. Keep your existing runtime installations.

## Distribution status

This beta is prepared for local testing. The current native-library redistribution
review has unresolved items; source publication and a passing build do not clear
public binary distribution. See the [release review](../packaging/windows/RELEASE_REVIEW.md).
Build-specific test results and source commit are recorded alongside the ZIP.
