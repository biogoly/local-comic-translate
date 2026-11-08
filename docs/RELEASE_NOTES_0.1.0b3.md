# Local Comic Translate 0.1.0b3

Beta 3 brings fitting, review and artwork repair into the manual page workflow.

On Windows x64, download **LCT-0.1.0b3-Windows-setup.zip** from the [beta 3 release](https://github.com/biogoly/local-comic-translate/releases/tag/v0.1.0b3). Extract the whole ZIP into a short, writable folder, then follow the [Windows setup instructions](../packaging/windows/SOURCE_SETUP.md). This download contains the editor source and a launcher. On first use, the launcher installs the pinned dependencies into a local environment. Allow time and disk space for the Qt, ONNX and NVIDIA runtime downloads. Open a copy of a project to try the new features.

- **Fit and review lettering.** Fit Text previews wrapping, size and placement for an editable text layer. Bubble interior uses the original artwork to find room for lettering, including when a detector box clips a narrow balloon tail. The Review navigator identifies missing or overflowing text, supports selected-region OCR retry, and tracks approved or protected work. [Fit Text](TEXT_FIT_M1B.md) · [Automatic fitting](AUTOMATIC_BUBBLE_FITTING.md) · [Review](REVIEW_NAVIGATOR.md)
- **Keep translations consistent.** Project context supports a glossary, character notes, approved translation examples and reviewed suggestions. Vision-capable models can receive text-box coordinates, with position hints for identical wording in different panels. [Project context](TRANSLATION_CONTEXT.md) · [Text-box coordinates](SPATIAL_TRANSLATION_CONTEXT.md)
- **Edit lettering on the canvas.** Save and share lettering presets, apply perspective or curves, and match tilted source text while retaining manual rotation. [Lettering styles](LETTERING_STYLES.md) · [Text shapes and rotation](LETTERING_TRANSFORMS.md)
- **Repair colored backgrounds.** Gradient accepts clicks and Shift-drag sample lines. Protect artwork beside the mask, or compare Clone and Heal on textured areas. Original-image sampling reads the source page even when earlier edits are in memory. [Gradient](GRADIENT_REPAIR.md) · [Protected repairs](PROTECTED_REPAIRS.md)
- **Compare Artistic Edit results.** Keep two generated versions, check their lettering and apply either one as an undoable edit. [Compare previews](ARTISTIC_EDIT_PREVIEWS.md) · [Check lettering](LETTERING_CHECK.md)

Local LLM translation connects to your llama.cpp installation. Artistic Edit supports the separate Diffusers runtime, local ComfyUI, and the configured provider API. Model files are obtained separately. [ComfyUI setup](COMFYUI_BACKEND.md)
