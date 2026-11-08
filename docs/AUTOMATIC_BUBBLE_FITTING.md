# Optional bubble fitting during rendering

Implemented November 3, 2026, as an extension of the existing [Fit Text preview](TEXT_FIT_M1B.md).

## Enable it

Open **Settings → Text Rendering** and enable **Fit to bubble when rendering**. It is off by default. **Bubble padding** starts at 8 page pixels; minimum and maximum font sizes use the existing controls on the same settings page. These are application preferences, saved between sessions.

With this option enabled:

- Manual **Render**, including selected pages, fits newly created lettering to estimated speech-bubble interiors.
- A selected-region translation that creates a new text item uses the same fitting logic.
- Automatic translation fits new lettering, including webtoon pages. Existing items are retained by region identity while other regions continue through processing.
- Existing lettering stays as it is. Repeating manual Render adds missing items without replacing or refitting moved, resized or individually formatted items.

**Existing lettering is preserved by region identity.** The [Review navigator](REVIEW_NAVIGATOR.md) adds independent source, translation, layout, and automatic-cleanup protections. Existing rendered items stay in place with automatic fitting enabled, while other regions can proceed through processing. OCR and target wording can still change unless their own protections are enabled; review retained lettering when its target changes. Use manual Render to fill missing regions, or the selected-item Fit Text preview to adjust existing lettering. Pages containing orphan lettering without saved region ownership still receive a conservative whole-page skip and an informational batch report entry. With automatic fitting disabled, explicit **Keep layout** controls still apply.

Turning the setting on, opening a project or switching pages does not reflow existing text.

## Fitting and fallback

The estimator reads original page pixels, never accumulated cleanup patches. It uses the source text/bubble bounds separately from the new lettering's position and width. The result remains editable text with ordinary undo/redo, project persistence and export.

Wording, Unicode, intentional line breaks and character formatting are retained. Legacy wrapping inserted only for the old rectangular renderer is not treated as intentional line breaks. A candidate must fit inside the padded boundary and respect the minimum readable font size. Balanced is preferred, with Larger available when necessary to keep smaller inline spans readable.

Uncertain contours, missing or mismatched originals, unsupported vertical/rotated/scaled text, page-spanning regions or impossible fits retain the existing rectangular rendering. No region is excluded from OCR or translation because fitting failed. Fitting does not change source pixels, cleanup masks or Artistic Edit masks.

The same page-local text properties are used for saved and live webtoon results, including horizontal centering of narrower pages. Existing unique region IDs identify lettering even after it has been moved. Legacy items without IDs retain conservative position matching.

## Initial fitting validation (before M2)

- 571 automated tests passed with real Windows fonts registered in the offscreen Qt runtime.
- New coverage includes original-pixel immutability, rich text and Unicode, minimum readability, Box fallback, manual and selected-page Render, repeated Render, undo/redo, webtoon centering and seam fallback, existing-page preservation before cleanup, pending-edit snapshots before worker startup, and settings persistence/defaults.
- All nine non-English interface catalogs pass source/runtime/compiled checks, with 1,014 active/runtime messages per locale.
- Source startup smoke passed all 18 checks, including all six themes, using an isolated profile.

These checks use synthetic images and deterministic pipeline fixtures. No live OCR/LLM inference or new Windows package build was performed for this delivery. Real-comic workflow testing remains the next user checkpoint.

See [Review navigator](REVIEW_NAVIGATOR.md) for the later per-region protection behavior and validation.
