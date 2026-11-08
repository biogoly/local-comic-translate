# Text fitting: first editor-roadmap delivery

Implemented in the working tree on November 2, 2026. This is the M0 identity foundation and M1a rectangular text-fitting feature. The representative-page editing-time evaluation remains pending. The subsequent optional bubble-fitting mode is documented in [TEXT_FIT_M1B.md](TEXT_FIT_M1B.md); the review navigator remains a later milestone.

## Try it

1. Restart the app from this checkout and open a page or a copy of an existing project.
2. Select one rendered text item. Click **Fit Text** beneath the formatting controls, or use its context menu while not editing characters.
3. Compare **Current**, **Balanced**, and **Larger text**. Adjust the fitting-area width/height and font-size limits if needed. The dashed boundary shows the chosen rectangular area.
4. Leave **Keep manual line breaks** checked to preserve intentional breaks. Unchecking it permits rewrapping those breaks; it does not rewrite the wording.
5. Click **Apply**. This creates one undo step. **Cancel** leaves the item unchanged. Ordinary resizing and selected-word formatting remain available afterward.

The selected-item controls also display overflow and readability warnings. An impossible fit disables Apply; enlarge the area or shorten the text yourself. A readability warning is separate from geometric overflow, and considers smaller inline text as well as the base font size.

The fitting area starts from the linked OCR rectangle, or the current item if no unique linkage exists. Applied preferences belong to that text item. The area uses the item's local dimensions and follows its rotation/scale; canvas zoom does not change the result. Changing it does not change the OCR region, cleanup mask, or Artistic Edit mask. Fitting never runs automatically while typing, dragging, switching pages, or opening a project.

## Preserved state and boundaries

- Rich-text spans, colors, explicit line breaks by default, outlines, vertical layout, position, rotation, and scale are retained. Font sizes and soft wrapping change only after Apply.
- Page and region UUIDs persist through project save/load and Save As. Undo snapshots preserve IDs; explicit duplicates receive new IDs. Legacy links are inferred only when unambiguous. An explicit stale ID cannot silently select another region by position.
- Text properties persist the document's default font and fitting preferences so the canvas, saved project, raster export, and editable export have the same font basis.
- A page switch or edit during a preview invalidates it. Undo targets the owning page, including an unloaded page, rather than whichever page is currently visible.
- Webtoon text contained within one page is supported. Items crossing a page seam are temporarily unavailable for Fit Text. Their existing editing behavior remains available.
- This version fits rectangles. It does not infer balloon contours, reshape bubbles, paraphrase translations, or regenerate artwork. Unsupported rich content such as embedded images is not fitted.
- New strings are present in all nine existing non-English interface catalogs.

## Validation record

- **495 automated tests passed**, including real Qt document shaping, UTF-16 rich-text ranges, French/Portuguese/Vietnamese text, Japanese and RTL cases, vertical preview parity, outlines, impossible fits, undo/redo, webtoon page ownership, save/recreation, and raster export.
- Existing OCR/translation, manual-region, inpainting undo, local repair, Artistic Edit, and provider tests passed as part of the full suite. These are regression tests, not new live-model evaluations.
- Source application smoke passed all 18 checks, including initialization of all six themes and offline guards. This was a source-runtime check, not a newly built Windows package.
- A separate native dark-theme Qt window was exercised with real UI input: a balanced fit, an impossible height with Apply disabled, restoring the area, and Apply. The rich emphasis remained visible.
- Webtoon seam/page-ownership checks use deterministic synthetic geometry. A real multi-page comic editing session and editing-time comparison are still needed before the next release.

Run the unit suite from the repository with its Python environment:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -B -c "from PySide6.QtWidgets import QApplication; app=QApplication([]); import unittest; r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not r.wasSuccessful())"
```

For manual evaluation, use independent copies of a starting project with frozen OCR and translations. Compare correction count, final font sizes, remaining overflow, and active editing time. Include a rectangular caption, a round balloon, long French, mixed emphasis, vertical text, and a webtoon seam. Keep private comic images and evaluation projects outside Git.
