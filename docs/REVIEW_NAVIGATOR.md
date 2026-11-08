# Review navigator and region protections

Implemented in the local working tree on November 3, 2026 (M2). This adds a manual review workflow alongside the existing page tools.

## Find and resolve issues

Expand **Review** below **Fit Text** in the right sidebar. Choose **All pages** or **Current page**, then optionally filter the issue list. Clicking an issue opens its page and selects the matching region. **Next / Previous**, or **Ctrl+Alt+Right / Ctrl+Alt+Left**, move through the filtered list. Navigation uses saved page and region identities, including after moving boxes, reordering pages, or reopening a project.

The panel lists missing or uncertain OCR, missing translation, overflow, small text, failed batch operations, and stages awaiting review. It evaluates the current text and layout incrementally. Bubble contours are checked when their page is open.

Select **Source**, **Translation**, **Layout**, or **Cleanup**, then **Mark reviewed**. **Ignore issue** dismisses only that particular finding. Reviews and dismissals are tied to relevant content: changing the source, translation, layout, or cleanup evidence can make the relevant finding appear again. Reviewing intentionally blank text can acknowledge that it should remain blank. Layout approval does not hide measured overflow or text below the configured minimum size.

**Fit Text** opens the existing preview for rendered text. **Report** opens the existing batch report, including failures elsewhere in the work. Missing regions that were never detected are not automatically discovered by the navigator.

## Review a questionable OCR reading

After **Recognize**, open **Review** and choose **Uncertain OCR** in the issue filter. PP-OCR reports a score for each recognized line. The navigator flags a region if the lowest score among the lines included in its source text is below **0.75**. The cutoff is a review heuristic. A high-scoring line can still contain a mistake. Earlier project files acquire scores as you recognize regions again.

Select a region and click **Retry OCR…**. Choose a compatible local recognizer. Continuing may download its model weights on first use. The recognizer reads the original page crop, so this also works after cleaning. Compare its proposed reading with the current source, edit the proposal if needed, then choose **Apply** or **Cancel**. Apply changes that region’s source text and supports **Undo**. Use **Translate** when you want a new translation and **Keep source** to retain your reading during a later page-wide recognition pass.

You can retry any selected region, including one with a high score or an older reading. The choices depend on source language, script and the recognizer that supplied the current reading. Spanish and Portuguese, for example, can switch between PP-OCRv5 Latin and PP-OCRv6 Multilingual. The panel explains when a language has no eligible alternative.

The score label identifies its model and what it measured. Microsoft OCR can supply word scores, and Pororo can supply a sequence score. Each uses a different scoring method. The **Uncertain OCR** filter uses PP-OCR line scores only. A reading with unknown evidence has no score label. Editing a reading clears its displayed machine score, and marking the source reviewed acknowledges the current wording.

## Independent protections

Select a region and use the Review panel checkboxes or its right-click menu:

| Control | Whole-page and batch behavior | Direct editing |
|---|---|---|
| Keep source | Retains its OCR text | The source field remains editable |
| Keep translation | Retains its target wording, including case and line breaks | The target field remains editable |
| Keep layout | Retains its existing lettering, geometry, and formatting; retains an intentionally absent text layer too | Dragging, resizing, formatting, and Fit Text remain available |
| Skip automatic cleanup | Excludes its area from page-wide segmentation and cleaning, including overlapping neighboring masks | Deliberate brushes and selected-region cleanup remain available |

For an artistic title, **Skip automatic cleanup** can protect the original artwork while OCR and translation remain available. These controls work for all detector classes, including `text_free`. They do not restrict detection to speech bubbles.

Protections and review decisions are saved with the project and are undoable. Approval is separate from protection: marking a translation reviewed does not prevent a later page-wide translation from replacing it. Use **Keep translation** for that.

A selected-region OCR or translation request on a kept stage opens a replacement preview with **Apply / Cancel**. Apply preserves the protection and is undoable. For a translation with existing lettering, it preserves the text layer's geometry and formatting while updating the wording. Without an existing layer, use the normal Clean / Render steps after applying. A stale preview cannot replace text changed while the request was running.

## Cleanup and automatic fitting

Automatic cleaning subtracts protected regions from the final effective mask and restores the captured, currently composed pixels outside that mask. This also applies after fast bubble filling, so overlapping masks cannot change protected areas indirectly. Rotated protected regions use a conservative enclosing rectangle. The control is automatic-operation protection, not an immutable paint layer; an intentional manual brush can still change those pixels.

With **Fit to bubble when rendering** enabled, existing rendered items continue to be retained by region identity, while other regions on the page can proceed. Source and target updates remain independently controlled. If a target changes underneath retained lettering, review its layout and render or refit deliberately. Pages containing orphan lettering without saved region ownership retain the conservative whole-page skip.

Cleanup review uses recorded cleanup patches to identify areas for inspection. Failed-operation findings come from the batch report.

## Validation

- **656 automated tests passed**, using the Windows build virtual environment and real Arial, Segoe UI, Meiryo, and Malgun font files registered for offscreen Qt tests.
- Source startup smoke passed all **18 checks**, including all six themes and isolated settings.
- All nine non-English interface catalogs passed source/runtime/compiled checks, with **1,069 active/runtime messages** per locale.
- Native Windows checks confirmed selection, next-issue and keyboard navigation, cross-page navigation, reviewed state, protection controls, and narrow-sidebar scrolling. A clipping issue found during this check was fixed and rechecked.

Automated coverage includes content-bound reviews and dismissals, independent protections, project/undo identity, stale replacement previews, Unicode and rich text replacement, page reordering, normal and webtoon navigation, moved or absent kept lettering, exact protected target wording, overlapping masks, fast fill, rotated protected bounds, and cleanup across webtoon seams. Models are mocked in workflow tests; the cleanup checks use deterministic synthetic images.

The native Windows check uses a disposable two-page project and isolated settings. It checks selection, next-issue navigation, page switching, review decisions, and independent protection controls. No real OCR/LLM inference, existing user project changes, or package build is part of this delivery. Real-comic workflow testing remains the next checkpoint.
