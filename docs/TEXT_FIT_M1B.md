# Bubble-aware Fit Text

Implemented in the local working tree on November 2, 2026, following the first rectangular Fit Text delivery. This is an optional fitting mode using the existing editable text engine. It does not add a model or change detection, OCR, translation, segmentation, cleaning, or Artistic Edit.

## Try it

1. Select a rendered text item and open **Fit Text**.
2. Change **Fit mode** from **Box** to **Bubble interior**.
3. Start with **Estimate from original**. The blue area in the preview is the usable interior after padding. Compare Current, Balanced, and Larger text.
4. Adjust **Padding** for more space around the lettering.
5. If the estimate is unsuitable, choose **Rectangle** or **Ellipse** under Boundary. Set its origin and dimensions in page pixels. These controls affect only the fitting area.
6. Apply the preview. Font size, wrapping and placement are one undoable change. Cancel leaves the page alone.

Box remains the initial default. A successful applied bubble fit remembers its mode, padding and boundary override in the project. Ordinary dragging, resizing and selected-word formatting remain available; selected-item diagnostics recheck the current text position against its boundary without automatically moving it back.

An optional [Fit to bubble when rendering setting](AUTOMATIC_BUBBLE_FITTING.md) is now available for new lettering. It is off by default and preserves existing lettering by region identity during automatic translation. The [Review navigator](REVIEW_NAVIGATOR.md) adds independent protections for source, translation, layout, and automatic cleanup.

## What it does

The estimator reads immutable original page pixels, separately from processed history and cleanup patches. It looks for an enclosed, mostly flat background around the source text. It checks connectedness, enclosure, usable area and the relation to the text, removes narrow tails, then applies the requested padding. Flat colored interiors can work; gradients, open outlines, textures, and uncertain artwork can be declined. An estimated boundary is a heuristic, not guaranteed bubble identification.

Fitting searches a bounded set of text widths, font sizes and positions. It checks complete shaped-line rectangles, glyph overhangs and outlines against the padded mask, including holes and concave corners. The selected candidate is validated again before being offered. Existing wording and character formatting are preserved; automatic wraps stay soft. Manual line breaks remain on by default.

The saved result is an ordinary editable text item with its position, HTML, default font and width. Canvas, normal/webtoon project state and export consume that same result. OCR rectangles, original images, cleanup masks and Artistic Edit masks are not modified. A page/source/geometry change during preview invalidates the operation. Fitting is disabled during queued page processing.

## Limits and fallbacks

- Initial contour support is horizontal, unrotated text at normal item scale. Vertical, rotated or scaled text uses Box with an explanation. Canvas zoom is unrestricted.
- Text items crossing a webtoon page seam retain the earlier Fit Text restriction.
- This validates and places rectangular Qt text layouts inside a contour. It is not variable-width text flow around arbitrary contours.
- An unreliable estimate falls back to the established Box area with an explanation. Applying that fallback stores Box mode. The region is never hidden, deleted or excluded from OCR/translation.
- When a detector box clips a narrow balloon tail, Fit Text checks just beyond that edge for a closed tip. The extra pixels are used only to verify the outline. If the opening reaches the page edge, stays open, or joins another wide area, Fit Text falls back to Box. You can supply a manual boundary for these cases.
- Local masks are limited to 1,536 pixels per side and two million pixels to bound interactive work. Larger requests use Box.
- Manual overrides use known page dimensions consistently, whether or not the original has been loaded. An entirely off-page boundary falls back; it cannot move the lettering off the export canvas.
- Changes to page dimensions make automatic estimation from an unmatched original unavailable. A manual boundary remains available. No cleanup mask is treated as a text boundary.

## Validation

- **532 automated tests passed with real Windows font files explicitly registered**, including Arial styles, Segoe UI, Meiryo and Malgun. This avoids the sandbox's empty system-font registry during offscreen testing.
- Boundary fixtures cover round, rectangular, flat colored, tailed and irregular interiors; leakage, gradients, padding, bounds, allocation limits and input immutability.
- Qt layout tests include independent raster containment, outlines, holes/concavities, impossible fits, rich text, Unicode and preserved manual breaks.
- Workflow tests cover Apply/Undo/Redo, Cancel, Box fallback, original-image loading, saved/recreated items, raster export, off-page overrides, stale previews, lazy project-image materialization, diagnostics after moving text, and webtoon horizontal/vertical coordinate offsets.
- All nine non-English catalogs pass source/runtime/compiled checks: **1,010 active/runtime messages per locale**.
- Source smoke passed all 18 checks, including startup under all six themes. A disposable native preview opened and accepted a bubble candidate; a real-font offscreen Qt render was visually inspected. No new Windows package was built.
- A four-region check on the supplied Spanish page accepted the caption and both balloon interiors, including one with a clipped tail. Its usable area differed by two pixels from the result with a box enclosing the entire tail. The estimator declined the ornamental title. This check used manually annotated regions on the original page.

Real comic editing-time, correction-count and readability comparisons remain pending user testing. Keep this mode opt-in until those results justify a broader default.
