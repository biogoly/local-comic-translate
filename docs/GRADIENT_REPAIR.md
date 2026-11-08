# Sampled gradient repair (M4a)

Gradient fill repairs a smooth colored background using clean samples you choose. It fits a color plane across the painted mask and makes no model calls. It does not reconstruct paper grain, screentones, or other texture; the fixed-sample Clone brush remains available for those areas.

## Workflow

1. Paint or segment the text/artifacts to remove. Work on one smooth background area at a time; inspect the mask so it does not include a panel border or artwork you want to keep.
2. Under Inpainting, choose **Fill method → Gradient**, then **Preview gradient fill…**.
3. In the left view, choose at least four clean sample locations surrounding the mask. Click for a single pixel, or **hold Shift and drag** to sample a straight line. On speckled paper, a short line through one background color can give a better reading than a click on a bright or dark fleck. Each line counts as one observation at its midpoint, using the median red, green and blue values along its trace. You can mix clicks and lines. Keep each trace clear of lettering, outlines and the painted mask.
4. Choose **Generate preview** and inspect the result on the right. Use the wheel/zoom controls, **Fit repair area**, or **Fit page** to inspect or reach other sample positions. Drag to pan. Remove the last sample or clear the samples if needed.
5. Choose **Apply** to insert the repair, or **Cancel** to leave the page and painted mask unchanged.

**Sample from** defaults to **Edited page**, the current background including edits and repair patches. Choose **Original page** to read the source file or the source image embedded in the project, even after the page has an edit history. Original lettering is visible in that source, so choose clean pixels. Changing the source clears the samples and invalidates the preview.

**Solid** retains the existing dropper-and-fill workflow. Restore, Pick color, and Clone retain their existing behavior and icons. No extra blur, feathering, or mask expansion is added by Gradient fill.

## Applying and undoing

Only pixels covered by the captured mask are blended into the current page. The mask's existing antialiasing is respected, and pixels outside it remain identical. Original-image sampling still applies onto the current edited background; it does not replace the rest of the page with the original image.

Applying the patch and consuming its mask is one page-level undo step named **Sampled gradient fill**. Undo restores both the previous pixels and the painted mask; Redo restores the accepted result. In webtoon mode, the repair and mask consumption stay within the current physical page. Accepted repairs use ordinary inpainting patch storage and survive project save/reopen. Revert Inpainting also includes these repairs.

The dialog is a fixed snapshot. If the page, mask, image pixels, project identity, or undo state changes before Apply, the result is rejected; open a new preview. When sampling Original page, changes to that source also invalidate the result. The unaccepted preview and sample markers are not part of project images or exports.

## When a fit is rejected

- Points in a line cannot describe a two-dimensional gradient. Spread them vertically and horizontally.
- Points on only one side of the mask would require uncontrolled extrapolation. Place samples around all sides of the repair.
- Conflicting colors can indicate samples on lettering, a border, or several different backgrounds. Remove those samples or repair a smaller area.

There is no automatic fallback that guesses a nearby border color. A gradient is appropriate for a smooth transition between colors; use Solid for a flat background and Clone for repeated texture.

Place the sample centers around the unprotected part of the painted mask. If the mask overlaps a protected letter or border, you can leave that protected area outside the sample perimeter. Every sample must still stay outside the entire painted mask. The preview shows the reason when more samples are needed.

For a textured area, preview Clone or Heal using a clean texture sample. You can protect a panel border before repairing the adjacent background. [Protected artwork and Heal](PROTECTED_REPAIRS.md).

## Validation

On November 3, 2026, all 762 automated tests and 18 source smoke checks passed. The focused repair checks cover outlier/rank/coverage handling, exact outside-mask pixels, original/current sampling, stale previews, undo/redo, webtoon page offsets, and accepted-patch save/reopen. All nine interface catalogs match their compiled resources, and the dialog was checked across all six themes.

A disposable native Windows test exercised four sample clicks, preview generation, Apply, Undo, and Redo on a colored panel beside a dark border. The synthetic repaired gradient differed from its known clean surface by at most one 8-bit color level. This does not establish quality on scanned or textured comic artwork; those pages need user comparison. No model, API, or GPU is required for this tool, and no package was rebuilt for this stage.

On six small patches of real comic artwork, short lines made the color estimate more consistent on cyan paper and purple print grain when the sample positions moved by one pixel. Clicks and lines each produced 18 usable fits across 30 placements. On an actual cover credit, both Gradient methods smoothed over the book-spine texture. Clone and Heal preserved more of the design when the clean sample came from the same spine.
