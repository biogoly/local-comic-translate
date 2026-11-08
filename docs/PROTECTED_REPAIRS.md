# Protected artwork and experimental Heal

Implemented locally, November 3, 2026. These additions preserve the existing manual workflow and ordinary fixed-sample Clone brush. They require no model or network connection.

## Protect artwork

1. Expand **Protect artwork** beneath the inpainting fill controls.
2. Select **Protect**, set the brush size, and paint borders or artwork that repairs must preserve. Purple shows the protected pixels.
3. Run cleanup, Restore, Clone, Solid/Gradient fill, Heal or Artistic Edit as usual. New raster repair patches preserve the protected background pixels exactly, including feathered edges.
4. Use **Unprotect** to make a small area editable again. This changes the protection mask, not the artwork. Each stroke supports Undo/Redo.

**Show protected pixels** controls visibility only; hiding the overlay does not disable protection. The overlay is excluded from cleanup masks and exports. Protection is saved per page in the project, including webtoon pages. Existing projects with no protection behave as before.

Protection covers background-pixel repairs, not editable lettering objects. It does not prevent moving or formatting text. Undo/Redo and explicit page-wide revert actions restore historical edits exactly; they do not reinterpret history using today's protection mask. Unprotect an area before deliberately repairing it. There is no automatic override.

## Heal comparison prototype

Restore and Clone's **Original** option now read the original page file or its embedded project image, including when the current page is displaying a later edit. A Heal preview uses the frozen texture captured by Clone. To sample several background colors for a smooth surface, use [Gradient repair](GRADIENT_REPAIR.md).

This is a small, deterministic color-adjusted clone preview. It preserves texture from your chosen sample and offsets its color toward an explicitly picked RGB value. It does not reconstruct arbitrary hidden artwork or sample neighboring borders automatically.

1. Select **Clone**, choose a brush size large enough to contain the repair, then **Alt-click** a clean texture sample. The current **Sample original** option determines whether it comes from the original or edited page. The captured sample stays fixed.
2. Use **Pick color** on an intact area with the desired background color.
3. Paint a small repair mask with the ordinary mask brush, or use an existing segmented mask. The entire mask must fit inside one captured circular sample centered on the mask. For distant areas, work on them separately.
4. Choose **Fill method → Heal (experimental)**, then **Preview Heal…**.
5. Compare **Current page**, **Ordinary Clone**, and **Heal preview**. Adjust **Color adjustment**; 0% matches ordinary Clone. Previewing never changes the page. Apply Heal only when it improves the result.

Oversized masks are rejected rather than tiled. Cancel and capture a larger clean sample or reduce the mask. Protected pixels remain unchanged. Applying the patch and consuming the repair mask is one Undo step; Undo restores both. Changed page pixels, masks, protection, project or page history invalidate the preview.

For smooth gradients without useful clean texture, Gradient fill remains the better starting point. The prototype needs comparison on real damaged comic panels before broader healing behavior is justified.

## Implementation and validation

- Portable protection records use a versioned compressed binary mask in page state. Decoding is bounded and malformed active masks stop protected operations instead of silently disabling protection.
- New `PatchInsertCommand` patches are clipped once, after compositing/feathering, against the current composed background. Stored patch pixels are immutable for Undo/Redo.
- Automatic cleaned-image exports also clip the generated image before writing it. Webtoon virtual slices and seam repairs crop the physical-page mask into the correct local coordinates.
- Gradient and Artistic Edit previews show the clipped result. Heal snapshots the explicit sample, RGB target, page, mask and protection.
- Source and compiled interface translations cover all nine existing UI locales.

Tests cover binary mask validation, final pixel preservation, mutating/dilating backends, automatic cleaned PNG export, virtual-page seams, project save/reopen, zoom, page navigation, overlay exclusion, text hit testing, preview rejection and repair Undo/Redo. Native Windows testing on a disposable synthetic page verified Heal preview/Apply/Undo/Redo with protected pixels unchanged. No live-model or real-comic quality comparison was performed for this stage.

Validation: **832 automated tests passed** in the actual app `.venv`, with Windows fonts registered for layout checks. All nine source/compiled locale catalogs cover 1,219 active/runtime messages each. `git diff --check` passed.

The current app environment lacks the optional `azure.ai.vision.imageanalysis` package: 17 of 18 source smoke checks pass there, including the main window in six themes. The separate build environment passes all 18. No dependency was installed or upgraded for this work, and no package was published.
