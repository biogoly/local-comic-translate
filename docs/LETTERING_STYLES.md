# Project lettering styles

Lettering styles reuse formatting from an existing editable text box. They are
project-specific presets for dialogue, narration, sound effects, or any other
lettering you want to keep consistent. No image model is involved.

## Create and apply a style

1. Format a text box using the normal font, color, outline, alignment, and spacing
   controls. Select that box, or select several text boxes on the current page.
2. Click **Lettering styles**, beneath **Fit Text** in the lettering controls.
3. Choose the source box in **Preview text box**, enter a **Style name**, and click
   **Save from previewed text box**. You can also select, rename, delete, or replace
   an existing preset.
4. Select the **Properties to apply**. Font, size, color, outline, line spacing,
   and alignment are initially selected. Bold, Italic, and Underline start at
   **Keep existing**, preserving emphasis on individual words. Choose **On** to
   enable an emphasis, **Off** to remove it, or **Use preset** to copy its saved
   value. Each choice updates the Styled preview immediately.
5. Inspect **Current** and **Styled**. The preview selector lets you check each
   selected box before applying the preset to all of them.
6. Choose **Apply to selection**, or **Save presets only** to retain your presets
   without changing any lettering. **Cancel** discards the dialog's changes.

**Fit preview** adjusts the comparison views; it does not fit text to a bubble.
Changing a font or its size can change wrapping and text height. Inspect the
result on the page and use the existing **Fit Text** tool or resize controls as
needed. Position, box width, rotation, scale, and writing direction are retained.

Bold, Italic, and Underline in this dialog affect the entire selected text boxes.
On and Off work without a saved preset; other formatting stays unchanged when no
preset is selected. Keep existing retains mixed word-level formatting. To format
individual words, select those words in the page's editable text box and use the
main toolbar's **B / I / U** buttons.

Changing an emphasis choice does not rewrite a saved preset. To store it, use
**Save from previewed text box** or **Replace style from previewed text box**;
these capture the current text box's base formatting plus your emphasis choices.
Then choose **Save presets only** or **Apply to selection**. Applying formatting
does not change the words or flatten editable text into an image.

## Saving and undo

Presets are saved with the project. Older projects start with an empty style
library. Opening another project or clearing the workspace resets the library.
Font files are not embedded; another computer needs the same fonts for the same
appearance. The preview warns about missing fonts and unsupported characters;
applying a result with font warnings requires an explicit acknowledgement.

Applying formatting to several selected boxes creates one page-local **Apply
lettering style** undo action. Undo restores their previous formatting; Redo
reapplies it. Preset-library edits are saved as project settings and are not
reverted by page Undo, so undoing a page edit cannot erase presets saved later on
another page.

Selection is limited to one current page, including in webtoon mode. Background
processing must finish before opening or applying the style preview. A preview
is rejected if its project, page, text, or style library changes underneath it.

## Reuse presets in another project

1. Open a page in the source project and choose **Lettering styles → Export presets…**.
   Pick a location for the JSON file. It contains every preset currently listed,
   including any changes you made in the open dialog.
2. Open a page in the destination project and choose **Lettering styles → Import presets…**.
   You can do this with no text boxes selected. The dialog reports how many
   presets it added, renamed, or skipped.
3. Choose **Save presets only** to keep the imported styles for later, then save
   the project. To use one immediately, open the dialog with a text box selected,
   select the preset, check **Styled**, and choose **Apply to selection**.
   **Cancel** discards pending library edits. An exported file is written as soon
   as you choose its destination.

If both projects have a style named **Caption**, matching formatting counts as
a duplicate. If the formatting differs, the existing **Caption** keeps its name
and the incoming version receives a name such as **Caption (2)**. The library
can hold 100 presets. If the additions would exceed that limit, the dialog
declines the whole file. Make room and import it again.

The file contains preset names and formatting such as font family, size,
color, outline, spacing, alignment, and emphasis. Page images, dialogue, and
font files stay out of the export. For the same appearance on another computer,
install the same font there. Rotation, perspective, and curvature belong to
individual text boxes. See [Editable text shape](LETTERING_TRANSFORMS.md).

## Original style editor validation

Validated on Windows with the source app runtime:

- 890 automated tests passed, including 58 lettering-style tests for schema,
  project persistence, rich-text formatting, previews, stale-result rejection,
  page-local undo, font diagnostics, page reload, and image export.
- Native clicks in a disposable two-bubble project created a preset, compared
  both boxes, applied it, and exercised toolbar Undo/Redo. Selected-word bold and
  italic formatting survived all three operations.
- Follow-up native testing enabled Bold, Italic, and Underline individually
  without a preset, verified each visible preview, and applied/undid/redid the
  result. Regression tests also verify Off, Keep existing, saved values, and
  main-toolbar selected-word formatting after applying a style.
- All nine source/compiled interface catalogs passed coverage and placeholder
  checks (1,277 active/runtime messages per language).
- Preview controls passed theme checks across all six themes, including selected
  preset text contrast. Text samples covered the supported scripts; missing
  glyphs are reported when the installed fonts lack them.
- All 18 source smoke checks passed using the isolated Windows build environment.
  This was not a rebuilt installer or a new PSD-effects validation.

The latest regression log is `work/m5a-emphasis-tests-full.log` in the development workspace;
the smoke report is `work/m5a-source-smoke.json`. Native test measurements are in
`work/m5a-native-profile/results.jsonl` and
`work/m5a-emphasis-native-profile/results.jsonl`. No live OCR, LLM, or image-model inference
was needed for this feature's checks.
