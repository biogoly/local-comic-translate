# Editable text shape

Select one text box and click **Text shape**, below **Lettering styles**. The dialog compares the current lettering with the proposed result at the same preview scale. Apply makes one undoable change; Cancel leaves the page unchanged.

- **Width / Height** stretch the lettering independently (25–400%). They do not rewrap the text.
- **Skew** slants the whole text block (−45° to 45°). This is separate from italic formatting on individual words.
- **Horizontal / Vertical perspective** taper the text block in either direction (−60% to 60%).
- **Curve** bends each horizontal line along a circular baseline (−120° to 120°). Positive values arch upward; negative values bend downward. Start around 30–60° for a gentle title or sound effect. Line breaks, wrapping and individual-word formatting stay editable.
- **Reset distortion** restores normal width, height, skew, perspective and curve. It keeps the box position and existing rotation. Apply the reset to make it an undoable edit.

You can still edit the text, select and format individual words, move the box, and use its resize and rotate handles. Canvas resizing changes the text box and font as before; the Text shape controls stretch its appearance. A shape edit affects lettering only: it does not enlarge OCR boxes or cleanup regions.

Fit Text is unavailable while a distortion is applied, and the review navigator does not claim a measured fit for distorted text. Reset the distortion before using Fit Text. Curves support horizontal lettering; the Curve control is disabled for vertical lettering. A curve can be combined with stretch, skew, perspective and canvas rotation. Each wrapped or explicitly separated line gets its own arc; this is not text flowing along an arbitrary hand-drawn path.

## Match original lettering angle

On a fresh page, run **Detect** while the source lettering is visible. Continue through Recognize, Translate, Segment and Clean as usual. **Render** rotates new editable text to follow a straight source baseline when several letters support the same tilt. Detection uses the original letters within each selected region. The current range is 3° to 35° clockwise or counterclockwise. Short exclamations, curved or vertical lettering, and busy artwork can give too little evidence, in which case new text uses the usual orientation.

To adjust the result, select the rendered text and drag its rotation handle just outside the text box. Release near horizontal to snap to 0°. Your chosen angle takes priority on subsequent renders and survives saving and reopening the project. **Undo** reverses the adjustment. You can also rotate the OCR box before rendering when you want to set the angle yourself.

Projects saved before this feature can use a fresh detection pass on the original lettering. You can rotate already typeset text directly.

## Projects and export

Save the project (`.ctpr`) to retain the original rich text and editable shape parameters. Project reload, page navigation, webtoon storage and Undo/Redo preserve these parameters. A transformed item crossing webtoon pages keeps one editable source owned by its original page; each exported page receives the visible portion of that lettering.

Image exports render the transformed lettering. In PSD exports, ordinary text and affine transforms (stretch, skew and rotation) remain editable text layers. Curved or perspective lettering becomes a clearly labeled transparent image layer; other text layers remain editable. Keep the original project to edit curved or perspective text later.

When reopening a PSD in this application, curved and perspective image layers are imported into the background, below the remaining editable text. The importer warns about this difference; the original project is the authoritative editable version.

## Validation boundary

Tests cover transform and curved glyph geometry, rich-word formatting, Unicode graphemes and bidirectional text, Qt input-method events, resize/rotate undo, empty/retyped text, actual project save/reopen, webtoon page seams, and rendered output. Native Windows checks on a disposable two-bubble page covered upward/downward curve previews, Apply, word selection, typing a replacement word, per-word bold and its toolbar state after Undo, resizing and resize undo. Resize handles activate just outside the text border, preserving text selection inside it.

The November 4 curved-lettering increment passed 983 automated tests (38 curve-specific tests), all nine language catalogs, and 18 source smoke checks. Japanese composition and commits were checked through Qt input-method events; this is not a manual Windows IME acceptance test. PSD tests use the installed PhotoshopAPI library and check layer data, pixel placement, worker-thread export and reimport; a visual inspection in Adobe Photoshop has not been performed.

The font-fallback follow-up passed 986 tests (41 curve-specific tests). It fixes partially stationary text when a comic font needs another font for an accented character, such as DigitalStrip BB with É. Curved layout now associates native shaped glyphs with their source graphemes without relying on Qt's incorrect fallback-run indexes; shared ligatures are retained and painted once. Original test fonts reproduce the failure without requiring a user's installed fonts. A native Windows preview also verified the reported four-line sentence with the actual DigitalStrip font and outline, using a disposable copy of the lettering state.
