# Experimental text-box coordinates

In **Settings → LLMs**, enable **Provide Image as Input to AI**, then **Provide text-box coordinates (experimental)**. This option is off by default. It helps a vision-capable translation model match each OCR entry to the artwork around it. Try it when the same word appears in different panels and could mean different things in each.

The coordinate control is unavailable when image input is off or the selected adapter/model does not accept images. Turning image input off or switching to a text-only model retains your preference without sending spatial metadata. A local model still needs vision support and its required projector. Text-only translation retains its ordinary input.

## What the model receives

The request includes the delivered image dimensions and a visual-reference entry for each requested region. `bbox_normalized` contains `[left, top, right, bottom]` values between 0 and 1, relative to that image, with the origin at its upper-left corner. Coordinates round outward by at most a tenth of a source pixel per edge to limit prompt length. Angles are clockwise degrees. For rotated text, the box encloses the visible rotated region. Reading order follows the supplied OCR sequence.

When the same OCR wording occurs more than once in a request, each visible occurrence also gets a plain-language position such as “bottom left.” This helps the model associate identical words with different panels. Wording matches across capitalization and whitespace. In **Request preview**, look for the `position` field beside the repeated entries.

Partially visible regions are clipped to the delivered image and marked as clipped. Fully off-image regions keep their OCR text and request ID but have no visual box. Invalid or unmappable geometry reports an error instead of sending misleading coordinates. Visible webtoon requests map text regions into the delivered view; full-page and combined-image requests use their corresponding image coordinates.

No numbered image overlays, extra crops, or additional image requests are added. OCR text remains the translation input and the image supplies context. Response keys remain `block_0`, `block_1`, and so on, with each response matched to the existing region identity. Enabling spatial context does not change detection, cleanup masks, or lettering geometry.

## Inspect and compare

Open **Translation context → Request preview → Build preview** to inspect the included spatial reference without sending a model request. Choose the relevant selected-region or current-page scope. In webtoon mode, a physical-page preview can differ from a visible-area or automatic chunk request.

Coordinate settings and included geometry participate in request/cache identity. Changing them does not rewrite existing translations; use an explicit translation action to obtain a new result. Existing response validation and stale-result safeguards still apply.

For an initial comparison, use the same page, model, language pair, and generation settings with spatial context off and on. Compare missed text, speaker associations, wording, correction effort, and processing time. Keep your preferred result; coordinates are an optional aid rather than a guarantee of better translation.

## Local Gemma trial

On two synthetic pages with the OCR entries supplied in reverse panel order, Gemma 4 12B correctly distinguished Spanish *banco* (a bank or a bench) and *pluma* (a pen or a feather) with the positional hints. The earlier coordinate format and the image alone both gave the meanings in panel order.

On three saved Spanish comic pages, Gemma returned all 22 translations with coordinates enabled and disabled. Both versions passed eight of nine checks on who was speaking, what the dialogue referred to, and what happened in the scene. Both misread an interrupted adjective as the address “my love.” The compact format saved 837 prompt tokens across those three pages compared with the earlier coordinate format. These results describe one local model and this set of pages.
