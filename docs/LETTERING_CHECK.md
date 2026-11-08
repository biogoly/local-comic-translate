# Check generated lettering

**Check lettering** is an optional OCR comparison in the Artistic Edit preview. It compares the generated writable area with the exact translation you requested, without changing the preview or deciding whether it should be applied.

## Use it

1. Select a text region with the wording you want in its target translation field.
2. In Artistic Edit, choose **Artistic lettering** and click **Use selected translation**.
3. Generate a preview, then click **Check lettering** in the preview window.
4. Open **Lettering details** to compare the requested wording, OCR reading and reported differences. Inspect the image and choose Apply, regenerate or discard as usual.

To keep a promising version while trying another seed, click **Generate alternative**. You can check each version separately and compare their lettering before applying one. [Compare Artistic Edit previews](ARTISTIC_EDIT_PREVIEWS.md).

The check uses the selected OCR route and the translation's **target language**, since that is the language expected in the edited image. The preview identifies the OCR service before you run it. Local OCR runs locally; a configured hosted OCR service sends the check image to that provider when you click the button. Pixels outside the writable area and protected pixels are excluded from the OCR input. Checking does not silently select a different provider or generate another image.

## Exact wording and results

The expected text is captured by **Use selected translation**. Editing its generated prompt clears that binding; the application does not guess the requested wording from quotation marks or other freeform instructions. Click **Use selected translation** again to establish a new exact-text request. Freeform image edits, empty-bubble edits and requests without explicit wording do not receive a lettering pass/fail result.

**Use selected translation** puts the intended capitalization directly in the quoted replacement sent to the image model. For example, an uppercase target for the mixed-case Spanish credit “Color de la ilustración de cubierta BRUSCO” becomes “Cover illustration coloring: BRUSCO” in the prompt. All-caps source lettering keeps an uppercase target. A target that already contains lowercase letters supplies its own casing. Review the quotation before generating, particularly for names whose spelling differs between languages. You can edit it directly. Click **Use selected translation** again to rebuild an older prompt.

The comparison uses the capitalization in the generated quotation and flags differences in spelling, case, accents and punctuation. It normalizes canonically equivalent Unicode characters and runs of whitespace, including line breaks. Missing words, extra OCR text and substitutions appear in the details. CJK and similar unspaced scripts can show individual character differences. Empty, symbol-only or unreadable OCR receives an uncertain result. Inspect the preview to judge the letter style.

“OCR text matches” means the recognized string agrees with the requested string under those rules. It does not certify the image: stylized letters can be misread, missed or incorrectly recognized as the intended wording. A mismatch can likewise be an OCR error rather than an error in the generated lettering. Always inspect the preview.

The check is advisory. A warning or an OCR failure does not disable Apply, repaint the image, rewrite target text or retranslate anything. Existing preview validity checks still apply if the page or project changes. Late results from a closed or superseded preview are discarded.

Automated comparison and request-routing checks do not establish real-model OCR accuracy on stylized artwork. Check representative lettering in your own comics before relying on the diagnostic.
