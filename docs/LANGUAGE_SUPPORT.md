# Portuguese and Vietnamese translation support

## Using the languages

1. Restart the source application after updating.
2. Select **Portuguese (Portugal)**, **Portuguese (Brazil)**, or **Vietnamese** in
   the source-language selector. The same choices are available as translation
   targets. Selections persist when changing pages in Manual mode.
3. Keep **Settings > Tools > OCR** on **Default** for local recognition. The
   existing optional provider OCR choices still work through their own services;
   this change does not introduce automatic cloud fallback.
4. Detect regions or draw a region manually, recognize the source text, review
   it, then translate/segment/clean/render as usual.
5. Use a font containing the target language's characters. Arial was verified
   for the Vietnamese/Portuguese test set; decorative comic fonts may omit
   accents. Review both the editable text and the exported page.

**Select Vietnamese explicitly.** Auto detects scripts, not individual
Latin-script languages. Its existing Latin route is unchanged and does not
select the Vietnamese recognizer. For a mixed-language page, recognize its
Vietnamese regions with Vietnamese selected, then change the source choice for
other regions as appropriate.

These are comic source/target languages, not new interface localizations. Menus
and settings still use the existing interface-language choices. The new
Portugal/Brazil labels are translated in all nine existing Qt catalogs.

## Portuguese dialects and compatibility

Portuguese source recognition reuses the existing Latin PP-OCRv5 model. No
replacement model or additional runtime dependency is needed.

Both local and cloud LLM translation prompts now explicitly request **European
Portuguese (Portugal)** or **Brazilian Portuguese (Brazil)**, rather than leaving
the meaning of generic Portuguese to the model. Translation quality still
depends on the selected model. Traditional provider adapters normalize codes:
DeepL uses `PT` for either Portuguese source and `PT-PT`/`PT-BR` for targets;
Microsoft uses `pt-pt`/`pt` and honors an explicit source selection. Auto keeps
provider language detection enabled.

Saved identifiers remain `Portuguese`, `Brazilian Portuguese`, and `Vietnamese`;
existing project/settings values and their previous localized labels remain
readable. No project migration or reset is required.

## Vietnamese local OCR

The stock Latin recognizer's alphabet lacks many Vietnamese letters. The newer
stock Latin and V6-small artifacts examined during development also lacked
complete coverage, despite general language-support tables. Changing a
dictionary alone cannot add output classes to trained weights.

Explicit Vietnamese recognition therefore uses the community
[OcrDoc Vietnamese PP-OCRv6-medium fine-tune](https://huggingface.co/truonggiabao7/ocrdoc-models),
whose publisher declares the weights Apache-2.0. It was trained for identity
documents/passports, not comics. Vietnamese comic OCR should be considered
**experimental pending representative comic-page testing**, not a guarantee of
accurate stylized lettering or diacritics.

- The app downloads just the recognizer and its matched charset on first use:
  62,289,039 bytes (about 59.4 MiB) plus a 680-byte charset.
- Revision: `9b5346432fb84a447fdc4d780a7741126b446b0f`.
- Recognizer SHA-256:
  `767305d37d06728fe9578ff8132595ce23f66139df5e2937fd4131bd1dc34f12`.
- Charset SHA-256:
  `5d801bf60472c4c1b184b6b6843aa2c754d8daf1429247e07658ee0bd79ab448`.
- Both checksums are verified before loading. No remote Python code is loaded.
- It is cached separately under the fork's user-data directory at
  `models/ocr/ppocr-v6-vietnamese`; other OCR model files are not replaced.
- After the first download, recognition works offline. Model files are not
  bundled with the source release. No extra Python packages are required.
- The recognizer uses 320/640-pixel input widths and at most four crops per batch;
  the existing detection, line preparation, region editing, and rendering remain
  in use. CPU inference was tested. GPU/provider configurations need separate
  qualification; the existing GPU toggle and CPU fallback behavior are retained.

The supplied charset contains 228 entries. CTC blank and space give 230 output
classes, checked at initialization. Malformed output shapes fail clearly rather
than being silently decoded with a mismatched alphabet.

## Validation — October 9, 2026

This is a source-code validation record, not a new Windows package qualification.

- **Regression suite:** 452 tests passed, including new OCR routing, isolated
  model registration, width/batch limits, Portuguese dialect/provider requests,
  selector localization, legacy identifiers, and page-language persistence.
- **Real local CPU OCR:** the production downloader fetched and verified the
  Vietnamese pair into an isolated test cache. The real OCR processor prepared
  manually drawn regions and recognized synthetic printed lines, using Arial
  at 42 pixels on a light background. Portuguese (Portugal) matched 4/4 lines;
  Portuguese (Brazil) matched 4/4; Vietnamese matched 5/8. Vietnamese errors
  included `lo` read as `Io`, `HẸN` as `HỆN` plus omitted punctuation, and a period
  read as a colon. This is not a real-comic accuracy benchmark.
- **Vocabulary:** all tested Vietnamese precomposed tone/vowel combinations,
  both uppercase and lowercase, were present in the downloaded charset.
- **Real Qt rendering/export:** manual target text in all three languages,
  precomposed and decomposed Unicode, HTML and project/canvas serialization,
  pixel-identical restored rendering, and lossless production PNG export passed.
  Resolved Arial covered all 175 tested characters, including combining marks.
- **Translation requests:** local/cloud LLM prompts and traditional provider
  codes were tested with mocked requests. No live translation-quality evaluation
  or paid provider calls were made in this validation.
- **UI catalogs:** all nine existing catalogs cover 966 active/runtime messages
  and match their compiled resources. No Vietnamese/Portuguese interface catalog
  was added.
- **Offline source smoke test:** main window, all six themes, image/archive
  round trips, and CPU ONNX checks passed without network requests or model
  downloads. No frozen Windows package was rebuilt.

Local development evidence is retained in the task workspace under
`work/language-support/language-validation.json`, `qt-rendering-report.json`,
and the corresponding input/exported PNGs. No user comics or credentials were
included in the repository.

Next qualification step: inspect real Vietnamese and Portuguese comic pages,
especially low-resolution scans, ornate fonts, and stacked Vietnamese accents.
