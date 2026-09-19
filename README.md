# Local Comic Translate

English | [한국어](docs/README_ko.md) | [Français](docs/README_fr.md) | [简体中文](docs/README_zh-CN.md)

<img src="https://i.imgur.com/QUVK6mK.png" alt="Comic Translate interface">

Local Comic Translate is a community fork of [Comic Translate](https://github.com/ogkalu2/comic-translate) focused on private, local translation and a safer hands-on desktop editing workflow. It keeps the upstream application's detection, OCR, inpainting, rendering, project, archive, and webtoon features while adding managed `llama.cpp` translation, FLUX.2 Klein artistic editing, optional direct provider APIs, and substantial editing and reliability improvements. No upstream account, subscription, or translation credits are required.

The inherited OCR pipeline supports English, Korean, Japanese, French, Simplified and Traditional Chinese, Russian, German, Dutch, Spanish, and Italian as source languages, with those and additional languages available as translation targets.

This fork is under active development. The changes described below currently require running from source; the downloads on the upstream Comic Translate website do not include them. The translated README files also still describe the upstream project and may not yet reflect the fork-specific features.

## Latest update — 0.1.0

- **Artistic Edit:** local FLUX.2 Klein 4B/9B with compatible LoRAs, or the Black Forest Labs API, using a preview-before-apply workflow with undo and project persistence.
- **Your choice of translation backend:** retain local llama-server settings while switching to OpenAI, Gemini, Claude, or Deepseek with your own API keys.
- **Better manual workflow:** OCR works directly on hand-drawn regions; completed manual translations and their formatting survive later whole-page processing; source and target language choices persist between pages.
- **Localized cleanup:** restore original pixels with a brush, clone a fixed sample of color and texture, or fill a mask with a sampled background color.
- **Desktop polish:** Midnight, Parchment, Lavender, and Mint themes alongside Light/Dark, clearer Artistic Edit controls, refreshed splash artwork, and updated GUI language catalogs.
- **Independent fork identity:** separate settings and application data, with upstream update prompts and paid-service routing removed.

## Highlights of this fork

### Local first, private LLM translation

- A **Local LLM** translator that does not require a hosted translation API or Comic Translate account.
- **Managed llama.cpp** mode launches a local `llama-server` process, chooses an unused loopback port, reuses the process between requests, and stops it when the app exits.
- **External server** mode supports OpenAI-compatible `/v1` endpoints such as Ollama, LM Studio, or a separately managed `llama-server`.
- Configurable GGUF model, optional multimodal projector (`mmproj`), context size, GPU layers, startup/request timeouts, maximum output tokens, temperature, Top P, and Top K.
- Optional page-image input for vision-capable models. The existing OCR remains the primary text source; the image gives a multimodal model additional context and a chance to correct obvious OCR mistakes.
- Strict block-ID validation and automatic retries prevent incomplete or malformed model output from silently shifting translations into the wrong bubbles.
- Page-level translation caching uses the original source image, so later inpainting does not unnecessarily invalidate a cached translation.

### Optional direct provider APIs

- **Local LLM** remains the default. No upstream Comic Translate sign-in, subscription, or credit purchase is required or used for translation or OCR.
- In **Settings > Tools > Translator**, choose **OpenAI**, **Google Gemini**, **Anthropic Claude**, or **Deepseek** to use a cloud provider directly.
- In **Settings > Provider APIs**, choose the matching provider, enter your own API key, and select or type its API model ID. The provider bills your account directly. Presets are starting points, not a complete list of available models.
- Microsoft Azure OCR credentials and the Gemini key are also available for the existing optional cloud OCR choices. Default OCR stays local.
- **Save Keys** is opt-in. Keys are stored in this fork's local application settings (not an encrypted vault), never in project files. Turn it off and save/close the app to remove persisted provider keys; model selections remain saved.
- Cloud translation sends recognized text and, when **Provide Image as Input to AI** is enabled and supported by the model, the page image. Provider policies and charges apply. There is no automatic local-to-cloud fallback.
- The former **Advanced / Custom** configuration is removed. An active saved Custom endpoint migrates to **Local LLM > External server**. Existing named cloud-model selections migrate to their direct provider, retaining their model choice; users must supply their own key.
- Provider credentials and local-server configuration are independent. Changing **Tools > Translator** switches the active backend without clearing either configuration; the provider dropdown under **Provider APIs** only selects which credentials to edit.

### Safer manual editing

- A practical page-by-page workflow for reviewing detected regions before anything is erased.
- Both source and target language selections remain in effect when moving between pages in Manual mode. An explicit **Auto** source-language selection also persists until changed.
- Right-click **OCR** works on a hand-drawn region without first running whole-page Detect. Local OCR prepares missing line boundaries, and an explicit region OCR action retries that region rather than returning an old cached reading or processing unrelated boxes.
- Whole-page Detect retains hand-drawn regions and suppresses duplicate detections of the same text. Later Recognize/Translate operations preserve their completed source/target text; Segment and Render retain their existing text layers and word-level formatting instead of deleting or duplicating them.
- Preserving a manual text layer does not bypass background cleanup: its region still participates in segmentation and cleaning. Untranslated manual regions continue through the normal workflow.
- Zoom In, Zoom Out, and Fit controls, plus configurable keyboard shortcuts.
- Natural page sorting by filename and sorting by modification time.
- Borderless replacement text appears immediately when a new text region is drawn and text is entered.
- Deleting and retyping text preserves the chosen font.
- Bold, italic, and underline can be applied to selected words instead of only to the entire text item.
- Inline formatting coexists with text-box moving/resizing, and manually entered target text is retained when focus changes, including Japanese IME input.
- Full-page manual translation creates any missing text layers. Translating a selected, not-yet-rendered region cleans it before creating its translated text layer.

### Inpainting and undo improvements

- Every manual cleanup pass is undoable and redoable with the normal Undo/Redo controls.
- **Restore brush** restores original base-page pixels only under the brush, leaving edits elsewhere intact. Original lettering is restored too; this does not remove editable text overlays. Set its size with the existing brush slider. Each stroke is undoable.
- **Clone brush** (stamp icon): set the brush size, Alt-click a clean area, then paint to repeat that fixed sample of color and texture. The dashed cyan circle stays at the source; the solid black/white outline follows the destination and scales with zoom. **Soft edge** controls blending. **Sample original** is enabled by default to sample beneath cleanup patches; turn it off to copy the edited page. Changing size, pages, or the source mode requires a new Alt-click sample. Each stroke is undoable; only the painted pixels change. No AI model is used.
- **Pick color**, then **Fill mask with sampled color**, fills the painted or segmented areas on the current page with an intact background color. It uses the visible mask without extra expansion or model inference. Use this for flat colored bubbles; gradients and textured artwork still need inpainting or artistic editing. Undo restores both the previous pixels and the mask.
- **Revert All Inpainting on the Current Page** removes all saved cleanup patches as one undoable action.
- Repeated cleanup starts from the currently composited page, preventing previously removed text from reappearing in later patches.
- Manual brush cleanup handles single clicks as well as strokes and uses integer-safe image bounds.
- Geometry-aware fallback masks distinguish rectangular caption boxes from round balloons, reducing leftover text in square corners while remaining conservative around artwork.
- Automatic page edits are grouped into predictable undo steps.

### FLUX.2 Klein artistic editing

- An optional, isolated Diffusers worker can edit a selected text box, a painted area, or a whole page with FLUX.2 Klein 4B or 9B.
- The editor uses the model's native BF16 precision with configurable GPU or CPU-offload policies.
- Compatible LoRAs can be selected from `models/loras/flux2-klein`, with per-edit strength control and model-size validation.
- **Use selected translation** can build an **Artistic lettering** prompt, or an **Empty speech bubble** prompt for placing editable text afterward.
- **Context padding** supplies surrounding artwork to the model; **Edit margin** expands the area allowed to change around a selected box so longer translated lettering is not clipped to its original bounds. Review the preview before applying an expanded edit.
- Generated results appear in a Before/After preview and are applied as non-destructive, undoable project patches.
- The FLUX worker is persistent, while model or device-policy changes intentionally reload it.
- CPU-offload policies expose the physical GPU index, which is useful when another local model already occupies one card.
- An optional **Black Forest Labs API** backend supports hosted Klein 4B and 9B without a local FLUX environment or GPU. Configure its key in **Settings > Provider APIs**, then select the backend at the top of Artistic Edit. **Local Diffusers** remains the default each launch.
- Cloud edits share the same selection, context, edit margin, preview, Apply, and undo workflow. Each generation (including regeneration) asks permission to upload the selected area plus context, or the whole page when selected. A preview is already a paid generation, even if discarded.
- Local LoRAs, steps, guidance, and GPU controls are unavailable for this hosted backend. Cloud processing is capped below 4 megapixels; outputs are validated before being restored to page coordinates and confined to the edit mask.
- Cancel stops waiting for a cloud result; it cannot guarantee cancellation of the provider's job or charges. Failed or uncertain submissions are never automatically resubmitted. After a timeout, check the provider dashboard before retrying. Content restrictions can differ from local models.

API reference: [BFL image editing](https://docs.bfl.ai/flux_2/flux2_image_editing). Cloud integration has automated mocked transport/controller coverage, and live Flux and LLM API requests have been confirmed during development. This is not a guarantee for every provider or model; account access, content policies, and charges still depend on your chosen service.

### Desktop appearance, languages, and fork isolation

- Six themes: **Dark**, **Light**, **Midnight**, **Parchment**, **Lavender**, and **Mint**. Artistic Edit buttons have theme-aware backgrounds, readable labels, and visible borders.
- Updated startup splash and fork branding.
- The interface offers English, Korean, French, Simplified Chinese, Russian, Japanese, German, Spanish, and Italian. New-feature translations are included in all nine Qt catalogs, including the additional Turkish catalog (not currently offered by the language selector). Coverage checks validate placeholders and compiled catalogs; this is separate from the older translated README files.
- The fork uses its own settings, application-data paths, and application identity to avoid interference from a separately installed upstream version. Legacy data migration is read-only with respect to the old profile.
- Automatic startup update checks are disabled, and the account/subscription sales UI is removed. Manual **Check for Updates** now targets this fork's GitHub releases rather than upstream.

### GPU and reliability work

- Windows dependencies include ONNX Runtime GPU with matching CUDA 13/cuDNN 9 runtime packages; TensorRT is no longer probed unless explicitly requested.
- CUDA failure cleanly falls back to CPU instead of requiring TensorRT.
- Fixes cover asynchronous page navigation, project patch identity and persistence, repeated inpainting composition, and missing render updates.
- A regression suite covers local LLM responses, detection labels, automatic/manual translation, text editing, image sorting, inpainting geometry and undo, translation caching, and device selection.

## Recommended workflow

Automatic translation remains available, but careful comic restoration usually benefits from processing one page at a time:

1. Load the comic and use **Sort** if the pages are not in reading order.
2. Select **Manual** mode, set the source and target languages, and click **Detect**. Use **Auto** for the source when the work genuinely mixes languages or its language is unknown.
3. Review the regions before processing: delete detections that belong to artwork and draw boxes around any missed dialogue or captions.
4. Click **Recognize**, then correct the source text if OCR made a mistake.
5. Click **Translate** and review or edit the target text.
6. Click **Segment**, inspect the proposed cleanup areas, then click **Clean**.
7. Click **Render** and adjust each text item for font, size, alignment, spacing, and emphasis.
8. Use the cleanup brush and **Apply Cleanup** for any remaining marks. Undo/Redo or **Revert All Inpainting on the Current Page** is available if cleanup damages the art.

You can also draw a missed region and right-click **OCR**, then **Translate**, without detecting the rest of the page first. Its completed translation remains intact if you subsequently run the whole-page manual workflow. For colored or textured backgrounds, try the sampled-color fill, restore brush, or clone brush before repeatedly inpainting the same area. Use Artistic Edit for lettering that should become part of the artwork rather than a normal editable text layer.

The detector intentionally keeps both speech-bubble text and free-floating text. This prevents manual mode from losing valid OCR regions, but it also means **Translate All** can inpaint artistic titles, signs, sound effects, or other lettering that is part of the art. Manual review before cleaning is the recommended approach when preserving artwork matters.

## Installation

### Requirements

- Python 3.12
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A local `llama-server` executable and a compatible GGUF model if you want to use managed local translation
- WinRAR or 7-Zip on `PATH` when opening CBR archives

### Run from source

```bash
git clone https://github.com/biogoly/local-comic-translate.git
cd local-comic-translate
uv init --python 3.12
uv add -r requirements.txt --compile-bytecode
uv run comic.py
```

On Windows, `run.bat` is also provided after the initial dependency installation:

```bat
run.bat
```

### Optional FLUX.2 Klein runtime

The **Local Diffusers** backend uses a separate Python environment so its newer PyTorch, Diffusers, and Transformers packages do not disturb the main application environment. The BFL API backend does not require this optional environment. From the repository root, create a Python 3.12 environment and install the pinned optional requirements:

```bat
py -3.12 -m venv .venv-flux
.venv-flux\Scripts\python.exe -m pip install -r requirements-flux.txt
```

In **Artistic Edit**, choose **Select FLUX Python…** and select `.venv-flux\Scripts\python.exe`. Model files must already be present in the Hugging Face cache because the integrated worker runs cache-only by default. The fork's profile isolation does not create a separate Hugging Face cache or duplicate user-selected model paths.

Place optional `.safetensors` LoRAs in `models/loras/flux2-klein/4b/` or `models/loras/flux2-klein/9b/` as appropriate. See the [LoRA folder guide](models/loras/flux2-klein/README.md) for compatibility and optional metadata. Weights are not bundled with the repository. The current local worker uses BF16; experimental bitsandbytes INT8/NF4 loading is not enabled.

To update an existing checkout:

```bash
git pull
uv add -r requirements.txt --compile-bytecode
```

### Configure a local LLM

`llama.cpp` itself is an external runtime and is not installed through `requirements.txt` or `uv`. Download or build [`llama.cpp`](https://github.com/ggml-org/llama.cpp), then obtain a chat/instruction GGUF model that fits your hardware.

The backend is model-agnostic; Gemma 4 is the primary model used during the fork's current development and testing.

In the application:

1. Open **Settings > Tools** and select **Local LLM** as the translator.
2. Open **Settings > LLMs**.
3. Choose one of the following runtimes:
   - **Managed llama.cpp**: select `llama-server` (optional if it is already on `PATH`), the main GGUF model, and the matching vision projector GGUF if the model requires one.
   - **External server**: enter the OpenAI-compatible base URL and exact model name exposed by the server. The default URL, `http://127.0.0.1:11434/v1`, targets Ollama.
4. Enable **Provide Image as Input to AI** only if the selected model and server support vision. In managed mode, configure the required `mmproj` file first.

The conservative translation defaults are Temperature `0.20`, Top P `0.90`, and Top K `40`. They are good starting points for translation; adjust them only if your model's own documentation recommends different sampling values. Set Top K to `0` to disable it.

### Switch between local and API backends

For text translation:

1. Open **Settings > Provider APIs**, choose the provider, enter its key, and select or type the model ID. Enable **Save Keys** only if you want the key retained between sessions.
2. Open **Settings > Tools > Translator** and choose that provider. The change applies to subsequent requests without restarting.
3. Select **Local LLM** again to return to the saved llama.cpp/external-server configuration. There is no need to replace the local endpoint or erase its model settings.

For artistic editing, enter the **Black Forest Labs** key in **Provider APIs**, then change the top dropdown in **Artistic Edit** from **Local Diffusers** to **Black Forest Labs API**. Choose 4B/9B and use **Generate Preview**. Switching back restores access to local controls. This choice is independent of the text translator.

When testing a different provider, use an untranslated page or region: existing completed manual translations are intentionally preserved, and cached results can avoid a new request. Cloud requests send data to the chosen provider and may incur charges; a discarded artistic preview is still a generation.

### GPU acceleration

There are two independent GPU settings:

- **Settings > Tools > Use GPU** controls supported ONNX/PyTorch work such as detection, OCR, and inpainting. On Windows, the dependency set installs ONNX Runtime GPU and matching CUDA 13/cuDNN 9 runtime packages. A compatible NVIDIA driver is still required, but a separate CUDA Toolkit or TensorRT installation is not.
- **Settings > LLMs > GPU layers** controls how many layers managed `llama.cpp` tries to offload. External servers manage their own GPU settings.

CPU fallback remains available. GPU speedups vary by model, page size, and which pipeline stage is the bottleneck; a full page workflow may not scale in proportion to raw GPU utilization.

## Usage tips

- `Ctrl` + mouse wheel: zoom
- `Ctrl` + `=`: zoom in
- `Ctrl` + `-`: zoom out
- `Ctrl` + `0`: fit page to the window
- Arrow Left/Right: move between pages
- Standard trackpad gestures work in the image viewer
- Make sure the selected font supports the target language
- Page sort options are **Name: A to Z**, **Name: Z to A**, **Modified: Oldest First**, and **Modified: Newest First**. Filename sorting is natural, so `page2` comes before `page10`.
- If a CBR file raises `RarCannotExec("Cannot find working tool")`, add the WinRAR or 7-Zip installation folder to `PATH`.

## How it works

### Speech-bubble detection and text segmentation

The application uses the upstream [bubble-and-text-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector), an RT-DETR-v2 model trained on manga, webtoons, and Western comics. Detection results remain editable before OCR, segmentation, or cleanup.

<img src="https://i.imgur.com/TlzVH3j.jpg" width="49%" alt="Detected comic text"> <img src="https://i.imgur.com/h18XrYT.jpg" width="49%" alt="Segmented comic text">

### OCR

The default OCR backends are:

- [manga-ocr](https://github.com/kha-white/manga-ocr) for Japanese
- [Pororo](https://github.com/yunwoong7/korean_ocr_using_pororo) for Korean
- [PPOCRv5](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html) for other supported languages

Gemini and Microsoft Azure Vision remain available as optional direct-provider OCR integrations using your own credentials.

### Translation

Translation uses either the local OpenAI-compatible path or the direct, user-key provider integrations described above; it does not route through the upstream paid service. Page-wide requests group eligible recognized regions for context. Completed manual regions are preserved, and selected-region requests may use cached results or a narrower request. Compatible multimodal models can optionally receive the page image as well.

### Inpainting and artistic editing

The local cleanup backends remain LaMa, MI-GAN, and AOT. The separate FLUX.2 Klein Artistic Edit tool handles generative changes such as translated artistic lettering, rebuilding or resizing a text bubble, and localized artwork repair. It is a supervised preview/apply workflow rather than an automatic replacement for cleanup.

<img src="https://i.imgur.com/cVVGVXp.jpg" width="49%" alt="Comic before inpainting"> <img src="https://i.imgur.com/bLkPyqG.jpg" width="49%" alt="Comic after inpainting">

### Text rendering

Translated or manually entered text is wrapped inside editable regions. Font, size, alignment, spacing, color, outline, direction, and character-level emphasis can be adjusted before export.

## Tests

Run the regression suite from the repository root:

```bash
uv run python -m unittest discover -s tests -v
```

Check GUI translation coverage and compiled catalogs separately:

```bash
uv run python resources/translations/check_translations.py
```

The release checks include manual-region OCR and preservation, local/API provider routing, artistic-edit contracts and previews, project patch persistence, cleanup/clone undo, themes, language persistence, and fork-profile isolation. Most model and network operations are mocked in the regression suite; they do not download weights or spend API credits.

## Current limitations

- This is an early source-only fork; broad testing across languages, models, layouts, and operating systems is still in progress.
- Automatic mode cannot always distinguish dialogue from lettering that belongs to the artwork. Use Manual mode when preservation is important.
- Detection, OCR, translation, segmentation, and inpainting can all require human correction on difficult pages.
- Vision input requires a genuinely multimodal model, a compatible server, and (when applicable) the matching projector file.
- Local generation quality and speed depend heavily on model choice, context size, and available RAM/VRAM.
- Artistic Edit currently requires single-page mode; it is not a book-wide automatic stage or a webtoon multi-page editor. Generated lettering must be checked for spelling and unwanted artwork changes before applying it.
- The other-language README files have not yet been updated with fork-specific documentation.

## Comic samples from the upstream project

These examples show Comic Translate's original hosted GPT workflow. Some titles also have official English translations.

- [The Wretched of the High Seas](https://www.drakoo.fr/bd/drakoo/les_damnes_du_grand_large/les_damnes_du_grand_large_-_histoire_complete/9782382330128)
- [Journey to the West](https://ac.qq.com/Comic/comicInfo/id/541812)
- [The Wormworld Saga](https://wormworldsaga.com/index.php)
- [Frieren: Beyond Journey's End](https://renta.papy.co.jp/renta/sc/frm/item/220775/title/742932/)
- [Days of Sand](https://9ekunst.nl/2021/05/20/nieuw-album-van-aimee-de-jongh-is-benauwd-als-een-zandstorm/)
- [Player (OH Hyeon-Jun)](https://comic.naver.com/webtoon/list?titleId=745876&page=1&sort=ASC&tab=fri)
- [Carbon & Silicon](https://www.amazon.com/Carbone-Silicium-French-Mathieu-Bablet-ebook/dp/B0C1LTGZ85/)

## Acknowledgements

This work is a fork of [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate). The original application, models, interface, and integrations are the foundation for the changes documented here.

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [lama-cleaner](https://github.com/Sanster/lama-cleaner)
- [dreMaz/AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting)
- [Pororo Korean OCR](https://github.com/yunwoong7/korean_ocr_using_pororo)
- [manga-ocr](https://github.com/kha-white/manga-ocr)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [dayu_widgets](https://github.com/phenom-films/dayu_widgets)

Licensed under the [Apache License 2.0](LICENSE).
