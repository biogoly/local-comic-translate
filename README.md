# Local Comic Translate

English | [한국어](docs/README_ko.md) | [Français](docs/README_fr.md) | [简体中文](docs/README_zh-CN.md)

<img src="https://i.imgur.com/QUVK6mK.png" alt="Comic Translate interface">

Local Comic Translate is a community fork of [Comic Translate](https://github.com/ogkalu2/comic-translate) focused on private, local translation and a safer hands-on desktop editing workflow. It keeps the upstream application's detection, OCR, inpainting, rendering, project, archive, and webtoon features while adding a managed `llama.cpp` translator and substantial editing and reliability improvements.

The inherited OCR pipeline supports English, Korean, Japanese, French, Simplified and Traditional Chinese, Russian, German, Dutch, Spanish, and Italian as source languages, with those and additional languages available as translation targets.

This fork is under active development. The changes described below currently require running from source; the downloads on the upstream Comic Translate website do not include them. The translated README files also still describe the upstream project and may not yet reflect the fork-specific features.

## Highlights of this fork

### Local first, private LLM translation

- A **Local LLM** translator that does not require a hosted translation API or Comic Translate account.
- **Managed llama.cpp** mode launches a local `llama-server` process, chooses an unused loopback port, reuses the process between requests, and stops it when the app exits.
- **External server** mode supports OpenAI-compatible `/v1` endpoints such as Ollama, LM Studio, or a separately managed `llama-server`.
- Configurable GGUF model, optional multimodal projector (`mmproj`), context size, GPU layers, startup/request timeouts, maximum output tokens, temperature, Top P, and Top K.
- Optional page-image input for vision-capable models. The existing OCR remains the primary text source; the image gives a multimodal model additional context and a chance to correct obvious OCR mistakes.
- Strict block-ID validation and automatic retries prevent incomplete or malformed model output from silently shifting translations into the wrong bubbles.
- Page-level translation caching uses the original source image, so later inpainting does not unnecessarily invalidate a cached translation.

### Safer manual editing

- A practical page-by-page workflow for reviewing detected regions before anything is erased.
- Zoom In, Zoom Out, and Fit controls, plus configurable keyboard shortcuts.
- Natural page sorting by filename and sorting by modification time.
- Borderless replacement text appears immediately when a new text region is drawn and text is entered.
- Deleting and retyping text preserves the chosen font.
- Bold, italic, and underline can be applied to selected words instead of only to the entire text item.
- Full-page manual translation creates any missing text layers. Translating a selected, not-yet-rendered region cleans it before creating its translated text layer.

### Inpainting and undo improvements

- Every manual cleanup pass is undoable and redoable with the normal Undo/Redo controls.
- **Revert All Inpainting on the Current Page** removes all saved cleanup patches as one undoable action.
- Repeated cleanup starts from the currently composited page, preventing previously removed text from reappearing in later patches.
- Manual brush cleanup handles single clicks as well as strokes and uses integer-safe image bounds.
- Geometry-aware fallback masks distinguish rectangular caption boxes from round balloons, reducing leftover text in square corners while remaining conservative around artwork.
- Automatic page edits are grouped into predictable undo steps.

### GPU and reliability work

- Windows dependencies include ONNX Runtime GPU with matching CUDA 13/cuDNN 9 runtime packages; TensorRT is no longer probed unless explicitly requested.
- CUDA failure cleanly falls back to CPU instead of requiring TensorRT.
- Fixes cover asynchronous page navigation, project patch identity and persistence, repeated inpainting composition, and missing render updates.
- A regression suite covers local LLM responses, detection labels, automatic/manual translation, text editing, image sorting, inpainting geometry and undo, translation caching, and device selection.

## Recommended workflow

Automatic translation remains available, but careful comic restoration usually benefits from processing one page at a time:

1. Load the comic and use **Sort** if the pages are not in reading order.
2. Select **Manual** mode and click **Detect**.
3. Review the regions before processing: delete detections that belong to artwork and draw boxes around any missed dialogue or captions.
4. Click **Recognize**, then correct the source text if OCR made a mistake.
5. Click **Translate** and review or edit the target text.
6. Click **Segment**, inspect the proposed cleanup areas, then click **Clean**.
7. Click **Render** and adjust each text item for font, size, alignment, spacing, and emphasis.
8. Use the cleanup brush and **Apply Cleanup** for any remaining marks. Undo/Redo or **Revert All Inpainting on the Current Page** is available if cleanup damages the art.

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

Gemini and Microsoft Azure Vision remain available as optional upstream OCR integrations.

### Translation

The upstream hosted LLM integrations remain available. This fork additionally provides the local OpenAI-compatible path described above. All LLM translators receive the page's recognized text together for context; compatible multimodal models can optionally receive the page image as well.

### Inpainting

The current local inpainting backends remain LaMa, MI-GAN, and AOT. The fork improves how cleanup masks and patches are applied, edited, persisted, and reversed; it does not yet replace these backends with a generative image-edit model.

<img src="https://i.imgur.com/cVVGVXp.jpg" width="49%" alt="Comic before inpainting"> <img src="https://i.imgur.com/bLkPyqG.jpg" width="49%" alt="Comic after inpainting">

### Text rendering

Translated or manually entered text is wrapped inside editable regions. Font, size, alignment, spacing, color, outline, direction, and character-level emphasis can be adjusted before export.

## Tests

Run the regression suite from the repository root:

```bash
uv run python -m unittest discover -s tests -v
```

## Current limitations

- This is an early source-only fork; broad testing across languages, models, layouts, and operating systems is still in progress.
- Automatic mode cannot always distinguish dialogue from lettering that belongs to the artwork. Use Manual mode when preservation is important.
- Detection, OCR, translation, segmentation, and inpainting can all require human correction on difficult pages.
- Vision input requires a genuinely multimodal model, a compatible server, and (when applicable) the matching projector file.
- Local generation quality and speed depend heavily on model choice, quantization, context size, and available RAM/VRAM.
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
