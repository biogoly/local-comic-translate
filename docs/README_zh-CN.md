# Local Comic Translate

[English](../README.md) | [한국어](README_ko.md) | [Français](README_fr.md) | 简体中文 | [Español](README_es.md)

<img width="1641" height="1001" alt="Comic Translate 界面" src="https://github.com/user-attachments/assets/4cc8c044-b8b4-4773-8c6d-ad7a84cdaca1" />

Local Comic Translate 是 [Comic Translate](https://github.com/ogkalu2/comic-translate) 的社区分支，专注于保护隐私的本地翻译，以及更安全、便于手动操作的桌面编辑流程。它保留了上游应用的检测、OCR、图像修复、渲染、项目、压缩包和条漫功能，同时加入了托管式 `llama.cpp` 翻译、FLUX.2 Klein 艺术编辑、可选的服务商直连 API，并大幅改进了编辑体验和可靠性。无需上游账号、订阅或翻译点数。

沿用的 OCR 流程支持英语、韩语、日语、法语、简体中文、繁体中文、俄语、德语、荷兰语、西班牙语和意大利语作为源语言；翻译目标语言包含上述语言以及更多语言。

此分支正在积极开发中。下文介绍的改动目前需要从源码运行；上游 Comic Translate 网站提供的下载版本不包含这些改动。上方的语言链接提供了本分支当前文档的译文。这些指南中的设置和按钮名称保留为英文，便于在不同界面语言中对应查找。

## 最新更新 — 0.1.0

- **Artistic Edit:** 使用本地 FLUX.2 Klein 4B/9B 及兼容的 LoRA，或使用 Black Forest Labs API；采用先预览再应用的流程，支持撤销，并随项目保存。
- **翻译后端自由选择：** 使用自己的 API 密钥切换到 OpenAI、Gemini、Claude 或 Deepseek，同时保留本地 llama-server 设置。
- **更完善的手动流程：** 可直接对手绘区域运行 OCR；已完成的手动翻译及其格式会在后续整页处理中保留；源语言和目标语言的选择会在切换页面时保持不变。
- **局部清理：** 使用画笔恢复原始像素，克隆固定采样区域的颜色与纹理，或用采样的背景色填充蒙版。
- **桌面体验改进：** 在 Light/Dark 之外新增 Midnight、Parchment、Lavender 和 Mint 主题，改进 Artistic Edit 控件的清晰度，更新启动画面插图及 GUI 语言目录。
- **独立的分支标识：** 使用独立的设置和应用数据，移除了上游更新提示及付费服务路由。

## 本分支的主要特点

### 本地优先、保护隐私的 LLM 翻译

- **Local LLM** 翻译器无需托管翻译 API，也无需 Comic Translate 账号。
- **Managed llama.cpp** 模式会启动本地 `llama-server` 进程，选择一个空闲的回环端口，在多次请求之间复用进程，并在应用退出时停止进程。
- **External server** 模式支持兼容 OpenAI 的 `/v1` 端点，例如 Ollama、LM Studio 或单独管理的 `llama-server`。
- 可配置 GGUF 模型、可选的多模态投影器（`mmproj`）、上下文大小、GPU 层数、启动及请求超时、最大输出 token 数、temperature、Top P 和 Top K。
- 可向支持视觉的模型提供页面图像。现有 OCR 仍是主要文本来源；图像为多模态模型提供额外上下文，并让模型有机会纠正明显的 OCR 错误。
- 严格的文本块 ID 校验和自动重试，可防止不完整或格式错误的模型输出在未提示的情况下将译文错放到其他气泡中。
- 页面级翻译缓存以原始源图像为依据，因此后续图像修复不会无谓地使已缓存的译文失效。

### 可选的服务商直连 API

- 默认仍使用 **Local LLM**。翻译和 OCR 均不需要、也不会使用上游 Comic Translate 登录、订阅或点数购买服务。
- 在 **Settings > Tools > Translator** 中选择 **OpenAI**、**Google Gemini**、**Anthropic Claude** 或 **Deepseek**，即可直接使用云服务商。
- 在 **Settings > Provider APIs** 中选择对应服务商，输入自己的 API 密钥，并选择或输入其 API 模型 ID。服务商会直接向你的账号计费。预设仅供起步参考，并非全部可用模型的完整列表。
- 现有的可选云端 OCR 也可以配置 Microsoft Azure OCR 凭据及 Gemini 密钥。默认 OCR 仍在本地运行。
- **Save Keys** 需要主动启用。密钥存储在本分支的本地应用设置中（并非加密保险库），不会写入项目文件。关闭此选项并保存或退出应用后，即可移除持久化保存的服务商密钥；模型选择仍会保留。
- 云端翻译会发送识别出的文本；若启用了 **Provide Image as Input to AI** 且模型支持图像输入，还会发送页面图像。使用时适用服务商的政策和收费标准。本地失败后不会自动回退到云端。
- 旧的 **Advanced / Custom** 配置已移除。已保存且处于启用状态的 Custom 端点会迁移至 **Local LLM > External server**。现有的具名云模型选择会迁移至对应的直连服务商，并保留模型选择；用户需要提供自己的密钥。
- 服务商凭据与本地服务器配置相互独立。修改 **Tools > Translator** 会切换当前使用的后端，而不会清空任一配置；**Provider APIs** 下的服务商下拉框仅用于选择要编辑哪家服务商的凭据。

### 更安全的手动编辑

- 提供实用的逐页处理流程，便于在擦除任何内容之前检查检测区域。
- 在 Manual 模式下切换页面时，源语言和目标语言的选择都会保留。明确选择 **Auto** 作为源语言后，该选择也会持续生效，直到再次修改。
- 无需先对整页运行 Detect，即可右键选择 **OCR**，识别手绘区域。本地 OCR 会补齐缺失的文本行边界；明确对某一区域执行 OCR 时，会重新识别该区域，而不会返回旧缓存结果或处理无关文本框。
- 整页 Detect 会保留手绘区域，并抑制对同一文本的重复检测。后续 Recognize/Translate 操作会保留这些区域中已完成的原文和译文；Segment 和 Render 会保留已有文本图层及词语级格式，而不会删除或重复创建它们。
- 保留手动文本图层并不会跳过背景清理：其所在区域仍会参与分割和清理。尚未翻译的手动区域继续按正常流程处理。
- 提供 Zoom In、Zoom Out 和 Fit 控件，以及可配置的键盘快捷键。
- 支持按文件名自然排序，以及按修改时间排序页面。
- 绘制新的文本区域并输入文字后，无边框的替换文字会立即显示。
- 删除并重新输入文字时会保留选定的字体。
- 可对选中的词语应用粗体、斜体和下划线，而不局限于整个文本对象。
- 行内格式与文本框移动、缩放功能可同时使用；焦点变化时，手动输入的译文会保留，日语输入法输入的文字也不例外。
- 手动整页翻译会创建所有缺失的文本图层。翻译选中的、尚未渲染的区域时，会先清理该区域，再创建译文文本图层。

### 图像修复与撤销改进

- 每次手动清理都可以通过常规 Undo/Redo 控件撤销和重做。
- **Restore brush** 仅恢复画笔覆盖范围内的原始页面像素，其他位置的编辑保持不变。原有文字也会恢复；此操作不会移除可编辑的叠加文本。使用现有画笔滑块调整大小。每一笔都可以撤销。
- **Clone brush**（印章图标）：设置画笔大小，按住 Alt 点击一处干净区域，然后涂画以重复使用该固定样本中的颜色与纹理。青色虚线圆停留在采样源位置；黑白实线轮廓跟随目标位置，并随缩放比例变化。**Soft edge** 控制边缘融合程度。默认启用 **Sample original**，以采样清理补丁下方的原始图像；关闭后则复制已编辑的页面。修改画笔大小、切换页面或更改采样源模式后，需要重新按住 Alt 点击采样。每一笔都可以撤销，且只改变涂画到的像素。此工具不使用 AI 模型。
- 先使用 **Pick color**，再使用 **Fill mask with sampled color**，即可用完好的背景色填充当前页面上涂画或分割出的区域。此操作直接使用可见蒙版，不额外扩张，也不进行模型推理。适用于纯色气泡；渐变色及带纹理的画面仍需图像修复或艺术编辑。撤销会同时恢复原有像素和蒙版。
- **Revert All Inpainting on the Current Page** 会以一次可撤销的操作移除所有已保存的清理补丁。
- 重复清理会从当前已合成的页面开始，避免已移除的文字在后续补丁中重新出现。
- 手动画笔清理同时支持单击和连续笔画，并使用安全的整数图像边界计算。
- 考虑区域几何形状的后备蒙版可区分矩形说明框与圆形气泡，在谨慎保护画面的同时，减少矩形角落残留的文字。
- 自动页面编辑会分组为行为明确的撤销步骤。

### FLUX.2 Klein 艺术编辑

- 可选的独立 Diffusers 工作进程可使用 FLUX.2 Klein 4B 或 9B，编辑选中的文本框、涂画区域或整页图像。
- 编辑器使用模型原生的 BF16 精度，并可配置 GPU 或 CPU 卸载策略。
- 可从 `models/loras/flux2-klein` 中选择兼容的 LoRA，每次编辑均可控制强度，并会验证其与模型规格是否匹配。
- **Use selected translation** 可以构建 **Artistic lettering** 提示词，或构建 **Empty speech bubble** 提示词，以便随后放置可编辑文本。
- **Context padding** 向模型提供周边画面作为上下文；**Edit margin** 扩大选中文本框周围允许修改的区域，避免较长的翻译文字被原有边界裁切。应用扩大范围的编辑之前，请先检查预览。
- 生成结果会显示在 Before/After 对比预览中，并以非破坏性、可撤销的项目补丁形式应用。
- FLUX 工作进程会持续运行；更改模型或设备策略时，会有意重新加载该进程。
- CPU 卸载策略可指定物理 GPU 索引，当另一个本地模型已占用某张显卡时尤其有用。
- 可选的 **Black Forest Labs API** 后端支持托管 Klein 4B 和 9B，无需本地 FLUX 环境或 GPU。在 **Settings > Provider APIs** 中配置密钥，然后在 Artistic Edit 顶部选择此后端。每次启动时，默认后端仍为 **Local Diffusers**。
- 云端编辑使用相同的选区、上下文、编辑边距、预览、Apply 和撤销流程。每次生成（包括重新生成）都会征求许可，以便上传选定区域及其上下文，或在选择整页时上传整页。即使丢弃预览，预览本身也已是一次付费生成。
- 此托管后端不提供本地 LoRA、步数、引导强度和 GPU 控件。云端处理的图像大小限制在 400 万像素以下；输出经过验证后才会恢复到页面坐标，并被限制在编辑蒙版范围内。
- Cancel 会停止等待云端结果，但无法保证取消服务商的任务或费用。失败或状态不确定的提交绝不会自动重新提交。超时后，请先检查服务商控制台，再决定是否重试。云端内容限制可能与本地模型不同。

API 参考：[BFL 图像编辑](https://docs.bfl.ai/flux_2/flux2_image_editing)。云端集成具备使用模拟对象的传输层与控制器自动化测试覆盖，开发期间也已验证真实的 Flux 和 LLM API 请求。这并不保证所有服务商或模型都可用；账号访问权限、内容政策和收费仍取决于所选服务。

艺术编辑示例 1

<img width="3261" height="1980" alt="艺术编辑示例 1" src="https://github.com/user-attachments/assets/32da6bac-01ef-4ed4-a5f0-5fd08772ceec" />

艺术编辑示例 2

<img width="3271" height="2069" alt="艺术编辑示例 2" src="https://github.com/user-attachments/assets/ba29d9fd-8c12-4d2c-8c8f-8854104e088e" />

艺术编辑示例 3

<img width="3273" height="1990" alt="艺术编辑示例 3" src="https://github.com/user-attachments/assets/99f4f497-87b1-405e-a5d4-06dec5b45840" />

### 桌面外观、语言与分支隔离

- 提供六种主题：**Dark**、**Light**、**Midnight**、**Parchment**、**Lavender** 和 **Mint**。Artistic Edit 按钮的背景会适应主题，文字清晰可读，边框易于辨认。
- 更新了启动画面和分支品牌标识。
- 界面提供英语、韩语、法语、简体中文、俄语、日语、德语、西班牙语和意大利语。全部九个 Qt 翻译目录均包含新功能译文，其中还包括额外的土耳其语目录（目前语言选择器尚未提供土耳其语）。覆盖检查会验证占位符和编译后的翻译目录；界面语言与上方链接的 README 译文相互独立。
- 本分支使用自己的设置、应用数据路径和应用标识，以避免与另行安装的上游版本相互干扰。迁移旧数据时，对旧配置文件仅进行读取。
- 已禁用启动时自动检查更新，并移除账号及订阅销售界面。手动执行 **Check for Updates** 时，现在会检查本分支的 GitHub 发布版本，而不是上游版本。

### GPU 与可靠性改进

- Windows 依赖包含 ONNX Runtime GPU 及配套的 CUDA 13/cuDNN 9 运行时软件包；除非明确要求，否则不再探测 TensorRT。
- CUDA 失败时会平稳回退到 CPU，无需依赖 TensorRT。
- 修复涉及异步页面导航、项目补丁标识与持久化、重复图像修复时的合成，以及缺失的渲染更新。
- 回归测试套件覆盖本地 LLM 响应、检测标签、自动及手动翻译、文本编辑、图像排序、图像修复几何与撤销、翻译缓存，以及设备选择。

## 推荐工作流程

自动翻译仍然可用，但如果希望仔细修整漫画，通常适合逐页处理：

1. 载入漫画；若页面未按阅读顺序排列，使用 **Sort** 排序。
2. 选择 **Manual** 模式，设置源语言和目标语言，然后点击 **Detect**。当作品确实混用多种语言，或无法确定语言时，可将源语言设为 **Auto**。
3. 处理前检查区域：删除属于画面内容的误检框，并为漏检的对白或说明文字绘制文本框。
4. 点击 **Recognize**；若 OCR 有误，修正原文。
5. 点击 **Translate**，检查或编辑译文。
6. 点击 **Segment**，检查拟清理的区域，再点击 **Clean**。
7. 点击 **Render**，为各文本对象调整字体、大小、对齐、间距和强调格式。
8. 使用清理画笔及 **Apply Cleanup** 去除剩余痕迹。若清理损伤了画面，可使用 Undo/Redo 或 **Revert All Inpainting on the Current Page**。

也可以为漏检文字绘制区域，右键选择 **OCR**，再选择 **Translate**，无需先检测页面其余部分。之后即使运行整页手动流程，已完成的译文也会保持不变。对于彩色或带纹理的背景，在反复修复同一区域之前，可以先尝试采样颜色填充、恢复画笔或克隆画笔。对于应融入画面本身的艺术字，可使用 Artistic Edit，而不是普通的可编辑文本图层。

检测器有意保留气泡内文字和气泡外的独立文字。这样可以避免手动模式丢失有效的 OCR 区域，但也意味着 **Translate All** 可能对艺术标题、招牌、拟声词或其他属于画面的文字进行图像修复。如果重视保留原画，建议在清理前进行手动检查。

## 安装

### 环境要求

- Python 3.12
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- 若要使用托管式本地翻译，需要本地 `llama-server` 可执行文件及兼容的 GGUF 模型
- 打开 CBR 压缩包时，需要将 WinRAR 或 7-Zip 添加到 `PATH`

### 从源码运行

```bash
git clone https://github.com/biogoly/local-comic-translate.git
cd local-comic-translate
uv init --python 3.12
uv add -r requirements.txt --compile-bytecode
uv run comic.py
```

在 Windows 上，首次安装依赖后，也可以使用提供的 `run.bat`：

```bat
run.bat
```

### 可选的 FLUX.2 Klein 运行环境

**Local Diffusers** 后端使用独立的 Python 环境，以免其较新的 PyTorch、Diffusers 和 Transformers 软件包影响主应用环境。BFL API 后端不需要此可选环境。在仓库根目录下创建 Python 3.12 环境，并安装版本已固定的可选依赖：

```bat
py -3.12 -m venv .venv-flux
.venv-flux\Scripts\python.exe -m pip install -r requirements-flux.txt
```

在 **Artistic Edit** 中选择 **Select FLUX Python…**，并选取 `.venv-flux\Scripts\python.exe`。模型文件必须预先存在于 Hugging Face 缓存中，因为集成的工作进程默认仅从缓存加载。本分支的配置隔离不会创建独立的 Hugging Face 缓存，也不会复制用户选定路径中的模型。

将可选的 `.safetensors` LoRA 按对应规格放入 `models/loras/flux2-klein/4b/` 或 `models/loras/flux2-klein/9b/`。兼容性和可选元数据说明见 [LoRA 文件夹指南](../models/loras/flux2-klein/README.md)。仓库不附带权重。当前本地工作进程使用 BF16；实验性的 bitsandbytes INT8/NF4 加载功能尚未启用。

更新已有的本地仓库：

```bash
git pull
uv add -r requirements.txt --compile-bytecode
```

### 配置本地 LLM

`llama.cpp` 本身是外部运行时，不会通过 `requirements.txt` 或 `uv` 安装。下载或编译 [`llama.cpp`](https://github.com/ggml-org/llama.cpp)，再获取适合你的硬件的对话或指令型 GGUF 模型。

此后端不限定具体模型；本分支当前开发和测试主要使用 Gemma 4。

在应用中：

1. 打开 **Settings > Tools**，将翻译器设为 **Local LLM**。
2. 打开 **Settings > LLMs**。
3. 选择以下运行方式之一：
   - **Managed llama.cpp**：选择 `llama-server`（若已在 `PATH` 中则可不选）、主 GGUF 模型，以及模型需要时所对应的视觉投影器 GGUF。
   - **External server**：输入兼容 OpenAI 的基础 URL，以及服务器提供的准确模型名称。默认 URL `http://127.0.0.1:11434/v1` 指向 Ollama。
4. 仅在所选模型和服务器支持视觉时启用 **Provide Image as Input to AI**。在托管模式下，需先配置所需的 `mmproj` 文件。

较保守的翻译默认值为 Temperature `0.20`、Top P `0.90` 和 Top K `40`。这些值适合作为翻译的起点；仅当模型自身文档建议采用其他采样值时再进行调整。将 Top K 设为 `0` 可禁用它。

### 在本地与 API 后端之间切换

文字翻译的设置方式：

1. 打开 **Settings > Provider APIs**，选择服务商，输入密钥，并选择或输入模型 ID。只有希望在下次启动时保留密钥，才启用 **Save Keys**。
2. 打开 **Settings > Tools > Translator**，选择该服务商。此更改会应用于后续请求，无需重启。
3. 再次选择 **Local LLM** 即可回到已保存的 llama.cpp 或外部服务器配置，无需替换本地端点或清除模型设置。

对于艺术编辑，在 **Provider APIs** 中输入 **Black Forest Labs** 密钥，然后将 **Artistic Edit** 顶部下拉框从 **Local Diffusers** 改为 **Black Forest Labs API**。选择 4B/9B，再使用 **Generate Preview**。切回本地后，即可重新使用本地控件。此选择与文字翻译器相互独立。

测试其他服务商时，请使用尚未翻译的页面或区域：已完成的手动译文会有意保留，而缓存结果也可能使应用无需发出新请求。云端请求会将数据发送给所选服务商，并可能产生费用；丢弃的艺术编辑预览仍算一次生成。

### GPU 加速

有两项相互独立的 GPU 设置：

- **Settings > Tools > Use GPU** 控制检测、OCR 和图像修复等受支持的 ONNX/PyTorch 任务。在 Windows 上，依赖集合会安装 ONNX Runtime GPU 及配套的 CUDA 13/cuDNN 9 运行时软件包。仍需兼容的 NVIDIA 驱动，但无需另行安装 CUDA Toolkit 或 TensorRT。
- **Settings > LLMs > GPU layers** 控制托管 `llama.cpp` 尝试卸载到 GPU 的层数。外部服务器自行管理其 GPU 设置。

仍可回退到 CPU。GPU 加速效果取决于模型、页面大小，以及处理流程中哪个阶段构成瓶颈；完整页面流程的速度提升可能不会与 GPU 原始利用率成正比。

## 使用技巧

- `Ctrl` + 鼠标滚轮：缩放
- `Ctrl` + `=`：放大
- `Ctrl` + `-`：缩小
- `Ctrl` + `0`：将页面缩放至适应窗口
- 左/右方向键：切换页面
- 图像查看器支持常用触控板手势
- 确保所选字体支持目标语言
- 页面排序选项为 **Name: A to Z**、**Name: Z to A**、**Modified: Oldest First** 和 **Modified: Newest First**。文件名采用自然排序，因此 `page2` 会排在 `page10` 前面。
- 如果 CBR 文件引发 `RarCannotExec("Cannot find working tool")` 错误，请将 WinRAR 或 7-Zip 的安装文件夹添加到 `PATH`。

## 工作原理

### 对话气泡检测与文本分割

应用使用上游的 [bubble-and-text-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)。这是一个在日漫、条漫及欧美漫画上训练的 RT-DETR-v2 模型。在 OCR、分割或清理之前，仍可编辑检测结果。

<img src="https://i.imgur.com/TlzVH3j.jpg" width="49%" alt="检测出的漫画文本"> <img src="https://i.imgur.com/h18XrYT.jpg" width="49%" alt="分割后的漫画文本">

### OCR

默认 OCR 后端为：

- 日语使用 [manga-ocr](https://github.com/kha-white/manga-ocr)
- 韩语使用 [Pororo](https://github.com/yunwoong7/korean_ocr_using_pororo)
- 其他受支持语言使用 [PPOCRv5](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)

Gemini 和 Microsoft Azure Vision 仍可作为可选的服务商直连 OCR 集成，使用时需提供自己的凭据。

### 翻译

翻译采用本地兼容 OpenAI 的接口，或使用上述由用户提供密钥的服务商直连集成；不会经过上游付费服务。整页请求会将符合条件的已识别区域分组，以提供上下文。已完成的手动区域会保留，而针对选定区域的请求可能使用缓存结果，或发送范围更小的请求。兼容的多模态模型也可按需接收页面图像。

### 图像修复与艺术编辑

本地清理后端仍为 LaMa、MI-GAN 和 AOT。独立的 FLUX.2 Klein Artistic Edit 工具负责生成式修改，例如翻译后的艺术字、重建或调整文字气泡大小，以及局部画面修复。它采用由用户检查预览后再应用的流程，并非清理功能的自动替代方案。

<img src="https://i.imgur.com/cVVGVXp.jpg" width="49%" alt="图像修复前的漫画"> <img src="https://i.imgur.com/bLkPyqG.jpg" width="49%" alt="图像修复后的漫画">

### 文本渲染

翻译或手动输入的文字会在可编辑区域内自动换行。导出前可调整字体、大小、对齐、间距、颜色、描边、方向，以及字符级强调格式。

## 测试

在仓库根目录运行回归测试套件：

```bash
uv run python -m unittest discover -s tests -v
```

单独检查 GUI 翻译覆盖率及编译后的翻译目录：

```bash
uv run python resources/translations/check_translations.py
```

发布检查涵盖手动区域 OCR 及内容保留、本地/API 服务商路由、艺术编辑接口约定及预览、项目补丁持久化、清理及克隆撤销、主题、语言选择保留，以及分支配置隔离。回归测试套件中的大多数模型和网络操作均使用模拟实现；不会下载权重或消耗 API 点数。

## 当前限制

- 这是一个仍处于早期阶段、仅提供源码运行方式的分支；针对不同语言、模型、版式和操作系统的广泛测试仍在进行中。
- 自动模式无法始终区分对白与属于画面的文字。当保留原画很重要时，请使用 Manual 模式。
- 对于复杂页面，检测、OCR、翻译、分割和图像修复均可能需要人工纠正。
- 视觉输入需要真正支持多模态的模型、兼容的服务器，以及适用时所需的匹配投影器文件。
- 本地生成的质量和速度在很大程度上取决于模型选择、上下文大小，以及可用的内存和显存。
- Artistic Edit 目前要求单页模式；它既不是全书自动处理阶段，也不是条漫多页编辑器。应用之前，必须检查生成文字的拼写，以及是否出现了不希望发生的画面改动。

## 上游项目的漫画示例

这些示例展示了 Comic Translate 原有的托管 GPT 工作流程。部分作品也有官方英文译本。

- [The Wretched of the High Seas](https://www.drakoo.fr/bd/drakoo/les_damnes_du_grand_large/les_damnes_du_grand_large_-_histoire_complete/9782382330128)
- [Journey to the West](https://ac.qq.com/Comic/comicInfo/id/541812)
- [The Wormworld Saga](https://wormworldsaga.com/index.php)
- [Frieren: Beyond Journey's End](https://renta.papy.co.jp/renta/sc/frm/item/220775/title/742932/)
- [Days of Sand](https://9ekunst.nl/2021/05/20/nieuw-album-van-aimee-de-jongh-is-benauwd-als-een-zandstorm/)
- [Player (OH Hyeon-Jun)](https://comic.naver.com/webtoon/list?titleId=745876&page=1&sort=ASC&tab=fri)
- [Carbon & Silicon](https://www.amazon.com/Carbone-Silicium-French-Mathieu-Bablet-ebook/dp/B0C1LTGZ85/)

## 致谢

本项目是 [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate) 的分支。原始应用、模型、界面和集成功能，为本文所述改动奠定了基础。

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [lama-cleaner](https://github.com/Sanster/lama-cleaner)
- [dreMaz/AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting)
- [Pororo Korean OCR](https://github.com/yunwoong7/korean_ocr_using_pororo)
- [manga-ocr](https://github.com/kha-white/manga-ocr)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [dayu_widgets](https://github.com/phenom-films/dayu_widgets)

本项目采用 [Apache License 2.0](../LICENSE) 许可证。
