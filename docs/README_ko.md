# Local Comic Translate

[English](../README.md) | 한국어 | [Français](README_fr.md) | [简体中文](README_zh-CN.md) | [Español](README_es.md)

<img src="https://i.imgur.com/QUVK6mK.png" alt="Comic Translate 인터페이스">

Local Comic Translate는 [Comic Translate](https://github.com/ogkalu2/comic-translate)의 커뮤니티 포크로, 개인정보를 보호하는 로컬 번역과 더 안전하게 직접 편집할 수 있는 데스크톱 작업 흐름에 중점을 둡니다. 원본 애플리케이션의 감지, OCR, 인페인팅, 렌더링, 프로젝트, 압축 파일, 웹툰 기능을 유지하면서 관리형 `llama.cpp` 번역, FLUX.2 Klein 아트 편집, 선택적으로 사용할 수 있는 제공업체 직접 API, 다양한 편집 및 안정성 개선을 추가했습니다. 원본 서비스의 계정이나 구독, 번역 크레딧은 필요하지 않습니다.

원본에서 이어받은 OCR 파이프라인은 영어, 한국어, 일본어, 프랑스어, 중국어 간체 및 번체, 러시아어, 독일어, 네덜란드어, 스페인어, 이탈리아어를 원문 언어로 지원하며, 이 언어들을 포함한 더 많은 언어를 번역 대상 언어로 사용할 수 있습니다.

이 포크는 활발히 개발 중입니다. 아래에 설명된 변경 사항을 사용하려면 현재는 소스 코드에서 실행해야 하며, 원본 Comic Translate 웹사이트의 다운로드에는 이 변경 사항이 포함되어 있지 않습니다. 위의 언어 링크에서 이 포크의 최신 문서를 번역한 내용을 볼 수 있습니다. 인터페이스 언어가 달라도 해당 항목을 찾을 수 있도록, 이 안내서의 설정 및 버튼 이름은 영어로 표기합니다.

## 최신 업데이트 — 0.1.0

- **Artistic Edit:** 호환되는 LoRA와 함께 로컬 FLUX.2 Klein 4B/9B를 사용하거나 Black Forest Labs API를 사용할 수 있습니다. 적용 전에 미리보기를 확인하며, 실행 취소와 프로젝트 저장을 지원합니다.
- **번역 백엔드 선택:** 로컬 llama-server 설정을 유지한 채 본인의 API 키로 OpenAI, Gemini, Claude, Deepseek로 전환할 수 있습니다.
- **개선된 수동 작업 흐름:** 직접 그린 영역에서 바로 OCR을 실행할 수 있습니다. 완료된 수동 번역과 서식은 이후 페이지 전체를 처리해도 유지되며, 원문 및 번역 대상 언어 선택도 페이지 간에 유지됩니다.
- **부분 정리:** 브러시로 원본 픽셀을 복원하거나, 고정된 색상 및 질감 샘플을 복제하거나, 샘플링한 배경색으로 마스크를 채울 수 있습니다.
- **데스크톱 사용성 개선:** Light/Dark 외에 Midnight, Parchment, Lavender, Mint 테마를 추가하고, Artistic Edit 컨트롤을 더 명확하게 다듬었으며, 시작 화면 이미지와 GUI 언어 카탈로그를 업데이트했습니다.
- **독립적인 포크 식별 정보:** 설정과 애플리케이션 데이터를 별도로 관리하고, 원본 서비스의 업데이트 알림과 유료 서비스 연결 경로를 제거했습니다.

## 이 포크의 주요 특징

### 로컬 우선, 개인정보를 보호하는 LLM 번역

- 호스팅 번역 API나 Comic Translate 계정이 필요 없는 **Local LLM** 번역기를 제공합니다.
- **Managed llama.cpp** 모드는 로컬 `llama-server` 프로세스를 실행하고, 사용하지 않는 루프백 포트를 선택하며, 요청 간에 프로세스를 재사용하고, 앱을 종료할 때 함께 중지합니다.
- **External server** 모드는 Ollama, LM Studio, 별도로 관리하는 `llama-server` 등 OpenAI 호환 `/v1` 엔드포인트를 지원합니다.
- GGUF 모델, 선택적 멀티모달 프로젝터(`mmproj`), 컨텍스트 크기, GPU 레이어 수, 시작 및 요청 시간 제한, 최대 출력 토큰 수, Temperature, Top P, Top K를 설정할 수 있습니다.
- 시각 입력을 지원하는 모델에는 선택적으로 페이지 이미지를 제공할 수 있습니다. 기존 OCR이 여전히 주요 텍스트 소스이며, 이미지는 멀티모달 모델에 추가 문맥을 제공하고 명백한 OCR 오류를 수정할 기회를 줍니다.
- 엄격한 블록 ID 검증과 자동 재시도를 통해, 불완전하거나 형식이 잘못된 모델 출력 때문에 번역이 아무런 경고 없이 엉뚱한 말풍선에 배치되는 일을 방지합니다.
- 페이지 단위 번역 캐시는 원본 이미지를 사용하므로, 이후 인페인팅을 하더라도 캐시된 번역이 불필요하게 무효화되지 않습니다.

### 선택적으로 사용하는 제공업체 직접 API

- 기본 번역기는 **Local LLM**입니다. 번역이나 OCR에 원본 Comic Translate의 로그인, 구독, 크레딧 구매가 필요하지 않으며, 해당 서비스를 이용하지도 않습니다.
- **Settings > Tools > Translator**에서 **OpenAI**, **Google Gemini**, **Anthropic Claude**, **Deepseek**를 선택하면 클라우드 제공업체를 직접 사용할 수 있습니다.
- **Settings > Provider APIs**에서 해당 제공업체를 선택하고, 본인의 API 키를 입력한 뒤 API 모델 ID를 선택하거나 직접 입력하세요. 요금은 제공업체가 사용자의 계정에 직접 청구합니다. 사전 설정은 시작을 돕기 위한 예시이며, 사용 가능한 모든 모델을 나열한 목록은 아닙니다.
- 기존의 선택적 클라우드 OCR을 위해 Microsoft Azure OCR 자격 증명과 Gemini 키도 설정할 수 있습니다. 기본 OCR은 계속 로컬에서 실행됩니다.
- **Save Keys**는 사용자가 선택하여 켜는 옵션입니다. 키는 이 포크의 로컬 애플리케이션 설정에 저장되며, 암호화된 보관소에 저장되는 것은 아닙니다. 프로젝트 파일에는 저장되지 않습니다. 저장된 제공업체 키를 제거하려면 이 옵션을 끄고 설정을 저장하거나 앱을 종료하세요. 모델 선택은 그대로 저장됩니다.
- 클라우드 번역은 인식된 텍스트를 전송하며, **Provide Image as Input to AI**가 켜져 있고 모델이 지원하는 경우에는 페이지 이미지도 전송합니다. 제공업체의 정책과 요금이 적용됩니다. 로컬 처리에서 클라우드로 자동 전환되지는 않습니다.
- 이전의 **Advanced / Custom** 설정은 제거되었습니다. 저장되어 있던 Custom 엔드포인트가 활성 상태였다면 **Local LLM > External server**로 이전됩니다. 기존에 선택한 이름 있는 클라우드 모델은 모델 선택을 유지한 채 해당 제공업체의 직접 연결 설정으로 이전되며, 사용자가 본인의 키를 입력해야 합니다.
- 제공업체 자격 증명과 로컬 서버 설정은 서로 독립적입니다. **Tools > Translator**를 변경하면 두 설정을 지우지 않고 활성 백엔드만 전환합니다. **Provider APIs**의 제공업체 드롭다운은 편집할 자격 증명만 선택합니다.

### 더 안전한 수동 편집

- 무엇이든 지우기 전에 감지된 영역을 검토할 수 있도록, 한 페이지씩 작업하는 실용적인 흐름을 제공합니다.
- Manual 모드에서 페이지를 이동해도 원문 언어와 번역 대상 언어 선택이 모두 유지됩니다. 원문 언어를 명시적으로 **Auto**로 선택한 경우에도 변경할 때까지 유지됩니다.
- 페이지 전체에 Detect를 먼저 실행하지 않아도, 직접 그린 영역에서 오른쪽 클릭 메뉴의 **OCR**을 사용할 수 있습니다. 로컬 OCR은 누락된 텍스트 줄 경계를 준비하며, 사용자가 특정 영역의 OCR을 실행하면 오래된 캐시 결과를 반환하거나 관계없는 상자를 처리하는 대신 해당 영역을 다시 인식합니다.
- 페이지 전체에 Detect를 실행해도 직접 그린 영역은 유지되며, 같은 텍스트의 중복 감지는 억제됩니다. 이후 Recognize/Translate 작업에서도 완료된 원문과 번역문이 보존됩니다. Segment와 Render는 기존 텍스트 레이어와 단어별 서식을 삭제하거나 중복 생성하지 않고 유지합니다.
- 수동 텍스트 레이어를 보존하더라도 배경 정리를 건너뛰지는 않습니다. 해당 영역은 계속 분할과 정리 작업에 포함됩니다. 아직 번역하지 않은 수동 영역은 일반 작업 흐름에 따라 처리됩니다.
- Zoom In, Zoom Out, Fit 컨트롤과 사용자 지정 키보드 단축키를 제공합니다.
- 파일 이름의 자연 정렬과 수정 시간순 정렬을 지원합니다.
- 새 텍스트 영역을 그리고 내용을 입력하면 테두리 없는 대체 텍스트가 즉시 표시됩니다.
- 텍스트를 지웠다가 다시 입력해도 선택한 글꼴이 유지됩니다.
- 텍스트 항목 전체뿐 아니라 선택한 단어에만 굵게, 기울임꼴, 밑줄을 적용할 수 있습니다.
- 텍스트 상자를 이동하거나 크기를 조절해도 부분 서식을 함께 사용할 수 있으며, 일본어 IME 입력을 포함하여 직접 입력한 번역문은 포커스가 바뀌어도 유지됩니다.
- 페이지 전체의 수동 번역은 누락된 텍스트 레이어를 생성합니다. 아직 렌더링하지 않은 선택 영역을 번역할 때는 해당 영역을 정리한 뒤 번역 텍스트 레이어를 생성합니다.

### 인페인팅 및 실행 취소 개선

- 모든 수동 정리 작업은 일반 Undo/Redo 컨트롤로 실행 취소하거나 다시 실행할 수 있습니다.
- **Restore brush**는 브러시가 지나간 부분의 원본 페이지 픽셀만 복원하고, 다른 곳의 편집은 그대로 둡니다. 원본의 글자도 복원되지만, 편집 가능한 텍스트 오버레이가 제거되지는 않습니다. 기존 브러시 슬라이더로 크기를 조절하세요. 각 획을 실행 취소할 수 있습니다.
- **Clone brush**(도장 아이콘): 브러시 크기를 설정하고 깨끗한 영역을 Alt-클릭한 다음, 칠하여 고정된 색상과 질감 샘플을 반복해서 복제합니다. 청록색 점선 원은 원본 위치에 남고, 검정/흰색 실선 윤곽은 복제할 위치를 따라가며 확대·축소 배율에 맞게 표시됩니다. **Soft edge**로 가장자리의 혼합 정도를 조절합니다. **Sample original**은 기본적으로 켜져 있어 정리 패치 아래의 원본을 샘플링합니다. 편집된 페이지를 복사하려면 끄세요. 크기, 페이지, 샘플 원본 모드를 변경하면 Alt-클릭으로 새로 샘플링해야 합니다. 각 획을 실행 취소할 수 있으며, 칠한 픽셀만 변경됩니다. AI 모델은 사용하지 않습니다.
- **Pick color**를 사용한 뒤 **Fill mask with sampled color**를 실행하면, 현재 페이지에서 브러시로 칠했거나 분할된 영역을 손상되지 않은 배경색으로 채웁니다. 추가 확장이나 모델 추론 없이 보이는 마스크를 사용합니다. 단색 말풍선에 사용하세요. 그라데이션이나 질감이 있는 그림에는 여전히 인페인팅이나 아트 편집이 필요합니다. 실행 취소하면 이전 픽셀과 마스크가 모두 복원됩니다.
- **Revert All Inpainting on the Current Page**는 저장된 모든 정리 패치를 한 번의 실행 취소 가능한 작업으로 제거합니다.
- 반복 정리는 현재 합성된 페이지를 기준으로 시작하므로, 이전에 지운 텍스트가 이후 패치에서 다시 나타나는 일을 방지합니다.
- 수동 브러시 정리는 드래그한 획뿐 아니라 한 번의 클릭도 처리하며, 정수 좌표를 안전하게 사용하여 이미지 경계를 처리합니다.
- 형상을 고려하는 대체 마스크는 직사각형 캡션 상자와 둥근 말풍선을 구분합니다. 그림 주변은 보수적으로 처리하면서 사각 모서리에 남는 텍스트를 줄입니다.
- 자동 페이지 편집은 예측 가능한 실행 취소 단계로 묶입니다.

### FLUX.2 Klein 아트 편집

- 선택적으로 사용하는 별도의 Diffusers 작업 프로세스가 FLUX.2 Klein 4B 또는 9B로 선택한 텍스트 상자, 브러시로 칠한 영역, 페이지 전체를 편집할 수 있습니다.
- 편집기는 모델의 기본 BF16 정밀도를 사용하며, GPU 또는 CPU 오프로딩 정책을 설정할 수 있습니다.
- `models/loras/flux2-klein`에서 호환되는 LoRA를 선택할 수 있으며, 편집별 강도 조절과 모델 크기 검증을 지원합니다.
- **Use selected translation**으로 **Artistic lettering** 프롬프트를 만들거나, 나중에 편집 가능한 텍스트를 배치하기 위한 **Empty speech bubble** 프롬프트를 만들 수 있습니다.
- **Context padding**은 주변 그림을 모델에 함께 제공합니다. **Edit margin**은 선택한 상자 주변에서 변경을 허용할 영역을 확장하여, 더 길어진 번역 글자가 원래 경계에 잘리지 않도록 합니다. 확장된 편집을 적용하기 전에 미리보기를 검토하세요.
- 생성 결과는 Before/After 미리보기에 표시되며, 원본을 손상시키지 않고 실행 취소할 수 있는 프로젝트 패치로 적용됩니다.
- FLUX 작업 프로세스는 계속 유지되며, 모델이나 장치 정책을 변경하면 의도적으로 다시 로드합니다.
- CPU 오프로딩 정책에서는 실제 GPU 인덱스를 선택할 수 있어, 다른 로컬 모델이 이미 한 GPU를 사용 중일 때 유용합니다.
- 선택적 **Black Forest Labs API** 백엔드는 로컬 FLUX 환경이나 GPU 없이 호스팅된 Klein 4B와 9B를 지원합니다. **Settings > Provider APIs**에서 키를 설정한 다음 Artistic Edit 상단에서 백엔드를 선택하세요. 앱을 실행할 때마다 기본값은 **Local Diffusers**입니다.
- 클라우드 편집도 동일한 선택, 컨텍스트, 편집 여백, 미리보기, Apply, 실행 취소 흐름을 사용합니다. 재생성을 포함한 매번의 생성 시, 선택 영역과 주변 컨텍스트 또는 전체 페이지를 선택한 경우 전체 페이지를 업로드할 권한을 요청합니다. 미리보기 역시 유료 생성이므로, 폐기하더라도 요금이 발생합니다.
- 이 호스팅 백엔드에서는 로컬 LoRA, 스텝 수, 가이던스, GPU 컨트롤을 사용할 수 없습니다. 클라우드 처리는 4메가픽셀 미만으로 제한됩니다. 출력은 검증을 거친 뒤 페이지 좌표에 맞게 복원되며, 편집 마스크 안으로 제한됩니다.
- 취소는 클라우드 결과를 기다리는 동작을 중지합니다. 제공업체의 작업이나 요금 청구까지 취소된다고 보장하지는 않습니다. 실패했거나 성공 여부가 불확실한 요청은 자동으로 다시 전송하지 않습니다. 시간 초과가 발생하면 재시도하기 전에 제공업체 대시보드를 확인하세요. 콘텐츠 제한은 로컬 모델과 다를 수 있습니다.

API 참고 문서: [BFL 이미지 편집](https://docs.bfl.ai/flux_2/flux2_image_editing). 클라우드 연동에는 전송 및 컨트롤러를 모의 구현한 자동화 테스트가 있으며, 개발 중 실제 Flux 및 LLM API 요청도 확인했습니다. 이것이 모든 제공업체나 모델에서의 동작을 보장하지는 않습니다. 계정 접근 권한, 콘텐츠 정책, 요금은 선택한 서비스에 따라 달라집니다.

### 데스크톱 외관, 언어, 포크 격리

- **Dark**, **Light**, **Midnight**, **Parchment**, **Lavender**, **Mint**의 여섯 가지 테마를 제공합니다. Artistic Edit 버튼은 테마에 맞는 배경, 읽기 쉬운 레이블, 선명한 테두리를 사용합니다.
- 시작 화면 이미지와 포크 브랜딩을 업데이트했습니다.
- 인터페이스는 영어, 한국어, 프랑스어, 중국어 간체, 러시아어, 일본어, 독일어, 스페인어, 이탈리아어를 제공합니다. 새 기능의 번역은 현재 언어 선택기에 표시되지 않는 추가 터키어 카탈로그를 포함해 아홉 개의 Qt 카탈로그에 모두 들어 있습니다. 번역 범위 검사는 자리표시자와 컴파일된 카탈로그를 검증합니다. 인터페이스 언어는 위에서 연결한 README 번역과 별개입니다.
- 별도로 설치된 원본 버전과 서로 간섭하지 않도록, 이 포크는 자체 설정, 애플리케이션 데이터 경로, 애플리케이션 식별 정보를 사용합니다. 이전 데이터의 이전 작업은 기존 프로필에 대해서는 읽기만 수행합니다.
- 시작 시 자동 업데이트 확인은 비활성화되어 있으며, 계정 및 구독 판매 UI는 제거되었습니다. 수동 **Check for Updates**는 이제 원본 프로젝트 대신 이 포크의 GitHub 릴리스를 확인합니다.

### GPU 및 안정성 개선

- Windows 종속성에는 ONNX Runtime GPU와 호환되는 CUDA 13/cuDNN 9 런타임 패키지가 포함됩니다. 명시적으로 요청하지 않는 한 TensorRT를 더 이상 탐색하지 않습니다.
- CUDA가 실패하면 TensorRT를 요구하지 않고 CPU로 정상 전환합니다.
- 비동기 페이지 이동, 프로젝트 패치 식별 및 저장, 반복 인페인팅 합성, 렌더링 갱신 누락 문제를 수정했습니다.
- 회귀 테스트는 로컬 LLM 응답, 감지 레이블, 자동 및 수동 번역, 텍스트 편집, 이미지 정렬, 인페인팅 형상과 실행 취소, 번역 캐싱, 장치 선택을 다룹니다.

## 권장 작업 흐름

자동 번역도 계속 사용할 수 있지만, 만화를 세심하게 복원하려면 보통 한 페이지씩 처리하는 편이 좋습니다.

1. 만화를 불러오고, 페이지가 읽는 순서대로 되어 있지 않다면 **Sort**를 사용하세요.
2. **Manual** 모드를 선택하고 원문 언어와 번역 대상 언어를 설정한 다음 **Detect**를 클릭하세요. 작품에 실제로 여러 언어가 섞여 있거나 언어를 알 수 없는 경우 원문 언어에 **Auto**를 사용하세요.
3. 처리하기 전에 영역을 검토하세요. 그림에 해당하는 감지 결과는 지우고, 놓친 대사나 캡션에는 상자를 그리세요.
4. **Recognize**를 클릭한 다음, OCR에 오류가 있다면 원문을 수정하세요.
5. **Translate**를 클릭하고 번역문을 검토하거나 편집하세요.
6. **Segment**를 클릭하고 제안된 정리 영역을 확인한 다음 **Clean**을 클릭하세요.
7. **Render**를 클릭하고 각 텍스트 항목의 글꼴, 크기, 정렬, 간격, 강조를 조절하세요.
8. 남은 흔적은 정리 브러시와 **Apply Cleanup**으로 처리하세요. 정리 과정에서 그림이 손상되었다면 Undo/Redo 또는 **Revert All Inpainting on the Current Page**를 사용할 수 있습니다.

페이지의 나머지 부분을 먼저 감지하지 않고도, 놓친 영역을 직접 그린 뒤 오른쪽 클릭 메뉴에서 **OCR**, 이어서 **Translate**를 실행할 수 있습니다. 이후 페이지 전체에 수동 작업 흐름을 실행해도 완료된 번역은 유지됩니다. 색상이나 질감이 있는 배경에서는 같은 영역에 인페인팅을 반복하기 전에 샘플 색상 채우기, 복원 브러시, 복제 브러시를 사용해 보세요. 일반적인 편집 가능 텍스트 레이어가 아니라 그림의 일부가 되어야 하는 글자에는 Artistic Edit를 사용하세요.

감지기는 말풍선 안의 텍스트와 말풍선 밖의 텍스트를 의도적으로 모두 유지합니다. 덕분에 수동 모드에서 유효한 OCR 영역이 누락되지 않지만, **Translate All**을 실행하면 장식적인 제목, 표지판, 효과음, 그 밖에 그림의 일부인 글자에도 인페인팅이 적용될 수 있습니다. 그림 보존이 중요하다면 정리 전에 수동으로 검토하는 것을 권장합니다.

## 설치

### 요구 사항

- Python 3.12
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- 관리형 로컬 번역을 사용하려면 로컬 `llama-server` 실행 파일과 호환되는 GGUF 모델
- CBR 압축 파일을 열려면 `PATH`에 등록된 WinRAR 또는 7-Zip

### 소스 코드에서 실행

```bash
git clone https://github.com/biogoly/local-comic-translate.git
cd local-comic-translate
uv init --python 3.12
uv add -r requirements.txt --compile-bytecode
uv run comic.py
```

Windows에서는 처음 종속성을 설치한 뒤 사용할 수 있도록 `run.bat`도 제공됩니다.

```bat
run.bat
```

### 선택적 FLUX.2 Klein 런타임

**Local Diffusers** 백엔드는 별도의 Python 환경을 사용하므로, 최신 PyTorch, Diffusers, Transformers 패키지가 기본 애플리케이션 환경에 영향을 주지 않습니다. BFL API 백엔드에는 이 선택적 환경이 필요하지 않습니다. 저장소 루트에서 Python 3.12 환경을 만들고, 버전이 고정된 선택적 요구 패키지를 설치하세요.

```bat
py -3.12 -m venv .venv-flux
.venv-flux\Scripts\python.exe -m pip install -r requirements-flux.txt
```

**Artistic Edit**에서 **Select FLUX Python…**을 누르고 `.venv-flux\Scripts\python.exe`를 선택하세요. 통합된 작업 프로세스는 기본적으로 캐시만 사용하므로, 모델 파일은 이미 Hugging Face 캐시에 있어야 합니다. 이 포크의 프로필 격리는 별도의 Hugging Face 캐시를 만들거나 사용자가 선택한 모델 경로의 파일을 복제하지 않습니다.

선택적으로 사용하는 `.safetensors` LoRA는 모델에 맞게 `models/loras/flux2-klein/4b/` 또는 `models/loras/flux2-klein/9b/`에 넣으세요. 호환성과 선택적 메타데이터에 대해서는 [LoRA 폴더 안내](../models/loras/flux2-klein/README.md)를 참고하세요. 모델 가중치는 저장소에 포함되어 있지 않습니다. 현재 로컬 작업 프로세스는 BF16을 사용하며, 실험적인 bitsandbytes INT8/NF4 로딩은 활성화되어 있지 않습니다.

기존에 복제한 저장소를 업데이트하려면 다음을 실행하세요.

```bash
git pull
uv add -r requirements.txt --compile-bytecode
```

### 로컬 LLM 설정

`llama.cpp` 자체는 외부 런타임이며 `requirements.txt`나 `uv`를 통해 설치되지 않습니다. [`llama.cpp`](https://github.com/ggml-org/llama.cpp)를 다운로드하거나 빌드한 다음, 하드웨어에 맞는 대화용 또는 지시 수행용 GGUF 모델을 준비하세요.

이 백엔드는 특정 모델에 종속되지 않습니다. 이 포크의 현재 개발과 테스트에는 주로 Gemma 4를 사용하고 있습니다.

애플리케이션에서 다음과 같이 설정하세요.

1. **Settings > Tools**를 열고 번역기로 **Local LLM**을 선택하세요.
2. **Settings > LLMs**를 여세요.
3. 다음 런타임 중 하나를 선택하세요.
   - **Managed llama.cpp**: `llama-server`(이미 `PATH`에 있다면 선택 사항), 기본 GGUF 모델, 모델에 필요한 경우 호환되는 시각 프로젝터 GGUF를 선택하세요.
   - **External server**: OpenAI 호환 기본 URL과 서버가 제공하는 정확한 모델 이름을 입력하세요. 기본 URL인 `http://127.0.0.1:11434/v1`은 Ollama용입니다.
4. 선택한 모델과 서버가 시각 입력을 지원할 때만 **Provide Image as Input to AI**를 켜세요. 관리형 모드에서는 먼저 필요한 `mmproj` 파일을 설정하세요.

안정적인 번역을 위한 기본값은 Temperature `0.20`, Top P `0.90`, Top K `40`입니다. 번역을 시작하기에 적합한 값이므로, 모델 자체 문서에서 다른 샘플링 값을 권장할 때만 조절하세요. Top K를 비활성화하려면 `0`으로 설정하세요.

### 로컬 백엔드와 API 백엔드 전환

텍스트 번역의 경우:

1. **Settings > Provider APIs**를 열고 제공업체를 선택한 다음, 키를 입력하고 모델 ID를 선택하거나 직접 입력하세요. 세션 사이에 키를 유지하려는 경우에만 **Save Keys**를 켜세요.
2. **Settings > Tools > Translator**를 열고 해당 제공업체를 선택하세요. 앱을 다시 시작하지 않아도 이후 요청부터 변경 사항이 적용됩니다.
3. 저장해 둔 llama.cpp 또는 외부 서버 설정으로 돌아가려면 **Local LLM**을 다시 선택하세요. 로컬 엔드포인트를 바꾸거나 모델 설정을 지울 필요는 없습니다.

아트 편집의 경우 **Provider APIs**에 **Black Forest Labs** 키를 입력한 다음, **Artistic Edit** 상단 드롭다운을 **Local Diffusers**에서 **Black Forest Labs API**로 바꾸세요. 4B/9B를 선택하고 **Generate Preview**를 사용하세요. 다시 전환하면 로컬 컨트롤을 사용할 수 있습니다. 이 선택은 텍스트 번역기와 별개입니다.

다른 제공업체를 테스트할 때는 아직 번역하지 않은 페이지나 영역을 사용하세요. 완료된 수동 번역은 의도적으로 보존되며, 캐시된 결과로 인해 새 요청이 전송되지 않을 수도 있습니다. 클라우드 요청은 선택한 제공업체로 데이터를 전송하고 요금이 발생할 수 있습니다. 아트 편집 미리보기를 폐기해도 이미 생성은 수행된 상태입니다.

### GPU 가속

서로 독립적인 두 가지 GPU 설정이 있습니다.

- **Settings > Tools > Use GPU**는 감지, OCR, 인페인팅 등 지원되는 ONNX/PyTorch 작업을 제어합니다. Windows에서는 종속성을 설치할 때 ONNX Runtime GPU와 호환되는 CUDA 13/cuDNN 9 런타임 패키지가 함께 설치됩니다. 호환되는 NVIDIA 드라이버는 여전히 필요하지만, CUDA Toolkit이나 TensorRT를 별도로 설치할 필요는 없습니다.
- **Settings > LLMs > GPU layers**는 관리형 `llama.cpp`가 GPU로 오프로딩을 시도할 레이어 수를 제어합니다. 외부 서버는 자체 GPU 설정을 관리합니다.

CPU로의 대체 실행도 계속 지원합니다. GPU의 속도 향상 정도는 모델, 페이지 크기, 파이프라인에서 병목이 되는 단계에 따라 달라집니다. 페이지 전체 작업의 속도가 단순한 GPU 사용률에 비례해 빨라지는 것은 아닙니다.

## 사용 팁

- `Ctrl` + 마우스 휠: 확대·축소
- `Ctrl` + `=`: 확대
- `Ctrl` + `-`: 축소
- `Ctrl` + `0`: 페이지를 창 크기에 맞춤
- 왼쪽/오른쪽 방향키: 페이지 간 이동
- 이미지 뷰어에서 일반적인 트랙패드 제스처를 사용할 수 있습니다.
- 선택한 글꼴이 번역 대상 언어를 지원하는지 확인하세요.
- 페이지 정렬 옵션은 **Name: A to Z**, **Name: Z to A**, **Modified: Oldest First**, **Modified: Newest First**입니다. 파일 이름에는 자연 정렬이 적용되므로 `page2`가 `page10`보다 먼저 옵니다.
- CBR 파일에서 `RarCannotExec("Cannot find working tool")` 오류가 발생하면 WinRAR 또는 7-Zip 설치 폴더를 `PATH`에 추가하세요.

## 작동 방식

### 말풍선 감지 및 텍스트 분할

이 애플리케이션은 원본 프로젝트의 [bubble-and-text-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)를 사용합니다. 일본 만화, 웹툰, 서양 만화로 학습된 RT-DETR-v2 모델입니다. 감지 결과는 OCR, 분할, 정리 작업을 실행하기 전에 편집할 수 있습니다.

<img src="https://i.imgur.com/TlzVH3j.jpg" width="49%" alt="감지된 만화 텍스트"> <img src="https://i.imgur.com/h18XrYT.jpg" width="49%" alt="분할된 만화 텍스트">

### OCR

기본 OCR 백엔드는 다음과 같습니다.

- 일본어: [manga-ocr](https://github.com/kha-white/manga-ocr)
- 한국어: [Pororo](https://github.com/yunwoong7/korean_ocr_using_pororo)
- 그 밖의 지원 언어: [PPOCRv5](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)

Gemini와 Microsoft Azure Vision도 본인의 자격 증명으로 제공업체에 직접 연결하는 선택적 OCR 연동으로 사용할 수 있습니다.

### 번역

번역에는 위에서 설명한 로컬 OpenAI 호환 경로나 사용자 키를 사용하는 제공업체 직접 연동 중 하나를 사용하며, 원본 프로젝트의 유료 서비스를 거치지 않습니다. 페이지 전체 요청은 처리 대상인 인식 영역을 묶어 문맥을 제공합니다. 완료된 수동 영역은 보존되며, 선택 영역 요청은 캐시된 결과를 사용하거나 더 좁은 범위로 요청할 수 있습니다. 호환되는 멀티모달 모델에는 선택적으로 페이지 이미지도 제공할 수 있습니다.

### 인페인팅 및 아트 편집

로컬 정리 백엔드는 계속 LaMa, MI-GAN, AOT를 사용합니다. 별도의 FLUX.2 Klein Artistic Edit 도구는 번역된 장식 글자 생성, 말풍선 재구성 또는 크기 조절, 그림의 부분 복구와 같은 생성형 변경을 처리합니다. 사용자가 미리보기를 검토하고 적용하는 작업 흐름이며, 기존 정리를 자동으로 대체하는 기능은 아닙니다.

<img src="https://i.imgur.com/cVVGVXp.jpg" width="49%" alt="인페인팅 전 만화"> <img src="https://i.imgur.com/bLkPyqG.jpg" width="49%" alt="인페인팅 후 만화">

### 텍스트 렌더링

번역하거나 직접 입력한 텍스트는 편집 가능한 영역 안에서 자동으로 줄바꿈됩니다. 내보내기 전에 글꼴, 크기, 정렬, 간격, 색상, 외곽선, 방향, 문자 단위 강조를 조절할 수 있습니다.

## 테스트

저장소 루트에서 회귀 테스트를 실행하세요.

```bash
uv run python -m unittest discover -s tests -v
```

GUI 번역 범위와 컴파일된 카탈로그는 별도로 확인하세요.

```bash
uv run python resources/translations/check_translations.py
```

릴리스 검사는 수동 영역의 OCR과 보존, 로컬/API 제공업체 연결 경로, 아트 편집의 데이터 규약과 미리보기, 프로젝트 패치 저장, 정리 및 복제 작업의 실행 취소, 테마, 언어 선택 유지, 포크 프로필 격리를 포함합니다. 회귀 테스트에서 대부분의 모델 및 네트워크 작업은 모의 구현을 사용하므로, 가중치를 다운로드하거나 API 크레딧을 사용하지 않습니다.

## 현재 제한 사항

- 이 포크는 아직 초기 단계이며 소스 코드에서만 실행할 수 있습니다. 다양한 언어, 모델, 레이아웃, 운영 체제 전반에 대한 광범위한 테스트는 진행 중입니다.
- 자동 모드는 대사와 그림의 일부인 글자를 항상 구분할 수는 없습니다. 원본 보존이 중요하다면 Manual 모드를 사용하세요.
- 복잡한 페이지에서는 감지, OCR, 번역, 분할, 인페인팅 모두 사람이 수정해야 할 수 있습니다.
- 시각 입력에는 실제로 멀티모달을 지원하는 모델, 호환 서버, 그리고 해당되는 경우 모델에 맞는 프로젝터 파일이 필요합니다.
- 로컬 생성의 품질과 속도는 모델 선택, 컨텍스트 크기, 사용 가능한 RAM/VRAM에 크게 좌우됩니다.
- Artistic Edit는 현재 단일 페이지 모드에서만 사용할 수 있습니다. 책 전체에 적용되는 자동 처리 단계나 웹툰의 여러 페이지를 편집하는 도구는 아닙니다. 생성된 글자를 적용하기 전에 철자와 원치 않는 그림 변경이 없는지 확인해야 합니다.

## 원본 프로젝트의 만화 예시

아래 예시는 Comic Translate의 원래 호스팅 GPT 작업 흐름을 보여 줍니다. 일부 작품은 공식 영어 번역본도 있습니다.

- [The Wretched of the High Seas](https://www.drakoo.fr/bd/drakoo/les_damnes_du_grand_large/les_damnes_du_grand_large_-_histoire_complete/9782382330128)
- [Journey to the West](https://ac.qq.com/Comic/comicInfo/id/541812)
- [The Wormworld Saga](https://wormworldsaga.com/index.php)
- [Frieren: Beyond Journey's End](https://renta.papy.co.jp/renta/sc/frm/item/220775/title/742932/)
- [Days of Sand](https://9ekunst.nl/2021/05/20/nieuw-album-van-aimee-de-jongh-is-benauwd-als-een-zandstorm/)
- [Player (OH Hyeon-Jun)](https://comic.naver.com/webtoon/list?titleId=745876&page=1&sort=ASC&tab=fri)
- [Carbon & Silicon](https://www.amazon.com/Carbone-Silicium-French-Mathieu-Bablet-ebook/dp/B0C1LTGZ85/)

## 감사의 말

이 작업은 [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate)를 포크한 것입니다. 원본 애플리케이션, 모델, 인터페이스, 연동 기능이 여기에 설명된 변경 사항의 기반입니다.

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [lama-cleaner](https://github.com/Sanster/lama-cleaner)
- [dreMaz/AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting)
- [Pororo Korean OCR](https://github.com/yunwoong7/korean_ocr_using_pororo)
- [manga-ocr](https://github.com/kha-white/manga-ocr)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [dayu_widgets](https://github.com/phenom-films/dayu_widgets)

[Apache License 2.0](../LICENSE)에 따라 배포됩니다.
