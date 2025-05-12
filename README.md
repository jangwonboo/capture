# E-book Page Capture Tool

한국어/영어 e-book 페이지를 캡처하고 OCR 처리하여 텍스트를 추출하는 유틸리티입니다. ML/LLM 기술을 활용한 텍스트 개선 기능도 포함하고 있습니다. LLM(Large Language Model)을 활용하여 OCR 결과를 개선하고 검색 가능한 PDF 파일을 생성할 수도 있습니다.

## 1. 프로그램 개요

이 도구는 e-book 리더(Ridibooks, Kindle 등)에서 렌더링된 페이지 이미지를 캡처하고, OCR(광학 문자 인식)을 수행하여 텍스트를 추출합니다. 다양한 책 판형에 맞게 창 크기를 조정하는 기능도 제공합니다.

## 2. 주요 기능

- 특정 창이나 전체 화면 캡처
- 다양한 책 판형에 맞게 창 크기 자동 조정
- 다중 모니터 환경 지원
- 자동 페이지 넘기기 (키보드 입력)
- OCR을 통한 텍스트 추출 (Tesseract)
- LLM을 활용한 텍스트 품질 개선
- Apple Silicon 기기에서 MLX 최적화 지원
- 검색 가능한 PDF 생성 및 병합

## 3. 작동 흐름도

```
[명령행 인자 파싱] → [로깅 설정] → [창 포커스 지정] → [페이지 캡처 시작]
     ↓
[화면 캡처] → [이미지 저장] → [OCR 처리] → [텍스트 추출]
     ↓
[LLM 텍스트 개선(선택적)] → [검색 가능 PDF 생성] → [다음 페이지로 이동] → [반복]
     ↓
[모든 PDF 병합] → [최종 결과 생성]
```

## 4. 설치 방법

1. 이 저장소를 클론합니다:
```bash
git clone https://github.com/yourusername/capture.git
cd capture
```

2. 의존성 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

3. Tesseract OCR을 설치합니다:
   - macOS: `brew install tesseract`
   - Windows: [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)에서 다운로드
   - Linux: `sudo apt-get install tesseract-ocr`

4. (선택사항) 다국어 OCR을 위한 언어 데이터 설치:
   - macOS: `brew install tesseract-lang`
   - Windows/Linux: [github.com/tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata)에서 언어 데이터 다운로드

5. (선택사항) LLM 텍스트 개선 기능을 위한 설정:
   - Apple Silicon 기기에서 MLX 가속화를 사용하려면: `pip install mlx>=0.0.10`
   - 사용할 모델 크기를 선택하려면 환경 변수 설정: `export MLX_MODEL_SIZE=small`

## 5. 사용 방법

기본 사용법:
```bash
python run.py --window "창 제목" --title "책 제목" --pages 10
```

책 판형에 맞게 창 크기 조정:
```bash
python run.py --window "리더앱" --title "내 책" --book-format A5 --pages 10
```

OCR 텍스트 추출:
```bash
python run.py --window "리더앱" --title "내 책" --pages 10 --ocr --language eng+kor
```

LLM으로 OCR 텍스트 개선:
```bash
python run.py --window "리더앱" --title "내 책" --pages 10 --ocr --enhance
```

이미지 영역 잘라내기:
```bash
python run.py --window "리더앱" --title "내 책" --pages 10 --crop "10,10,90,90"
```

특정 모니터에서 캡처:
```bash
python run.py --window "리더앱" --title "내 책" --monitor 1 --pages 10
```

## 6. 핵심 구성 요소 및 동작 원리

### 6.1. 시스템 및 환경 초기화

- **플랫폼 감지**: `platform_utils.py`에서 Windows, macOS, Linux 중 어떤 환경에서 실행 중인지 확인
- **Apple Silicon 감지**: Apple Silicon 기기에서 실행 중인 경우 MLX 가속화 가능 여부 확인
- **Tesseract OCR 경로 설정**: `config.py`에서 플랫폼별로 적절한 Tesseract 경로 설정
- **로깅 설정**: `logging_utils.py`에서 로그 레벨, 파일 출력 등 로깅 관련 설정

### 6.2. 창 제어 및 화면 캡처

- **`get_window_by_title(window_title)`**: 지정된 제목의 창을 찾아 포커스 설정
  - Windows: `platforms/windows.py`에서 win32gui를 사용하여 창 핸들 찾기
  - macOS: `platforms/macos.py`에서 AppleScript를 사용하여 앱 활성화
  - Linux: `platforms/linux.py`에서 xdotool을 사용하여 창 활성화

- **`capture_screen(window_title)`**: 화면 또는 특정 창 캡처
  - 플랫폼별 캡처 방식 구현 (윈도우 창, macOS 앱, 전체 화면 등)
  - 실패 시 대체 방법으로 fallback

- **`resize_window_to_book_format(window_title, book_format)`**: 지정된 책 판형에 맞게 창 크기 조정
  - `config/book_formats.yaml`에서 지정된 치수 사용
  - 플랫폼별 창 크기 조정 구현

- **`get_monitor_info()`**: 다중 모니터 환경에서 모니터 정보 수집
  - 각 모니터의 위치와 해상도 검색
  - 특정 모니터 타겟팅 지원

### 6.3. 페이지 이동

- **`send_keystroke(key_name)`**: 키 입력으로 페이지 이동
  - pyautogui를 사용하여 키보드 입력 시뮬레이션
  - 다양한 키 입력 방식 지원 (화살표, 스페이스, 엔터 등)

### 6.4. 이미지 처리 및 OCR

- **`preprocess_image(image)`**: OCR 정확도 향상을 위한 이미지 전처리
  - 그레이스케일 변환, 적응형 임계값 처리, 노이즈 감소 등

- **`ocr_image_file(image_path, lang, use_llm)`**: 이미지 파일에서 텍스트 추출
  - Tesseract OCR 실행
  - 옵션에 따라 LLM 텍스트 개선 적용

- **`ocr_multiple_files(image_files, lang, use_llm)`**: 여러 이미지 파일 일괄 처리
  - 병렬 처리 옵션 지원

- **`crop_image(image, crop_percentage)`**: 이미지 잘라내기
  - 지정된 비율에 따라 이미지 영역 추출

### 6.5. LLM 텍스트 개선

- **`LLMEnhancer` 클래스**: OCR 텍스트를 LLM으로 개선
  - `enhance_text()`: OCR 오류 수정 및 텍스트 품질 향상
  - `load_model()`: 지정된 크기의 모델 로드
  - Apple Silicon에서 MLX 가속화 지원
  - CPU 최적화 모드 지원

### 6.6. PDF 생성 및 병합

- **`image_to_searchable_pdf(image_path, output_pdf_path, lang, enhance_ocr)`**: 
  단일 이미지를 검색 가능한 PDF로 변환
  - OCR 텍스트 추출 및 LLM 개선
  - 이미지와 텍스트 레이어를 포함한 PDF 생성

- **`merge_pdfs(pdf_files, output_file)`**: 여러 PDF 파일을 하나로 병합
  - PyPDF2를 사용하여 PDF 파일 병합

## 7. 명령행 옵션

### 창 옵션
- `--window`, `-w`: 캡처할 창 제목
- `--book-format`, `-f`: 창 크기 조정용 책 판형
- `--scale`: 창 크기 조정 배율 (0.0-1.0, 기본값: 0.8)
- `--padding`: 창 크기 조정 시 여백 비율 (기본값: 5%)

### 캡처 옵션
- `--title`, `-t`: e-book 제목 (출력 디렉토리 이름으로 사용)
- `--pages`, `-p`: 캡처할 페이지 수 (기본값: 1)
- `--start`, `-s`: 시작 페이지 번호 (기본값: 1)
- `--delay`, `-d`: 캡처 전 대기 시간(초) (기본값: 0.5)
- `--key`, `-k`: 페이지 넘김용 키 (기본값: right)
- `--page-delay`: 페이지 넘김 후 대기 시간(초) (기본값: 0.3)
- `--output-dir`, `-o`: 출력 디렉토리
- `--format`: 출력 이미지 형식 (기본값: png)
- `--quality`: JPEG/WebP 이미지 품질 (1-100)
- `--crop`: 자르기 영역(비율): left,top,right,bottom

### 모니터 옵션
- `--monitor`: 캡처할 모니터 인덱스 지정
- `--list-monitors`: 사용 가능한 모니터 목록 표시 후 종료

### OCR 옵션
- `--ocr`: 캡처된 이미지에서 OCR 수행
- `--language`, `-l`: OCR 언어 코드 (기본값: eng+kor)
- `--psm`: Tesseract 페이지 세그먼트 모드 (기본값: 3)
- `--dpi`: OCR 처리 DPI (기본값: 300)

### LLM 개선 옵션
- `--enhance`: LLM을 사용하여 OCR 텍스트 개선
- `--model-size`: LLM 모델 크기 (tiny, small, medium)
- `--no-mlx`: Apple Silicon에서 MLX 최적화 비활성화

### 기타 옵션
- `--list-formats`: 사용 가능한 책 판형 목록 표시 후 종료
- `--log-level`: 로그 레벨 (기본값: INFO)
- `--log-file`: 로그 파일 경로
- `--quiet`, `-q`: 콘솔 출력 억제

## 8. 지원하는 책 판형

`--list-formats` 옵션으로 사용 가능한 모든 책 판형을 확인할 수 있습니다:

- ISO 표준 판형 (A3, A4, A5, A6)
- 한국 표준 판형 (신국판, 국판, 크라운판 등)
- 미국/영국 판형 (Paperback, Trade 등)
- 주요 e-reader 해상도 (Kindle, iPad 등)
- 모바일 기기 해상도:
  - iPhone 시리즈 (iPhone SE부터 iPhone 15 Pro Max까지)
  - iPad 시리즈 (iPad, iPad Air, iPad Pro, iPad mini)
  - Samsung Galaxy Tab 시리즈 (A8, S6 Lite, S7/S8, S8 Ultra 등)

## 9. 오류 처리 및 예외 상황

- **창 포커스 실패**: 창 이름을 찾지 못하면 대체 방법으로 전체 화면 캡처 시도
- **스크린샷 실패**: 기본 방법 실패 시 대체 캡처 방법 사용 (pyautogui → 플랫폼 고유 API → 시스템 스크린샷 명령)
- **OCR 오류**: Tesseract 오류 발생 시 로깅하고 빈 문자열 반환, 처리 계속 진행
- **LLM 처리 실패**: 모델 로딩 실패 또는 처리 오류 발생 시 원본 OCR 텍스트를 그대로 사용
- **페이지 캡처 실패**: 연속 5회 이상 실패할 경우 캡처 프로세스 중단하고 로깅
- **라이브러리 의존성 문제**: 선택적 기능 비활성화 (예: MLX 없으면 CPU 모드로 전환)
- **메모리 관리**: 대량의 이미지 처리 시 메모리 사용량 모니터링 및 정리

## 10. 프로젝트 구조

```
capture/
├── config/               # 설정 파일
│   ├── book_formats.yaml # 책 판형 정의
│   └── settings.yaml     # 일반 설정
├── core/                 # 핵심 기능 모듈
│   ├── capture.py        # 화면 캡처 기능
│   ├── llm.py            # LLM 텍스트 개선
│   ├── ocr.py            # OCR 처리
│   └── window.py         # 창 관리 및 크기 조정
├── platforms/            # 플랫폼별 구현
│   ├── macos.py          # macOS 특화 구현
│   ├── windows.py        # Windows 특화 구현
│   └── linux.py          # Linux 특화 구현
├── utils/                # 유틸리티 모듈
│   ├── config.py         # 설정 로드 및 관리
│   ├── logging_utils.py  # 로깅 유틸리티
│   ├── mlx_utils.py      # MLX 최적화 유틸리티
│   └── platform_utils.py # 플랫폼 감지 유틸리티
├── temp/                 # 미사용 파일 보관
├── run.py                # 메인 실행 스크립트
├── requirements.txt      # 의존성 패키지 목록
└── README.md             # 프로젝트 문서
```

## 라이선스

MIT

## 크레딧

이 도구는 Tesseract OCR, PyAutoGUI 및 다양한 ML 라이브러리를 활용하여 기능을 제공합니다.