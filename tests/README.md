# Test Scripts for E-book Capture Tool

이 디렉토리에는 E-book Capture Tool의 기능 테스트를 위한 스크립트가 포함되어 있습니다.

## 테스트 스크립트 개요

1. **test_capture.py**: 전반적인 기능 테스트 (윈도우 크기 조절, 모니터 선택, 캡처, OCR, LLM 기능 등)
2. **test_window_resize.py**: 다양한 책 형식과 크기 조절 옵션을 테스트
3. **test_ocr_llm.py**: OCR 및 LLM 텍스트 개선 기능 테스트

## 테스트 실행 방법

### 일반 기능 테스트

기본적인 기능 테스트를 위해:

```bash
python test_capture.py
```

특정 창을 대상으로 테스트하려면:

```bash
python test_capture.py --target-window "창 제목"
```

특정 테스트를 건너뛰려면:

```bash
python test_capture.py --skip-resize --skip-ocr
```

### 윈도우 크기 조절 테스트

윈도우 크기 조절 테스트는 특정 창 제목이 필요합니다:

```bash
python test_window_resize.py --window "창 제목"
```

테스트할 책 형식을 지정할 수도 있습니다:

```bash
python test_window_resize.py --window "창 제목" --formats "a4,paperback,kindle"
```

사용 가능한 모든 책 형식을 확인하려면:

```bash
python test_window_resize.py --list-formats
```

테스트 간 대기 시간을 조정하려면:

```bash
python test_window_resize.py --window "창 제목" --delay 5
```

### OCR 및 LLM 개선 테스트

OCR 및 LLM 텍스트 개선 기능을 테스트하려면:

```bash
python test_ocr_llm.py
```

테스트에 사용할 자체 이미지를 지정할 수 있습니다:

```bash
python test_ocr_llm.py --input-image "경로/이미지.png"
```

LLM 개선 테스트를 건너뛰려면:

```bash
python test_ocr_llm.py --skip-llm
```

## 테스트 결과

모든 테스트 스크립트는 실행된 테스트와 결과에 대한 요약을 제공합니다. 테스트 결과 파일은 다음 디렉토리에 저장됩니다:

- 일반 테스트: `test_results/`
- OCR 테스트: `test_ocr_results/`

테스트가 성공하면 스크립트는 종료 코드 0을 반환하고, 하나 이상의 테스트가 실패하면 종료 코드 1을 반환합니다. 