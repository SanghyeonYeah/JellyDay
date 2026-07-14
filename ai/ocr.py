"""
JellyDay OCR 모듈 — 약봉투/처방전 사진에서 약물명 텍스트를 추출한다.

EasyOCR(무료, 로컬 실행, 한글 지원)을 감싼 추론 유틸리티다. 학습은
하지 않는다 — 사전학습된 기성 엔진으로 train_pipeline.py의
DrugAnalyzer.analyze(ocr_names)에 넣을 입력(raw 텍스트 리스트)을
만드는 게 목적이다.
"""

from typing import List

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

_reader = None


def get_reader(languages=('ko', 'en')):
    """
    EasyOCR Reader를 lazy 초기화한다.
    최초 호출 시 감지/인식 모델 가중치를 인터넷에서 내려받아 캐싱하고,
    이후 호출부터는 캐시된 Reader 인스턴스를 재사용한다.
    """
    global _reader
    if not _EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr 미설치 -> pip install easyocr")
    if _reader is None:
        import torch
        gpu = torch.cuda.is_available()
        _reader = easyocr.Reader(list(languages), gpu=gpu)
    return _reader


def read_drug_names(image_path: str, min_confidence: float = 0.4) -> List[str]:
    """
    약봉투/처방전 사진에서 텍스트 라인을 추출한다.

    반환값은 DrugAnalyzer.analyze(ocr_names)에 바로 넣을 수 있는
    문자열 리스트다. min_confidence 미만인 저신뢰 인식 결과는 제외한다.
    """
    reader = get_reader()
    results = reader.readtext(image_path)
    return [text for (_, text, conf) in results if conf >= min_confidence and text.strip()]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("사용법: python ocr.py <이미지 경로>")
        sys.exit(1)
    names = read_drug_names(sys.argv[1])
    print("인식된 텍스트:", names)
