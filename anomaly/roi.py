"""
관심 영역(ROI) / 무시 마스크 공통 기능
- ROI 마스크: 원본 이미지와 같은 크기의 흑백 PNG. 흰색(>0) = 검사할 영역, 검정 = 무시.
- 모델 폴더에 roi.png 로 복사해 두고, 학습과 검사 양쪽에서 같은 마스크를 쓴다.
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_roi(path: str | Path | None, size_wh: tuple[int, int] | None = None) -> np.ndarray | None:
    """마스크를 0/1 uint8 배열로 읽는다. size_wh 를 주면 그 크기로 맞춘다. 없으면 None."""
    if not path or not Path(path).exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if size_wh and (m.shape[1], m.shape[0]) != tuple(size_wh):
        m = cv2.resize(m, size_wh, interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def apply_to_image(img: np.ndarray, roi: np.ndarray | None) -> np.ndarray:
    """ROI 바깥을 중간 회색으로 채운다 (모델이 그 영역에서 아무것도 배우지/보지 않도록)."""
    if roi is None:
        return img
    out = img.copy()
    out[roi == 0] = 128
    return out


def apply_to_map(amap: np.ndarray, roi: np.ndarray | None) -> np.ndarray:
    """점수 지도의 ROI 바깥을 0 으로 만든다."""
    if roi is None:
        return amap
    return amap * roi.astype(amap.dtype)


def copy_into_model(roi_path: str | Path | None, model_dir: Path) -> str | None:
    """ROI 파일을 모델 폴더에 roi.png 로 복사하고 상대 경로 문자열을 돌려준다."""
    if not roi_path or not Path(roi_path).exists():
        return None
    dst = model_dir / "roi.png"
    shutil.copyfile(roi_path, dst)
    return "roi.png"


def model_roi_path(model_dir: Path, info: dict) -> Path | None:
    rel = info.get("roi")
    if not rel:
        return None
    p = model_dir / rel
    return p if p.exists() else None


def editor_to_mask(editor_value: dict | None) -> Image.Image | None:
    """Gradio ImageEditor 값(배경 + 그린 레이어)에서 마스크를 만든다. 붓으로 칠한 곳 = ROI."""
    if not editor_value:
        return None
    layers = editor_value.get("layers") or []
    bg = editor_value.get("background")
    if bg is None:
        return None
    W, H = Image.fromarray(bg).size if isinstance(bg, np.ndarray) else bg.size
    mask = np.zeros((H, W), np.uint8)
    for layer in layers:
        arr = np.asarray(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:
            mask |= (arr[..., 3] > 0).astype(np.uint8)
        elif arr.ndim == 3:
            mask |= (arr.max(axis=2) > 0).astype(np.uint8)
    if mask.sum() == 0:
        return None
    return Image.fromarray(mask * 255)


def draw_roi_outline(img: Image.Image, roi: np.ndarray | None) -> Image.Image:
    """검사 결과 이미지에 ROI 경계를 파란 선으로 표시."""
    if roi is None:
        return img
    arr = np.asarray(img).copy()
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr, contours, -1, (30, 120, 255), 2)
    return Image.fromarray(arr)
