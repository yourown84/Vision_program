"""
어노멀리 검사 API 서버 (FastAPI)
- 라인 프로그램(C#, LabVIEW, Python 등)이 이미지를 보내면 판정 결과를 JSON 으로 돌려준다.
- 화면(app.py) 없이 단독으로 동작하며, 검사 로직은 core.py 를 그대로 사용한다.

실행: start_api.bat  (기본 주소 http://127.0.0.1:8000, 설명서 http://127.0.0.1:8000/docs)

주요 주소:
  GET  /health                 서버 상태, 현재 모델
  GET  /models                 사용 가능한 모델 목록
  POST /load?model=이름        모델 미리 올리기 (첫 검사 지연을 없앨 때)
  POST /inspect                이미지 1장 어노멀리 검사 (multipart 파일 업로드)
  GET  /models_cls             분류 모델 목록
  POST /classify               이미지 1장 분류 (클래스, 확률, 선택적으로 Grad-CAM 히트맵)
  GET  /models_seg             세그 모델 목록
  POST /segment                이미지 1장 세그멘테이션 (불량 영역 박스/마스크)
  POST /load_model?kind=&model= 모델 교체 (에이전트가 사용)
  GET  /status                 올라온 모델과 사용 가능한 모델 목록
환경 변수: MANAGER_URL (관리 서버 주소, 지정하면 검사 결과를 자동 보고), AGENT_ID (이 PC 이름), API_PORT
"""
import base64
import io
import os
import threading
import time

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

import core

os.environ.setdefault("TRUST_REMOTE_CODE", "1")

app = FastAPI(title="비전 검사 API", version="0.2")

_inspector: core.Inspector | None = None
_lock = threading.Lock()  # 모델 교체와 검사가 동시에 일어나지 않도록


def get_inspector(model_name: str | None) -> core.Inspector:
    global _inspector
    with _lock:
        if model_name is None:
            if _inspector is None:
                models = core.list_models()
                if not models:
                    raise HTTPException(400, "학습된 모델이 없습니다. 먼저 학습하세요.")
                model_name = models[0]
            else:
                return _inspector
        if _inspector is None or _inspector.model_name != model_name:
            if model_name not in core.list_models():
                raise HTTPException(404, f"모델을 찾을 수 없습니다: {model_name}")
            _inspector = core.Inspector(model_name)
        return _inspector


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": os.environ.get("INSPECT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"),
        "loaded_model": _inspector.model_name if _inspector else None,
    }


@app.get("/models")
def models():
    return {"models": [core.load_info(m) | {"name": m} for m in core.list_models()]}


@app.post("/load")
def load(model: str):
    t0 = time.perf_counter()
    insp = get_inspector(model)
    return {"loaded_model": insp.model_name, "load_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/inspect")
async def inspect(
    file: UploadFile = File(..., description="검사할 이미지 (png/jpg/bmp)"),
    model: str | None = Form(None, description="모델 이름. 생략하면 현재 올라온 모델"),
    threshold: float | None = Form(None, description="0~1 NG 임계값. 생략하면 모델에 저장된 값"),
    review_threshold: float | None = Form(None, description="0~1 재검 임계값 (재검~NG 사이면 REVIEW). 생략하면 모델 저장값"),
    pixel_threshold: float | None = Form(None, description="0~1 픽셀 임계값 (불량 영역 표시용). 생략하면 모델 저장값"),
    return_heatmap: bool = Form(False, description="true 면 히트맵 이미지를 base64 PNG 로 함께 반환"),
):
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "이미지 파일을 읽을 수 없습니다.")

    insp = get_inspector(model)
    with _lock:
        r = insp.inspect(image, threshold, review_threshold, pixel_threshold)

    body = {
        "result": r["verdict"],                        # "OK" | "REVIEW" | "NG"
        "score": round(r["score"] * 100, 1),          # 0~100
        "threshold": round(r["threshold"] * 100, 1),  # 0~100
        "review_threshold": round(r["review_threshold"] * 100, 1),
        "boxes": [list(b) for b in r["boxes"]],       # [x, y, w, h] 원본 좌표
        "elapsed_ms": round(r["elapsed_ms"], 1),
        "model": insp.model_name,
        "filename": file.filename,
    }
    if return_heatmap:
        buf = io.BytesIO()
        r["heatmap"].save(buf, format="PNG")
        body["heatmap_png_base64"] = base64.b64encode(buf.getvalue()).decode()
    _report(body | {"kind": "anomaly"})
    return body


# ----------------------------------------------------------------------
# 분류 (Classification)
# ----------------------------------------------------------------------
import core_cls  # noqa: E402

_classifier: core_cls.Classifier | None = None


def get_classifier(model_name: str | None) -> core_cls.Classifier:
    global _classifier
    with _lock:
        if model_name is None:
            if _classifier is not None:
                return _classifier
            models = core_cls.list_models()
            if not models:
                raise HTTPException(400, "학습된 분류 모델이 없습니다.")
            model_name = models[0]
        if _classifier is None or _classifier.model_name != model_name:
            if model_name not in core_cls.list_models():
                raise HTTPException(404, f"분류 모델을 찾을 수 없습니다: {model_name}")
            _classifier = core_cls.Classifier(model_name)
        return _classifier


@app.get("/models_cls")
def models_cls():
    return {"models": [core_cls.load_info(m) for m in core_cls.list_models()]}


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    review_threshold: float | None = Form(None, description="0~1. 최고 확률이 이 값 미만이면 REVIEW. 생략하면 모델 저장값"),
    return_heatmap: bool = Form(False, description="true 면 Grad-CAM 근거 히트맵을 base64 PNG 로 반환"),
):
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "이미지 파일을 읽을 수 없습니다.")
    clf = get_classifier(model)
    with _lock:
        r = clf.predict(image, with_cam=return_heatmap, review_threshold=review_threshold)
    body = {
        "result": r["verdict"],   # "OK" | "REVIEW"
        "label": r["label"],
        "confidence": round(r["confidence"] * 100, 1),
        "probs": {k: round(v * 100, 1) for k, v in r["probs"].items()},
        "elapsed_ms": round(r["elapsed_ms"], 1),
        "model": clf.model_name,
        "filename": file.filename,
    }
    if return_heatmap and r["heatmap"] is not None:
        buf = io.BytesIO()
        r["heatmap"].save(buf, format="PNG")
        body["heatmap_png_base64"] = base64.b64encode(buf.getvalue()).decode()
    _report(body | {"kind": "cls"})
    return body


# ----------------------------------------------------------------------
# 세그멘테이션 (Segmentation)
# ----------------------------------------------------------------------
import core_seg  # noqa: E402

_segmenter: core_seg.Segmenter | None = None


def get_segmenter(model_name: str | None) -> core_seg.Segmenter:
    global _segmenter
    with _lock:
        if model_name is None:
            if _segmenter is not None:
                return _segmenter
            models = core_seg.list_models()
            if not models:
                raise HTTPException(400, "학습된 세그 모델이 없습니다.")
            model_name = models[0]
        if _segmenter is None or _segmenter.model_name != model_name:
            if model_name not in core_seg.list_models():
                raise HTTPException(404, f"세그 모델을 찾을 수 없습니다: {model_name}")
            _segmenter = core_seg.Segmenter(model_name)
        return _segmenter


@app.get("/models_seg")
def models_seg():
    return {"models": [core_seg.load_info(m) for m in core_seg.list_models()]}


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    threshold: float | None = Form(None, description="0~1 픽셀 임계값. 생략하면 모델 저장값"),
    min_area: int | None = Form(None, description="이보다 작은 조각(픽셀 수)은 무시. 생략하면 모델 저장값"),
    return_overlay: bool = Form(False, description="true 면 불량 영역을 칠한 이미지를 base64 PNG 로 반환"),
    return_mask: bool = Form(False, description="true 면 흑백 마스크를 base64 PNG 로 반환"),
):
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "이미지 파일을 읽을 수 없습니다.")
    seg = get_segmenter(model)
    with _lock:
        r = seg.segment(image, threshold, min_area)
    body = {
        "result": r["verdict"],
        "defect_ratio": round(r["defect_ratio"], 3),   # 불량 픽셀 비율 (%)
        "max_prob": round(r["max_prob"] * 100, 1),
        "boxes": [list(b) for b in r["boxes"]],
        "elapsed_ms": round(r["elapsed_ms"], 1),
        "model": seg.model_name,
        "filename": file.filename,
    }
    for flag, key, img in ((return_overlay, "overlay_png_base64", r["overlay"]), (return_mask, "mask_png_base64", r["mask"])):
        if flag:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            body[key] = base64.b64encode(buf.getvalue()).decode()
    _report(body | {"kind": "seg"})
    return body


# ----------------------------------------------------------------------
# 공통: 모델 교체, 상태, 관리 서버 보고
# ----------------------------------------------------------------------
@app.post("/load_model")
def load_model(kind: str, model: str):
    """kind: anomaly | cls | seg. 에이전트가 새 모델을 배포한 뒤 호출한다."""
    t0 = time.perf_counter()
    if kind == "anomaly":
        name = get_inspector(model).model_name
    elif kind == "cls":
        name = get_classifier(model).model_name
    elif kind == "seg":
        name = get_segmenter(model).model_name
    else:
        raise HTTPException(400, "kind 는 anomaly, cls, seg 중 하나여야 합니다.")
    return {"kind": kind, "loaded_model": name, "load_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.get("/status")
def status():
    return {
        "device": os.environ.get("INSPECT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"),
        "loaded": {
            "anomaly": _inspector.model_name if _inspector else None,
            "cls": _classifier.model_name if _classifier else None,
            "seg": _segmenter.model_name if _segmenter else None,
        },
        "available": {"anomaly": core.list_models(), "cls": core_cls.list_models(), "seg": core_seg.list_models()},
    }


# 검사 결과를 관리 서버로 보내는 백그라운드 큐 (검사 지연에 영향을 주지 않도록 별도 스레드)
import queue  # noqa: E402
import socket  # noqa: E402

MANAGER_URL = os.environ.get("MANAGER_URL", "").rstrip("/")
AGENT_ID = os.environ.get("AGENT_ID") or socket.gethostname()
_report_q: "queue.Queue[dict]" = queue.Queue(maxsize=1000)


def _report(body: dict) -> None:
    if not MANAGER_URL:
        return
    item = {k: v for k, v in body.items() if not k.endswith("_base64")}
    item |= {"agent_id": AGENT_ID, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        _report_q.put_nowait(item)
    except queue.Full:
        pass


def _report_worker() -> None:
    import httpx

    with httpx.Client(base_url=MANAGER_URL, timeout=5) as c:
        while True:
            item = _report_q.get()
            try:
                c.post("/api/results", json=item)
            except Exception:  # noqa: BLE001
                pass


if MANAGER_URL:
    threading.Thread(target=_report_worker, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    # 서버 시작 시 첫 모델을 미리 올려 첫 검사 지연을 없앤다
    if core.list_models():
        get_inspector(None)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", "8000")))
