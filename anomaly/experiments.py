"""
모델 비교 실험
- 같은 데이터로 여러 설정을 차례로 학습하고, 같은 검증 기준으로 성능과 속도를 표로 비교한다.
- 결과 모델은 "<이름>__<설정>" 으로 저장되므로 마음에 드는 것을 그대로 쓰면 된다.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import core
import core_cls
import core_seg

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "experiments"
RESULTS_DIR.mkdir(exist_ok=True)

# 비교 후보 (UI 체크박스 이름 -> 학습 인자)
ANOMALY_CANDIDATES = {
    "patchcore_256": {"model_type": "patchcore", "image_size": 256, "coreset_ratio": 0.1},
    "patchcore_fast_256": {"model_type": "patchcore_fast", "image_size": 256, "coreset_ratio": 0.1},
    "patchcore_320": {"model_type": "patchcore", "image_size": 320, "coreset_ratio": 0.1},
    "patchcore_256_dense": {"model_type": "patchcore", "image_size": 256, "coreset_ratio": 0.25},
    "dinomaly_256": {"model_type": "dinomaly", "image_size": 256, "epochs": 20},
}
CLS_CANDIDATES = {
    "resnet18_224": {"backbone": "resnet18", "image_size": 224},
    "resnet50_224": {"backbone": "resnet50", "image_size": 224},
    "efficientnet_b0_224": {"backbone": "efficientnet_b0", "image_size": 224},
    "convnext_tiny_224": {"backbone": "convnext_tiny", "image_size": 224},
}
SEG_CANDIDATES = {
    "unet_resnet18_256": {"encoder": "resnet18", "image_size": 256},
    "unet_resnet34_256": {"encoder": "resnet34", "image_size": 256},
    "unet_efficientnet-b0_256": {"encoder": "efficientnet-b0", "image_size": 256},
    "unet_resnet18_384": {"encoder": "resnet18", "image_size": 384},
}
CANDIDATES = {"anomaly": ANOMALY_CANDIDATES, "cls": CLS_CANDIDATES, "seg": SEG_CANDIDATES}


def _time_inference(fn, n=20) -> float:
    for _ in range(3):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def run_comparison(kind: str, base_name: str, data: dict, candidates: list[str], common: dict | None = None, log=print) -> list[dict]:
    """kind: anomaly | cls | seg.  data: 학습 함수에 넘길 데이터 경로들.  candidates: CANDIDATES[kind] 의 키 목록.
    common: 모든 후보에 공통으로 적용할 추가 인자 (roi_path, aug, batch_size 등)."""
    rows = []
    common = common or {}
    for i, cand in enumerate(candidates, 1):
        cfg = CANDIDATES[kind][cand]
        name = f"{base_name}__{cand}"
        log(f"===== [{i}/{len(candidates)}] {cand} 학습 시작 =====")
        t0 = time.time()
        try:
            if kind == "anomaly":
                info = core.train_model(name, data["normal_dir"], data.get("abnormal_dir"), log=log, **{**common, **cfg})
                insp = core.Inspector(name)
                img = Image.open(core.list_images(Path(data["normal_dir"]))[0]).convert("RGB")
                ms = _time_inference(lambda: insp.inspect(img))
                m = info["metrics"]
                row = {"설정": cand, "이미지 AUROC": m.get("image_AUROC"), "이미지 F1": m.get("image_F1Score"),
                       "픽셀 AUROC": m.get("pixel_AUROC"), "검사 ms": round(ms, 1)}
                score = m.get("image_F1Score") or m.get("image_AUROC") or 0
            elif kind == "cls":
                info = core_cls.train_classifier(name, data["data_dir"], log=log, **{**common, **cfg})
                clf = core_cls.Classifier(name)
                img = Image.open(next(Path(data["data_dir"]).rglob("*.png"))).convert("RGB")
                ms = _time_inference(lambda: clf.predict(img, with_cam=False))
                row = {"설정": cand, "검증 정확도": info["val_accuracy"], "검사 ms": round(ms, 1)}
                score = info["val_accuracy"]
            else:
                info = core_seg.train_segmenter(name, data["data_dir"], log=log, **{**common, **cfg})
                seg = core_seg.Segmenter(name)
                img = Image.open(next((Path(data["data_dir"]) / "images").glob("*.png"))).convert("RGB")
                ms = _time_inference(lambda: seg.segment(img))
                row = {"설정": cand, "검증 IoU": info["val_iou"], "검사 ms": round(ms, 1)}
                score = info["val_iou"]
            row |= {"학습 초": round(time.time() - t0, 1), "모델 이름": name, "_score": score, "오류": ""}
        except Exception as e:  # noqa: BLE001
            row = {"설정": cand, "모델 이름": name, "_score": -1, "오류": str(e)[:120], "학습 초": round(time.time() - t0, 1)}
            log(f"  실패: {e}")
        rows.append(row)
    rows.sort(key=lambda r: (-(r["_score"] or 0), r.get("검사 ms", 1e9)))
    out = RESULTS_DIR / f"{kind}_{base_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"kind": kind, "data": data, "common": common, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"비교 결과 저장: {out}")
    return rows
