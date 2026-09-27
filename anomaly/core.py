"""
어노멀리 디텍션 핵심 기능 (UI와 분리)
- train_model()   : 정상 이미지 폴더로 학습 -> models/<이름>/... /model.pt 저장
- Inspector       : 저장된 모델을 불러와 이미지 1장 검사 (스코어, OK/재검/NG 판정, 히트맵)
이 파일의 함수는 app.py(UI)뿐 아니라 API 서버(server.py)에서도 그대로 사용한다.

지원 모델 (ANOMALY_MODELS):
  patchcore       정상 조각을 기억해 두고 비교. 빠르고 데이터가 적어도 안정적 (기본)
  patchcore_fast  같은 방식, 작은 백본(resnet18). 더 빠르고 가벼움
  dinomaly        DINOv2 파운데이션 모델 기반. 정확도 최상위, 학습 시간이 더 걸림
"""
import json
import os

os.environ.setdefault("HF_HOME", str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".cache" / "torch"))
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import roi as roi_util

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"
MODELS_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

ANOMALY_MODELS = {
    "patchcore": "PatchCore (기본, 빠름)",
    "patchcore_fast": "PatchCore 경량 (resnet18)",
    "dinomaly": "Dinomaly (DINOv2 파운데이션, 정확도 우선)",
}

# 증강 기본값 (UI 에서 바꿀 수 있음)
DEFAULT_AUG = {"hflip": False, "vflip": False, "rotate": 0, "brightness": 0.0, "contrast": 0.0}


def _roi_fill_transform(roi_path: str | None):
    """ROI 바깥을 회색으로 채우는 anomalib 용 transform. 마스크(gt_mask)는 건드리지 않는다."""
    if not roi_path or not Path(roi_path).exists():
        return None
    from torchvision import tv_tensors
    from torchvision.transforms import v2

    class RoiFill(v2.Transform):
        def __init__(self, path):
            super().__init__()
            self.path = path
            self.cache: dict[tuple[int, int], torch.Tensor | None] = {}

        def transform(self, inpt, params):
            if isinstance(inpt, tv_tensors.Mask) or not isinstance(inpt, torch.Tensor) or inpt.ndim < 3 or inpt.shape[-3] != 3:
                return inpt
            H, W = inpt.shape[-2:]
            if (W, H) not in self.cache:
                r = roi_util.load_roi(self.path, (W, H))
                self.cache[(W, H)] = None if r is None else torch.from_numpy(r).bool()
            r = self.cache[(W, H)]
            if r is None:
                return inpt
            fill = 128 if inpt.dtype == torch.uint8 else 0.5
            out = inpt.clone()
            out[..., :, ~r] = fill
            return tv_tensors.wrap(out, like=inpt) if isinstance(inpt, tv_tensors.TVTensor) else out

    return RoiFill(roi_path)


def build_augmentations(aug: dict | None, roi_path: str | None = None):
    """증강 설정 -> torchvision v2 transform (없으면 None). roi_path 가 있으면 ROI 채우기를 맨 앞에 넣는다."""
    from torchvision.transforms import v2

    ops = []
    roi_t = _roi_fill_transform(roi_path)
    if roi_t is not None:
        ops.append(roi_t)
    if not aug:
        return v2.Compose(ops) if ops else None
    if aug.get("hflip"):
        ops.append(v2.RandomHorizontalFlip())
    if aug.get("vflip"):
        ops.append(v2.RandomVerticalFlip())
    if aug.get("rotate", 0):
        ops.append(v2.RandomRotation(float(aug["rotate"])))
    if aug.get("brightness", 0) or aug.get("contrast", 0):
        ops.append(v2.ColorJitter(brightness=float(aug.get("brightness", 0)), contrast=float(aug.get("contrast", 0))))
    if aug.get("scale", 0):
        s = float(aug["scale"])
        ops.append(v2.RandomAffine(degrees=0, scale=(1 - s, 1 + s)))
    return v2.Compose(ops) if ops else None


# ----------------------------------------------------------------------
# 학습
# ----------------------------------------------------------------------
def train_model(
    model_name: str,
    normal_dir: str,
    abnormal_dir: str | None = None,
    image_size: int = 256,
    coreset_ratio: float = 0.1,
    model_type: str = "patchcore",
    batch_size: int = 8,
    epochs: int = 20,
    aug: dict | None = None,
    roi_path: str | None = None,
    log=print,
) -> dict:
    """정상 이미지 폴더로 학습하고 배포용 model.pt 로 내보낸다.

    abnormal_dir 이 없으면 anomalib 이 정상 이미지에 가짜 결함을 합성해서 임계값(threshold)을 자동으로 정한다.
    epochs 는 dinomaly 에만 쓰인다 (patchcore 는 1회 통과 학습).
    """
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Dinomaly, Patchcore

    if model_type not in ANOMALY_MODELS:
        raise ValueError(f"model_type 은 {list(ANOMALY_MODELS)} 중 하나여야 합니다.")
    normal_dir = Path(normal_dir)
    if not normal_dir.is_dir():
        raise FileNotFoundError(f"정상 이미지 폴더가 없습니다: {normal_dir}")
    n_normal = len(list_images(normal_dir))
    if n_normal < 10:
        raise ValueError(f"정상 이미지가 너무 적습니다 ({n_normal}장). 최소 10장 이상 필요합니다.")

    has_abnormal = bool(abnormal_dir) and Path(abnormal_dir).is_dir()
    log(f"정상 이미지 {n_normal}장 발견, 모델 {ANOMALY_MODELS[model_type]}")
    if has_abnormal:
        log(f"불량 이미지 {len(list_images(Path(abnormal_dir)))}장 발견 (임계값 결정과 성능 평가에 사용)")
    else:
        log("불량 이미지 없음 -> 정상 이미지에 가짜 결함을 합성해서 임계값을 정합니다")
    if roi_path and Path(roi_path).exists():
        log(f"ROI 마스크 사용: {roi_path}")
    if aug and any(aug.get(k) for k in ("hflip", "vflip", "rotate", "brightness", "contrast", "scale")):
        log(f"증강: {aug}")

    datamodule = Folder(
        name=model_name,
        normal_dir=str(normal_dir),
        abnormal_dir=str(abnormal_dir) if has_abnormal else None,
        test_split_mode="from_dir" if has_abnormal else "synthetic",
        val_split_mode="from_test",
        val_split_ratio=0.5,
        train_batch_size=int(batch_size),
        eval_batch_size=int(batch_size),
        num_workers=0,  # Windows 에서는 0 이 안전
        train_augmentations=build_augmentations(aug, roi_path),
        val_augmentations=_roi_fill_transform(roi_path),
        test_augmentations=_roi_fill_transform(roi_path),
        seed=42,
    )

    engine_kwargs: dict = {}
    if model_type == "dinomaly":
        # DINOv2 는 14픽셀 패치 단위라 입력 크기가 14의 배수여야 한다 -> 가장 가까운 배수로 맞춤
        adj = max(14, round(image_size / 14) * 14)
        if adj != image_size:
            log(f"Dinomaly 입력 크기를 {image_size} -> {adj} 로 조정 (14의 배수)")
            image_size = adj
        model = Dinomaly(pre_processor=Dinomaly.configure_pre_processor(image_size=(image_size, image_size), crop_size=image_size))
        engine_kwargs["max_epochs"] = int(epochs)
    else:
        backbone = "resnet18" if model_type == "patchcore_fast" else "wide_resnet50_2"
        model = Patchcore(
            backbone=backbone,
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=coreset_ratio,
            pre_processor=Patchcore.configure_pre_processor(image_size=(image_size, image_size)),
        )

    engine = Engine(default_root_dir=str(RUNS_DIR), logger=False, enable_progress_bar=False, **engine_kwargs)

    t0 = time.time()
    log("학습 시작...")
    engine.fit(model=model, datamodule=datamodule)
    log(f"학습 완료 ({time.time() - t0:.1f}초)")

    log("검증 데이터로 성능 평가 중...")
    metrics = engine.test(model=model, datamodule=datamodule, verbose=False)
    metrics = {k: float(v) for k, v in (metrics[0] if metrics else {}).items()}

    out_dir = MODELS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    engine.export(model=model, export_type=ExportType.TORCH, export_root=str(out_dir), model_file_name="model")
    model_path = _find_model_file(out_dir)

    info = {
        "model_name": model_name,
        "model_type": model_type,
        "normal_dir": str(normal_dir),
        "abnormal_dir": str(abnormal_dir) if has_abnormal else None,
        "image_size": image_size,
        "coreset_ratio": coreset_ratio,
        "batch_size": int(batch_size),
        "epochs": int(epochs) if model_type == "dinomaly" else 1,
        "aug": aug or {},
        "roi": roi_util.copy_into_model(roi_path, out_dir),
        "n_normal": n_normal,
        "train_seconds": round(time.time() - t0, 1),
        "metrics": metrics,
        "threshold": 0.5,          # NG 임계값 (anomalib 정규화 기준 0.5 = 자동 임계값)
        "review_threshold": 0.5,   # 이 값 이상 ~ threshold 미만이면 '재검'. threshold 와 같으면 재검 구간 없음
        "pixel_threshold": 0.5,    # 히트맵/불량 영역 표시용 픽셀 임계값
        "model_path": str(model_path),
    }
    (out_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"모델 저장: {model_path}")
    return info


def _find_model_file(model_dir: Path) -> Path:
    candidates = list(model_dir.rglob("model.pt"))
    if not candidates:
        raise FileNotFoundError(f"model.pt 를 찾지 못했습니다: {model_dir}")
    return candidates[0]


# ----------------------------------------------------------------------
# 모델 관리
# ----------------------------------------------------------------------
def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def list_models() -> list[str]:
    return sorted(p.name for p in MODELS_DIR.iterdir() if (p / "info.json").exists())


def load_info(model_name: str) -> dict:
    return json.loads((MODELS_DIR / model_name / "info.json").read_text(encoding="utf-8"))


def save_info(model_name: str, **updates) -> dict:
    info = load_info(model_name)
    info.update(updates)
    (MODELS_DIR / model_name / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def save_threshold(model_name: str, threshold: float, review_threshold: float | None = None, pixel_threshold: float | None = None) -> None:
    upd = {"threshold": float(threshold)}
    if review_threshold is not None:
        upd["review_threshold"] = min(float(review_threshold), float(threshold))
    if pixel_threshold is not None:
        upd["pixel_threshold"] = float(pixel_threshold)
    save_info(model_name, **upd)


def verdict(score: float, threshold: float, review_threshold: float | None) -> str:
    """OK / REVIEW / NG 3단계 판정."""
    if score > threshold:
        return "NG"
    if review_threshold is not None and review_threshold < threshold and score > review_threshold:
        return "REVIEW"
    return "OK"


# ----------------------------------------------------------------------
# 검사 (추론)
# ----------------------------------------------------------------------
class Inspector:
    """저장된 모델 하나를 메모리에 올려두고 이미지를 검사한다."""

    def __init__(self, model_name: str):
        # anomalib 이 저장한 model.pt 는 pickle 형식이라 신뢰 표시가 필요하다 (우리가 직접 학습한 파일)
        os.environ.setdefault("TRUST_REMOTE_CODE", "1")
        from anomalib.deploy import TorchInferencer

        self.model_name = model_name
        self.info = load_info(model_name)
        model_dir = MODELS_DIR / model_name
        model_path = Path(self.info["model_path"])
        if not model_path.exists():  # 다른 PC 에서 복사해 온 경우 절대 경로가 다르다
            model_path = _find_model_file(model_dir)
        self.inferencer = TorchInferencer(path=model_path, device=os.environ.get("INSPECT_DEVICE", "auto"))
        self.roi_path = roi_util.model_roi_path(model_dir, self.info)
        self._roi_cache: dict[tuple[int, int], np.ndarray | None] = {}
        size = self.info["image_size"]
        self.inferencer.predict(np.zeros((size, size, 3), np.uint8))  # 워밍업

    def _roi(self, W: int, H: int) -> np.ndarray | None:
        key = (W, H)
        if key not in self._roi_cache:
            self._roi_cache[key] = roi_util.load_roi(self.roi_path, (W, H))
        return self._roi_cache[key]

    def inspect(self, image, threshold: float | None = None, review_threshold: float | None = None,
                pixel_threshold: float | None = None) -> dict:
        """이미지 1장 검사.

        반환:
          score            : 0~1 이상 점수 (0.5 = 학습 시 자동 임계값)
          verdict          : "OK" | "REVIEW" | "NG"
          is_ng            : NG 여부
          heatmap          : 원본 위에 이상 영역을 색으로 겹친 이미지 (PIL)
          mask             : 불량 영역 흑백 마스크 (PIL)
          boxes            : 불량 영역 바운딩 박스 [(x, y, w, h), ...] (원본 좌표)
          elapsed_ms       : 모델 처리 시간
        """
        info = self.info
        threshold = info.get("threshold", 0.5) if threshold is None else float(threshold)
        review_threshold = info.get("review_threshold", threshold) if review_threshold is None else float(review_threshold)
        pixel_threshold = info.get("pixel_threshold", 0.5) if pixel_threshold is None else float(pixel_threshold)

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGB")
        W, H = image.size
        roi = self._roi(W, H)
        base = np.asarray(image)
        model_in = roi_util.apply_to_image(base, roi)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.inferencer.predict(Image.fromarray(model_in))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        amap = pred.anomaly_map.detach().squeeze().float().cpu().numpy()  # 0~1 정규화된 픽셀 점수
        amap = cv2.resize(amap, (W, H), interpolation=cv2.INTER_LINEAR)
        amap = roi_util.apply_to_map(amap, roi)  # 표시용 지도만 ROI 바깥을 0 으로
        # 이미지 점수는 모델 값을 그대로 쓴다. 학습/검사 모두 ROI 바깥을 회색으로 채우므로 바깥 영역은 점수에 영향이 없다.
        score = float(pred.pred_score.detach().flatten()[0])

        mask = (amap > pixel_threshold).astype(np.uint8)
        boxes = _mask_to_boxes(mask)

        heat = cv2.applyColorMap((np.clip(amap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        for (x, y, w, h) in boxes:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
        heatmap = roi_util.draw_roi_outline(Image.fromarray(overlay), roi)

        v = verdict(score, threshold, review_threshold)
        return {
            "score": score,
            "verdict": v,
            "is_ng": v == "NG",
            "threshold": threshold,
            "review_threshold": review_threshold,
            "pixel_threshold": pixel_threshold,
            "heatmap": heatmap,
            "mask": Image.fromarray(mask * 255),
            "boxes": boxes,
            "elapsed_ms": elapsed_ms,
        }


def _mask_to_boxes(mask: np.ndarray, min_area: int = 30) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes
