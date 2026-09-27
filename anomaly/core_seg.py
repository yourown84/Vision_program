"""
세그멘테이션(Segmentation) 핵심 기능 (UI와 분리)
- train_segmenter()  : 이미지 + 마스크 쌍으로 U-Net 학습 -> models_seg/<이름>/model.pt
- Segmenter          : 저장된 모델로 이미지 1장의 불량 영역을 픽셀 단위로 찾음 (마스크, 박스, 면적, 오버레이)
- make_masks_from_anomaly() : 어노멀리 모델로 초기 마스크를 자동 생성 (라벨링 보조)

데이터 폴더 구조:
  <root>/images/*.png   원본 이미지
  <root>/masks/*.png    같은 이름의 마스크 (흑백, 0=배경, 0이 아닌 값=불량)
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
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import roi as roi_util

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models_seg"
MODELS_DIR.mkdir(exist_ok=True)
DEVICE = os.environ.get("INSPECT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DEFAULT_AUG = {"hflip": True, "vflip": True, "rotate": 90, "brightness": 0.15, "contrast": 0.0, "scale": 0.0}


def _find_pairs(root: Path) -> list[tuple[Path, Path]]:
    img_dir, mask_dir = root / "images", root / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"{root} 안에 images/ 와 masks/ 폴더가 필요합니다.")
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    pairs = [(p, masks[p.stem]) for p in sorted(img_dir.iterdir())
             if p.suffix.lower() in IMAGE_EXTS and p.stem in masks]
    if not pairs:
        raise ValueError("이미지와 이름이 같은 마스크 쌍을 찾지 못했습니다.")
    return pairs


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class _SegDataset(Dataset):
    def __init__(self, pairs, image_size, train, aug=None, roi_path=None):
        self.pairs, self.size, self.train = pairs, image_size, train
        self.aug = {**DEFAULT_AUG, **(aug or {})}
        self.roi_path = roi_path
        self.roi_cache: dict[tuple[int, int], np.ndarray | None] = {}

    def __len__(self):
        return len(self.pairs)

    def _roi(self, wh):
        if wh not in self.roi_cache:
            self.roi_cache[wh] = roi_util.load_roi(self.roi_path, wh)
        return self.roi_cache[wh]

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        r = self._roi((img.shape[1], img.shape[0]))
        if r is not None:
            img = roi_util.apply_to_image(img, r)
            mask = mask * r
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.train:
            a = self.aug
            if a.get("hflip") and np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if a.get("vflip") and np.random.rand() < 0.5:
                img, mask = img[::-1], mask[::-1]
            rot = float(a.get("rotate", 0))
            if rot >= 90:
                k = np.random.randint(4)
                img, mask = np.rot90(img, k), np.rot90(mask, k)
            elif rot > 0:
                ang = np.random.uniform(-rot, rot)
                M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, 1.0)
                img = cv2.warpAffine(np.ascontiguousarray(img), M, (self.size, self.size), borderValue=(128, 128, 128))
                mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (self.size, self.size), flags=cv2.INTER_NEAREST)
            if a.get("scale", 0):
                s = np.random.uniform(1 - a["scale"], 1 + a["scale"])
                M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), 0, s)
                img = cv2.warpAffine(np.ascontiguousarray(img), M, (self.size, self.size), borderValue=(128, 128, 128))
                mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (self.size, self.size), flags=cv2.INTER_NEAREST)
            b, c = float(a.get("brightness", 0)), float(a.get("contrast", 0))
            if b or c:
                alpha = np.random.uniform(1 - c, 1 + c) if c else 1.0
                beta = np.random.uniform(-b, b) * 255 if b else 0.0
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        return _to_tensor(img), torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)


# ----------------------------------------------------------------------
# 학습
# ----------------------------------------------------------------------
def train_segmenter(
    model_name: str,
    data_dir: str,
    encoder: str = "resnet18",
    image_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 8,
    patience: int = 0,
    aug: dict | None = None,
    roi_path: str | None = None,
    log=print,
) -> dict:
    import segmentation_models_pytorch as smp

    pairs = _find_pairs(Path(data_dir))
    n_defect = sum(cv2.imread(str(m), cv2.IMREAD_GRAYSCALE).max() > 0 for _, m in pairs)
    log(f"이미지-마스크 쌍 {len(pairs)}개 (불량 포함 {n_defect}장, 정상 {len(pairs) - n_defect}장)")
    if len(pairs) < 10:
        raise ValueError("최소 10쌍 이상 필요합니다.")
    if roi_path and Path(roi_path).exists():
        log(f"ROI 마스크 사용: {roi_path}")

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(pairs))
    n_val = max(2, int(len(pairs) * 0.2))
    val_pairs = [pairs[i] for i in idx[:n_val]]
    train_pairs = [pairs[i] for i in idx[n_val:]]
    train_loader = DataLoader(_SegDataset(train_pairs, image_size, True, aug, roi_path), batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(_SegDataset(val_pairs, image_size, False, None, roi_path), batch_size=int(batch_size))
    log(f"학습 {len(train_pairs)}장 / 검증 {len(val_pairs)}장, U-Net({encoder}), {epochs} 에폭, 배치 {batch_size}, 학습률 {lr}")

    model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_iou, best_state, t0, since_best, history = -1.0, None, time.time(), 0, []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                out = model(x)
                loss = bce(out, y) + dice(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()
        iou = _evaluate(model, val_loader)
        history.append({"epoch": epoch, "loss": round(total / n, 4), "val_iou": round(iou, 4)})
        log(f"[{epoch}/{epochs}] loss={total / n:.4f}  검증 IoU={iou * 100:.1f}%")
        if iou > best_iou:
            best_iou, since_best = iou, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if patience and since_best >= int(patience):
                log(f"{patience} 에폭 동안 개선 없음 -> 조기 종료")
                break

    out_dir = MODELS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model.pt")
    info = {
        "model_name": model_name,
        "encoder": encoder,
        "image_size": image_size,
        "n_pairs": len(pairs),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "patience": int(patience),
        "aug": {**DEFAULT_AUG, **(aug or {})},
        "roi": roi_util.copy_into_model(roi_path, out_dir),
        "val_iou": round(best_iou, 4),
        "history": history,
        "threshold": 0.5,
        "min_area": 20,
        "train_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"최고 검증 IoU {best_iou * 100:.1f}%  모델 저장: {out_dir / 'model.pt'}")
    log("  (IoU = 예측 영역과 정답 영역이 겹치는 비율. 70% 이상이면 실용적, 50% 이하면 데이터 보강 필요)")
    return info


@torch.no_grad()
def _evaluate(model, loader, thr=0.5):
    model.eval()
    inter, union = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = (torch.sigmoid(model(x)) > thr).float()
        inter += (p * y).sum().item()
        union += ((p + y) > 0).float().sum().item()
    return inter / union if union > 0 else 1.0


# ----------------------------------------------------------------------
# 모델 관리
# ----------------------------------------------------------------------
def list_models() -> list[str]:
    return sorted(p.name for p in MODELS_DIR.iterdir() if (p / "info.json").exists())


def load_info(model_name: str) -> dict:
    return json.loads((MODELS_DIR / model_name / "info.json").read_text(encoding="utf-8"))


def save_info(model_name: str, **updates) -> dict:
    info = load_info(model_name)
    info.update(updates)
    (MODELS_DIR / model_name / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def save_threshold(model_name: str, threshold: float, min_area: int | None = None) -> None:
    upd = {"threshold": float(threshold)}
    if min_area is not None:
        upd["min_area"] = int(min_area)
    save_info(model_name, **upd)


# ----------------------------------------------------------------------
# 추론
# ----------------------------------------------------------------------
class Segmenter:
    def __init__(self, model_name: str):
        import segmentation_models_pytorch as smp

        self.model_name = model_name
        self.info = load_info(model_name)
        self.model = smp.Unet(encoder_name=self.info["encoder"], encoder_weights=None, in_channels=3, classes=1)
        self.model.load_state_dict(torch.load(MODELS_DIR / model_name / "model.pt", map_location="cpu"))
        self.model.to(DEVICE).eval()
        self.size = self.info["image_size"]
        self.roi_path = roi_util.model_roi_path(MODELS_DIR / model_name, self.info)
        self._roi_cache: dict[tuple[int, int], np.ndarray | None] = {}
        self.segment(Image.new("RGB", (64, 64)))  # 워밍업

    def _roi(self, wh):
        if wh not in self._roi_cache:
            self._roi_cache[wh] = roi_util.load_roi(self.roi_path, wh)
        return self._roi_cache[wh]

    @torch.no_grad()
    def segment(self, image, threshold: float | None = None, min_area: int | None = None) -> dict:
        """반환: is_ng, verdict, defect_ratio(불량 픽셀 비율 %), boxes[(x,y,w,h)], mask(PIL), overlay(PIL), max_prob, elapsed_ms"""
        if threshold is None:
            threshold = self.info.get("threshold", 0.5)
        if min_area is None:
            min_area = int(self.info.get("min_area", 20))
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGB")
        W, H = image.size
        arr = np.asarray(image)
        roi = self._roi((W, H))
        model_in = roi_util.apply_to_image(arr, roi)
        x = _to_tensor(cv2.resize(model_in, (self.size, self.size), interpolation=cv2.INTER_AREA)).unsqueeze(0).to(DEVICE)

        t0 = time.perf_counter()
        prob = torch.sigmoid(self.model(x))[0, 0].float().cpu().numpy()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        prob = roi_util.apply_to_map(prob, roi)
        mask = (prob > threshold).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area >= min_area:
                boxes.append((int(bx), int(by), int(bw), int(bh)))
            else:
                mask[labels == i] = 0  # 너무 작은 조각은 노이즈로 보고 제거
        overlay = arr.copy()
        red = np.zeros_like(arr); red[..., 0] = 255
        m3 = mask.astype(bool)
        overlay[m3] = (0.5 * overlay[m3] + 0.5 * red[m3]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

        is_ng = len(boxes) > 0
        return {
            "is_ng": is_ng,
            "verdict": "NG" if is_ng else "OK",
            "defect_ratio": float(mask.mean() * 100),
            "boxes": boxes,
            "mask": Image.fromarray(mask * 255),
            "overlay": roi_util.draw_roi_outline(Image.fromarray(overlay), roi),
            "max_prob": float(prob.max()),
            "threshold": threshold,
            "elapsed_ms": elapsed_ms,
        }


# ----------------------------------------------------------------------
# 라벨링 보조: 어노멀리 모델로 초기 마스크 생성
# ----------------------------------------------------------------------
def make_masks_from_anomaly(anomaly_model: str, image_dir: str, out_dir: str, threshold: float = 0.5, log=print) -> dict:
    """불량 이미지 폴더를 어노멀리 모델로 검사해서 images/ + masks/ 데이터셋을 만든다.
    자동 생성된 마스크는 사람이 확인/수정하는 것을 전제로 한 초안이다."""
    import core

    src = Path(image_dir)
    dst = Path(out_dir)
    paths = core.list_images(src)
    if not paths:
        raise ValueError(f"이미지가 없습니다: {src}")
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    insp = core.Inspector(anomaly_model)
    n_ng = 0
    for i, p in enumerate(paths):
        r = insp.inspect(p, pixel_threshold=threshold)
        img = Image.open(p).convert("RGB")
        img.save(dst / "images" / f"{p.stem}.png")
        r["mask"].save(dst / "masks" / f"{p.stem}.png")
        n_ng += int(len(r["boxes"]) > 0)
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(paths)} 처리")
    log(f"완료: {len(paths)}장 -> {dst}  (불량 영역이 잡힌 이미지 {n_ng}장)")
    log("masks/ 폴더의 흑백 이미지를 그림판 등으로 열어 잘못 잡힌 부분을 수정한 뒤 세그 학습에 사용하세요.")
    return {"n_images": len(paths), "n_ng": n_ng, "out_dir": str(dst)}
