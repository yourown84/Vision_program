"""
분류(Classification) 핵심 기능 (UI와 분리)
- train_classifier() : 클래스별 폴더로 CNN 미세조정 -> models_cls/<이름>/model.pt
- Classifier         : 저장된 모델로 이미지 1장 분류 (클래스, 확률, 재검 판정, Grad-CAM 근거 히트맵)

데이터 폴더 구조: <root>/<클래스이름>/*.png  (폴더 이름이 정답 라벨)
"""
import json
import os

os.environ.setdefault("HF_HOME", str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".cache" / "torch"))
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import roi as roi_util

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models_cls"
MODELS_DIR.mkdir(exist_ok=True)
DEVICE = os.environ.get("INSPECT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")  # INSPECT_DEVICE=cpu 로 강제 가능
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEFAULT_AUG = {"hflip": True, "vflip": True, "rotate": 10, "brightness": 0.15, "contrast": 0.15, "scale": 0.0}


class _RoiTransform:
    """ROI 바깥을 회색으로 채우는 전처리 (학습/검사 동일하게 적용)."""

    def __init__(self, roi_path):
        self.roi_path = roi_path
        self.cache: dict[tuple[int, int], np.ndarray | None] = {}

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.roi_path:
            return img
        key = img.size
        if key not in self.cache:
            self.cache[key] = roi_util.load_roi(self.roi_path, key)
        r = self.cache[key]
        return img if r is None else Image.fromarray(roi_util.apply_to_image(np.asarray(img.convert("RGB")), r))


def _transforms(image_size: int, train: bool, aug: dict | None = None, roi_path=None):
    aug = {**DEFAULT_AUG, **(aug or {})}
    ops = [_RoiTransform(roi_path), transforms.Resize((image_size, image_size))]
    if train:
        if aug.get("hflip"):
            ops.append(transforms.RandomHorizontalFlip())
        if aug.get("vflip"):
            ops.append(transforms.RandomVerticalFlip())
        if aug.get("rotate", 0):
            ops.append(transforms.RandomRotation(float(aug["rotate"])))
        if aug.get("scale", 0):
            s = float(aug["scale"])
            ops.append(transforms.RandomAffine(degrees=0, scale=(1 - s, 1 + s)))
        if aug.get("brightness", 0) or aug.get("contrast", 0):
            ops.append(transforms.ColorJitter(float(aug.get("brightness", 0)), float(aug.get("contrast", 0))))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


# ----------------------------------------------------------------------
# 학습
# ----------------------------------------------------------------------
def train_classifier(
    model_name: str,
    data_dir: str,
    backbone: str = "resnet18",
    image_size: int = 224,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 16,
    patience: int = 0,
    class_weight: bool = True,
    aug: dict | None = None,
    roi_path: str | None = None,
    log=print,
) -> dict:
    """patience > 0 이면 검증 정확도가 그 에폭 수 동안 오르지 않을 때 조기 종료.
    class_weight=True 이면 이미지 수가 적은 클래스에 가중치를 줘 불균형을 보정."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {data_dir}")

    full = datasets.ImageFolder(str(data_dir))
    classes = full.classes
    if len(classes) < 2:
        raise ValueError("클래스 폴더가 2개 이상 필요합니다 (예: good, ng).")
    counts = {c: 0 for c in classes}
    for _, y in full.samples:
        counts[classes[y]] += 1
    log("클래스별 이미지 수: " + ", ".join(f"{c}={n}" for c, n in counts.items()))
    for c, n in counts.items():
        if n < 5:
            raise ValueError(f"클래스 '{c}' 이미지가 너무 적습니다 ({n}장). 최소 5장 이상 필요합니다.")
    if roi_path and Path(roi_path).exists():
        log(f"ROI 마스크 사용: {roi_path}")

    n_val = max(1, int(len(full) * 0.2))
    gen = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(len(full)), [len(full) - n_val, n_val], generator=gen)
    train_ds = datasets.ImageFolder(str(data_dir), transform=_transforms(image_size, True, aug, roi_path))
    val_ds = datasets.ImageFolder(str(data_dir), transform=_transforms(image_size, False, None, roi_path))
    train_loader = DataLoader(torch.utils.data.Subset(train_ds, list(train_idx)), batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, list(val_idx)), batch_size=int(batch_size))
    log(f"학습 {len(train_idx)}장 / 검증 {len(val_idx)}장, 모델 {backbone}, {epochs} 에폭, 배치 {batch_size}, 학습률 {lr}")

    model = timm.create_model(backbone, pretrained=True, num_classes=len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    weights = None
    if class_weight:
        tot = sum(counts.values())
        weights = torch.tensor([tot / (len(classes) * counts[c]) for c in classes], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    out_dir = MODELS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, best_state, t0, since_best, history = -1.0, None, time.time(), 0, []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()
        acc, _ = _evaluate(model, val_loader)
        history.append({"epoch": epoch, "loss": round(total / n, 4), "val_acc": round(acc, 4)})
        log(f"[{epoch}/{epochs}] loss={total / n:.4f}  검증 정확도={acc * 100:.1f}%")
        if acc > best_acc:
            best_acc, since_best = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if patience and since_best >= int(patience):
                log(f"{patience} 에폭 동안 개선 없음 -> 조기 종료")
                break

    model.load_state_dict(best_state)
    _, confusion = _evaluate(model, val_loader, n_classes=len(classes))
    torch.save(best_state, out_dir / "model.pt")
    info = {
        "model_name": model_name,
        "backbone": backbone,
        "image_size": image_size,
        "classes": classes,
        "counts": counts,
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "patience": int(patience),
        "class_weight": bool(class_weight),
        "aug": {**DEFAULT_AUG, **(aug or {})},
        "roi": roi_util.copy_into_model(roi_path, out_dir),
        "val_accuracy": round(best_acc, 4),
        "confusion": confusion,  # confusion[정답][예측]
        "history": history,
        "review_threshold": 0.0,  # 최고 확률이 이 값(0~1) 미만이면 '재검'
        "train_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"최고 검증 정확도 {best_acc * 100:.1f}%  모델 저장: {out_dir / 'model.pt'}")
    return info


@torch.no_grad()
def _evaluate(model, loader, n_classes: int | None = None):
    model.eval()
    correct, n = 0, 0
    conf = [[0] * n_classes for _ in range(n_classes)] if n_classes else None
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        n += y.numel()
        if conf is not None:
            for t, p in zip(y.tolist(), pred.tolist()):
                conf[t][p] += 1
    return correct / max(n, 1), conf


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


# ----------------------------------------------------------------------
# 추론 + Grad-CAM
# ----------------------------------------------------------------------
class Classifier:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.info = load_info(model_name)
        self.classes = self.info["classes"]
        self.model = timm.create_model(self.info["backbone"], pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(MODELS_DIR / model_name / "model.pt", map_location="cpu"))
        self.model.to(DEVICE).eval()
        self.roi_path = roi_util.model_roi_path(MODELS_DIR / model_name, self.info)
        self.tf = _transforms(self.info["image_size"], False, None, self.roi_path)
        self.predict(Image.new("RGB", (64, 64)))  # 워밍업

    def predict(self, image, with_cam: bool = True, review_threshold: float | None = None) -> dict:
        """반환: label, confidence, verdict("OK"|"REVIEW"), probs{클래스:확률}, heatmap(PIL, Grad-CAM), elapsed_ms"""
        if review_threshold is None:
            review_threshold = float(self.info.get("review_threshold", 0.0))
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGB")
        W, H = image.size
        x = self.tf(image).unsqueeze(0).to(DEVICE)

        t0 = time.perf_counter()
        feats = self.model.forward_features(x)
        if with_cam:
            feats.retain_grad()
        logits = self.model.forward_head(feats)
        probs = torch.softmax(logits, 1)[0]
        idx = int(probs.argmax())
        cam = None
        if with_cam:
            self.model.zero_grad()
            logits[0, idx].backward()
            w = feats.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * feats).sum(1))[0].detach().float().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        heatmap = None
        if cam is not None:
            cam = cv2.resize(cam, (W, H))
            roi = roi_util.load_roi(self.roi_path, (W, H))
            cam = roi_util.apply_to_map(cam, roi)
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            heatmap = roi_util.draw_roi_outline(Image.fromarray(cv2.addWeighted(np.asarray(image), 0.55, heat, 0.45, 0)), roi)

        conf = float(probs[idx])
        return {
            "label": self.classes[idx],
            "confidence": conf,
            "verdict": "REVIEW" if conf < review_threshold else "OK",
            "probs": {c: float(p) for c, p in zip(self.classes, probs.tolist())},
            "heatmap": heatmap,
            "elapsed_ms": elapsed_ms,
        }
