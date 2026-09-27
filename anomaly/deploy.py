"""
배포 도우미: 학습된 모델을 zip 으로 묶어 관리 서버에 올리거나, 관리 서버에서 받아 설치한다.
- package_model(kind, name) -> zip bytes, version(내용 해시)
- upload_model(manager_url, kind, name)
- install_model(kind, name, zip_bytes, version)
UI(app.py 배포 탭)와 agent.py 가 함께 사용한다.
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).parent
MODEL_DIRS = {"anomaly": BASE_DIR / "models", "cls": BASE_DIR / "models_cls", "seg": BASE_DIR / "models_seg"}
KIND_LABELS = {"anomaly": "어노멀리", "cls": "분류", "seg": "세그"}


def model_dir(kind: str, name: str) -> Path:
    return MODEL_DIRS[kind] / name


def list_local(kind: str) -> list[str]:
    d = MODEL_DIRS[kind]
    return sorted(p.name for p in d.iterdir() if (p / "info.json").exists()) if d.exists() else []


def local_version(kind: str, name: str) -> str | None:
    p = model_dir(kind, name) / ".version"
    return p.read_text().strip() if p.exists() else None


def package_model(kind: str, name: str) -> tuple[bytes, str]:
    """모델 폴더 전체를 zip 으로 묶는다. version 은 파일 내용 해시 (같은 내용이면 같은 버전)."""
    d = model_dir(kind, name)
    if not (d / "info.json").exists():
        raise FileNotFoundError(f"모델이 없습니다: {kind}/{name}")
    buf = io.BytesIO()
    h = hashlib.sha256()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name != ".version":
                rel = p.relative_to(d).as_posix()
                data = p.read_bytes()
                h.update(rel.encode()); h.update(data)
                z.writestr(rel, data)
    return buf.getvalue(), h.hexdigest()[:12]


def upload_model(manager_url: str, kind: str, name: str) -> dict:
    data, version = package_model(kind, name)
    info = json.loads((model_dir(kind, name) / "info.json").read_text(encoding="utf-8"))
    with httpx.Client(base_url=manager_url.rstrip("/"), timeout=120) as c:
        r = c.post("/api/models/upload", data={"kind": kind, "name": name, "version": version,
                                              "info": json.dumps(info, ensure_ascii=False)},
                   files={"file": (f"{name}.zip", data, "application/zip")})
        r.raise_for_status()
        return r.json()


def install_model(kind: str, name: str, zip_bytes: bytes, version: str) -> Path:
    """zip 을 풀어 로컬 모델 폴더로 설치한다 (기존 폴더는 덮어씀)."""
    d = model_dir(kind, name)
    d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(d)
    (d / ".version").write_text(version)
    return d


def download_model(manager_url: str, kind: str, name: str) -> bytes:
    with httpx.Client(base_url=manager_url.rstrip("/"), timeout=300) as c:
        r = c.get(f"/api/models/{kind}/{name}/download")
        r.raise_for_status()
        return r.content
