"""
비전 검사 UI (Gradio)
- 1~3 어노멀리 : 정상 이미지만으로 학습 / 검사 / 일괄 검사
- 4~5 분류     : 클래스별 폴더로 학습 / 검사 (Grad-CAM)
- 6~8 세그     : 마스크 초안 생성 / 학습 / 검사 (픽셀 단위 불량 영역)
- 9   배포     : 관리 서버에 모델 업로드, 라인 PC 배포 지정, 결과 조회
- 10  ROI      : 검사 영역을 붓으로 칠해 마스크 파일로 저장
- 11  모델 비교 : 같은 데이터로 여러 모델을 학습해 성능/속도 비교
실행: start_ui.bat 더블클릭
"""
import os
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import httpx
import numpy as np

import core
import core_cls
import core_seg
import deploy
import experiments
import roi as roi_util

# 현재 메모리에 올라온 검사기 (모델을 바꿀 때만 다시 로드)
_inspector: core.Inspector | None = None
_classifier: core_cls.Classifier | None = None
_segmenter: core_seg.Segmenter | None = None
_lock = threading.Lock()

SAMPLE_NORMAL = str(core.BASE_DIR / "data" / "sample" / "train" / "good")
SAMPLE_DEFECT = str(core.BASE_DIR / "data" / "sample" / "test" / "defect")
SAMPLE_CLS = str(core.BASE_DIR / "data" / "sample_cls")
SAMPLE_SEG = str(core.BASE_DIR / "data" / "sample_seg")
ROI_DIR = core.BASE_DIR / "data" / "roi"
DEFAULT_MANAGER = os.environ.get("MANAGER_URL", "http://127.0.0.1:9000")

VERDICT_KO = {"OK": "OK (정상)", "REVIEW": "재검 (경계)", "NG": "NG (불량)"}
VERDICT_COLOR = {"OK": "#2e7d32", "REVIEW": "#ef6c00", "NG": "#d32f2f"}


def get_inspector(model_name: str) -> core.Inspector:
    global _inspector
    with _lock:
        if _inspector is None or _inspector.model_name != model_name:
            _inspector = core.Inspector(model_name)
        return _inspector


def get_classifier(model_name: str) -> core_cls.Classifier:
    global _classifier
    with _lock:
        if _classifier is None or _classifier.model_name != model_name:
            _classifier = core_cls.Classifier(model_name)
        return _classifier


def get_segmenter(model_name: str) -> core_seg.Segmenter:
    global _segmenter
    with _lock:
        if _segmenter is None or _segmenter.model_name != model_name:
            _segmenter = core_seg.Segmenter(model_name)
        return _segmenter


def drop_cached(kind: str, name: str) -> None:
    """같은 이름으로 다시 학습했을 때 예전 모델이 메모리에 남지 않도록."""
    global _inspector, _classifier, _segmenter
    with _lock:
        if kind == "anomaly" and _inspector and _inspector.model_name == name:
            _inspector = None
        if kind == "cls" and _classifier and _classifier.model_name == name:
            _classifier = None
        if kind == "seg" and _segmenter and _segmenter.model_name == name:
            _segmenter = None


def run_in_thread(fn, lines: list[str]):
    """긴 작업을 스레드로 돌리며 로그를 0.5초마다 내보내는 제너레이터."""
    result: dict = {}

    def worker():
        try:
            result["info"] = fn()
        except Exception as e:  # noqa: BLE001
            result["error"] = f"{e}\n\n{traceback.format_exc()}"

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        yield "\n".join(lines), None
        time.sleep(0.5)
    if "error" in result:
        lines.append("오류 발생:\n" + result["error"])
        yield "\n".join(lines), None
        return
    yield "\n".join(lines), result["info"]


def make_logger(lines: list[str]):
    def log(msg):
        lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    return log


def aug_dict(hflip, vflip, rotate, brightness, contrast, scale) -> dict:
    return {"hflip": bool(hflip), "vflip": bool(vflip), "rotate": float(rotate),
            "brightness": float(brightness), "contrast": float(contrast), "scale": float(scale)}


def roi_or_none(path: str) -> str | None:
    path = (path or "").strip()
    return path if path and Path(path).exists() else None


def verdict_html(v: str, extra: str) -> str:
    return (f"<div style='font-size:2.2em;font-weight:bold;color:{VERDICT_COLOR[v]};'>{VERDICT_KO[v]}</div>"
            f"<div style='font-size:1.2em;'>{extra}</div>")


def aug_controls(defaults: dict):
    """증강 설정 UI 묶음. 반환: 컴포넌트 6개."""
    with gr.Row():
        hflip = gr.Checkbox(defaults.get("hflip", False), label="좌우 뒤집기")
        vflip = gr.Checkbox(defaults.get("vflip", False), label="상하 뒤집기")
        rotate = gr.Slider(0, 180, value=defaults.get("rotate", 0), step=1, label="회전 범위 (도)")
    with gr.Row():
        brightness = gr.Slider(0, 0.5, value=defaults.get("brightness", 0), step=0.01, label="밝기 변동")
        contrast = gr.Slider(0, 0.5, value=defaults.get("contrast", 0), step=0.01, label="대비 변동")
        scale = gr.Slider(0, 0.3, value=defaults.get("scale", 0), step=0.01, label="크기 변동")
    return hflip, vflip, rotate, brightness, contrast, scale


# ----------------------------------------------------------------------
# 1. 어노멀리 학습
# ----------------------------------------------------------------------
def run_training(model_name, normal_dir, abnormal_dir, model_type, image_size, coreset_ratio, batch_size, epochs,
                 hflip, vflip, rotate, brightness, contrast, scale, roi_path):
    model_name = (model_name or "").strip()
    if not model_name:
        yield "모델 이름을 입력하세요.", gr.update()
        return
    if not (normal_dir or "").strip():
        yield "정상 이미지 폴더를 입력하세요.", gr.update()
        return
    lines: list[str] = []
    info = None
    for text, info in run_in_thread(
        lambda: core.train_model(model_name, normal_dir.strip(), (abnormal_dir or "").strip() or None,
                                 int(image_size), float(coreset_ratio), model_type, int(batch_size), int(epochs),
                                 aug_dict(hflip, vflip, rotate, brightness, contrast, scale), roi_or_none(roi_path),
                                 log=make_logger(lines)),
        lines,
    ):
        if info is None:
            yield text, gr.update()
    if info is None:
        return
    drop_cached("anomaly", model_name)
    m = info["metrics"]
    lines.append("")
    lines.append(f"완료. 학습 {info['train_seconds']}초, 정상 {info['n_normal']}장")
    if m:
        lines.append("검증 성능: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
        lines.append("  (image_AUROC 1.0 = 정상/불량을 완벽히 구분, F1Score = 판정 정확도)")
    lines.append("검사 탭에서 이 모델을 선택해 사용하세요.")
    yield "\n".join(lines), gr.update(choices=core.list_models(), value=model_name)


# ----------------------------------------------------------------------
# 2. 어노멀리 검사
# ----------------------------------------------------------------------
def on_model_change(model_name):
    if not model_name:
        return gr.update(), gr.update(), gr.update(), "모델을 선택하세요."
    info = core.load_info(model_name)
    m = info.get("metrics", {})
    desc = (f"{core.ANOMALY_MODELS.get(info.get('model_type', 'patchcore'), '')} | 학습 정상 {info['n_normal']}장, "
            f"입력 {info['image_size']}px" + (" | ROI 적용" if info.get("roi") else ""))
    if m:
        desc += " | " + ", ".join(f"{k}={v:.3f}" for k, v in m.items())
    thr = float(info.get("threshold", 0.5))
    return (gr.update(value=thr), gr.update(value=float(info.get("review_threshold", thr))),
            gr.update(value=float(info.get("pixel_threshold", 0.5))), desc)


def run_inspect(model_name, image, threshold, review_threshold, pixel_threshold):
    if not model_name:
        return None, None, "모델을 먼저 선택하세요.", ""
    if image is None:
        return None, None, "이미지를 올려주세요.", ""
    r = get_inspector(model_name).inspect(image, float(threshold), float(review_threshold), float(pixel_threshold))
    html = verdict_html(r["verdict"], f"이상 점수 <b>{r['score'] * 100:.1f}</b> / 100 "
                        f"(재검 {float(review_threshold) * 100:.0f} · NG {float(threshold) * 100:.0f})")
    boxes = ", ".join(f"({x},{y},{w}x{h})" for x, y, w, h in r["boxes"]) or "없음"
    detail = f"불량 영역 {len(r['boxes'])}곳: {boxes}\n처리 시간 {r['elapsed_ms']:.1f} ms"
    return r["heatmap"], r["mask"], html, detail


def save_threshold(model_name, threshold, review_threshold, pixel_threshold):
    if not model_name:
        return "모델을 먼저 선택하세요."
    core.save_threshold(model_name, float(threshold), float(review_threshold), float(pixel_threshold))
    return (f"'{model_name}' 저장: NG 임계값 {float(threshold):.2f}, 재검 임계값 {min(float(review_threshold), float(threshold)):.2f}, "
            f"픽셀 임계값 {float(pixel_threshold):.2f}. API 서버도 이 값을 기본으로 씁니다.")


# ----------------------------------------------------------------------
# 3. 일괄 검사
# ----------------------------------------------------------------------
def run_batch(model_name, folder, threshold, review_threshold):
    if not model_name:
        return [], "모델을 먼저 선택하세요."
    folder = (folder or "").strip()
    if not Path(folder).is_dir():
        return [], f"폴더를 찾을 수 없습니다: {folder}"
    paths = core.list_images(Path(folder))
    if not paths:
        return [], "폴더에 이미지가 없습니다."
    insp = get_inspector(model_name)
    rows, counts, total_ms = [], {"OK": 0, "REVIEW": 0, "NG": 0}, 0.0
    for p in paths:
        r = insp.inspect(p, float(threshold), float(review_threshold))
        counts[r["verdict"]] += 1
        total_ms += r["elapsed_ms"]
        rows.append([p.name, round(r["score"] * 100, 1), VERDICT_KO[r["verdict"]].split(" ")[0], len(r["boxes"]), round(r["elapsed_ms"], 1)])
    rows.sort(key=lambda x: -x[1])
    summary = (f"총 {len(paths)}장 | NG {counts['NG']} / 재검 {counts['REVIEW']} / OK {counts['OK']} | "
               f"평균 {total_ms / len(paths):.1f} ms/장")
    return rows, summary


# ----------------------------------------------------------------------
# 4~5. 분류
# ----------------------------------------------------------------------
def run_cls_training(model_name, data_dir, backbone, image_size, epochs, lr, batch_size, patience, class_weight,
                     hflip, vflip, rotate, brightness, contrast, scale, roi_path):
    model_name = (model_name or "").strip()
    if not model_name:
        yield "모델 이름을 입력하세요.", gr.update()
        return
    lines: list[str] = []
    info = None
    for text, info in run_in_thread(
        lambda: core_cls.train_classifier(model_name, (data_dir or "").strip(), backbone, int(image_size), int(epochs),
                                          float(lr), int(batch_size), int(patience), bool(class_weight),
                                          aug_dict(hflip, vflip, rotate, brightness, contrast, scale), roi_or_none(roi_path),
                                          log=make_logger(lines)),
        lines,
    ):
        if info is None:
            yield text, gr.update()
    if info is None:
        return
    drop_cached("cls", model_name)
    lines.append("")
    lines.append(f"완료. 최고 검증 정확도 {info['val_accuracy'] * 100:.1f}%  ({info['train_seconds']}초)")
    lines.append("혼동 행렬 (행=정답, 열=예측): " + " / ".join(info["classes"]))
    for c, row in zip(info["classes"], info["confusion"]):
        lines.append(f"  {c:>10}: {row}")
    lines.append("분류 검사 탭에서 이 모델을 선택해 사용하세요.")
    yield "\n".join(lines), gr.update(choices=core_cls.list_models(), value=model_name)


def on_cls_model_change(model_name):
    if not model_name:
        return gr.update(), "모델을 선택하세요."
    info = core_cls.load_info(model_name)
    return (gr.update(value=float(info.get("review_threshold", 0.0))),
            f"클래스: {', '.join(info['classes'])} | 검증 정확도 {info['val_accuracy'] * 100:.1f}% | "
            f"모델 {info['backbone']}, 입력 {info['image_size']}px" + (" | ROI 적용" if info.get("roi") else ""))


def run_classify(model_name, image, review_threshold):
    if not model_name:
        return None, "모델을 먼저 선택하세요.", {}
    if image is None:
        return None, "이미지를 올려주세요.", {}
    r = get_classifier(model_name).predict(image, review_threshold=float(review_threshold))
    html = verdict_html(r["verdict"], f"판정 클래스 <b>{r['label']}</b>, 확신도 <b>{r['confidence'] * 100:.1f}%</b> "
                        f"<span style='color:#888'>{r['elapsed_ms']:.1f} ms</span>")
    return r["heatmap"], html, r["probs"]


def save_cls_threshold(model_name, review_threshold):
    if not model_name:
        return "모델을 먼저 선택하세요."
    core_cls.save_info(model_name, review_threshold=float(review_threshold))
    return f"'{model_name}' 재검 기준 저장: 확신도 {float(review_threshold) * 100:.0f}% 미만이면 재검"


# ----------------------------------------------------------------------
# 6~8. 세그멘테이션
# ----------------------------------------------------------------------
def run_make_masks(anomaly_model, image_dir, out_dir, threshold):
    if not anomaly_model:
        yield "어노멀리 모델을 선택하세요."
        return
    lines: list[str] = []
    for text, _ in run_in_thread(
        lambda: core_seg.make_masks_from_anomaly(anomaly_model, (image_dir or "").strip(), (out_dir or "").strip(),
                                                 float(threshold), log=make_logger(lines)),
        lines,
    ):
        yield text


def run_seg_training(model_name, data_dir, encoder, image_size, epochs, lr, batch_size, patience,
                     hflip, vflip, rotate, brightness, contrast, scale, roi_path):
    model_name = (model_name or "").strip()
    if not model_name:
        yield "모델 이름을 입력하세요.", gr.update()
        return
    lines: list[str] = []
    info = None
    for text, info in run_in_thread(
        lambda: core_seg.train_segmenter(model_name, (data_dir or "").strip(), encoder, int(image_size), int(epochs),
                                         float(lr), int(batch_size), int(patience),
                                         aug_dict(hflip, vflip, rotate, brightness, contrast, scale), roi_or_none(roi_path),
                                         log=make_logger(lines)),
        lines,
    ):
        if info is None:
            yield text, gr.update()
    if info is None:
        return
    drop_cached("seg", model_name)
    lines.append("")
    lines.append(f"완료. 최고 검증 IoU {info['val_iou'] * 100:.1f}%  ({info['train_seconds']}초)")
    lines.append("세그 검사 탭에서 이 모델을 선택해 사용하세요.")
    yield "\n".join(lines), gr.update(choices=core_seg.list_models(), value=model_name)


def on_seg_model_change(model_name):
    if not model_name:
        return gr.update(), gr.update(), "모델을 선택하세요."
    info = core_seg.load_info(model_name)
    return (gr.update(value=float(info.get("threshold", 0.5))), gr.update(value=int(info.get("min_area", 20))),
            f"학습 데이터 {info['n_pairs']}쌍 | 검증 IoU {info['val_iou'] * 100:.1f}% | U-Net({info['encoder']}), "
            f"입력 {info['image_size']}px" + (" | ROI 적용" if info.get("roi") else ""))


def run_segment(model_name, image, threshold, min_area):
    if not model_name:
        return None, None, "모델을 먼저 선택하세요.", ""
    if image is None:
        return None, None, "이미지를 올려주세요.", ""
    r = get_segmenter(model_name).segment(image, float(threshold), int(min_area))
    html = verdict_html(r["verdict"], f"불량 영역 <b>{len(r['boxes'])}</b>곳, 면적 비율 <b>{r['defect_ratio']:.2f}%</b>")
    boxes = ", ".join(f"({x},{y},{w}x{h})" for x, y, w, h in r["boxes"]) or "없음"
    detail = f"불량 박스: {boxes}\n최대 확률 {r['max_prob'] * 100:.1f}%  처리 시간 {r['elapsed_ms']:.1f} ms"
    return r["overlay"], r["mask"], html, detail


def save_seg_threshold(model_name, threshold, min_area):
    if not model_name:
        return "모델을 먼저 선택하세요."
    core_seg.save_threshold(model_name, float(threshold), int(min_area))
    return f"'{model_name}' 저장: 픽셀 임계값 {float(threshold):.2f}, 최소 면적 {int(min_area)} 픽셀"


# ----------------------------------------------------------------------
# 9. 배포
# ----------------------------------------------------------------------
def deploy_upload(manager_url, kind, name):
    if not name:
        return "모델을 선택하세요."
    try:
        meta = deploy.upload_model(manager_url, kind, name)
        return f"업로드 완료: {deploy.KIND_LABELS[kind]} '{name}' v{meta['version']} ({meta['size_mb']} MB)"
    except Exception as e:  # noqa: BLE001
        return f"업로드 실패: {e}"


def deploy_refresh(manager_url):
    try:
        with httpx.Client(base_url=manager_url.rstrip("/"), timeout=10) as c:
            agents = c.get("/api/agents").json()["agents"]
            models = c.get("/api/models").json()["models"]
            summ = c.get("/api/summary").json()
    except Exception as e:  # noqa: BLE001
        return [], gr.update(choices=[]), gr.update(choices=[]), f"관리 서버에 연결할 수 없습니다: {e}"
    rows = []
    for a in agents:
        L, A = a.get("loaded", {}), a.get("assignments", {})

        def cell(k):
            cur = L.get(k) or "-"
            want = (A.get(k) or {}).get("name")
            return cur if not want or want == cur else f"{cur} (대기: {want})"
        rows.append([a["agent_id"], "온라인" if a["online"] else "오프라인", a.get("last_seen_str", ""),
                     a.get("device") or "-", cell("anomaly"), cell("cls"), cell("seg")])
    agent_ids = [a["agent_id"] for a in agents]
    model_choices = [f"{m['kind']}/{m['name']}" for m in models]
    msg = f"에이전트 {len(agents)}대, 업로드된 모델 {len(models)}개, 수집된 검사 결과 {summ['total']}건"
    return (rows, gr.update(choices=agent_ids, value=(agent_ids or [None])[0]),
            gr.update(choices=model_choices, value=(model_choices or [None])[0]), msg)


def deploy_assign(manager_url, agent_id, kind_name):
    if not agent_id or not kind_name:
        return "에이전트와 모델을 선택하세요."
    kind, name = kind_name.split("/", 1)
    try:
        with httpx.Client(base_url=manager_url.rstrip("/"), timeout=10) as c:
            r = c.post("/api/deploy", json={"agent_id": agent_id, "kind": kind, "name": name})
            r.raise_for_status()
        return f"'{agent_id}' 에 {deploy.KIND_LABELS[kind]} 모델 '{name}' 배포를 지정했습니다. 에이전트가 몇 초 안에 내려받아 교체합니다."
    except Exception as e:  # noqa: BLE001
        return f"배포 지정 실패: {e}"


def deploy_results(manager_url, limit):
    try:
        with httpx.Client(base_url=manager_url.rstrip("/"), timeout=10) as c:
            rows = c.get("/api/results", params={"limit": int(limit)}).json()["results"]
    except Exception as e:  # noqa: BLE001
        return [], f"조회 실패: {e}"
    table = []
    for x in rows:
        score = x.get("score")
        if score is None and x.get("confidence") is not None:
            score = f"{x.get('label')} {x['confidence']}%"
        elif score is None and x.get("defect_ratio") is not None:
            score = f"{x['defect_ratio']}%"
        table.append([x.get("time", ""), x.get("agent_id", ""), deploy.KIND_LABELS.get(x.get("kind"), x.get("kind", "")),
                      x.get("result") or x.get("label", ""), score, len(x.get("boxes") or []), x.get("elapsed_ms", ""), x.get("filename", "")])
    return table, f"{len(table)}건"


# ----------------------------------------------------------------------
# 10. ROI 그리기
# ----------------------------------------------------------------------
def roi_load_sample(folder):
    folder = (folder or "").strip()
    if not Path(folder).is_dir():
        return None, f"폴더를 찾을 수 없습니다: {folder}"
    imgs = core.list_images(Path(folder))
    if not imgs:
        return None, "폴더에 이미지가 없습니다."
    return str(imgs[0]), f"{imgs[0].name} 을 불러왔습니다. 붓으로 검사할 영역을 칠하세요 (칠한 곳만 검사)."


def roi_save(editor_value, name):
    name = (name or "").strip()
    if not name:
        return "ROI 이름을 입력하세요.", None
    mask = roi_util.editor_to_mask(editor_value)
    if mask is None:
        return "칠해진 영역이 없습니다. 붓 도구로 검사할 영역을 칠하세요.", None
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    path = ROI_DIR / f"{name}.png"
    mask.save(path)
    ratio = np.asarray(mask).mean() / 255 * 100
    return (f"저장: {path}  (전체의 {ratio:.1f}% 를 검사). 학습 탭의 'ROI 마스크 파일' 칸에 이 경로를 넣으세요.", str(path))


# ----------------------------------------------------------------------
# 11. 모델 비교
# ----------------------------------------------------------------------
def run_compare(kind, base_name, path1, path2, candidates, roi_path):
    base_name = (base_name or "").strip()
    if not base_name:
        yield "이름을 입력하세요.", gr.update()
        return
    if not candidates:
        yield "비교할 설정을 하나 이상 고르세요.", gr.update()
        return
    data = {"normal_dir": path1, "abnormal_dir": (path2 or "").strip() or None} if kind == "anomaly" else {"data_dir": path1}
    common = {"roi_path": roi_or_none(roi_path)}
    lines: list[str] = []
    rows = None
    for text, rows in run_in_thread(lambda: experiments.run_comparison(kind, base_name, data, list(candidates), common, log=make_logger(lines)), lines):
        if rows is None:
            yield text, gr.update()
    if rows is None:
        return
    keys = [k for k in rows[0].keys() if not k.startswith("_")] if rows else []
    table = [[r.get(k, "") for k in keys] for r in rows]
    lines.append("")
    lines.append("표는 성능 높은 순(같으면 빠른 순)으로 정렬되어 있습니다. 마음에 드는 '모델 이름'을 검사 탭에서 그대로 선택하면 됩니다.")
    yield "\n".join(lines), gr.update(value=table, headers=keys)


def compare_choices(kind):
    c = list(experiments.CANDIDATES[kind].keys())
    default_path = {"anomaly": SAMPLE_NORMAL, "cls": SAMPLE_CLS, "seg": SAMPLE_SEG}[kind]
    label = {"anomaly": "정상 이미지 폴더", "cls": "데이터 폴더 (클래스 폴더들의 상위)", "seg": "데이터 폴더 (images/, masks/ 의 상위)"}[kind]
    return (gr.update(choices=c, value=c[:2]), gr.update(value=default_path, label=label),
            gr.update(visible=(kind == "anomaly")))


# ======================================================================
# 화면 구성
# ======================================================================
with gr.Blocks(title="비전 검사") as demo:
    gr.Markdown("# 비전 검사 시스템\n어노멀리(정상만 학습) · 분류(불량 종류) · 세그멘테이션(불량 영역 픽셀) 세 가지 방식을 제공합니다.")

    # ---------------- 1. 어노멀리 학습 ----------------
    with gr.Tab("1. 학습"):
        gr.Markdown("정상 이미지가 들어 있는 폴더 경로를 넣고 학습 버튼을 누르세요. "
                    "불량 폴더는 선택 사항이며, 있으면 임계값이 더 정확해지고 성능 점수도 나옵니다.")
        with gr.Row():
            with gr.Column():
                t_name = gr.Textbox(label="모델 이름", value="sample", placeholder="예: 부품A_상면")
                t_normal = gr.Textbox(label="정상 이미지 폴더", value=SAMPLE_NORMAL)
                t_abnormal = gr.Textbox(label="불량 이미지 폴더 (선택)", value=SAMPLE_DEFECT)
                t_type = gr.Dropdown(choices=[(v, k) for k, v in core.ANOMALY_MODELS.items()], value="patchcore", label="모델 종류")
                with gr.Row():
                    t_size = gr.Dropdown([224, 256, 320, 384, 448, 512], value=256, label="입력 이미지 크기 (px)",
                                         info="클수록 작은 결함을 잘 잡지만 느려집니다")
                    t_coreset = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="메모리 샘플링 비율 (PatchCore)",
                                          info="낮을수록 빠르고 가벼움, 높을수록 정밀")
                with gr.Accordion("고급 설정 (배치, 에폭, 증강, ROI)", open=False):
                    with gr.Row():
                        t_batch = gr.Slider(1, 64, value=8, step=1, label="배치 크기", info="GPU 메모리 부족 시 줄이세요")
                        t_epochs = gr.Slider(1, 200, value=20, step=1, label="에폭 수 (Dinomaly 만 사용)")
                    gr.Markdown("**증강** (학습 이미지를 무작위로 변형해 데이터를 늘림. 라인에서 실제로 생기는 변동만 켜세요)")
                    t_aug = aug_controls(core.DEFAULT_AUG)
                    t_roi = gr.Textbox(label="ROI 마스크 파일 (선택, 10번 탭에서 만든 PNG 경로)", placeholder="비우면 전체 이미지를 검사")
                t_btn = gr.Button("학습 시작", variant="primary")
            t_log = gr.Textbox(label="진행 상황", lines=22, interactive=False)

    # ---------------- 2. 어노멀리 검사 ----------------
    with gr.Tab("2. 검사"):
        with gr.Row():
            i_model = gr.Dropdown(core.list_models(), label="모델 선택", value=(core.list_models() or [None])[0])
            i_refresh = gr.Button("목록 새로고침", scale=0)
        i_info = gr.Markdown("")
        with gr.Row():
            with gr.Column():
                i_image = gr.Image(type="pil", label="검사할 이미지", height=360)
                i_thr = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="NG 임계값", info="이 점수를 넘으면 NG. 0.5 = 학습 시 자동 결정된 값")
                i_rev = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="재검 임계값", info="재검~NG 사이 점수는 '재검'. NG 와 같게 두면 재검 없음")
                i_pix = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="픽셀 임계값 (히트맵 감도)", info="낮추면 불량 영역이 넓게 표시됨")
                with gr.Row():
                    i_btn = gr.Button("검사", variant="primary")
                    i_save = gr.Button("이 임계값들을 기본값으로 저장")
                i_msg = gr.Markdown("")
            with gr.Column():
                i_verdict = gr.HTML()
                i_heat = gr.Image(label="불량 영역 히트맵 (파란 선 = ROI)", height=360)
                with gr.Accordion("마스크 / 상세", open=False):
                    i_mask = gr.Image(label="불량 마스크", height=240)
                    i_detail = gr.Textbox(label="상세", lines=3, interactive=False)

    # ---------------- 3. 일괄 검사 ----------------
    with gr.Tab("3. 일괄 검사"):
        gr.Markdown("폴더 안의 모든 이미지를 검사해 점수 순으로 보여줍니다. 임계값을 어디에 둘지 정할 때 쓰세요.")
        with gr.Row():
            b_folder = gr.Textbox(label="이미지 폴더", value=SAMPLE_DEFECT)
            b_thr = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="NG 임계값")
            b_rev = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="재검 임계값")
            b_btn = gr.Button("일괄 검사", variant="primary", scale=0)
        b_summary = gr.Markdown("")
        b_table = gr.Dataframe(headers=["파일", "이상 점수", "판정", "불량 영역 수", "처리 ms"],
                               datatype=["str", "number", "str", "number", "number"], interactive=False)

    # ---------------- 4. 분류 학습 ----------------
    with gr.Tab("4. 분류 학습"):
        gr.Markdown("폴더 이름이 곧 정답(클래스)입니다. 예) `data/부품A/good`, `data/부품A/scratch`, `data/부품A/dent` 처럼 "
                    "클래스별 폴더에 이미지를 넣고, 그 상위 폴더 경로를 입력하세요. 클래스당 최소 5장, 권장 50장 이상.")
        with gr.Row():
            with gr.Column():
                c_name = gr.Textbox(label="모델 이름", value="sample_cls")
                c_dir = gr.Textbox(label="데이터 폴더 (클래스 폴더들의 상위)", value=SAMPLE_CLS)
                with gr.Row():
                    c_backbone = gr.Dropdown(["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"], value="resnet18",
                                             label="모델 종류", info="resnet18 이 가장 빠름, 아래로 갈수록 정밀하지만 느림")
                    c_size = gr.Dropdown([224, 256, 320, 384], value=224, label="입력 크기 (px)")
                    c_epochs = gr.Slider(3, 100, value=15, step=1, label="에폭 수")
                with gr.Accordion("고급 설정 (학습률, 배치, 조기 종료, 증강, ROI)", open=False):
                    with gr.Row():
                        c_lr = gr.Dropdown([1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, label="학습률", info="학습이 불안정하면 낮추세요")
                        c_batch = gr.Slider(4, 64, value=16, step=4, label="배치 크기")
                        c_patience = gr.Slider(0, 20, value=0, step=1, label="조기 종료 (에폭)", info="0 = 끄기. n 에폭 동안 개선 없으면 중단")
                    c_weight = gr.Checkbox(True, label="클래스 불균형 보정 (이미지 수가 적은 클래스에 가중치)")
                    gr.Markdown("**증강**")
                    c_aug = aug_controls(core_cls.DEFAULT_AUG)
                    c_roi = gr.Textbox(label="ROI 마스크 파일 (선택)")
                c_btn = gr.Button("분류 학습 시작", variant="primary")
            c_log = gr.Textbox(label="진행 상황", lines=22, interactive=False)

    # ---------------- 5. 분류 검사 ----------------
    with gr.Tab("5. 분류 검사"):
        with gr.Row():
            k_model = gr.Dropdown(core_cls.list_models(), label="분류 모델 선택", value=(core_cls.list_models() or [None])[0])
            k_refresh = gr.Button("목록 새로고침", scale=0)
        k_info = gr.Markdown("")
        with gr.Row():
            with gr.Column():
                k_image = gr.Image(type="pil", label="검사할 이미지", height=360)
                k_rev = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="재검 기준 (확신도)", info="최고 확률이 이 값 미만이면 '재검'. 0 = 끄기")
                with gr.Row():
                    k_btn = gr.Button("분류", variant="primary")
                    k_save = gr.Button("재검 기준 저장")
                k_msg = gr.Markdown("")
            with gr.Column():
                k_verdict = gr.HTML()
                k_probs = gr.Label(label="클래스별 확률", num_top_classes=6)
                k_heat = gr.Image(label="판단 근거 영역 (Grad-CAM)", height=300)

    # ---------------- 6. 세그 데이터 만들기 ----------------
    with gr.Tab("6. 세그 데이터 만들기"):
        gr.Markdown("세그멘테이션은 불량 영역을 픽셀 단위로 칠한 **마스크**가 필요합니다. "
                    "직접 그리기 전에, 학습된 어노멀리 모델로 불량 이미지 폴더를 검사해 마스크 초안을 자동으로 만듭니다. "
                    "결과의 masks/ 폴더를 그림판 등으로 열어 틀린 부분만 고치면 됩니다 (흰색=불량, 검정=정상).")
        with gr.Row():
            with gr.Column():
                s_anom = gr.Dropdown(core.list_models(), label="어노멀리 모델", value=(core.list_models() or [None])[0])
                s_src = gr.Textbox(label="불량 이미지 폴더", value=SAMPLE_DEFECT)
                s_dst = gr.Textbox(label="결과 폴더 (images/ 와 masks/ 가 생성됨)", value=str(core.BASE_DIR / "data" / "auto_seg"))
                s_thr = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="픽셀 임계값")
                s_btn = gr.Button("마스크 초안 생성", variant="primary")
            s_log = gr.Textbox(label="진행 상황", lines=12, interactive=False)

    # ---------------- 7. 세그 학습 ----------------
    with gr.Tab("7. 세그 학습"):
        gr.Markdown("`<폴더>/images/*.png` 와 같은 이름의 `<폴더>/masks/*.png` 쌍이 필요합니다. 정상 이미지(마스크 전부 검정)도 섞어 주세요.")
        with gr.Row():
            with gr.Column():
                g_name = gr.Textbox(label="모델 이름", value="sample_seg")
                g_dir = gr.Textbox(label="데이터 폴더 (images/, masks/ 의 상위)", value=SAMPLE_SEG)
                with gr.Row():
                    g_enc = gr.Dropdown(["resnet18", "resnet34", "efficientnet-b0", "mobilenet_v2"], value="resnet18",
                                        label="인코더", info="resnet18 이 빠르고 무난")
                    g_size = gr.Dropdown([256, 320, 384, 512], value=256, label="입력 크기 (px)")
                    g_epochs = gr.Slider(5, 100, value=20, step=1, label="에폭 수")
                with gr.Accordion("고급 설정 (학습률, 배치, 조기 종료, 증강, ROI)", open=False):
                    with gr.Row():
                        g_lr = gr.Dropdown([1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, label="학습률")
                        g_batch = gr.Slider(2, 32, value=8, step=2, label="배치 크기")
                        g_patience = gr.Slider(0, 20, value=0, step=1, label="조기 종료 (에폭)", info="0 = 끄기")
                    gr.Markdown("**증강** (회전 90 이상 = 90도 단위 회전)")
                    g_aug = aug_controls(core_seg.DEFAULT_AUG)
                    g_roi = gr.Textbox(label="ROI 마스크 파일 (선택)")
                g_btn = gr.Button("세그 학습 시작", variant="primary")
            g_log = gr.Textbox(label="진행 상황", lines=22, interactive=False)

    # ---------------- 8. 세그 검사 ----------------
    with gr.Tab("8. 세그 검사"):
        with gr.Row():
            e_model = gr.Dropdown(core_seg.list_models(), label="세그 모델 선택", value=(core_seg.list_models() or [None])[0])
            e_refresh = gr.Button("목록 새로고침", scale=0)
        e_info = gr.Markdown("")
        with gr.Row():
            with gr.Column():
                e_image = gr.Image(type="pil", label="검사할 이미지", height=360)
                with gr.Row():
                    e_thr = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="픽셀 임계값")
                    e_area = gr.Slider(1, 500, value=20, step=1, label="최소 불량 면적 (픽셀)", info="이보다 작은 조각은 무시")
                with gr.Row():
                    e_btn = gr.Button("검사", variant="primary")
                    e_save = gr.Button("이 값들을 기본값으로 저장")
                e_msg = gr.Markdown("")
            with gr.Column():
                e_verdict = gr.HTML()
                e_over = gr.Image(label="불량 영역 (빨강, 파란 선 = ROI)", height=360)
                with gr.Accordion("마스크 / 상세", open=False):
                    e_mask = gr.Image(label="불량 마스크", height=240)
                    e_detail = gr.Textbox(label="상세", lines=3, interactive=False)

    # ---------------- 9. 배포 ----------------
    with gr.Tab("9. 배포 (여러 PC 관리)"):
        gr.Markdown("서버 PC 에서 `start_manager.bat`, 각 라인 PC 에서 `start_api.bat` + `start_agent.bat` 을 실행해 두면 "
                    "여기서 모델을 올리고 어느 PC 에 배포할지 정할 수 있습니다. 라인 PC 의 검사 결과도 여기 모입니다.")
        d_url = gr.Textbox(label="관리 서버 주소", value=DEFAULT_MANAGER)
        with gr.Row():
            with gr.Column():
                gr.Markdown("**1) 이 PC 의 모델을 관리 서버에 올리기**")
                d_kind = gr.Radio(["anomaly", "cls", "seg"], value="anomaly", label="종류 (anomaly=어노멀리, cls=분류, seg=세그)")
                d_name = gr.Dropdown(deploy.list_local("anomaly"), label="모델", value=(deploy.list_local("anomaly") or [None])[0])
                d_up = gr.Button("관리 서버에 업로드", variant="primary")
                d_up_msg = gr.Markdown("")
            with gr.Column():
                gr.Markdown("**2) 라인 PC 에 배포 지정**")
                d_agent = gr.Dropdown([], label="라인 PC (에이전트)")
                d_model = gr.Dropdown([], label="배포할 모델 (종류/이름)")
                d_go = gr.Button("배포 지정", variant="primary")
                d_go_msg = gr.Markdown("")
        with gr.Row():
            d_refresh = gr.Button("새로고침", scale=0)
            d_status = gr.Markdown("")
        d_agents = gr.Dataframe(headers=["에이전트", "상태", "마지막 신호", "장치", "어노멀리", "분류", "세그"], interactive=False, label="라인 PC 현황")
        with gr.Accordion("수집된 검사 결과", open=False):
            with gr.Row():
                d_limit = gr.Slider(10, 500, value=50, step=10, label="최근 n건")
                d_res_btn = gr.Button("조회", scale=0)
                d_res_msg = gr.Markdown("")
            d_results = gr.Dataframe(headers=["시각", "PC", "종류", "판정", "점수/클래스", "불량 영역", "ms", "파일"], interactive=False)

    # ---------------- 10. ROI 그리기 ----------------
    with gr.Tab("10. ROI 그리기"):
        gr.Markdown("검사할 영역만 붓으로 칠해 마스크 파일로 저장합니다. 지그, 배경, 라벨처럼 검사와 무관한 부분을 빼면 오검이 줄어듭니다. "
                    "저장된 파일 경로를 각 학습 탭의 'ROI 마스크 파일' 칸에 넣으면 학습과 검사 모두 그 영역만 봅니다.")
        with gr.Row():
            r_folder = gr.Textbox(label="샘플 이미지 폴더 (첫 이미지를 불러옴)", value=SAMPLE_NORMAL)
            r_load = gr.Button("이미지 불러오기", scale=0)
        r_msg = gr.Markdown("")
        r_editor = gr.ImageEditor(label="붓으로 검사 영역을 칠하세요 (도구 모음의 붓 아이콘, 굵기 조절 가능)", type="numpy", height=520)
        with gr.Row():
            r_name = gr.Textbox(label="ROI 이름", value="roi1")
            r_save = gr.Button("ROI 저장", variant="primary", scale=0)
        r_out = gr.Markdown("")
        r_preview = gr.Image(label="저장된 마스크 (흰색 = 검사 영역)", height=240)

    # ---------------- 11. 모델 비교 ----------------
    with gr.Tab("11. 모델 비교"):
        gr.Markdown("같은 데이터로 여러 설정을 차례로 학습해 성능과 검사 속도를 표로 비교합니다. "
                    "각 결과는 `이름__설정` 모델로 저장되므로 마음에 드는 것을 검사 탭에서 바로 쓸 수 있습니다. 시간이 오래 걸릴 수 있습니다.")
        with gr.Row():
            with gr.Column():
                x_kind = gr.Radio([("어노멀리", "anomaly"), ("분류", "cls"), ("세그", "seg")], value="anomaly", label="종류")
                x_name = gr.Textbox(label="비교 이름", value="exp")
                x_path1 = gr.Textbox(label="정상 이미지 폴더", value=SAMPLE_NORMAL)
                x_path2 = gr.Textbox(label="불량 이미지 폴더 (어노멀리, 선택)", value=SAMPLE_DEFECT, visible=True)
                x_cands = gr.CheckboxGroup(list(experiments.ANOMALY_CANDIDATES), value=list(experiments.ANOMALY_CANDIDATES)[:2], label="비교할 설정")
                x_roi = gr.Textbox(label="ROI 마스크 파일 (선택, 모든 설정에 공통 적용)")
                x_btn = gr.Button("비교 시작", variant="primary")
            x_log = gr.Textbox(label="진행 상황", lines=22, interactive=False)
        x_table = gr.Dataframe(label="비교 결과 (성능 높은 순)", interactive=False)

    # ================= 이벤트 연결 =================
    t_btn.click(run_training, [t_name, t_normal, t_abnormal, t_type, t_size, t_coreset, t_batch, t_epochs, *t_aug, t_roi], [t_log, i_model])
    i_model.change(on_model_change, i_model, [i_thr, i_rev, i_pix, i_info])
    i_refresh.click(lambda: gr.update(choices=core.list_models()), None, i_model)
    i_btn.click(run_inspect, [i_model, i_image, i_thr, i_rev, i_pix], [i_heat, i_mask, i_verdict, i_detail])
    i_save.click(save_threshold, [i_model, i_thr, i_rev, i_pix], i_msg)
    b_btn.click(run_batch, [i_model, b_folder, b_thr, b_rev], [b_table, b_summary])
    demo.load(on_model_change, i_model, [i_thr, i_rev, i_pix, i_info])

    c_btn.click(run_cls_training, [c_name, c_dir, c_backbone, c_size, c_epochs, c_lr, c_batch, c_patience, c_weight, *c_aug, c_roi], [c_log, k_model])
    k_model.change(on_cls_model_change, k_model, [k_rev, k_info])
    k_refresh.click(lambda: gr.update(choices=core_cls.list_models()), None, k_model)
    k_btn.click(run_classify, [k_model, k_image, k_rev], [k_heat, k_verdict, k_probs])
    k_save.click(save_cls_threshold, [k_model, k_rev], k_msg)
    demo.load(on_cls_model_change, k_model, [k_rev, k_info])

    s_btn.click(run_make_masks, [s_anom, s_src, s_dst, s_thr], s_log)
    g_btn.click(run_seg_training, [g_name, g_dir, g_enc, g_size, g_epochs, g_lr, g_batch, g_patience, *g_aug, g_roi], [g_log, e_model])
    e_model.change(on_seg_model_change, e_model, [e_thr, e_area, e_info])
    e_refresh.click(lambda: gr.update(choices=core_seg.list_models()), None, e_model)
    e_btn.click(run_segment, [e_model, e_image, e_thr, e_area], [e_over, e_mask, e_verdict, e_detail])
    e_save.click(save_seg_threshold, [e_model, e_thr, e_area], e_msg)
    demo.load(on_seg_model_change, e_model, [e_thr, e_area, e_info])

    d_kind.change(lambda k: gr.update(choices=deploy.list_local(k), value=(deploy.list_local(k) or [None])[0]), d_kind, d_name)
    d_up.click(deploy_upload, [d_url, d_kind, d_name], d_up_msg)
    d_refresh.click(deploy_refresh, d_url, [d_agents, d_agent, d_model, d_status])
    d_go.click(deploy_assign, [d_url, d_agent, d_model], d_go_msg)
    d_res_btn.click(deploy_results, [d_url, d_limit], [d_results, d_res_msg])

    r_load.click(roi_load_sample, r_folder, [r_editor, r_msg])
    r_save.click(roi_save, [r_editor, r_name], [r_out, r_preview])

    x_kind.change(compare_choices, x_kind, [x_cands, x_path1, x_path2])
    x_btn.click(run_compare, [x_kind, x_name, x_path1, x_path2, x_cands, x_roi], [x_log, x_table])


if __name__ == "__main__":
    # server_port=None: 7860 이 사용 중이면 7861, 7862... 로 자동 선택 (포트 충돌로 죽지 않도록)
    demo.launch(server_name="127.0.0.1", server_port=None, inbrowser=True, show_error=True)
