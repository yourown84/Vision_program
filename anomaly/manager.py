"""
관리 서버 (Manager) - 서버 PC 에서 실행
- 여러 라인 PC(에이전트)의 상태를 모으고, 모델을 보관하고, 어느 PC 에 어떤 모델을 배포할지 정한다.
- 라인 PC 의 검사 결과를 수집한다 (재학습 데이터의 출발점).
- 브라우저 대시보드: http://<서버IP>:9000/

실행: start_manager.bat   (환경 변수 MANAGER_PORT 로 포트 변경 가능, 기본 9000)

저장 위치: manager_store/
  models/<kind>/<name>.zip      업로드된 모델
  models/<kind>/<name>.json     모델 정보
  state.json                    에이전트 목록과 배포 지정
  results.jsonl                 검사 결과 (한 줄 = 한 건)
"""
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

BASE = Path(__file__).parent / "manager_store"
MODELS = BASE / "models"
STATE = BASE / "state.json"
RESULTS = BASE / "results.jsonl"
KINDS = ("anomaly", "cls", "seg")
OFFLINE_AFTER = 30  # 초. 이 시간 동안 heartbeat 가 없으면 오프라인으로 표시

app = FastAPI(title="검사 관리 서버", version="0.1")
_lock = threading.Lock()


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"agents": {}, "assignments": {}}


def _save_state(state: dict) -> None:
    BASE.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# 모델 저장소
# ----------------------------------------------------------------------
def _model_meta_path(kind: str, name: str) -> Path:
    return MODELS / kind / f"{name}.json"


@app.get("/api/models")
def list_models():
    out = []
    for kind in KINDS:
        for meta in sorted((MODELS / kind).glob("*.json")) if (MODELS / kind).exists() else []:
            out.append(json.loads(meta.read_text(encoding="utf-8")))
    return {"models": out}


@app.post("/api/models/upload")
async def upload_model(
    kind: str = Form(...),
    name: str = Form(...),
    version: str = Form(...),
    info: str = Form("{}", description="모델 info.json 내용"),
    file: UploadFile = File(..., description="모델 폴더를 압축한 zip"),
):
    if kind not in KINDS:
        raise HTTPException(400, f"kind 는 {KINDS} 중 하나여야 합니다.")
    (MODELS / kind).mkdir(parents=True, exist_ok=True)
    data = await file.read()
    (MODELS / kind / f"{name}.zip").write_bytes(data)
    meta = {
        "kind": kind, "name": name, "version": version, "size_mb": round(len(data) / 2**20, 1),
        "uploaded": time.strftime("%Y-%m-%d %H:%M:%S"), "info": json.loads(info),
    }
    _model_meta_path(kind, name).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


@app.get("/api/models/{kind}/{name}/download")
def download_model(kind: str, name: str):
    p = MODELS / kind / f"{name}.zip"
    if not p.exists():
        raise HTTPException(404, "모델이 없습니다.")
    return FileResponse(p, filename=f"{name}.zip")


# ----------------------------------------------------------------------
# 에이전트
# ----------------------------------------------------------------------
@app.post("/api/agents/heartbeat")
def heartbeat(body: dict):
    agent_id = body.get("agent_id")
    if not agent_id:
        raise HTTPException(400, "agent_id 가 필요합니다.")
    with _lock:
        state = _load_state()
        state["agents"][agent_id] = body | {"last_seen": time.time()}
        _save_state(state)
        return {"assignments": state["assignments"].get(agent_id, {})}


@app.get("/api/agents")
def list_agents():
    state = _load_state()
    now = time.time()
    agents = []
    for aid, a in state["agents"].items():
        agents.append(a | {
            "agent_id": aid,
            "online": now - a.get("last_seen", 0) < OFFLINE_AFTER,
            "last_seen_str": time.strftime("%H:%M:%S", time.localtime(a.get("last_seen", 0))),
            "assignments": state["assignments"].get(aid, {}),
        })
    return {"agents": sorted(agents, key=lambda x: x["agent_id"])}


@app.post("/api/deploy")
def deploy(body: dict):
    """{agent_id, kind, name}  ->  해당 에이전트에 그 모델을 배포하도록 지정. name 이 null 이면 지정 해제."""
    agent_id, kind, name = body.get("agent_id"), body.get("kind"), body.get("name")
    if kind not in KINDS:
        raise HTTPException(400, f"kind 는 {KINDS} 중 하나여야 합니다.")
    with _lock:
        state = _load_state()
        if agent_id not in state["agents"]:
            raise HTTPException(404, f"에이전트를 찾을 수 없습니다: {agent_id}")
        a = state["assignments"].setdefault(agent_id, {})
        if name:
            meta_p = _model_meta_path(kind, name)
            if not meta_p.exists():
                raise HTTPException(404, f"업로드된 모델이 없습니다: {kind}/{name}")
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            a[kind] = {"name": name, "version": meta["version"], "assigned": time.strftime("%Y-%m-%d %H:%M:%S")}
        else:
            a.pop(kind, None)
        _save_state(state)
        return {"agent_id": agent_id, "assignments": a}


# ----------------------------------------------------------------------
# 검사 결과 수집
# ----------------------------------------------------------------------
@app.post("/api/results")
def add_result(body: dict):
    BASE.mkdir(exist_ok=True)
    with _lock, RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/api/results")
def get_results(limit: int = 100, agent_id: str | None = None):
    if not RESULTS.exists():
        return {"results": [], "total": 0}
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    if agent_id:
        rows = [r for r in rows if r.get("agent_id") == agent_id]
    return {"results": rows[-limit:][::-1], "total": len(rows)}


@app.get("/api/summary")
def summary():
    rows = get_results(limit=100000)["results"]
    per_agent: dict[str, dict] = {}
    for r in rows:
        a = per_agent.setdefault(r.get("agent_id", "?"), {"total": 0, "ng": 0})
        a["total"] += 1
        a["ng"] += int(r.get("result") == "NG")
    return {"per_agent": per_agent, "total": len(rows)}


# ----------------------------------------------------------------------
# 대시보드
# ----------------------------------------------------------------------
DASHBOARD = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>검사 관리 서버</title>
<style>
body{font-family:Segoe UI,Malgun Gothic,sans-serif;margin:20px;background:#f6f7f9;color:#222}
h1{font-size:1.5em} h2{font-size:1.15em;margin:24px 0 8px}
table{border-collapse:collapse;background:#fff;width:100%;box-shadow:0 1px 3px #0002}
th,td{padding:7px 10px;border-bottom:1px solid #e5e7eb;text-align:left;font-size:.93em}
th{background:#eef1f5}
.on{color:#2e7d32;font-weight:bold}.off{color:#999}.ng{color:#d32f2f;font-weight:bold}.ok{color:#2e7d32}
select,button{padding:4px 8px;font-size:.9em}
button.p{background:#f97316;color:#fff;border:0;border-radius:4px;cursor:pointer}
.muted{color:#777;font-size:.85em}
</style></head><body>
<h1>검사 관리 서버 <span class="muted" id="clock"></span></h1>
<h2>라인 PC (에이전트)</h2>
<table id="agents"><thead><tr><th>이름</th><th>상태</th><th>마지막 신호</th><th>장치</th>
<th>어노멀리</th><th>분류</th><th>세그</th><th>배포</th></tr></thead><tbody></tbody></table>
<h2>업로드된 모델</h2>
<table id="models"><thead><tr><th>종류</th><th>이름</th><th>버전</th><th>크기</th><th>업로드</th><th>성능</th></tr></thead><tbody></tbody></table>
<h2>최근 검사 결과 <span class="muted" id="summary"></span></h2>
<table id="results"><thead><tr><th>시각</th><th>PC</th><th>종류</th><th>판정</th><th>점수/클래스</th><th>불량 영역</th><th>ms</th><th>파일</th></tr></thead><tbody></tbody></table>
<script>
const KINDS={anomaly:'어노멀리',cls:'분류',seg:'세그'};
let models=[];
function opt(kind,cur){const ms=models.filter(m=>m.kind===kind);
 return `<select id="sel_${kind}"><option value="">(해제)</option>`+ms.map(m=>`<option value="${m.name}" ${cur===m.name?'selected':''}>${m.name} v${m.version}</option>`).join('')+`</select>`;}
async function refresh(){
 const [a,m,r,s]=await Promise.all(['/api/agents','/api/models','/api/results?limit=30','/api/summary'].map(u=>fetch(u).then(x=>x.json())));
 models=m.models;
 document.querySelector('#models tbody').innerHTML=models.map(x=>{const i=x.info||{};const perf=i.metrics?Object.entries(i.metrics).map(([k,v])=>k+'='+(+v).toFixed(3)).join(', '):(i.val_accuracy!=null?'정확도 '+(i.val_accuracy*100).toFixed(1)+'%':(i.val_iou!=null?'IoU '+(i.val_iou*100).toFixed(1)+'%':''));
  return `<tr><td>${KINDS[x.kind]}</td><td>${x.name}</td><td>${x.version}</td><td>${x.size_mb} MB</td><td>${x.uploaded}</td><td>${perf}</td></tr>`}).join('')||'<tr><td colspan=6 class=muted>없음</td></tr>';
 document.querySelector('#agents tbody').innerHTML=a.agents.map(x=>{const L=x.loaded||{};const A=x.assignments||{};
  const cell=k=>`${L[k]||'-'}`+(A[k]&&A[k].name!==L[k]?` <span class=muted>(배포 대기: ${A[k].name})</span>`:'');
  return `<tr data-id="${x.agent_id}"><td>${x.agent_id}<br><span class=muted>${x.host||''}</span></td><td class="${x.online?'on':'off'}">${x.online?'온라인':'오프라인'}</td><td>${x.last_seen_str}</td><td>${x.device||'-'}</td>
  <td>${cell('anomaly')}</td><td>${cell('cls')}</td><td>${cell('seg')}</td>
  <td>${['anomaly','cls','seg'].map(k=>`${KINDS[k]} ${opt(k,(A[k]||{}).name)}`).join('<br>')}<br><button class=p onclick="deploy('${x.agent_id}',this)">배포 지정</button></td></tr>`}).join('')||'<tr><td colspan=8 class=muted>연결된 에이전트 없음</td></tr>';
 document.querySelector('#results tbody').innerHTML=r.results.map(x=>`<tr><td>${x.time||''}</td><td>${x.agent_id||''}</td><td>${KINDS[x.kind]||x.kind||''}</td>
  <td class="${x.result==='NG'?'ng':'ok'}">${x.result||x.label||''}</td><td>${x.score!=null?x.score:(x.confidence!=null?x.label+' '+x.confidence+'%':(x.defect_ratio!=null?x.defect_ratio+'%':''))}</td>
  <td>${(x.boxes||[]).length}</td><td>${x.elapsed_ms||''}</td><td>${x.filename||''}</td></tr>`).join('')||'<tr><td colspan=8 class=muted>없음</td></tr>';
 document.getElementById('summary').textContent=`총 ${s.total}건 · `+Object.entries(s.per_agent).map(([k,v])=>`${k}: NG ${v.ng}/${v.total}`).join(' · ');
 document.getElementById('clock').textContent=new Date().toLocaleTimeString();
}
async function deploy(id,btn){const tr=btn.closest('tr');
 for(const k of ['anomaly','cls','seg']){const name=tr.querySelector('#sel_'+k).value||null;
  const res=await fetch('/api/deploy',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({agent_id:id,kind:k,name})});
  if(!res.ok){alert(await res.text());return;}}
 refresh();}
refresh();setInterval(refresh,5000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD


if __name__ == "__main__":
    import uvicorn

    BASE.mkdir(exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MANAGER_PORT", "9000")))
